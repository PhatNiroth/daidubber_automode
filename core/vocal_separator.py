import os
import sys
import subprocess
import shutil
import threading
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as signal
from utils.gpu_utils import setup_cuda_dlls
from utils.logger import logger

setup_cuda_dlls()

# GPU Semaphore to ensure only 1 separation runs on GPU at a time to prevent VRAM Out-of-Memory
_gpu_separation_lock = threading.Semaphore(1)

# Module level cache for BS-Roformer to avoid re-loading weights for every video
_cached_roformer_separator = None
_cached_roformer_model_name = None

class VocalSeparator:
    """Separates vocal track from background music (BGM) and sound effects using BS-Roformer AI or Demucs on CUDA GPU."""
    
    def __init__(self, engine: str = "roformer", model_name: Optional[str] = None, device: str = "cuda"):
        self.engine = engine.lower() if engine else "roformer"
        self.device = device
        if self.engine == "roformer":
            self.model_name = model_name or "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
        else:
            self.model_name = model_name or "htdemucs"

    def extract_audio_from_video(self, video_path: str, output_wav: str) -> bool:
        """Extract high quality stereo WAV audio from video using FFmpeg."""
        from utils.file_utils import has_audio_stream
        if not has_audio_stream(video_path):
            video_name = Path(video_path).name
            logger.error(
                f"❌ វីដេអូ '{video_name}' គ្មានខ្សែសំឡេង (No Audio Track) ទេ! "
                f"ឯកសារនេះជាប្រភេទ Video-Only (ដូចជា YouTube Format .f399)។ "
                f"សូមជ្រើសរើស Folder ឬឯកសារវីដេអូដែលមានសំឡេងដើម (Full Audio & Video)!"
            )
            return False

        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", "44100",
                "-ac", "2",
                str(output_wav)
            ]
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace"
            )
            if result.returncode != 0:
                logger.error(f"FFmpeg audio extraction error: {result.stderr}")
                return False
            return True
        except Exception as e:
            logger.error(f"Failed to extract audio from video: {e}")
            return False

    @staticmethod
    def clean_foley_and_bgm(vocals_path: str, bgm_path: str, output_path: str) -> bool:
        """
        Pure Foley Impact Gate:
        Eliminates 100% of continuous Chinese songs, piano, Guzheng, strings, and musical chords.
        Selectively isolates and preserves physical action impacts:
        1. Sword clashes, metal blade ringing, weapon whooshes (HF energy impulse > 1800Hz).
        2. Horse hooves, foot stomps, body punches, falls (LF energy impulse < 200Hz).
        Between strikes, continuous music and singing are 100% MUTED (0% volume).
        """
        try:
            sr, data = wavfile.read(bgm_path)
            sr_v, v = wavfile.read(vocals_path)

            min_len = min(len(data), len(v))
            data = data[:min_len]
            v = v[:min_len]

            data_f = data.astype(np.float32) / 32768.0
            v_f = v.astype(np.float32) / 32768.0

            mono = data_f.mean(axis=1) if len(data_f.shape) > 1 else data_f
            v_mono = v_f.mean(axis=1) if len(v_f.shape) > 1 else v_f

            # 1. High-frequency strikes (> 1800Hz: sword clashes, blade slicing, metal impacts)
            sos_hf = signal.butter(4, 1800, 'highpass', fs=sr, output='sos')
            mono_hf = signal.sosfilt(sos_hf, mono)

            win_len = int(0.005 * sr)  # 5ms fast attack window
            kernel = np.ones(win_len) / win_len
            energy_hf = signal.convolve(mono_hf**2, kernel, mode='same')
            diff_hf = np.diff(energy_hf, prepend=0)

            # Threshold for sharp physical attacks
            pos_diff = diff_hf[diff_hf > 0]
            thresh_hf = np.percentile(pos_diff, 99.0) if len(pos_diff) > 0 else 1e-4
            peaks_hf, _ = signal.find_peaks(diff_hf, height=thresh_hf * 1.6, distance=int(0.060 * sr))

            # 2. Low-frequency impacts (< 200Hz: horse hooves, foot stomps, body punches)
            sos_lf = signal.butter(4, 200, 'lowpass', fs=sr, output='sos')
            mono_lf = signal.sosfilt(sos_lf, mono)
            energy_lf = signal.convolve(mono_lf**2, kernel, mode='same')
            diff_lf = np.diff(energy_lf, prepend=0)

            pos_diff_lf = diff_lf[diff_lf > 0]
            thresh_lf = np.percentile(pos_diff_lf, 98.8) if len(pos_diff_lf) > 0 else 1e-4
            peaks_lf, _ = signal.find_peaks(diff_lf, height=thresh_lf * 1.8, distance=int(0.080 * sr))

            # 3. Build composite gate envelope (250ms envelope per physical strike)
            gate = np.zeros(len(mono), dtype=np.float32)

            gate_len_hf = int(0.250 * sr)
            env_hf = np.ones(gate_len_hf, dtype=np.float32)
            att_hf = int(0.005 * sr)
            dec_hf = int(0.120 * sr)
            env_hf[:att_hf] = np.linspace(0.0, 1.0, att_hf)
            env_hf[-dec_hf:] = np.linspace(1.0, 0.0, dec_hf)

            for p in peaks_hf:
                start = max(0, p - att_hf)
                end = min(len(gate), start + gate_len_hf)
                gate[start:end] = np.maximum(gate[start:end], env_hf[:end-start])

            gate_len_lf = int(0.200 * sr)
            env_lf = np.ones(gate_len_lf, dtype=np.float32)
            att_lf = int(0.008 * sr)
            dec_lf = int(0.100 * sr)
            env_lf[:att_lf] = np.linspace(0.0, 1.0, att_lf)
            env_lf[-dec_lf:] = np.linspace(1.0, 0.0, dec_lf)

            for p in peaks_lf:
                start = max(0, p - att_lf)
                end = min(len(gate), start + gate_len_lf)
                gate[start:end] = np.maximum(gate[start:end], env_lf[:end-start])

            # 4. Vocal suppression: if human singing or dialogue is active, clamp gate to 0 to prevent voice leaks
            sos_mid = signal.butter(4, [250, 3200], 'bandpass', fs=sr, output='sos')
            v_mid = signal.sosfilt(sos_mid, v_mono)
            win_voc = int(0.060 * sr)
            k_voc = np.hanning(win_voc) / np.hanning(win_voc).sum()
            v_env = signal.convolve(np.abs(v_mid), k_voc, mode='same')
            voc_active = v_env > 0.015
            gate[voc_active] = np.minimum(gate[voc_active], 0.01)

            # 5. Multiply gate (0.00 continuous Chinese music/song, 1.00 on strikes)
            gate_2d = gate[:, None] if len(data_f.shape) > 1 else gate
            out_clean = data_f * gate_2d

            wavfile.write(output_path, sr, np.clip(out_clean * 32768.0, -32768, 32767).astype(np.int16))
            open_pct = np.mean(gate > 0.05) * 100
            logger.info(f"⚔️ Foley Impact Gate: ចាប់បានស្នូរកាប់ដាវ/សេះ/វាយគ្នា {len(peaks_hf) + len(peaks_lf)} កន្លែង (លុបបទចម្រៀងចិនបាន {100-open_pct:.1f}%)")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Clean Foley filtering failed ({e}), keeping original BGM.")
            return False

    def separate_vocals(self, audio_wav: str, output_dir: str, clean_foley: bool = False) -> Tuple[Optional[str], Optional[str]]:
        """
        Separates vocal track from background music and sound effects.
        Uses BS-Roformer AI by default (gold standard vocal isolation, SDR > 12.9 dB)
        with automatic fallback to Demucs if needed.
        """
        if self.engine == "roformer":
            try:
                v_path, b_path = self.separate_roformer(audio_wav, output_dir)
                if v_path and b_path and os.path.exists(v_path) and os.path.exists(b_path):
                    return v_path, b_path
                logger.warning("⚠️ BS-Roformer output files missing, attempting Demucs fallback...")
            except Exception as e:
                logger.warning(f"⚠️ BS-Roformer separation failed ({e}), falling back to Demucs...")

        return self.separate_demucs(audio_wav, output_dir, clean_foley=clean_foley)

    def separate_roformer(self, audio_wav: str, output_dir: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Runs BS-Roformer AI on NVIDIA CUDA GPU for clean studio-grade vocal removal.
        Isolates singing and speech cleanly without any choppy or stuttering audio.
        """
        global _cached_roformer_separator, _cached_roformer_model_name
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        with _gpu_separation_lock:
            models_dir = os.environ.get("AUDIO_SEPARATOR_MODELS_DIR", "D:/tmp/audio-separator-models")
            if not os.path.exists(models_dir):
                os.makedirs(models_dir, exist_ok=True)

            from audio_separator.separator import Separator

            if _cached_roformer_separator is None or _cached_roformer_model_name != self.model_name:
                logger.info(f"⚡ កំពុងផ្ទុកម៉ូដែល BS-Roformer AI ({self.model_name}) លើ GPU {self.device}...")
                sep = Separator(
                    output_dir=str(output_path),
                    output_format="WAV",
                    model_file_dir=models_dir,
                    use_autocast=False
                )
                sep.load_model(self.model_name)
                _cached_roformer_separator = sep
                _cached_roformer_model_name = self.model_name
            else:
                sep = _cached_roformer_separator
                sep.output_dir = str(output_path)
                if hasattr(sep, "model_instance") and sep.model_instance:
                    sep.model_instance.output_dir = str(output_path)

            logger.info("⚡ កំពុងបំបែកសំឡេងចម្រៀង/និយាយ និង BGM/Sound FX ដោយ BS-Roformer AI (គុណភាព Studio)...")
            output_files = sep.separate(
                str(audio_wav),
                custom_output_names={"Vocals": "vocals", "Instrumental": "no_vocals"}
            )

            cand_v = output_path / "vocals.wav"
            cand_nv = output_path / "no_vocals.wav"
            vocals_path = str(cand_v) if cand_v.exists() else None
            no_vocals_path = str(cand_nv) if cand_nv.exists() else None

            if not (vocals_path and no_vocals_path) and output_files:
                for f in output_files:
                    p = Path(f) if Path(f).is_absolute() else output_path / f
                    if not p.exists():
                        continue
                    stem_l = p.stem.lower()
                    if "vocals" in stem_l and "no_vocals" not in stem_l and "instrumental" not in stem_l:
                        vocals_path = str(p)
                    elif "no_vocals" in stem_l or "instrumental" in stem_l:
                        no_vocals_path = str(p)

            if vocals_path and no_vocals_path:
                logger.success("✅ បានបំបែកសំឡេង BGM & Foley FX តាមរយៈ BS-Roformer AI ស្អាតល្អ ១០០% (គ្មានសំឡេងចម្រៀង/និយាយចិន និងមិនដាច់ៗ)!")
                return str(vocals_path), str(no_vocals_path)
            else:
                logger.error(f"❌ រកមិនឃើញឯកសារបំបែកសំឡេងរបស់ Roformer ក្នុង {output_path}")
                return None, None

    def separate_demucs(self, audio_wav: str, output_dir: str, clean_foley: bool = False) -> Tuple[Optional[str], Optional[str]]:
        """
        Runs Demucs on NVIDIA CUDA GPU for vocal removal.
        Uses GPU semaphore lock to prevent CUDA Out-of-Memory during multi-video parallel dubbing.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        audio_name = Path(audio_wav).stem

        # Acquire GPU lock for separation stage
        with _gpu_separation_lock:
            logger.info(f"⚡ កំពុងបំបែកសំឡេង BGM លើ GPU {self.device} (Demucs {self.model_name})...")

            demucs_exe = shutil.which("demucs")
            base_cmd = [demucs_exe] if demucs_exe else [sys.executable, "-m", "demucs"]

            cmd = base_cmd + [
                "--two-stems", "vocals",
                "-n", self.model_name,
                "-d", self.device,
                "--shifts", "1",
                "-o", str(output_path),
                str(audio_wav)
            ]

            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"

            try:
                process = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace"
                )

                output_lines = []
                for line in process.stdout:
                    line_str = line.strip()
                    output_lines.append(line_str)
                    if "%" in line_str or "Separating" in line_str:
                        logger.info(f"Demucs: {line_str}")

                process.wait()

                if process.returncode != 0:
                    err_tail = "\n".join(output_lines[-8:])
                    logger.warning(f"Demucs returned code {process.returncode} ({err_tail}), attempting CPU fallback...")
                    if self.device == "cuda":
                        fallback_cmd = base_cmd + [
                            "--two-stems", "vocals",
                            "-n", self.model_name,
                            "-d", "cpu",
                            "--shifts", "0",
                            "-o", str(output_path),
                            str(audio_wav)
                        ]
                        sub_res = subprocess.run(
                            fallback_cmd,
                            env=env,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            encoding="utf-8",
                            errors="replace"
                        )
                        if sub_res.returncode != 0:
                            logger.error(f"Demucs CPU fallback error: {sub_res.stderr}")
                            return None, None

                model_folder = output_path / self.model_name / audio_name
                vocals_path = model_folder / "vocals.wav"
                no_vocals_path = model_folder / "no_vocals.wav"

                if vocals_path.exists() and no_vocals_path.exists():
                    final_bgm_path = str(no_vocals_path)
                    if clean_foley:
                        clean_bgm_path = str(model_folder / "no_vocals_clean.wav")
                        if self.clean_foley_and_bgm(str(vocals_path), str(no_vocals_path), clean_bgm_path):
                            final_bgm_path = clean_bgm_path
                            logger.success("✅ បានលុបបទចម្រៀងចិន និងរក្សាទុកសំឡេងកាប់ដាវ សំឡេងសេះ សំឡេងវាយគ្នា (Foley FX) ១០០%!")

                    logger.success("បំបែកសំឡេង BGM/Sound FX រួចរាល់ស្អាតល្អ គ្មានសំឡេងរំខាន!")
                    return str(vocals_path), final_bgm_path
                else:
                    logger.error(f"Missing demucs output files in {model_folder}")
                    return None, None

            except Exception as e:
                logger.error(f"Demucs separation error: {e}")
                return None, None
