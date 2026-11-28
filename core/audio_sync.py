import os
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from pydub import AudioSegment
from utils.logger import logger
from utils.file_utils import get_media_duration

def trim_tts_silence(audio: AudioSegment, pad_start_ms: int = 45, pad_end_ms: int = 90) -> AudioSegment:
    """
    Trims Edge-TTS leading (~250ms) and trailing (~900ms) dead air while preserving
    natural speech onset (pad_start_ms) and decay (pad_end_ms).
    Uses conservative -62dBFS threshold so soft initial Khmer consonants are NEVER cut off.
    """
    try:
        from pydub.silence import detect_leading_silence
        # Trim leading dead air with gentle threshold (-62dBFS)
        start_trim = detect_leading_silence(audio, silence_threshold=-62.0)
        start_pos = max(0, start_trim - pad_start_ms)
        trimmed = audio[start_pos:]

        # Trim trailing dead air with gentle threshold (-62dBFS)
        rev = trimmed.reverse()
        end_trim = detect_leading_silence(rev, silence_threshold=-62.0)
        end_pos = max(0, end_trim - pad_end_ms)
        clean = rev[end_pos:].reverse()
        # Safety guard: if trimming was too aggressive, return original audio so no speech is lost
        if len(clean) < 200 and len(audio) >= 200:
            return audio
        return clean
    except Exception as e:
        logger.warning(f"Error trimming TTS silence: {e}")
        return audio

class AudioSyncEngine:
    """
    Synchronizes translated Khmer speech with video actions and timestamps.
    Prevents any audio cut-off through multi-threaded time-stretching,
    preserves background audio/sound effects (BGM), and applies dynamic ducking.
    """

    def __init__(
        self,
        enable_time_stretch: bool = True,
        max_stretch_ratio: float = 1.30,
        bgm_volume: float = 0.85,
        voice_volume: float = 1.35,
        ducking_factor: float = 0.48
    ):
        self.enable_time_stretch = enable_time_stretch
        # Safe acoustic ceiling: Khmer syllables remain clear and natural up to 1.35x
        self.max_stretch_ratio = min(1.35, max(1.0, max_stretch_ratio))
        self.bgm_volume = bgm_volume
        self.voice_volume = voice_volume
        self.ducking_factor = ducking_factor

    def time_stretch_audio(self, input_audio: str, output_audio: str, speed_factor: float) -> bool:
        """
        Stretches or speeds up audio without altering pitch using FFmpeg atempo.
        Supports speed factors up to 3.0x (e.g. 2.5x) using chained atempo filters.
        speed_factor > 1.0 means faster (shorter duration).
        """
        speed_factor = max(0.5, min(3.0, speed_factor))

        # Build chained atempo filter (FFmpeg atempo is bounded to [0.5, 2.0] per filter)
        filters = []
        rem = speed_factor
        while rem > 2.0:
            filters.append("atempo=2.0")
            rem /= 2.0
        while rem < 0.5:
            filters.append("atempo=0.5")
            rem /= 0.5
        filters.append(f"atempo={rem:.4f}")
        filter_str = ",".join(filters)

        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", str(input_audio),
                "-filter:a", filter_str,
                "-vn",
                str(output_audio)
            ]
            res = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace"
            )
            return res.returncode == 0
        except Exception as e:
            logger.error(f"Error time-stretching audio: {e}")
    @staticmethod
    def enhance_dialogue_clarity(input_wav: str, output_wav: str) -> bool:
        """
        Applies professional broadcast speech clarity filtering via FFmpeg:
        1. highpass=f=85 (eliminates low-end rumble and boomy mud)
        2. equalizer=f=220:t=q:w=1.0:g=1.8 (adds natural chest warmth, fullness, and broadcast studio body)
        3. equalizer=f=3500:t=q:w=1.2:g=3.2 (enhances consonant presence, clarity, and articulation)
        4. loudnorm=I=-16:TP=-1.5:LRA=11 (broadcast-standard loudness leveling so dialogue is clear and sits in front)
        """
        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", str(input_wav),
                "-af", "highpass=f=85,equalizer=f=220:t=q:w=1.0:g=1.8,equalizer=f=3500:t=q:w=1.2:g=3.2,loudnorm=I=-16:TP=-1.5:LRA=11",
                "-vn",
                str(output_wav)
            ]
            res = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace"
            )
            return res.returncode == 0 and os.path.exists(output_wav)
        except Exception as e:
            logger.warning(f"Voice clarity filter warning ({e}), continuing with raw master...")
            return False

    def build_synchronized_dialogue_track(
        self,
        segments: List[Dict[str, Any]],
        total_duration: float,
        temp_dir: str
    ) -> Optional[Tuple[str, List[Tuple[int, int]]]]:
        """
        Places each Khmer TTS audio segment onto a timeline matching exact timestamps.
        Trims dead air, applies gentle parallel time-stretching within natural limits,
        and enforces sequential placement so speech from previous segments is NEVER cut off!
        """
        temp_path = Path(temp_dir)
        temp_path.mkdir(parents=True, exist_ok=True)

        total_ms = int(total_duration * 1000) + 2000
        master_track = AudioSegment.silent(duration=total_ms, frame_rate=44100)

        logger.info(f"⚡ កំពុងតម្រឹមចង្វាក់សំឡេង និងល្បឿននិយាយល្បឿនលឿន (Zero-Cut Audio Sync)...")
        num_segments = len(segments)

        # 1. First Pass: Trim dead air from all TTS audio segments to obtain true speech duration
        cleaned_audio_map = {}
        for i, seg in enumerate(segments):
            audio_path = seg.get("audio_file")
            if not audio_path or not os.path.exists(audio_path):
                continue
            try:
                raw_audio = AudioSegment.from_file(audio_path)
                clean_audio = trim_tts_silence(raw_audio, pad_start_ms=50, pad_end_ms=90)
                clean_file = str(temp_path / f"clean_seg_{i:04d}.wav")
                clean_audio.export(clean_file, format="wav")
                cleaned_audio_map[i] = (clean_audio, clean_file)
            except Exception as e:
                logger.warning(f"Error preparing TTS audio for segment {i}: {e}")
                try:
                    clean_file = str(temp_path / f"clean_seg_{i:04d}.wav")
                    raw_audio.export(clean_file, format="wav")
                    cleaned_audio_map[i] = (raw_audio, clean_file)
                except Exception:
                    pass

        # 2. Second Pass: Calculate safe time-stretch tasks targeting the available dialogue window
        safe_max_stretch = min(1.65, max(1.0, self.max_stretch_ratio * 1.25))
        stretch_tasks = []

        for i, seg in enumerate(segments):
            if i not in cleaned_audio_map:
                continue
            clean_audio, clean_file = cleaned_audio_map[i]

            start_sec = max(0.0, float(seg.get("start", 0.0)))
            end_sec = max(start_sec + 0.4, float(seg.get("end", start_sec + 0.4)))
            nominal_dur = end_sec - start_sec

            # Available duration before the NEXT subtitle line starts:
            # A line should fit before the next speaker begins to prevent overlapping speech
            if i + 1 < len(segments):
                next_start = float(segments[i + 1].get("start", end_sec))
                if next_start > start_sec + 0.35:
                    available_dur = next_start - start_sec - 0.035
                    target_dur = max(0.6, min(nominal_dur, available_dur))
                else:
                    target_dur = max(0.6, nominal_dur)
            else:
                target_dur = max(0.6, nominal_dur)

            actual_dur = len(clean_audio) / 1000.0

            if actual_dur > target_dur and self.enable_time_stretch:
                speed_ratio = actual_dur / target_dur
                clamped_ratio = min(safe_max_stretch, speed_ratio)
                if clamped_ratio > 1.04:
                    stretched_file = str(temp_path / f"stretched_seg_{seg.get('id', i):04d}.wav")
                    stretch_tasks.append((clean_file, stretched_file, clamped_ratio, i))

        # 3. Third Pass: Run parallel time-stretching with ThreadPoolExecutor
        stretched_map = {}
        if stretch_tasks:
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [
                    executor.submit(self.time_stretch_audio, src, dst, ratio)
                    for src, dst, ratio, idx in stretch_tasks
                ]
                for (src, dst, ratio, idx), fut in zip(stretch_tasks, futures):
                    if fut.result():
                        stretched_map[idx] = dst

        # 4. Fourth Pass: Subtitle-Anchored Audio Placement (Zero Overlap & Zero Lost Audio)
        # Speech onset is anchored strictly to subtitle timestamps without voices overlapping!
        import math
        db_change = 20 * math.log10(max(0.1, self.voice_volume)) if self.voice_volume != 1.0 else 0
        speech_intervals = []
        current_pos_ms = 0

        for i, seg in enumerate(segments):
            if i not in cleaned_audio_map:
                continue

            if i in stretched_map and os.path.exists(stretched_map[i]):
                segment_audio = AudioSegment.from_file(stretched_map[i])
            else:
                segment_audio = cleaned_audio_map[i][0]

            if db_change != 0:
                segment_audio = segment_audio + db_change

            nominal_start_ms = int(max(0.0, float(seg.get("start", 0.0))) * 1000)

            # Start time: anchored to mouth and scene action, never before previous audio has ended
            start_ms = max(nominal_start_ms, current_pos_ms)

            # Full speech integrity: NEVER slice or fade out segment audio mid-speech or at tail
            end_ms = start_ms + len(segment_audio)
            # Enforce clean non-overlapping boundary (at least 15ms natural acoustic gap)
            current_pos_ms = end_ms + 15

            # Dynamically ensure master track has sufficient length
            if end_ms > len(master_track):
                master_track = master_track + AudioSegment.silent(
                    duration=(end_ms - len(master_track) + 3000),
                    frame_rate=master_track.frame_rate
                )

            master_track = master_track.overlay(segment_audio, position=start_ms)
            speech_intervals.append((start_ms, end_ms))

        # 5. Export master track: keep full video duration without truncating trailing speech
        total_ms = int(total_duration * 1000)
        target_len = max(total_ms, current_pos_ms + 250)
        master_track = master_track[:target_len]

        raw_dialogue_output = temp_path / "khmer_dialogue_raw.wav"
        master_track.export(str(raw_dialogue_output), format="wav")

        dialogue_output = temp_path / "khmer_dialogue_master.wav"
        if AudioSyncEngine.enhance_dialogue_clarity(str(raw_dialogue_output), str(dialogue_output)):
            dialogue_final_path = str(dialogue_output)
            logger.success("🔊 បានបង្កើនកម្រិតច្បាស់នៃសំឡេងនិយាយខ្មែរ (Dialogue Clarity & Presence Boost: High-Pass 85Hz + Presence +3.5dB + Leveling)!")
        else:
            dialogue_final_path = str(raw_dialogue_output)

        logger.success("បង្កើតខ្សែសំឡេងនិយាយខ្មែរ (Dialogue Track) ត្រូវតាមចង្វាក់រួចរាល់ និងគ្មានការដាច់សំឡេង (Zero Cut-Off)!")
        return dialogue_final_path, speech_intervals

    @staticmethod
    def apply_dynamic_ducking(
        audio: AudioSegment,
        speech_intervals: List[Tuple[int, int]],
        duck_factor: float = 0.22,
        base_volume: float = 1.0,
        fade_ms: int = 200
    ) -> AudioSegment:
        """
        Dynamically attenuates background audio during speech intervals with smooth crossfades,
        leaving non-speech intervals (pauses, music breaks, action scenes) at full 100% volume.
        """
        if not speech_intervals or duck_factor >= 1.0:
            if base_volume != 1.0:
                import math
                db = 20 * math.log10(max(0.05, base_volume))
                return audio + db
            return audio

        import numpy as np
        sample_rate = audio.frame_rate
        num_channels = audio.channels
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        num_frames = len(samples) // num_channels

        gain = np.full(num_frames, base_volume, dtype=np.float32)
        duck_gain = base_volume * duck_factor
        fade_samples = int((fade_ms / 1000.0) * sample_rate)

        for start_ms, end_ms in speech_intervals:
            s_idx = max(0, int((start_ms / 1000.0) * sample_rate))
            e_idx = min(num_frames, int((end_ms / 1000.0) * sample_rate))
            if s_idx >= e_idx:
                continue

            # Smooth fade-in into ducked volume
            f_in_start = max(0, s_idx - fade_samples)
            if s_idx > f_in_start:
                ramp = np.linspace(base_volume, duck_gain, s_idx - f_in_start, dtype=np.float32)
                gain[f_in_start:s_idx] = np.minimum(gain[f_in_start:s_idx], ramp)

            # Ducked region during speech
            gain[s_idx:e_idx] = np.minimum(gain[s_idx:e_idx], duck_gain)

            # Smooth fade-out back to base volume
            f_out_end = min(num_frames, e_idx + fade_samples)
            if f_out_end > e_idx:
                ramp = np.linspace(duck_gain, base_volume, f_out_end - e_idx, dtype=np.float32)
                gain[e_idx:f_out_end] = np.minimum(gain[e_idx:f_out_end], ramp)

        if num_channels > 1:
            gain = np.repeat(gain, num_channels)

        ducked_samples = np.clip(samples * gain, -32768, 32767).astype(np.int16)
        return AudioSegment(
            ducked_samples.tobytes(),
            frame_rate=sample_rate,
            sample_width=audio.sample_width,
            channels=num_channels
        )

    def mix_final_audio(
        self,
        dialogue_wav: str,
        bgm_wav: Optional[str],
        output_wav: str,
        mode: str = "separate_bgm",
        speech_intervals: Optional[List[Tuple[int, int]]] = None
    ) -> bool:
        """
        Mixes master dialogue with background music/effects according to the 2 modes:
        1. 'separate_bgm': Keeps pure Demucs BGM/FX 100% without Chinese speech, subtle ducking during speech.
        2. 'mute_original': Mutes 100% of original audio, outputting only Khmer dialogue.
        """
        try:
            logger.info(f"🎛️ កំពុងលាយសំឡេង Master Audio (របៀបចាត់ចែង: {mode})...")

            # Condition 2: Mute original audio 100%
            if mode == "mute_original":
                dialogue = AudioSegment.from_file(dialogue_wav)
                dialogue.export(output_wav, format="wav")
                logger.success("✅ រក្សាទុកតែសំឡេងខ្មែរ (Mute សំឡេងដើមទាំងអស់ ១០០%) រួចរាល់។")
                return True

            # If BGM is missing despite not being mute_original, fallback to dialogue only with warning
            if not bgm_wav or not os.path.exists(bgm_wav):
                logger.warning("⚠️ រកមិនឃើញ BGM track ទេ កំពុងរក្សាទុកតែសំឡេងខ្មែរ...")
                dialogue = AudioSegment.from_file(dialogue_wav)
                dialogue.export(output_wav, format="wav")
                return True

            dialogue = AudioSegment.from_file(dialogue_wav)
            bgm = AudioSegment.from_file(bgm_wav)

            # Match lengths
            if len(bgm) < len(dialogue):
                silence = AudioSegment.silent(duration=len(dialogue) - len(bgm), frame_rate=bgm.frame_rate)
                bgm = bgm + silence
            else:
                bgm = bgm[:len(dialogue)]

            # Extract speech intervals if not provided
            if speech_intervals is None:
                try:
                    from pydub.silence import detect_nonsilent
                    speech_intervals = detect_nonsilent(dialogue, min_silence_len=250, silence_thresh=-45)
                except Exception:
                    speech_intervals = []

            # Condition 1: Keep BGM & Sound FX (Demucs / BS-Roformer AI vocal removal)
            # Cinema dynamic ducking: Keeps Foley FX (swords, hits, horses) and BGM audible at ~48% during dialogue,
            # and full impact (85%) during action and speech pauses.
            duck_factor = min(0.65, max(0.35, self.ducking_factor))
            base_volume = max(0.60, self.bgm_volume)
            bgm_processed = self.apply_dynamic_ducking(
                bgm,
                speech_intervals,
                duck_factor=duck_factor,
                base_volume=base_volume,
                fade_ms=150
            )
            final_mix = bgm_processed.overlay(dialogue)
            final_mix.export(output_wav, format="wav")
            logger.success("✅ រក្សាទុកសំឡេង Sound FX (កាប់ដាវ សេះ វាយគ្នា) និងលាយសំឡេងខ្មែរបានជោគជ័យ!")
            return True

        except Exception as e:
            logger.error(f"Error mixing final audio: {e}")
            return False
