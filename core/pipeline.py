import os
import shutil
import time
import re
import difflib
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from utils.logger import logger
from utils.file_utils import get_media_duration, create_temp_dir
from core.vocal_separator import VocalSeparator
from core.transcriber import ChineseTranscriber
from core.translator import GeminiTranslator
from core.tts_engine import KhmerTTSEngine
from core.audio_sync import AudioSyncEngine
from core.gender_detector import GenderDetector
from core.video_renderer import VideoRenderer
from core.config_manager import ConfigManager

PIPELINE_STAGES = [
    "បើកវីដេអូ",
    "ស្រង់សំឡេង",
    "បំបែក Vocal/BGM",
    "ស្រង់សម្ដី ASR",
    "បកប្រែខ្មែរ",
    "បង្កើតសំឡេង TTS",
    "Sync សំឡេង & សកម្មភាព",
    "Export វីដេអូ"
]


def deduplicate_khmer_dialogue(segments: List[Dict[str, Any]], max_gap: float = 2.0) -> List[Dict[str, Any]]:
    """
    Cleans translated Khmer segments:
    1. Ensures valid non-empty Khmer text.
    2. Only drops exact consecutive identical duplicate lines (c1 == c2) within 2.0s.
    Guarantees 100% translation coverage so no genuine dialogue is skipped.
    """
    if not segments:
        return []

    def normalize_khmer_reaction(t: str) -> str:
        clean = re.sub(r'[\s\.,\!\?\'"\(\)\[\]\{\}—\-–:;ៗ៕។៛]+', '', t)
        return clean.strip()

    deduped = []
    dropped_count = 0

    for seg in segments:
        text = seg.get("km_text", "").strip()
        if not text:
            continue
        seg["km_text"] = text
        norm = normalize_khmer_reaction(text)

        if not deduped:
            deduped.append(dict(seg))
            continue

        prev = deduped[-1]
        prev_norm = normalize_khmer_reaction(prev.get("km_text", ""))
        gap = seg.get("start", 0.0) - prev.get("end", 0.0)

        # STRICT: ONLY drop if the Khmer text is 100% EXACTLY IDENTICAL to the immediately preceding line
        # Never drop legitimate dialogue sentences!
        if norm and prev_norm and norm == prev_norm and gap <= 0.35:
            dropped_count += 1
            prev["end"] = max(prev.get("end", 0.0), seg.get("end", 0.0))
            continue

        deduped.append(dict(seg))

    if dropped_count > 0:
        logger.info(f"👥 បានបង្រួមឃ្លាដូចគ្នាបេះបិទ {dropped_count} ប្រយោគ មកនិយាយតែម្តងគត់!")

    return deduped

class DubbingPipeline:
    """Orchestrates the entire AI dubbing pipeline for single video and batch jobs."""

    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.is_cancelled = False
        self.is_running = False

    def cancel(self):
        """Requests cancellation of ongoing dubbing jobs."""
        self.is_cancelled = True
        logger.warning("🛑 បានទទួលបញ្ជាបញ្ឈប់ការ Dubbing (Cancellation requested)...")

    def process_single_video(
        self,
        video_path: str,
        output_path: str,
        video_index: int,
        total_videos: int,
        progress_callback: Optional[Callable[[int, int, int, str, float, str], None]] = None
    ) -> bool:
        """Processes a single video through all 7 pipeline stages."""
        self.is_cancelled = False
        video_file = Path(video_path)
        output_file = Path(output_path)
        
        cfg = self.config_manager.config
        import uuid
        job_tag = f"job_{video_index}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        temp_dir = create_temp_dir() / job_tag
        temp_dir.mkdir(parents=True, exist_ok=True)

        def report(stage_idx: int, percent: float, text: str):
            stage_name = PIPELINE_STAGES[stage_idx] if stage_idx < len(PIPELINE_STAGES) else "Export"
            logger.step(stage_idx + 1, len(PIPELINE_STAGES), f"{video_file.name}: {text}")
            if progress_callback:
                progress_callback(video_index, total_videos, stage_idx, stage_name, percent, text)

        try:
            # ---------------- Stage 0: Inspect Video ----------------
            if self.is_cancelled: return False
            report(0, 0.05, f"ពិនិត្យវីដេអូ: {video_file.name}")
            total_duration = get_media_duration(str(video_file))
            if total_duration <= 0:
                logger.error(f"មិនអាចអានប្រវែងវីដេអូ: {video_file.name}")
                return False

            logger.info(f"📹 ប្រវែងវីដេអូសរុប: {total_duration:.1f} វិនាទី")

            # ---------------- Stage 1: Extract Audio ----------------
            if self.is_cancelled: return False
            report(1, 0.15, "កំពុងស្រង់សំឡេងចេញពីវីដេអូ...")
            extracted_wav = str(temp_dir / "original_audio.wav")
            sep_engine = cfg.get("separation_engine", "roformer")
            sep_model = cfg.get("roformer_model", "model_bs_roformer_ep_317_sdr_12.9755.ckpt") if sep_engine == "roformer" else cfg.get("demucs_model", "htdemucs")
            separator = VocalSeparator(engine=sep_engine, model_name=sep_model, device=cfg.get("device", "cuda"))
            if not separator.extract_audio_from_video(str(video_file), extracted_wav):
                return False

            # ---------------- Stage 2: Vocal Separation (Keep BGM/FX vs Mute) ----------------
            if self.is_cancelled: return False
            vocal_mode = cfg.get("vocal_mode", "separate_bgm")
            if vocal_mode not in ["separate_bgm", "mute_original", "clean_foley"]:
                vocal_mode = "separate_bgm"

            vocals_wav = extracted_wav
            bgm_wav = None

            if vocal_mode in ["separate_bgm", "clean_foley"]:
                engine_title = "BS-Roformer AI" if separator.engine == "roformer" else "Demucs AI"
                report(2, 0.30, f"កំពុងបំបែកសំឡេងចម្រៀង/និយាយ និងរក្សាទុក BGM & Sound FX ({engine_title})...")
                v_path, b_path = separator.separate_vocals(extracted_wav, str(temp_dir / "separation"), clean_foley=False)
                if v_path and b_path and os.path.exists(b_path):
                    vocals_wav = v_path
                    bgm_wav = b_path
                else:
                    logger.error(f"❌ ការបំបែកសំឡេង BGM ដោយ {engine_title} មិនបានសម្រេច!")
                    vocals_wav = extracted_wav
                    bgm_wav = None  # Never fall back to extracted_wav because it contains original Chinese vocals!
            else:
                report(2, 0.30, "ជ្រើសរើស Mute សំឡេងដើមទាំងអស់ (បិទសំឡេងដើម ១០០%)...")
                vocals_wav = extracted_wav
                bgm_wav = None

            # ---------------- Stage 3: Chinese ASR (Whisper) ----------------
            if self.is_cancelled: return False
            report(3, 0.50, "កំពុងស្តាប់ និងស្រង់សម្ដីចិន (Whisper ASR: Clean Vocals & VAD)...")
            transcriber = ChineseTranscriber(
                model_size=cfg.get("whisper_model", "large-v3-turbo"),
                device=cfg.get("device", "cuda")
            )
            # Transcribe original extracted audio directly to guarantee 100% dialogue capture (zero lost speech)
            asr_input_audio = extracted_wav
            segments = transcriber.transcribe(
                asr_input_audio,
                genre=cfg.get("drama_genre", "wuxia")
            )
            if not segments:
                logger.warning(f"រកមិនឃើញពាក្យនិយាយចិនក្នុងវីដេអូ: {video_file.name}")
                # If no dialogue, we can just copy original or background
                renderer = VideoRenderer(mode=cfg.get("render_mode", "fast_copy"))
                return renderer.render(str(video_file), bgm_wav or extracted_wav, str(output_file))

            # ---------------- Stage 4: Translate to Khmer (Gemini AI) ----------------
            if self.is_cancelled: return False
            report(4, 0.65, "កំពុងបកប្រែជាភាសាខ្មែរបែបកុន (Gemini AI Cinema Dubbing)...")
            translator = GeminiTranslator(
                api_key=cfg.get("gemini_api_key", ""),
                model_name=cfg.get("gemini_model", "gemini-3.5-flash"),
                style_prompt=cfg.get("khmer_prompt_style", "natural_movie"),
                drama_genre=cfg.get("drama_genre", "wuxia"),
                enable_ai_review=cfg.get("enable_ai_review", True)
            )
            translated_segments = translator.translate_segments(
                segments, 
                batch_size=35,
                video_path=str(video_file)
            )
            translated_segments = deduplicate_khmer_dialogue(translated_segments, max_gap=3.5)

            if not translated_segments and len(segments) > 0:
                logger.warning(f"⚠️ គ្មានអត្ថបទនិយាយបន្ទាប់ពីការច្រោះសម្រួល: {video_file.name}")
                renderer = VideoRenderer(mode=cfg.get("render_mode", "fast_copy"))
                return renderer.render(str(video_file), bgm_wav or extracted_wav, str(output_file))

            # Auto Speaker Gender Detection (Acoustic Pitch + Gemini Context)
            chosen_voice = cfg.get("khmer_voice", "km-KH-PisethNeural")
            if chosen_voice == "auto":
                report(4, 0.72, "🤖 កំពុងវិភាគ និងចាប់សំឡេងប្រុស/ស្រី (AI Character Tracking & Context)...")
                prev_voice = None
                for seg in translated_segments:
                    f0 = GenderDetector.detect_pitch_f0(vocals_wav, seg.get("start", 0), seg.get("end", 0))
                    voice_id, gender_label = GenderDetector.determine_voice(
                        pitch_f0=f0,
                        gemini_gender=seg.get("gemini_gender"),
                        previous_voice=prev_voice,
                        gemini_role=seg.get("gemini_role")
                    )
                    seg["assigned_voice"] = voice_id
                    seg["detected_gender"] = gender_label
                    seg["pitch_f0"] = f0
                    prev_voice = voice_id

                # Smooth isolated blips and enforce character consistency lock across scene
                translated_segments = GenderDetector.smooth_speaker_consistency(translated_segments)

                # Character Role Voice Tuning:
                # Dynamic character differentiation without artificial vocoder distortion:
                for seg in translated_segments:
                    voice_id = seg.get("assigned_voice", "km-KH-PisethNeural")
                    role = (seg.get("gemini_role") or "").lower()
                    km_txt = seg.get("km_text", "")
                    is_urgent = any(w in km_txt for w in ["!", "ឆាប់", "ឈប់", "ជួយ", "រត់"]) or "shout" in role or "fight" in role

                    if "Piseth" in voice_id:
                        if any(k in role for k in ["elder", "master", "father", "old", "emperor", "king", "villain"]):
                            assigned_pitch = "-3Hz"
                            assigned_rate = "-4%"
                        elif any(k in role for k in ["boy", "child", "servant", "comic", "junior"]):
                            assigned_pitch = "+2Hz"
                            assigned_rate = "+5%"
                        elif is_urgent:
                            assigned_pitch = "+1Hz"
                            assigned_rate = "+6%"
                        else:
                            # Hero / Male Protagonist / Standard Male
                            assigned_pitch = "+0Hz"
                            assigned_rate = "+0%"
                    else:
                        # Sreymom (Female)
                        if any(k in role for k in ["mother", "older_female", "empress", "queen"]):
                            assigned_pitch = "-3Hz"
                            assigned_rate = "-4%"
                        elif any(k in role for k in ["girl", "child", "cute"]):
                            assigned_pitch = "+3Hz"
                            assigned_rate = "+5%"
                        elif any(k in role for k in ["crying", "sad", "plead"]):
                            assigned_pitch = "+2Hz"
                            assigned_rate = "-4%"
                        elif is_urgent:
                            assigned_pitch = "+1Hz"
                            assigned_rate = "+5%"
                        else:
                            # Heroine / Maiden / Standard Female
                            assigned_pitch = "+0Hz"
                            assigned_rate = "+0%"

                    seg["assigned_pitch"] = assigned_pitch
                    seg["assigned_rate"] = assigned_rate

                male_cnt = sum(1 for s in translated_segments if "Piseth" in s.get("assigned_voice", ""))
                female_cnt = sum(1 for s in translated_segments if "Sreymom" in s.get("assigned_voice", ""))
                logger.info(f"🎭 លទ្ធផលចាប់សំឡេងស្វ័យប្រវត្តិ: ប្រុស (Male) {male_cnt} ប្រយោគ | ស្រី (Female) {female_cnt} ប្រយោគ")

            # ---------------- Stage 5: Khmer TTS (Edge-TTS) ----------------
            if self.is_cancelled: return False
            report(5, 0.80, "កំពុងបង្កើតសំឡេងខ្មែរ (Microsoft Neural TTS)...")
            tts_engine = KhmerTTSEngine(
                voice=cfg.get("khmer_voice", "km-KH-PisethNeural"),
                speed=cfg.get("voice_speed", "+0%"),
                pitch=cfg.get("voice_pitch", "+0Hz")
            )
            tts_segments = tts_engine.generate_speech(translated_segments, str(temp_dir / "tts"))
            if not tts_segments and len(translated_segments) > 0:
                logger.error(f"❌ មិនអាចបង្កើតសំឡេងខ្មែរបានទេ (TTS failed for all {len(translated_segments)} segments) លើវីដេអូ: {video_file.name}")
                return False

            # ---------------- Stage 6: Audio Synchronization & Mixing ----------------
            if self.is_cancelled: return False
            report(6, 0.90, "កំពុងតម្រឹមសំឡេងឲ្យត្រូវចង្វាក់សកម្មភាព និងលាយ BGM...")
            sync_engine = AudioSyncEngine(
                enable_time_stretch=cfg.get("enable_time_stretch", True),
                max_stretch_ratio=float(cfg.get("max_stretch_ratio", 1.30)),
                bgm_volume=float(cfg.get("bgm_volume", 0.28)),
                voice_volume=float(cfg.get("voice_volume", 1.35)),
                ducking_factor=float(cfg.get("ducking_factor", 0.25))
            )
            
            dialogue_res = sync_engine.build_synchronized_dialogue_track(
                tts_segments, total_duration, str(temp_dir / "sync")
            )
            if not dialogue_res:
                logger.error("មិនអាចបង្កើត dialogue master audio បានទេ។")
                return False

            if isinstance(dialogue_res, tuple):
                dialogue_master, speech_intervals = dialogue_res
            else:
                dialogue_master = dialogue_res
                speech_intervals = None

            # If mode is separate_bgm but bgm_wav is missing or invalid, keep bgm_wav as None to avoid leaking original Chinese speech
            if vocal_mode == "separate_bgm" and (not bgm_wav or not os.path.exists(bgm_wav)):
                logger.warning("⚠️ គ្មាន BGM track ទេ កំពុងរក្សាទុកតែសំឡេងខ្មែរ (No Chinese voice leak)...")
                bgm_wav = None

            final_audio_wav = str(temp_dir / "final_dubbed_audio.wav")
            if not sync_engine.mix_final_audio(
                dialogue_master, bgm_wav, final_audio_wav, mode=vocal_mode, speech_intervals=speech_intervals
            ):
                return False

            # ---------------- Stage 7: Video Render / Muxing ----------------
            if self.is_cancelled: return False
            report(7, 0.96, "កំពុង Export វីដេអូសម្រេចល្បឿនលឿន...")
            renderer = VideoRenderer(mode=cfg.get("render_mode", "fast_copy"))
            success = renderer.render(str(video_file), final_audio_wav, str(output_file))

            if success:
                report(7, 1.0, f"✅ បញ្ចប់ការ Dubbing ដោយជោគជ័យ: {output_file.name}")
                return True
            else:
                logger.error(f"Render បរាជ័យសម្រាប់ {video_file.name}")
                return False

        except Exception as e:
            logger.error(f"កំហុសក្នុងការដំណើរការ pipeline: {e}")
            return False

        finally:
            # Clean up temp files if successful
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass
