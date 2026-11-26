import asyncio
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import edge_tts
from utils.logger import logger
from utils.file_utils import get_media_duration
from core.translator import contains_chinese, fallback_translate_text

KHMER_VOICES = {
    "km-KH-PisethNeural": "សំឡេងប្រុស (Piseth - Male)",
    "km-KH-SreymomNeural": "សំឡេងស្រី (Sreymom - Female)",
    "auto": "🤖 ចាប់សំឡេងប្រុស/ស្រីស្វ័យប្រវត្តិ (Auto Male/Female Detect)"
}

class KhmerTTSEngine:
    """Generates high quality Khmer speech with multi-threaded concurrent async synthesis."""

    def __init__(self, voice: str = "km-KH-PisethNeural", speed: str = "+0%", pitch: str = "+0Hz", max_concurrent: int = 4):
        self.voice = voice if voice in KHMER_VOICES else "km-KH-PisethNeural"
        self.speed = speed
        self.pitch = pitch
        self.max_concurrent = max_concurrent

    def set_voice(self, voice: str):
        self.voice = voice

    def set_speed(self, speed: str):
        self.speed = speed

    def set_pitch(self, pitch: str):
        self.pitch = pitch

    async def _synthesize_single(
        self,
        text: str,
        output_path: str,
        voice_name: str,
        sem: asyncio.Semaphore,
        pitch: Optional[str] = None,
        rate: Optional[str] = None
    ) -> bool:
        """
        Synthesize a single text sentence into an audio file strictly using Microsoft Neural Edge-TTS.
        Guarantees 100% voice purity:
        - Male lines are strictly km-KH-PisethNeural.
        - Female lines are strictly km-KH-SreymomNeural.
        - ZERO third-party / robot voice fallbacks (NO gTTS).
        - Pitch is strictly locked to +0Hz (Microsoft Azure SSML for Khmer rejects non-zero pitch).
        """
        target_voice = voice_name if voice_name in ["km-KH-PisethNeural", "km-KH-SreymomNeural"] else "km-KH-PisethNeural"
        safe_pitch = "+0Hz"

        # Clean text: remove any non-Khmer foreign characters or control codes
        clean_text = text.strip()
        clean_text = re.sub(r'[\u4e00-\u9fff\u0e00-\u0e7f\x00-\x1f\x7f-\x9f]', '', clean_text).strip()
        if not clean_text:
            clean_text = "បាទ" if "Piseth" in target_voice else "ចាស"

        # Microsoft Azure Khmer Neural voices strictly require pitch="+0Hz" and rate="+0%".
        # Any non-zero pitch or rate causes "No audio was received" from the Azure backend.
        safe_pitch = "+0Hz"
        safe_rate = "+0%"

        async with sem:
            # 4 resilient retry attempts strictly using target_voice with +0Hz pitch and +0% rate
            attempts = [
                (target_voice, safe_pitch, safe_rate, clean_text),
                (target_voice, safe_pitch, safe_rate, clean_text),
                (target_voice, safe_pitch, safe_rate, clean_text),
                (target_voice, safe_pitch, safe_rate, re.sub(r'[^\u1780-\u17ff\s\.,!?]', '', clean_text).strip() or ("បាទ" if "Piseth" in target_voice else "ចាស"))
            ]

            for attempt_idx, (v_name, p_val, r_val, txt_to_speak) in enumerate(attempts):
                try:
                    # Clean up any lingering 0-byte or corrupted file
                    if os.path.exists(output_path):
                        try:
                            os.remove(output_path)
                        except Exception:
                            pass

                    communicate = edge_tts.Communicate(
                        text=txt_to_speak,
                        voice=v_name,
                        rate=r_val,
                        pitch=p_val
                    )
                    await communicate.save(output_path)
                    if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
                        return True
                except Exception as e:
                    if attempt_idx == len(attempts) - 1:
                        logger.error(f"TTS error on [{v_name}] '{txt_to_speak[:25]}...': {e}")
                    await asyncio.sleep(0.4 * (attempt_idx + 1))

            return False

    async def _generate_all(self, segments: List[Dict[str, Any]], output_dir: str) -> List[Dict[str, Any]]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"⚡ កំពុងបង្កើតសំឡេងខ្មែរល្បឿនលឿន (Concurrent Edge-TTS: {self.max_concurrent} threads) សរុប {len(segments)} ប្រយោគ...")
        
        sem = asyncio.Semaphore(self.max_concurrent)
        is_auto_voice = self.voice == "auto"

        tasks = []
        indexed_segments = []

        for idx, seg in enumerate(segments):
            km_text = seg.get("km_text", "").strip()
            if not km_text:
                continue

            # Critical Language Guard: Edge-TTS Khmer voices fail if text contains Chinese
            if contains_chinese(km_text):
                logger.warning(f"⚠️ រកឃើញអក្សរចិនក្នុងសំឡេងខ្មែរ (#{seg.get('id', idx+1)}): '{km_text[:20]}...' -> កំពុងបកប្រែជាភាសាខ្មែរបន្ទាន់...")
                translated = fallback_translate_text(km_text)
                if translated and not contains_chinese(translated):
                    km_text = translated
                    seg["km_text"] = km_text
                else:
                    # Strip lingering Chinese characters so Edge-TTS can pronounce Khmer without failing
                    cleaned = re.sub(r'[\u4e00-\u9fff]', '', translated or km_text).strip()
                    if cleaned and len(cleaned) >= 2:
                        km_text = cleaned
                        seg["km_text"] = km_text
                    else:
                        km_text = "បាទ"
                        seg["km_text"] = km_text

            seg_id = seg.get("id", idx + 1)
            audio_file = str(output_path / f"seg_{seg_id:04d}.mp3")

            if is_auto_voice:
                voice_to_use = seg.get("assigned_voice") or "km-KH-PisethNeural"
            else:
                voice_to_use = self.voice

            # Microsoft Azure Khmer Neural voices strictly require pitch="+0Hz" and rate="+0%"
            seg_pitch = "+0Hz"
            seg_rate = "+0%"

            task = self._synthesize_single(km_text, audio_file, voice_to_use, sem, pitch=seg_pitch, rate=seg_rate)
            tasks.append(task)
            indexed_segments.append((seg, audio_file, voice_to_use, seg_pitch, seg_rate))

        # Run all TTS tasks concurrently in parallel!
        results_flags = await asyncio.gather(*tasks)

        final_segments = []
        for (seg, audio_file, voice_used, pitch_used, rate_used), success in zip(indexed_segments, results_flags):
            if not (success and os.path.exists(audio_file) and os.path.getsize(audio_file) > 100):
                # Pure Edge-TTS Sequential Recovery (strictly voice_used, pitch=+0Hz, rate=+0%)
                logger.warning(f"⚠️ កំពុងព្យាយាមបង្កើតសំឡេងឡើងវិញដោយប្រុងប្រយ័ត្នសម្រាប់ប្រយោគ #{seg.get('id')} ({voice_used})...")
                try:
                    await asyncio.sleep(0.5)
                    retry_sem = asyncio.Semaphore(1)
                    await self._synthesize_single(
                        text=seg.get("km_text", ""),
                        output_path=audio_file,
                        voice_name=voice_used,
                        sem=retry_sem,
                        pitch="+0Hz",
                        rate="+0%"
                    )
                except Exception as e_rec:
                    logger.error(f"Pure Edge-TTS recovery error for #{seg.get('id')}: {e_rec}")

            if os.path.exists(audio_file) and os.path.getsize(audio_file) > 50:
                actual_dur = get_media_duration(audio_file)
                seg_res = dict(seg)
                seg_res["audio_file"] = audio_file
                seg_res["actual_duration"] = actual_dur
                seg_res["voice_used"] = voice_used
                seg_res["pitch_used"] = "+0Hz"
                seg_res["rate_used"] = rate_used
                final_segments.append(seg_res)
            else:
                logger.error(f"❌ មិនអាចបង្កើតសំឡេងសម្រាប់ប្រយោគ #{seg.get('id')}")

        logger.success(f"បង្កើតសំឡេងខ្មែរបានជោគជ័យល្បឿនលឿន {len(final_segments)}/{len(segments)} ប្រយោគ!")
        return final_segments

    def generate_speech(self, segments: List[Dict[str, Any]], output_dir: str) -> List[Dict[str, Any]]:
        """Synchronous wrapper to run concurrent async TTS generation."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
                return loop.run_until_complete(self._generate_all(segments, output_dir))
            else:
                return loop.run_until_complete(self._generate_all(segments, output_dir))
        except RuntimeError:
            return asyncio.run(self._generate_all(segments, output_dir))
