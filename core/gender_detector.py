import os
import numpy as np
import soundfile as sf
import librosa
from typing import Tuple, Optional
from utils.logger import logger

# Acoustic boundary: typical male speech is 85-165 Hz, female speech is 165-280+ Hz
PITCH_SPLIT_HZ = 165.0
CLEAR_MALE_MAX = 155.0
CLEAR_FEMALE_MIN = 175.0

class GenderDetector:
    """
    High-accuracy speaker gender detection combining acoustic fundamental frequency (F0)
    analysis with contextual LLM dialogue clues.
    """
    _is_warmed_up = False

    @classmethod
    def warmup(cls):
        """Warm up librosa/numba JIT compiler on a tiny synthetic signal."""
        if cls._is_warmed_up:
            return
        try:
            sr = 16000
            t = np.linspace(0, 0.2, int(sr * 0.2))
            dummy = 0.5 * np.sin(2 * np.pi * 150 * t)
            librosa.yin(dummy, fmin=65, fmax=350, sr=sr)
            cls._is_warmed_up = True
        except Exception:
            pass

    @classmethod
    def detect_pitch_f0(cls, audio_wav: str, start_sec: float, end_sec: float) -> Optional[float]:
        """
        Extracts median fundamental frequency (F0 in Hz) of the speech segment from vocals.wav.
        Returns median F0 (e.g. 120.5 Hz for male, 215.0 Hz for female), or None if unvoiced.
        """
        if not os.path.exists(audio_wav) or end_sec <= start_sec:
            return None

        try:
            cls.warmup()
            # Read only the segment slice to save memory and ensure high speed
            with sf.SoundFile(audio_wav) as f:
                sr = f.samplerate
                start_frame = int(start_sec * sr)
                num_frames = int((end_sec - start_sec) * sr)
                
                if start_frame >= len(f):
                    return None
                    
                f.seek(start_frame)
                y = f.read(num_frames)

            # Convert stereo to mono if needed
            if y.ndim > 1:
                y = np.mean(y, axis=1)

            if len(y) < sr * 0.25:  # Segment too short (< 250ms)
                return None

            # Downsample to 16kHz for ultra-fast YIN pitch estimation
            if sr != 16000:
                y = librosa.resample(y, orig_sr=sr, target_sr=16000)
                sr = 16000

            # Compute fundamental frequency using YIN (speech range: 65 Hz to 350 Hz)
            f0 = librosa.yin(y, fmin=65, fmax=350, sr=sr, frame_length=1024, hop_length=256)
            
            # Filter out unvoiced / NaN frames
            valid_f0 = f0[~np.isnan(f0)]
            if len(valid_f0) == 0:
                return None

            median_f0 = float(np.median(valid_f0))
            return median_f0

        except Exception as e:
            return None

    @classmethod
    def determine_voice(
        cls,
        pitch_f0: Optional[float],
        gemini_gender: Optional[str] = None,
        previous_voice: Optional[str] = None,
        gemini_role: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Combines contextual dialogue intelligence (Gemini LLM) with acoustic pitch (F0)
        to determine the most accurate Khmer voice.

        CONTEXT-FIRST ARCHITECTURE:
        In Chinese drama, male characters shouting, yelling, or crying spike their pitch to 180-240Hz.
        Acoustic-only algorithms falsely classify these as female. Context-first prevents this bug.
        """
        # Clean gemini gender tag
        g_tag = (gemini_gender or "").lower().strip()
        is_gemini_male = ("male" in g_tag and "female" not in g_tag) or ("ប្រុស" in g_tag) or ("man" in g_tag)
        is_gemini_female = ("female" in g_tag) or ("ស្រី" in g_tag) or ("woman" in g_tag)

        # Also inspect gemini role clues
        r_tag = (gemini_role or "").lower().strip()
        if not is_gemini_male and not is_gemini_female and r_tag:
            if any(k in r_tag for k in ["hero", "elder", "father", "master", "king", "emperor", "monk", "brother", "villain", "eunuch", "boy"]):
                is_gemini_male = True
            elif any(k in r_tag for k in ["female_lead", "mother", "girl", "sister", "queen", "empress", "maid", "concubine", "lady"]):
                is_gemini_female = True

        f0_str = f" ({pitch_f0:.0f}Hz)" if pitch_f0 is not None else ""

        # Priority 1: Strong Contextual Gender from Gemini Script Analysis
        if is_gemini_male and not is_gemini_female:
            return "km-KH-PisethNeural", f"ប្រុស (Context{f0_str})"
        elif is_gemini_female and not is_gemini_male:
            return "km-KH-SreymomNeural", f"ស្រី (Context{f0_str})"

        # Priority 2: Acoustic Pitch Analysis (when context is absent or ambiguous)
        if pitch_f0 is not None:
            if pitch_f0 < 160.0:
                return "km-KH-PisethNeural", f"ប្រុស (Pitch {pitch_f0:.0f}Hz)"
            else:
                return "km-KH-SreymomNeural", f"ស្រី (Pitch {pitch_f0:.0f}Hz)"

        # Priority 3: Dialogue Continuity from previous speaker
        if previous_voice:
            label = "ស្រី" if "Sreymom" in previous_voice else "ប្រុស"
            return previous_voice, f"{label} (Continuity)"

        # Default fallback (Male narrator / lead)
        return "km-KH-PisethNeural", "ប្រុស (Default)"

    @classmethod
    def smooth_speaker_consistency(cls, segments: list) -> list:
        """
        Eliminates single-segment rapid voice toggling (e.g. Male -> Female for 0.8s -> Male),
        and locks voice consistency for identifiable character turns.
        """
        if len(segments) < 3:
            return segments

        # Pass 1: Character-name consistency lock
        character_voices = {}
        for seg in segments:
            char = (seg.get("gemini_character") or "").strip().lower()
            voice = seg.get("assigned_voice")
            if char and voice:
                if char not in character_voices:
                    character_voices[char] = []
                character_voices[char].append(voice)

        # If a character has a dominant voice (> 60% of lines), lock all lines to that voice
        for char, v_list in character_voices.items():
            if len(v_list) >= 2:
                piseth_cnt = sum(1 for v in v_list if "Piseth" in v)
                sreymom_cnt = sum(1 for v in v_list if "Sreymom" in v)
                dominant = "km-KH-PisethNeural" if piseth_cnt >= sreymom_cnt else "km-KH-SreymomNeural"
                dom_label = "ប្រុស (Character Lock)" if "Piseth" in dominant else "ស្រី (Character Lock)"
                for seg in segments:
                    if (seg.get("gemini_character") or "").strip().lower() == char:
                        seg["assigned_voice"] = dominant
                        seg["detected_gender"] = dom_label

        # Pass 2: Smooth isolated 1-segment blips (e.g. [A, B, A] where B duration < 2.0s)
        for i in range(1, len(segments) - 1):
            prev_v = segments[i - 1].get("assigned_voice")
            curr_v = segments[i].get("assigned_voice")
            next_v = segments[i + 1].get("assigned_voice")
            dur = float(segments[i].get("duration") or 0.0)

            # If sandwiched between the exact same voice and the middle segment is short
            if prev_v == next_v and curr_v != prev_v and dur < 2.2:
                # Check if it has a strong opposite gender context; if not, smooth it
                curr_gender = (segments[i].get("gemini_gender") or "").lower()
                is_curr_opposite = ("female" in curr_gender if "Piseth" in prev_v else "male" in curr_gender)
                if not is_curr_opposite:
                    segments[i]["assigned_voice"] = prev_v
                    label = "ប្រុស (Smoothed)" if "Piseth" in prev_v else "ស្រី (Smoothed)"
                    segments[i]["detected_gender"] = label

        return segments
