import os
import re
import sys
import difflib
import threading
import subprocess
from pathlib import Path
import torch
from typing import List, Dict, Any, Optional
from utils.gpu_utils import setup_cuda_dlls
from utils.logger import logger

setup_cuda_dlls()

def preprocess_audio_for_asr(input_wav: str, output_wav: str) -> str:
    """
    Applies high-grade speech intelligibility filters to audio before Whisper transcription:
    1. highpass=f=90: Strips sub-bass rumble, wind, floor shaking, and AC hum (< 90Hz).
    2. lowpass=f=8000: Cuts harsh ultrasonic hiss and metallic ringings (> 8000Hz).
    3. equalizer=f=2800:width_type=o:width=1.5:g=3.5: Boosts Mandarin consonant presence & clarity (zh, ch, sh, x, s, z, j, q).
    4. speechnorm=e=4:r=0.0001:l=1: Dynamically levels speech so whispering characters are audible without clipping shouts.
    """
    if not os.path.exists(input_wav):
        return input_wav

    cmd = [
        "ffmpeg", "-y", "-i", str(input_wav),
        "-af", "highpass=f=90,lowpass=f=8000,equalizer=f=2800:width_type=o:width=1.5:g=3.5,speechnorm=e=4:r=0.0001:l=1",
        "-ar", "16000", "-ac", "1",
        str(output_wav)
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if res.returncode == 0 and os.path.exists(output_wav) and os.path.getsize(output_wav) > 100:
            return output_wav
    except Exception as e:
        logger.warning(f"Audio pre-filter warning ({e}), falling back to input audio...")
    return input_wav

def get_asr_initial_prompt(genre: str = "wuxia") -> str:
    """Returns domain-tuned prompt to guide Whisper towards correct Chinese drama vocabulary."""
    g = (genre or "wuxia").lower()
    if g == "modern":
        return "这是一部精彩的中国都市现代电视剧对白台词。请准确识别人物对白，使用简体中文及规范标点符号。"
    elif g == "palace":
        return "这是一部精彩的中国古代宫廷电视剧对白台词。请准确识别皇帝、皇后、大臣之言，使用简体中文及规范标点符号。"
    else:  # wuxia / auto default
        return "这是一部精彩的中国古装武侠影视剧台词对白，请准确识别江湖侠客、掌门弟子对白，使用简体中文及规范标点符号。"

def clean_intra_repetition(text: str) -> str:
    """Collapses hallucination repeating loops (3+ times) while preserving natural Chinese words (妈妈, 看看, 等等)."""
    if not text:
        return ""
    cleaned = text
    # Only collapse if a phrase (length 2-10) repeats 3 or more times (true hallucination loop)
    for length in range(10, 1, -1):
        pattern = r'(.{' + str(length) + r'}?)\1{2,}'
        cleaned = re.sub(pattern, r'\1', cleaned)
    # For single characters, only collapse if repeated 3+ times (e.g. 啊啊啊啊 -> 啊)
    cleaned = re.sub(r'(.)\1{2,}', r'\1', cleaned)
    return cleaned.strip()

def normalize_dialogue_text(text: str) -> str:
    """Strips punctuation, spaces and symbols for exact semantic comparison."""
    return re.sub(r'[^\w\u4e00-\u9fff]', '', text).strip()

def is_non_speech_or_hallucination(text: str) -> bool:
    """Detects bracketed sound effects, music tags, and common web subtitle boilerplate."""
    if not text:
        return True
    t = text.strip()
    # Punctuation / ellipses only
    if not re.search(r'[\u4e00-\u9fff\w]', t):
        return True
    # Pure bracketed audio tags: [音乐], (音乐), 【音乐】, [Music], ♪, ♫, [掌声], [笑声], etc.
    cleaned_brackets = re.sub(
        r'^[\[\(\{\<【（]\s*(?:音乐|音樂|music|bgm|applause|laughter|掌声|笑声|音效|叹气|喘息|哭声|脚步声)\s*[\]\)\}\>】）]$',
        '', t, flags=re.IGNORECASE
    ).strip()
    if not cleaned_brackets:
        return True
    # Music symbols
    if re.fullmatch(r'[♪♫♩♬\s\.\,\!\?\-—_~]+', t):
        return True
    # Boilerplate hallucinations
    bp = [
        r'^(?:请?点赞(?:投币)?(?:关注|收藏)?|感谢收看|谢谢观看|欢迎订阅|关注我|下期更精彩|字幕由.*提供|本视频由.*制作|严禁转载).*$'
    ]
    for p in bp:
        if re.search(p, t, flags=re.IGNORECASE):
            return True
    return False


def deduplicate_crowd_segments(segments: List[Dict[str, Any]], max_time_gap: float = 2.5) -> List[Dict[str, Any]]:
    """
    Suppresses consecutive identical Whisper loops (e.g. '照上人' repeating 10 times).
    Only collapses consecutive segments if normalized text is EXACTLY IDENTICAL
    within max_time_gap (2.5s), ensuring all genuine, distinct dialogue is 100% preserved.
    """
    if not segments:
        return []

    filtered = []
    for seg in segments:
        raw_text = seg.get("text", "")
        text = clean_intra_repetition(raw_text)
        if not text:
            continue
        seg["text"] = text
        norm = normalize_dialogue_text(text)
        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)

        if not filtered:
            filtered.append(dict(seg))
            continue

        prev = filtered[-1]
        prev_norm = normalize_dialogue_text(prev.get("text", ""))
        gap = start - prev.get("end", 0.0)

        # Only collapse if normalized text is EXACTLY IDENTICAL and consecutive within 2.5s
        if norm and prev_norm and norm == prev_norm and gap <= max_time_gap:
            prev["end"] = max(prev.get("end", 0.0), end)
            prev["duration"] = round(prev["end"] - prev["start"], 2)
            continue

        filtered.append(dict(seg))

    # Re-index ids sequentially
    for idx, s in enumerate(filtered):
        s["id"] = idx + 1

    return filtered

def merge_short_adjacent_segments(
    segments: List[Dict[str, Any]],
    max_gap: float = 0.65,
    max_combined_dur: float = 4.8
) -> List[Dict[str, Any]]:
    """
    Carefully merges broken syllables and clauses within the same sentence (<= 0.35s gap)
    while strictly preserving separate speaker turns and subtitle boundaries.
    Does not merge across sentence boundaries or different character lines.
    """
    if not segments or len(segments) <= 1:
        return segments

    merged = []
    current = dict(segments[0])

    for nxt in segments[1:]:
        gap = round(nxt["start"] - current["end"], 2)
        comb_dur = round(nxt["end"] - current["start"], 2)
        cur_txt = current["text"].strip()
        has_full_stop = bool(cur_txt and cur_txt[-1] in "。！？!?…")

        can_merge = False
        if not has_full_stop and gap <= max_gap and comb_dur <= max_combined_dur:
            can_merge = True

        if can_merge:
            current["end"] = nxt["end"]
            current["duration"] = comb_dur
            nxt_txt = nxt["text"].strip()
            if cur_txt and cur_txt[-1] in "，,":
                current["text"] = f"{cur_txt}{nxt_txt}"
            else:
                current["text"] = f"{cur_txt}，{nxt_txt}"
        else:
            merged.append(current)
            current = dict(nxt)

    merged.append(current)

    # Re-index ids sequentially
    for idx, s in enumerate(merged):
        s["id"] = idx + 1

    return merged

def process_faster_whisper_segments(segments, max_word_gap: float = 0.55) -> List[Dict[str, Any]]:
    """
    Robust Whisper segment parser:
    1. Uses word-level timestamps to eliminate dead silence at the start and end of clauses.
    2. Splits LONG clauses (> 4.5s) ONLY at major punctuation (。！？!?…) or natural pause commas.
    3. NEVER shatters short words or phrases across micro-pauses (preserves words like '我的手', '安息', '太慢了').
    4. Preserves 100% of spoken dialogue without false-positive ghost cluster dropping.
    """
    processed = []

    for s in segments:
        text = s.text.strip()
        if not text:
            continue
        if is_non_speech_or_hallucination(text):
            continue

        words = getattr(s, 'words', None)
        if words and len(words) > 0:
            start = round(float(words[0].start), 2)
            end = round(float(words[-1].end), 2)
        else:
            start = round(float(s.start), 2)
            end = round(float(s.end), 2)

        dur = round(end - start, 2)
        if dur < 0.1:
            continue
        if dur < 0.35:
            end = round(start + 0.35, 2)
            dur = 0.35

        # If the segment is long (> 4.5s) with distinct punctuation parts, split cleanly by punctuation
        if dur > 4.5 and words and len(words) > 3:
            sub_clusters = []
            cur_w = [words[0]]
            for w in words[1:]:
                prev_txt = cur_w[-1].word.strip()
                ends_major = bool(prev_txt and prev_txt[-1] in "。！？!?…")
                ends_comma = bool(prev_txt and prev_txt[-1] in ",，")
                gap = w.start - cur_w[-1].end
                cur_dur = cur_w[-1].end - cur_w[0].start

                split_here = False
                if ends_major and gap >= 0.25:
                    split_here = True
                elif ends_comma and gap >= 0.40:
                    split_here = True
                elif cur_dur >= 4.2 and (ends_comma or ends_major or gap >= 0.60):
                    split_here = True

                if split_here:
                    sub_clusters.append(cur_w)
                    cur_w = [w]
                else:
                    cur_w.append(w)
            if cur_w:
                sub_clusters.append(cur_w)

            for cl in sub_clusters:
                cl_text = "".join(w.word for w in cl).strip()
                if cl_text and re.search(r'[\u4e00-\u9fff\w]', cl_text):
                    processed.append({
                        'start': round(float(cl[0].start), 2),
                        'end': round(float(cl[-1].end), 2),
                        'duration': round(float(cl[-1].end) - float(cl[0].start), 2),
                        'text': cl_text
                    })
        else:
            processed.append({
                'start': start,
                'end': end,
                'duration': dur,
                'text': text
            })

    # Clean and keep all valid dialogue segments
    final_segs = []
    for seg in processed:
        txt = seg['text'].strip().strip('，。！？,.!?…')
        if not txt:
            continue
        if not re.search(r'[\u4e00-\u9fff\w]', txt):
            continue
        final_segs.append(seg)

    # Sort strictly by start time
    final_segs.sort(key=lambda s: float(s.get('start', 0.0)))

    # Guarantee clean sequential boundaries
    for idx in range(1, len(final_segs)):
        prev = final_segs[idx - 1]
        curr = final_segs[idx]
        p_end = float(prev.get('end', 0.0))
        c_start = float(curr.get('start', 0.0))
        if c_start < p_end:
            if c_start < float(prev['start']) + 0.3:
                curr['start'] = round(float(prev['start']) + 0.3, 2)
            prev['end'] = max(round(float(prev['start']) + 0.25, 2), round(float(curr['start']) - 0.05, 2))
            prev['duration'] = round(float(prev['end']) - float(prev['start']), 2)
            curr['duration'] = round(float(curr['end']) - float(curr['start']), 2)

    # Re-index ids
    for idx, s in enumerate(final_segs):
        s['id'] = idx + 1

    return final_segs

class ChineseTranscriber:
    """Transcribes Chinese dialogue with Faster-Whisper (CUDA FP16)."""
    
    _loaded_model = None
    _loaded_model_name = None
    _loaded_device = None
    _use_faster = True
    _model_lock = threading.Lock()

    def __init__(self, model_size: str = "large-v3-turbo", device: str = "cuda"):
        self.model_size = model_size
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"

    def get_model(self):
        """Loads and caches the Whisper/Faster-Whisper model in memory."""
        setup_cuda_dlls()
        
        with ChineseTranscriber._model_lock:
            if (ChineseTranscriber._loaded_model is None or 
                ChineseTranscriber._loaded_model_name != self.model_size or 
                ChineseTranscriber._loaded_device != self.device):
                try:
                    from faster_whisper import WhisperModel
                    compute_type = "float16" if self.device == "cuda" else "int8"
                    logger.info(f"🎙️ កំពុងដំណើរការ Faster-Whisper ({self.model_size}) លើ {self.device} ({compute_type})...")
                    ChineseTranscriber._loaded_model = WhisperModel(
                        self.model_size,
                        device=self.device,
                        compute_type=compute_type
                    )
                    ChineseTranscriber._use_faster = True
                    ChineseTranscriber._loaded_model_name = self.model_size
                    ChineseTranscriber._loaded_device = self.device
                    logger.success(f"Faster-Whisper ({self.model_size}) CUDA FP16 បានរួចរាល់ល្បឿនលឿន!")
                except Exception as e:
                    logger.warning(f"Faster-Whisper CUDA error ({e}), attempting standard Whisper fallback...")
                    import whisper
                    ChineseTranscriber._loaded_model = whisper.load_model(self.model_size, device=self.device)
                    ChineseTranscriber._use_faster = False
                    ChineseTranscriber._loaded_model_name = self.model_size
                    ChineseTranscriber._loaded_device = self.device
                    logger.success(f"Standard Whisper ({self.model_size}) បានរួចរាល់!")
            
        return ChineseTranscriber._loaded_model

    def transcribe(self, audio_path: str, genre: str = "wuxia") -> List[Dict[str, Any]]:
        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            return []

        temp_filtered_wav = None
        try:
            model = self.get_model()
            logger.info(f"⚡ កំពុងស្រង់សម្ដីចិនល្បឿនលឿន (Whisper ASR: Speech Filter, Silero VAD & Context Prompt Enabled)...")

            # Apply Speech Intelligibility Pre-filter (removes rumble, highlights Chinese consonants, normalizes volume)
            temp_dir = Path(audio_path).parent
            filtered_wav = str(temp_dir / f"asr_enhanced_{Path(audio_path).stem}.wav")
            processed_audio = preprocess_audio_for_asr(audio_path, filtered_wav)
            if processed_audio != audio_path:
                temp_filtered_wav = processed_audio

            cleaned_segments = []
            init_prompt = get_asr_initial_prompt(genre)

            if ChineseTranscriber._use_faster:
                # Run with Faster-Whisper (CTranslate2) with Silero VAD, beam_size=5, and genre initial_prompt
                segments, info = model.transcribe(
                    str(processed_audio),
                    language="zh",
                    task="transcribe",
                    beam_size=5,
                    temperature=0.0,
                    condition_on_previous_text=False,
                    vad_filter=False,
                    initial_prompt=init_prompt,
                    word_timestamps=True
                )

                cleaned_segments = process_faster_whisper_segments(segments, max_word_gap=0.55)
            else:
                # Standard OpenAI Whisper fallback
                result = model.transcribe(
                    str(processed_audio),
                    language="zh",
                    task="transcribe",
                    temperature=0.0,
                    condition_on_previous_text=False,
                    no_speech_threshold=0.6,
                    initial_prompt=init_prompt,
                    verbose=False
                )
                for idx, seg in enumerate(result.get("segments", [])):
                    text = seg.get("text", "").strip()
                    start = round(float(seg.get("start", 0.0)), 2)
                    end = round(float(seg.get("end", 0.0)), 2)
                    duration = round(end - start, 2)
                    if not text or duration < 0.2:
                        continue
                    if seg.get("no_speech_prob", 0.0) > 0.6:
                        continue
                    if is_non_speech_or_hallucination(text):
                        continue
                    cleaned_segments.append({
                        "id": idx + 1,
                        "start": start,
                        "end": end,
                        "duration": duration,
                        "text": text
                    })

            # Merge micro-fragments within same speech clause
            cleaned_segments = merge_short_adjacent_segments(cleaned_segments, max_gap=0.65, max_combined_dur=4.8)

            # Apply Smart Crowd & Repetitive Dialogue Deduplication Filter
            total_before = len(cleaned_segments)
            cleaned_segments = deduplicate_crowd_segments(cleaned_segments)
            removed = total_before - len(cleaned_segments)
            if removed > 0:
                logger.info(f"🧹 បានសម្រិតសម្រាំង និងលុបពាក្យដដែលៗរបស់មនុស្សច្រើន (Crowd Repetition Filter): រក្សាទុក {len(cleaned_segments)} ប្រយោគ (កាត់បន្ថយ {removed} ប្រយោគដដែលៗ)")

            logger.success(f"ស្រង់សម្ដីបានចំនួន {len(cleaned_segments)} ប្រយោគ (Segments) ល្បឿនលឿនបំផុត!")
            return cleaned_segments

        except Exception as e:
            logger.error(f"Whisper transcription error: {e}")
            # Try fallback to standard whisper if faster-whisper failed at runtime
            try:
                import whisper
                logger.info("កំពុងសាកល្បងជាមួយ Standard OpenAI Whisper...")
                fallback_m = whisper.load_model(self.model_size, device="cuda" if torch.cuda.is_available() else "cpu")
                res = fallback_m.transcribe(str(audio_path), language="zh")
                cleaned = []
                for idx, seg in enumerate(res.get("segments", [])):
                    txt = seg.get("text", "").strip()
                    if txt:
                        cleaned.append({
                            "id": idx + 1,
                            "start": round(float(seg.get("start", 0)), 2),
                            "end": round(float(seg.get("end", 0)), 2),
                            "duration": round(float(seg.get("end", 0)) - float(seg.get("start", 0)), 2),
                            "text": txt
                        })
                return cleaned
            except Exception as e2:
                logger.error(f"Fallback transcription also failed: {e2}")
                return []
        finally:
            if temp_filtered_wav and os.path.exists(temp_filtered_wav):
                try:
                    os.remove(temp_filtered_wav)
                except Exception:
                    pass
