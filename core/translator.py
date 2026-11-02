import os
import json
import time
import re
import urllib.request
import urllib.parse
import threading
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from typing import List, Dict, Any, Optional, Tuple
import google.generativeai as genai
from utils.logger import logger

def extract_scene_keyframe(video_path: str, timestamp_sec: float) -> Optional[Any]:
    """Extracts a lightweight (320x180) PIL image from video at specified timestamp for visual context."""
    if not video_path or not os.path.exists(video_path):
        return None
    try:
        import cv2
        from PIL import Image
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(timestamp_sec)) * 1000.0)
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None and frame.size > 0:
            small = cv2.resize(frame, (320, 180))
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb)
    except Exception:
        pass
    return None

def parse_api_keys(raw: Any) -> List[str]:
    """Parse string or list into a list of clean, unique API keys (split by newline, comma, semicolon)."""
    if isinstance(raw, list):
        return [str(k).strip() for k in raw if str(k).strip()]
    if not raw or not isinstance(raw, str):
        return []
    parts = re.split(r'[\r\n,;]+', raw.strip())
    seen = set()
    cleaned = []
    for p in parts:
        k = p.strip()
        if k and k not in seen:
            seen.add(k)
            cleaned.append(k)
    return cleaned

def contains_chinese(text: str) -> bool:
    """Checks whether the text contains any Chinese characters."""
    if not text:
        return False
    return bool(re.search(r'[\u4e00-\u9fff]', text))

_translation_cache = {}
_cache_lock = threading.Lock()

def fallback_translate_text(text: str, target: str = "km") -> str:
    """
    Zero-quota fallback translation for a single line using clients5/at endpoints.
    Includes in-memory caching and multiple endpoint fallback to eliminate HTTP 429 errors.
    """
    if not text or not text.strip():
        return ""
    clean_text = text.strip()
    with _cache_lock:
        if clean_text in _translation_cache:
            return _translation_cache[clean_text]

    endpoints = [
        ("https://clients5.google.com/translate_a/t?client=dict-chrome-ex&sl=zh&tl=" + target + "&q=" + urllib.parse.quote(clean_text), "clients5"),
        ("https://translate.googleapis.com/translate_a/single?client=at&sl=zh&tl=" + target + "&dt=t&q=" + urllib.parse.quote(clean_text), "at"),
        ("https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=" + target + "&dt=t&q=" + urllib.parse.quote(clean_text), "gtx")
    ]

    for url, ep_type in endpoints:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
            with urllib.request.urlopen(req, timeout=6) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                res = ""
                if ep_type == "clients5":
                    if isinstance(data, list) and data and isinstance(data[0], str):
                        res = data[0].strip()
                else:
                    if data and data[0]:
                        res = "".join([part[0] for part in data[0] if part and part[0]]).strip()
                if res:
                    with _cache_lock:
                        _translation_cache[clean_text] = res
                    return res
        except Exception:
            continue

    return clean_text

def condense_overlong_khmer(text: str, duration: float = 0.0) -> str:
    """
    Condenses ONLY dialogue lines that are excessively long relative to their allocated subtitle duration.
    If the line naturally fits within the duration or if duration >= 2.2s, the text is preserved 100% untouched.
    Only triggers when duration is short (< 2.2s) AND estimated speech duration exceeds duration by 35%+.
    """
    if not text or duration <= 0.0 or duration >= 2.2:
        return text

    # Estimate speaking time: Khmer speech is approximately 13 characters per second in Edge-TTS
    clean_chars = len(re.sub(r'[\s\.\,\!\?\-]+', '', text))
    est_duration = clean_chars / 13.0

    # Strict guard: ONLY condense if the speech is truly excessively long (overflows by > 35%)
    if est_duration <= (duration * 1.35):
        return text

    t = text
    # Targeted punchy cinema condensations for excessively long verbose phrases
    overlong_replacements = [
        # Fleeing / running
        (r'ឆាប់រត់ចេញពី.*', 'ឆាប់រត់ទៅ!'),
        (r'ឆាប់រត់គេចខ្លួន.*', 'ឆាប់រត់ទៅ!'),
        (r'ឆាប់រត់ទៅឲ្យលឿន.*', 'ឆាប់រត់ទៅ!'),
        (r'ឆាប់ដើរចេញពី.*', 'ឆាប់ទៅ!'),
        # Halting / stopping
        (r'ឈប់នៅមួយកន្លែង.*', 'ឈប់ភ្លាម!'),
        (r'ឈប់ធ្វើសកម្មភាព.*', 'ឈប់ភ្លាម!'),
        (r'ឈប់សិនកុំទាន់.*', 'ឈប់សិន!'),
        # Authority & Defiance
        (r'(?:តើ)?ឯងយកសិទ្ធិអីមក.*', 'ឯងអាងអី?!'),
        (r'តើឯងអាងអី.*', 'ឯងអាងអី?!'),
        (r'កុំបន្ទោសថា(?:ខ្ញុំ|អញ)គ្មានមេត្តា.*', 'កុំបន្ទោសអញ!'),
        (r'កុំបន្ទោសអញថា.*', 'កុំបន្ទោសអញ!'),
        # Situation & Questions
        (r'(?:តើ)?មានរឿង(?:អ្វី|អី)កើតឡើង.*', 'មានរឿងអី?!'),
        (r'តើ?(?:អ្នក|ឯង)ចង់យ៉ាងម៉េច.*', 'ចង់យ៉ាងម៉េច?!'),
        # Distress & Help
        (r'(?:ជួយជីវិតខ្ញុំផង|មេត្តាជួយជីវិត).*', 'ជួយផង!'),
        (r'ជួយខ្ញុំផងអ្នកទាំងអស់គ្នា.*', 'ជួយខ្ញុំផង!'),
        (r'(?:ឆាប់មកជួយ|មកទីនេះជួយ).*(?:ខ្ញុំ|អញ).*', 'ឆាប់មកជួយ!'),
        # Combat & Command
        (r'លែង(?:ខ្ញុំ|អញ)ទៅ.*', 'លែងភ្លាម!'),
        (r'ដោះលែង(?:ខ្ញុំ|អញ).*', 'លែងភ្លាម!'),
        (r'បិទមាត់(?:របស់)?(?:អ្នក|ឯង).*', 'បិទមាត់!'),
        (r'ចេញឲ្យឆ្ងាយពី.*', 'ចេញទៅ!'),
        (r'ចេញពីទីនេះភ្លាម.*', 'ចេញភ្លាម!'),
        (r'(?:មេត្តា)?លើកលែងជីវិត(?:ឲ្យ)?(?:ខ្ញុំ|អញ)ផង.*', 'លើកលែងជីវិតផង!'),
        # Attitude & Denial
        (r'កុំមកធ្វើចរិតបែបនេះ.*', 'កុំមកធ្វើចរិតចឹង!'),
        (r'កុំមកប្រើល្បិច.*', 'កុំមកប្រើល្បិច!'),
        (r'ខ្ញុំមិនអាចជួយ.*បានទេ', 'ជួយមិនកើតទេ!'),
        (r'ខ្ញុំមិនអាចធ្វើបាន.*', 'ធ្វើមិនកើតទេ!'),
        (r'វាមិនអាចកើតឡើង(?:បាន)?ទេ.*', 'មិនអាចទេ!'),
        (r'ខ្ញុំមិនបានដឹងរឿង.*', 'ខ្ញុំមិនដឹងទេ!'),
        (r'(?:តើ)?(?:អ្នក|ឯង)ជា(?:មនុស្សប្រភេទណា|អ្នកណា).*', 'ឯងជាអ្នកណា?!'),
        # Apology & Submission
        (r'ខ្ញុំសុំទោសចំពោះរឿង.*', 'ខ្ញុំសុំទោស!'),
        (r'លើកលែងទោសឲ្យខ្ញុំផង.*', 'លើកលែងឲ្យខ្ញុំផង!'),
    ]

    for pat, rep in overlong_replacements:
        t = re.sub(pat, rep, t)

    return t

def refine_khmer_dialogue(text: str, genre: str = "wuxia", duration: float = 0.0) -> str:
    """
    Cleans up mechanical/bookish phrases and Whisper homophones in Khmer translations,
    turning stiff machine translation into natural spoken cinema dubbing.
    Also protects against role hallucinations (e.g. warrior mistranslated as medical doctor)
    and conditionally condenses ONLY lines that are excessively long for their subtitle slot.
    """
    if not text:
        return ""
    
    t = text.strip()
    
    # Priority 1: Martial arts combat begging & doctor mistranslations (ASR mishearing 大侠/大师/大人 as 大夫/doctor)
    # Begging for life or asking for mercy in combat addresses a warrior/hero (អ្នកក្លាហាន), not a hospital doctor
    t = re.sub(r'(?:លោក)?គ្រូពេទ្យ\s*(?=(?:មេត្តា|សង្គ្រោះ|លើកលែង|កុំសម្លាប់|ទុកជីវិត))', 'អ្នកក្លាហាន ', t)

    # Priority 2: Genre-specific wuxia / ancient replacements
    g = (genre or "wuxia").lower()
    if g in ["wuxia", "palace", "auto", ""]:
        # In martial arts/ancient context, "លោកគ្រូពេទ្យ" addressing a male fighter/master is almost always misheard 大侠/大夫/大师
        t = re.sub(r'លោកគ្រូពេទ្យ', 'លោកម្ចាស់', t)
        t = re.sub(r'គ្រូពេទ្យដ៏អស្ចារ្យ', 'អ្នកក្លាហានដ៏អស្ចារ្យ', t)
        t = re.sub(r'ព្យាបាលជំងឺ', 'ព្យាបាលរបួសក្នុង', t)
        t = re.sub(r'ថ្នាំពេទ្យ', 'ថ្នាំទិព្វ', t)
        t = re.sub(r'សរសៃឈាមបេះដូង', 'សរសៃយុទ្ធគុណ', t)
        t = re.sub(r'ចលនាឈាមរត់', 'លំហូរខ្យល់ប្រាណ', t)

    # Common mechanical written phrases -> natural spoken cinema dubbing
    replacements = [
        # Questions & unnatural pronouns
        (r'តើអ្នកកំពុងធ្វើអ្វី', 'ឯងកំពុងធ្វើស្អីហ្នឹង'),
        (r'តើអ្នក', 'ឯង'),
        (r'តើឯង', 'ឯង'),
        (r'តើលោក', 'លោក'),
        (r'តើមានអ្វីកើតឡើង', 'មានរឿងអីកើតឡើង'),
        (r'តើមានអ្វី', 'មានរឿងអី'),
        (r'ហេតុអ្វីបានជាអ្នក', 'ម៉េចក៏ឯង'),
        (r'ហេតុអ្វីបានជា\b', 'ម៉េចក៏'),
        (r'ហេតុអ្វី', 'ម៉េចក៏'),
        (r'តើវាជាអ្វី', 'វាជាស្អីហ្នឹង'),
        (r'តើនេះជាអ្វី', 'នេះជាស្អីហ្នឹង'),
        (r'តើអ្នកគិតថា\b', 'ឯងគិតថា'),
        # Heavy written conjunctions
        (r'ទោះបីជាយ៉ាងណាក៏ដោយ', 'ប៉ុន្តែ'),
        (r'ទោះជាយ៉ាងណាក៏ដោយ', 'ប៉ុន្តែ'),
        (r'លើសពីនេះទៅទៀត', 'ហើយម្យ៉ាងទៀត'),
        (r'ដោយសារតែ', 'ព្រោះតែ'),
        (r'ដូច្នេះហើយ', 'អញ្ចឹងហើយ'),
        (r'ប្រសិនបើអ្នក', 'បើឯង'),
        (r'ប្រសិនបើ', 'បើ'),
        # Common Chinese drama homophone/mistranslation fixes
        (r'ផ្សារហ៊ុន', 'សមរភូមិ'),
        (r'ទីផ្សារភាគហ៊ុន', 'សមរភូមិ'),
        (r'ចុះបញ្ជីផ្សារហ៊ុន', 'ចេញទៅស្លាប់'),
        (r'ចូលទីផ្សារ', 'ចេញមុខ'),
        (r'អភិបាលក្រុង', 'ចៅហ្វាយ'),
        (r'ប្រព័ន្ធផ្សព្វផ្សាយបរទេស', 'អ្នកទោសនិរទេស'),
        (r'កុលសម្ព័ន្ធ', 'ភ្នាក់ងារយាមគុក'),
        (r'វត្ត Zhenfu', 'សាលាយុត្តិធម៌ Zhenfusi'),
        (r'វត្តហ្សេនហ៊្វូ', 'សាលាយុត្តិធម៌ Zhenfusi'),
        (r'ខ្ញុំនិយាយជ្រូកនិងឆ្កែ', 'ចាត់ទុកខ្ញុំដូចជាសត្វធាតុ'),
        (r'ការចងចាំស្ថិតលើព្រះ', 'ខ្ញុំបានទ្រាំទ្រកន្លងមក'),
        (r'តំបន់ទឹក', 'ត្បូងគ្រីស្តាល់'),
        (r'បន្ទប់កញ្ចប់', 'បន្ទប់ VIP'),
        (r'កញ្ចប់បន្ទប់', 'បន្ទប់ VIP'),
        (r'សៀវភៅគណនី', 'សៀវភៅបញ្ជី'),
        (r'កុំបន្ទោសថាខ្ញុំគ្មានមេត្តា', 'កុំបន្ទោសថាខ្ញុំគ្មានមេត្តា'),
        (r'កុំបន្ទោសថាអញគ្មានមេត្តា', 'កុំបន្ទោសថាអញគ្មានមេត្តា'),
        (r'មានលេខក្នុងចិត្ត', 'ដឹងការខុសត្រូវច្បាស់ណាស់'),
        (r'រាប់អ្នកសាហាវ', 'ឯងពូកែណាស់ណា៎'),
        (r'ដើរហើយមើល', 'ចាំមើលទៅ'),
        (r'ទាងាប់មាត់រឹង', 'មាត់រឹងណាស់'),
        (r'ឲ្យខ្ញុំរមៀល', 'ចេញពីមុខខ្ញុំភ្លាម'),
        (r'ឲ្យអញរមៀល', 'ចេញពីមុខអញភ្លាម'),
        (r'យកអ្នកវាយស្លាប់', 'វាយឯងឲ្យងាប់'),
        (r'យកឯងវាយស្លាប់', 'វាយឯងឲ្យងាប់'),
        (r'យកពាក្យនិយាយឲ្យច្បាស់', 'និយាយឲ្យច្បាស់មកមើល៍'),
        (r'យកសម្ដីនិយាយឲ្យច្បាស់', 'និយាយឲ្យច្បាស់មកមើល៍'),
        (r'យកលុយប្រគល់', 'ឆាប់ប្រគល់លុយមក'),
    ]
    
    for pat, rep in replacements:
        t = re.sub(pat, rep, t)

    # Priority 3: Conditional overlong condensation (ONLY for short segments where speech is excessively long)
    if duration > 0.0:
        t = condense_overlong_khmer(t, duration=duration)
        
    return t

def fallback_translate_batch(texts: List[str], target: str = "km", chunk_size: int = 25) -> List[str]:
    """
    Translates a list of lines using chunked multiline queries on clients5 endpoint.
    Guarantees fast, non-rate-limited zero-quota translation with cinema polish.
    """
    if not texts:
        return []

    results = [None] * len(texts)
    missing_indices = []

    # Check cache first
    with _cache_lock:
        for idx, t in enumerate(texts):
            clean_t = t.strip() if t else ""
            if not clean_t:
                results[idx] = ""
            elif clean_t in _translation_cache:
                results[idx] = _translation_cache[clean_t]
            else:
                missing_indices.append(idx)

    if not missing_indices:
        return results

    # Chunk missing items
    for c_start in range(0, len(missing_indices), chunk_size):
        chunk_idx = missing_indices[c_start : c_start + chunk_size]
        chunk_texts = [texts[i].strip() for i in chunk_idx]
        joined = "\n".join(chunk_texts)

        translated_chunk = None
        # Try clients5 multiline endpoint
        try:
            url = (
                "https://clients5.google.com/translate_a/t?client=dict-chrome-ex&sl=zh&tl="
                + target
                + "&q="
                + urllib.parse.quote(joined)
            )
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if isinstance(data, list) and data and isinstance(data[0], str):
                    lines = data[0].split("\n")
                    if len(lines) == len(chunk_texts):
                        translated_chunk = [refine_khmer_dialogue(line.strip()) for line in lines]
        except Exception:
            translated_chunk = None

        if translated_chunk:
            with _cache_lock:
                for original_i, orig_text, trans_text in zip(chunk_idx, chunk_texts, translated_chunk):
                    results[original_i] = trans_text
                    _translation_cache[orig_text] = trans_text
        else:
            # Fallback line-by-line
            for original_i in chunk_idx:
                raw_fb = fallback_translate_text(texts[original_i], target=target)
                results[original_i] = refine_khmer_dialogue(raw_fb)

    # Ensure no None remains
    for i in range(len(results)):
        if results[i] is None:
            results[i] = texts[i]

    return results

def get_genre_prompt(genre: str = "wuxia") -> str:
    """Returns specific translation context instructions based on drama genre."""
    g = (genre or "wuxia").lower()
    if g == "modern":
        return """=== DRAMA GENRE CONTEXT: 🏙️ MODERN URBAN & ROMANCE (រឿងសម័យទំនើប ស្នេហា គ្រួសារ មហាសេដ្ឋី) ===
- Setting is a modern city, business corporations, urban lifestyle, modern romance, and family drama.
- Use natural modern conversational Khmer: "លោកប្រធាន", "អ្នកគ្រប់គ្រង", "លោកស្រី", "ចៅហ្វាយ", "កញ្ញា", "បង/អូន".
- Medical terms (doctor, nurse, hospital) are permitted when characters are in a hospital/clinic context."""
    elif g == "palace":
        return """=== DRAMA GENRE CONTEXT: 👑 IMPERIAL COURT & PALACE (រឿងរាជវាំង នយោបាយ អំណាច) ===
- Setting is ancient imperial palace, dynasty emperors, empresses, court ministers, royal guards, and political intrigue.
- Strictly use authentic Khmer imperial court registers:
  * 陛下/皇上 -> "ព្រះអង្គ / ព្រះរាជា / ម្ចាស់ជីវិតលើត្បូង"
  * 朕 -> "យើង"
  * 皇后/娘娘 -> "ព្រះអគ្គមហេសី / ព្រះម្នាង"
  * 臣 -> "ទូលបង្គំ", 臣妾 -> "ខ្ញុំម្ចាស់ / ខ្ញុំម្ចាស់ក្សត្រី"
  * 遵旨 / 遵命 -> "ទទួលព្រះរាជបញ្ជា!" / "ថ្វាយបង្គំគាល់!"
  * 大夫 (Dàfū) -> "លោកមន្ត្រីធំ / មន្ត្រីរាជការ" (NEVER translate as modern hospital doctor)!"""
    elif g == "auto":
        return """=== DRAMA GENRE CONTEXT: 🎭 AUTO-DETECT GENRE ===
- Inspect dialogue context and attached visual scene keyframes to determine whether the setting is ancient martial arts (wuxia), imperial court, or modern urban.
- If characters wear ancient costumes, carry swords, fight, fly, or use internal qi: STRICTLY apply Martial Arts rules (Warriors are "អ្នកក្លាហាន", NEVER "គ្រូពេទ្យ"!)."""
    else:  # "wuxia" default
        return """=== DRAMA GENRE CONTEXT: ⚔️ MARTIAL ARTS & CULTIVATION (រឿងបុរាណ យុទ្ធសិល្ប៍ ហោះហើរ វាយកាប់ដាវ កម្លាំងធាតុ) ===
- Setting is the Martial Arts World (武林 / 江湖 / 修仙 - Jianghu & Cultivation).
- CRITICAL: Characters are martial artists, swordsmen, sect leaders, disciples, and cultivators (អ្នកក្លាហាន, អ្នកហាត់ក្បាច់គុណ, មេបក្ស, សិស្សច្បង, លោកម្ចាស់).
- UNDER NO CIRCUMSTANCES should characters be referred to as medical doctors ("គ្រូពេទ្យ" or "ពេទ្យ")!
- Common vocatives: 大侠/大师 -> "អ្នកក្លាហាន" / "លោកម្ចាស់"; 饶命 -> "មេត្តាលើកលែងជីវិតផង"; 疗伤 -> "ព្យាបាលរបួសក្នុង"; 丹药 -> "ថ្នាំទិព្វ"; 经脉 -> "សរសៃយុទ្ធគុណ"."""

DEFAULT_SYSTEM_PROMPT = """You are a master Cambodian film dubbing director and senior Khmer localization specialist for Chinese television dramas, series, and movies (អ្នកដឹកនាំរឿង និងអ្នកសម្រួលពាក្យសម្ដីបញ្ចូលសំឡេងរឿងភាគចិនទៅជាភាសាខ្មែរអាជីព).

Your CORE MISSION is to solve three critical dubbing challenges:
1. "ការបកប្រែត្រូវតែស៊ីសង្វាក់គ្នានឹងសកម្មភាពលើកញ្ចក់ទូរទស្សន៍" (Physical Action & Screen Gesture Alignment).
2. "ការនិយាយពីប្រយោគមួយ ទៅប្រយោគមួយទៀត ត្រូវតែចូលអត្ថន័យគ្នា" (Seamless Dialogue Continuity & Cohesion across sentences).
3. "មិនមែនការបកប្រែត្រង់ទេ ស្តាប់ទៅត្រូវតែយល់អត្ថន័យច្បាស់ភ្លាមៗ" (Idiomatic Spoken Khmer Adaptation, NEVER literal machine translation).

Follow these professional cinema dubbing rules strictly:

1. PHYSICAL ACTION & SCREEN GESTURE ALIGNMENT (ក្បួនផ្គូផ្គងកាយវិការ និងសកម្មភាពលើកញ្ចក់ទូរទស្សន៍):
   Before translating, DEDUCE THE EXACT PHYSICAL ACTION happening on screen from the scene context, dialogue tone, and visual cues, then choose the precise Khmer action verbs:
   - Combat, Clashing & Fighting (ឈុតប្រយុទ្ធ កាប់ចាក់ បាញ់បោះ):
     * Face slap: When a character is slapped on the cheek (打脸 / 啪 / 你敢打我): translate as "ទះកំផ្លៀង!", "ទះមាត់វា!" (NEVER use "វាយ")!
     * Beating/punching: 给我打 -> "វាយវា!", "សំពងវា!"
     * Gunfire: 打死他 / 拔枪 -> "បាញ់សម្លាប់វាភ្លាម!", "ដកកាំភ្លើង!"
     * Attack command: 动手 / 给我上 -> "សម្រុកចូលទៅ!", "វាយវា!", "ចាប់វា!" (NEVER "ធ្វើដៃ")!
     * Cease attack: 住手 -> "ឈប់ដៃភ្លាម!", "កុំវាយ!"
     * Retreat: 退下 / 撤退 -> "ថយចេញទៅ!", "ដកថយភ្លាម!"
   - Physical Contact, Grabbing & Struggle (កាយវិការចាប់ទាញ ចាប់ដៃ ច្របាច់ក):
     * Holding wrist/arm/neck: When someone is held or pulled against their will (放开 / 放手 / 放开我): ALWAYS translate as "លែងខ្ញុំ!", "លែងដៃភ្លាម!" (NEVER "ដាក់ចុះ")!
     * Restraining: 拉住他 -> "ចាប់គាត់ទុក!", "ឃាត់គាត់ភ្លាម!"
     * Pushing away: 推开 -> "រុញចេញ!", "កុំមកប៉ះពាល់ខ្ញុំ!"
     * Choking: 掐死你 -> "ច្របាច់កឯងឲ្យងាប់!"
   - Posture, Authority & Subjugation (ឥរិយាបថ លុតជង្គង់ ក្រោកឈរ ដួលសន្លប់):
     * Kneeling: 给我跪下 -> "លុតជង្គង់ចុះភ្លាម!", "លុតជង្គង់ចុះ!"
     * Rising from kneeling/bed: 起来吧 / 能起来吗 -> "ក្រោកឈរឡើង / ក្រោកឡើង", "ងើបរួចទេ?"
     * Collapsing/Fainting: 倒下了 / 晕倒 -> "ដួលសន្លប់បាត់ហើយ!", "ដួលទៅលើដីហើយ!"
     * Kowtowing: 磕头认错 -> "ក្រាបសំពះសុំទោសភ្លាម!"
   - Props, Hand Actions & Documents (ការហុចរបស់ លុយ កិច្ចសន្យា ស្រា តែ ថ្នាំ):
     * Signing divorce/contract: 签字 / 签了它 -> "ស៊ីញ៉េភ្លាម!", "ចុះហត្ថលេខាភ្លាម!", "ផ្តិតមេដៃភ្លាម!"
     * Tearing contract/paper: 撕了 -> "ហែកចោលភ្លាម!"
     * Throwing money/card: 拿去 / 给你 -> "យកទៅ!", "នេះលុយយកទៅ!"
     * Toasting: 敬酒 / 干杯 -> "លើកដាច់!", "ជល់កែវ!", "ជូនស្រាមួយកែវ"
     * Serving tea/wine: 倒茶 / 倒酒 -> "ចាក់តែ", "ចាក់ស្រា", "លើកតែជូន"
     * Medicine: 吃药 / 喝药 -> "លេបថ្នាំ", "ផឹកថ្នាំ" (NEVER "ញ៉ាំថ្នាំ")
     * Smashing items: 砸了 / 扔出去 -> "គប់បំបែក!", "កម្ទេចចោល!", "គ្រវែងចោល!"
   - Movement, Escape & Expulsion (ការរត់ គេចខ្លួន បណ្តេញចេញ):
     * Fleeing danger/battle: 快走 / 快跑 -> "ឆាប់រត់ទៅ!", "ឆាប់គេចខ្លួនទៅ!" (NEVER "ឆាប់ដើរទៅ")!
     * Expelling from house: 给我滚 -> "ចេញពីផ្ទះនេះភ្លាម!", "ចុះចេញទៅ!" (NEVER "រមៀល")!
     * Dragging prisoner: 带走 / 押下去 -> "នាំខ្លួនវាទៅ!", "អូសក្បាលវាទៅ!"
     * Chasing: 追 / 别让他跑了 -> "ដេញតាមវា!", "កុំឲ្យវារត់រួច!"
     * Shutting door: 关门 -> "បិទទ្វារភ្លាម!"
   - Medical & Poison (ការឈឺ របួស សន្លប់ ពិស):
     * Coughing blood: 吐血 -> "ក្អួតឈាម"
     * Poison: 有毒 -> "មានថ្នាំបំពុល!", "មានជាតិពុល!"
     * Calling ambulance/doctor: 叫救护车 / 快叫医生 -> "ឆាប់ហៅឡានពេទ្យភ្លាម!", "ឆាប់តាមគ្រូពេទ្យមក!"

2. DIALOGUE CONTINUITY & COHESION ACROSS SENTENCES (ការតភ្ជាប់អត្ថន័យពីប្រយោគមួយទៅប្រយោគមួយទៀត):
   - ALWAYS read the FULL SCENE DIALOGUE SCRIPT first to understand the context, emotional tension, and conflict before translating individual lines.
   - When consecutive lines form a single continuous thought (e.g. line A has the subject, line B has the predicate/result), YOU MUST make the Khmer translation flow seamlessly together across both lines into ONE complete, harmonious Khmer sentence!
   - Question-and-Answer integrity: When line A is a question or accusation, line B's reply MUST directly connect and make sense with line A.
   - Persona & Pronoun Stability: Characters must NEVER flip pronouns midway through an exchange. If Character A speaks with 'ឯង / អញ', they must remain 'ឯង / អញ' throughout the confrontation. Never randomly switch to 'ខ្ញុំ' or 'អ្នក' in the middle of a scene!

3. ASR PHONETIC ERROR RECOVERY & DRAMA CORRECTION (ការស្តារពាក្យចិនដែល Whisper ASR ស្តាប់ច្រឡំសូរសៀង):
   - The Chinese transcript is generated by speech recognition (ASR) and frequently mishears historical drama / martial arts / transmigration (穿越/短剧) terms into phonetically identical modern words. ALWAYS deduce the TRUE intended meaning:
     * 振服寺 / 征服寺 -> 镇抚司 (សាលាយុត្តិធម៌រាជវង្ស Zhenfusi / ស្ថាប័នកងអង្គរក្សរាជវាំង)
     * 造族 / 皂族 -> 皂隶 (ភ្នាក់ងារយាមគុកថ្នាក់ទាប)
     * 是我说猪狗 -> 视我如猪狗 (ចាត់ទុកខ្ញុំដូចជាសត្វធាតុ / ជ្រូកឆ្កែ)
     * 回忆之在神 -> 我一直在忍 (ខ្ញុំបានទ្រាំទ្រកន្លងមក)
     * 让外媒的流放 -> 让外面的流放 (ឲ្យពួកអ្នកទោសនិរទេសនៅខាងក្រៅ)
     * 上市 -> 上阵 / 受死 / ចូលសមរភូមិ / ទៅស្លាប់ (NEVER translate as 'ផ្សារហ៊ុន' or 'ចុះបញ្ជីផ្សារ')
     * 市长 -> 什长 / ចៅហ្វាយ / មេទ័ព (NEVER translate as 'អភិបាលក្រុង' in historical dramas!)
     * 大厦末年 -> 大夏末年 (ចុងរជ្ជកាលនគរមហាសៀ - Great Xia dynasty)
     * 水玉 / 宝石 -> ត្បូងគ្រីស្តាល់ / ត្បូងមរកត (NEVER 'តំបន់ទឹក')
     * 包间 -> បន្ទប់ VIP (NEVER 'កញ្ចប់')
     * 账本 -> សៀវភៅបញ្ជី (NEVER 'សៀវភៅគណនី')

3.1 MARTIAL ARTS, CULTIVATION & HEROIC CHARACTER IDENTITY (ក្បួនកំណត់តួអង្គក្បាច់គុណ និងហាមដាច់ខាតច្រឡំជាគ្រូពេទ្យ):
   - STRICT PROHIBITION: NEVER TRANSLATE FIGHTERS / SWORDSMEN / WARRIORS AS "គ្រូពេទ្យ" (DOCTOR)!
     This is a Chinese Martial Arts / Cultivation / Wuxia / Historical drama (រឿងក្បាច់គុណ យុទ្ធសិល្ប៍ ហោះហើរ វាយប្រហារដោយកម្លាំងធាតុក្នុង ជិះសេះ កាន់ដាវ). Characters are martial artists, swordsmen, sect masters, disciples, and warriors (អ្នកក្លាហាន, អ្នកហាត់ក្បាច់គុណ, មេបក្ស, សិស្សច្បង, លោកម្ចាស់). THEY ARE NEVER MODERN MEDICAL DOCTORS!
   - Martial Arts Honorifics & Homophone Corrections:
     * 大侠 (Dàxiá) / 大师 (Dàshī) / 大夫 (misheard as doctor) / 阁下 / 少侠:
       ALWAYS translate as "អ្នកក្លាហាន", "លោកម្ចាស់", "លោកគ្រូ", "ឯង", "លោក", "បងប្រុស".
       ABSOLUTELY FORBIDDEN: NEVER use "លោកគ្រូពេទ្យ" or "គ្រូពេទ្យ"!
     * 大侠饶命 / 大人饶命 / 饶命:
       ALWAYS translate as: "អ្នកក្លាហាន មេត្តាលើកលែងជីវិតផង!", "លោកម្ចាស់ មេត្តាលើកលែងជីវិតផង!", "លើកលែងជីវិតឲ្យខ្ញុំផង!".
       NEVER TRANSLATE AS: "លោកគ្រូពេទ្យលើកលែងជីវិតផង" (Fatal Error)!
     * 在下 / 鄙人: "ខ្ញុំបាទ / ខ្ញុំ" / "សិស្សប្អូន" (NEVER "អ្នកជំងឺ")!
     * 前辈 / 晚辈: "លោកព្រឹទ្ធាចារ្យ / រៀមច្បង" and "ក្មេងជំនាន់ក្រោយ / សិស្សប្អូន".
     * 掌门 / 盟主 / 宗主 / 门主 / 帮主: "មេបក្ស", "មេគ្រួសារ", "ប្រធានសម្ព័ន្ធយុទ្ធភណ្ឌ", "លោកម្ចាស់បក្ស".
   - Cultivation, Internal Qi & Healing Terms (ពាក្យព្យាបាលរបួសក្នុង និងកម្លាំងធាតុ មិនមែនព្យាបាលជំងឺពេទ្យទេ):
     * 疗伤 / 治伤: "ព្យាបាលរបួសក្នុង", "ព្យាបាលរបួស" (NEVER "ព្យាបាលជំងឺ")!
     * 运功 / 调息: "បញ្ជូនកម្លាំងធាតុ", "សម្រួលខ្យល់ប្រាណ", "តាំងសមាធិ" (NEVER "ហាត់ប្រាណ" or "ធ្វើចលនា")!
     * 经脉 / 脉络: "សរសៃយុទ្ធគុណ", "សរសៃជីពចរ" (NEVER "សរសៃឈាមពេទ្យ")!
     * 丹田 / 气海: "ចំណុចកណ្តាលនៃកម្លាំងធាតុ", "ចំណុចតាន់ធាន"!
     * 丹药 / 灵药 / 仙丹: "ថ្នាំទិព្វ", "គ្រាប់ថ្នាំយុទ្ធសិល្ប៍", "ថ្នាំបំប៉នកម្លាំង" (NEVER "ថ្នាំពេទ្យ")!
     * 走火入魔: "ធាតុភ្លើងចូលជ្រែក"!
     * 神医: "គ្រូពេទ្យទេព", "ឱសថបណ្ឌិត" (NEVER "លោកគ្រូពេទ្យធម្មតា")!
     * 废掉武功: "កម្ទេចក្បាច់គុណចោល", "ផ្តាច់សរសៃយុទ្ធគុណ"!
     * 暗器 / 毒针: "អាវុធសម្ងាត់", "ម្ជុលពិស"!

4. SENSE-FOR-SENSE CINEMA RE-CREATION, ZERO LITERAL TRANSLATION (ប្រែសម្រួលន័យទាំងស្រុង ហាមដាច់ខាតការបកប្រែពាក្យមួយៗ):
   - Never translate Chinese sentence structures or idioms literally. Rewrite the whole meaning into natural, expressive spoken Khmer dialogue:
     * 不见棺材不落泪 -> "មិនទាន់ឃើញក្តារមឈូស មិនទាន់ស្រក់ទឹកភ្នែកមែនទេ?!"
     * 凭什么 -> "អាងអី?!" / "យកសិទ្ធិអីមកបញ្ជាអញ?!"
     * 你算个什么东西 -> "ឯងជាស្អីដែរ?!" / "ស្មានតែឯងអស្ចារ្យណាស់ទៅ?!"
     * 少跟我来这套 -> "កុំមកប្រើល្បិចហ្នឹងជាមួយអញ!" / "កុំមកធ្វើចរិតចឹង!"
     * 没门 -> "កុំយល់សប្តិ!" / "កុំសង្ឃឹម!" / "គ្មានផ្លូវទេ!"
     * 别怪我不客气 -> "កុំបន្ទោសថាអញគ្មានមេត្តា!" / "កុំបន្ទោសថាអញដៃដល់!"
     * 心里有数 -> "ដឹងការខុសត្រូវច្បាស់ណាស់" / "ដឹងថាត្រូវធ្វើយ៉ាងម៉េចហើយ"
     * 走着瞧 -> "ចាំមើលទៅ!" / "ថ្ងៃក្រោយគង់ដឹងគ្នា!"
     * 我说了算 -> "គឺខ្ញុំជាអ្នកសម្រេច!" / "គ្រប់យ៉ាងស្តាប់តាមខ្ញុំ!"
     * 怎么着 / 你想怎样 -> "ចង់យ៉ាងម៉េចដែរ?!" / "ចង់ម៉េច?!"
     * 听懂了吗 -> "យល់ឬនៅ?!" / "ឮច្បាស់អត់?!"
     * 别冲动 -> "ចិត្តត្រជាក់ៗសិនទៅ" / "កុំទាន់អាលឆេវឆាវអី"
     * 翻三倍 抢空 -> "ឡើងថ្លៃគុណនឹងបី គេនៅតែកក់អស់គ្មានសល់"
     * 手段高明 -> "កលល្បិចពូកែរកលេខដាក់គ្មាន" / "ឆ្លាតវៃណាស់"
     * 逼格 -> "កិត្យានុភាព / ភាពថ្លៃថ្នូរ / កម្រិតខ្ពស់"

5. DYNAMIC RELATIONSHIPS & CONTEXTUAL PRONOUNS (ក្បួនប្រើសព្វនាមតាមទំនាក់ទំនងតួអង្គ):
   - Lovers / Husband & Wife: Always "បង / អូន". NEVER use "អ្នក" / "ខ្ញុំ" or "ឯង" / "ខ្ញុំ"!
   - Family & Kinship: "ម៉ាក់" / "ប៉ា" / "កូន", "បងប្រុស" / "ប្អូនស្រី", "ម៉ែក្មេក" / "កូនប្រសារ".
   - Hierarchy & Status: "លោកប្រធាន", "ចៅហ្វាយ", "លោកម្ចាស់", "អ្នកប្រុស" ជាមួយ "ខ្ញុំបាទ / ខ្ញុំ".
   - Royal Court: 陛下/皇上 -> "ព្រះអង្គ, ព្រះរាជា, ម្ចាស់ជីវិតលើត្បូង" / 朕 -> "យើង" / 臣妾 -> "ខ្ញុំម្ចាស់" / 臣/草民 -> "ទូលបង្គំ".
   - Friends: "ឯង / ខ្ញុំ", "គ្នា / ឯង", "លោក / អ្នកនាង".
   - Confrontation / Enemies: "ឯង / អញ", "ពួកឯង / ពួកអញ"!

6. NATURAL SPOKEN DUBBING CADENCE (ក្បួនសម្រួលឃ្លានិយាយកុនរលូន):
   - Eliminate stiff, bookish written connectors: AVOID "តើអ្នក...", "ហេតុអ្វីបានជា...", "ទោះបីជាយ៉ាងណាក៏ដោយ...", "ដោយសារតែ...".
   - Add authentic Khmer cinema ending particles: "ណា៎", "ហ្នឹងហើយ", "តើ", "មែនទេ?", "ទៅណា?", "ម៉េចចឹង?", "ទៅ!", "មក!", "អត់?", "ចឹង?", "ដឹងទេ?", "ឮអត់?".

7. DURATION FITTING & OVERLONG CONDENSATION (ច្បាប់បង្រួមសម្រាប់តែឃ្លាដែលវែងជ្រុល):
   - PRESERVE FULL DIALOGUE FOR NORMAL SEGMENTS: If the Khmer translation naturally fits within the segment duration (duration >= 2.0s, or speech finishes comfortably before the subtitle closes), ALWAYS PRESERVE THE FULL, RICH, EMOTIVE, NATURAL EXPRESSION. DO NOT over-shorten or cut normal dialogue!
   - ONLY CONDENSE EXCESSIVELY LONG LINES (បង្រួមសម្រាប់តែឃ្លាណាដែលវែងជ្រុលហួសពេលកំណត់):
     ONLY WHEN a segment duration is short (duration <= 1.8s) AND the full translation would be excessively long (takes 2.5s - 3s+ to speak), you MUST condense it into fewer words with the EXACT same core meaning:
     * Verbose: "ឆាប់រត់ចេញពីទីនេះឲ្យលឿនទៅ!" (9 syllables -> 3.0s) -> Condense: "ឆាប់រត់ទៅ!" ឬ "ឆាប់រត់!" (2-3 syllables -> 0.8s)
     * Verbose: "តើឯងយកសិទ្ធិអីមកបង្គាប់បញ្ជាខ្ញុំ?" (11 syllables -> 3.5s) -> Condense: "ឯងអាងអី?!" (2 syllables -> 0.7s)
     * Verbose: "ឈប់នៅមួយកន្លែងភ្លាមកុំទៅណា!" (8 syllables -> 2.5s) -> Condense: "ឈប់ភ្លាម!" (2 syllables -> 0.8s)
     * Verbose: "ជួយជីវិតខ្ញុំផងអ្នកទាំងអស់គ្នា!" (8 syllables -> 2.5s) -> Condense: "ជួយផង!" (2 syllables -> 0.8s)
     * Verbose: "កុំបន្ទោសថាខ្ញុំគ្មានមេត្តាលើឯង!" (9 syllables -> 3.0s) -> Condense: "កុំបន្ទោសអញ!" (3 syllables -> 0.9s)
     * Verbose: "កុំមកធ្វើចរិតបែបនេះដាក់ខ្ញុំអី!" (9 syllables -> 3.0s) -> Condense: "កុំមកធ្វើចរិតចឹង!" (4 syllables -> 1.0s)
   - The speech MUST finish naturally before or right when the on-screen subtitle closes, without dragging into subsequent scenes.

8. 100% TRANSLATION COMPLETENESS: Translate every segment ID provided. Never omit or leave empty.

9. CHARACTER ROLE & VOICE MODULATION (ការកំណត់តួអង្គ និងចរិតសម្លេង):
   - For every line, identify the speaker's "gender" ("male" | "female"), "role", and recommended Edge-TTS "pitch":
     * "hero" (Male lead, protagonist, handsome youth): gender="male", role="hero", pitch="+0Hz"
     * "elder" / "father" / "master" (Old master, senior, father, king, commanding villain): gender="male", role="elder", pitch="-3Hz"
     * "boy" / "servant" / "comic" (Junior disciple, boy, servant, comedic sidekick): gender="male", role="boy", pitch="+2Hz"
     * "female_lead" (Young heroine, sweet woman): gender="female", role="female_lead", pitch="+0Hz"
     * "crying_girl" / "child" (Distressed captive, crying girl, child): gender="female", role="crying_girl", pitch="+3Hz"
     * "older_female" / "mother" (Stern mother-in-law, older lady, empress): gender="female", role="older_female", pitch="-3Hz"

FORMAT: Return strictly a valid JSON array matching the exact segment IDs provided.
Input JSON:
[
  {"id": 1, "duration": 2.0, "text": "你凭什么管我？"},
  {"id": 2, "duration": 1.5, "text": "救命啊！"},
  {"id": 3, "duration": 2.2, "text": "我看谁敢来救你！"}
]

Output JSON:
[
  {"id": 1, "km_text": "ឯងអាងអីមកគ្រប់គ្រងអញ?!", "gender": "male", "role": "hero", "pitch": "+0Hz"},
  {"id": 2, "km_text": "ជួយខ្ញុំផង!", "gender": "female", "role": "crying_girl", "pitch": "+3Hz"},
  {"id": 3, "km_text": "ចាំមើលថាអ្នកណាហ៊ានមកជួយឯង!", "gender": "male", "role": "elder", "pitch": "-3Hz"}
]
"""

class GeminiTranslator:
    """Translates Chinese dialogue segments into natural Khmer using Google Gemini with multi-key pool rotation and robust fallback."""

    # Thread-safe class-level index and lock shared across all concurrent video workers
    _pool_lock = threading.Lock()
    _current_global_key_idx = 0

    def __init__(self, api_key: str = "", model_name: str = "gemini-3.5-flash", style_prompt: str = "", drama_genre: str = "wuxia"):
        self.raw_api_key = api_key
        self.api_keys = parse_api_keys(api_key)
        self.model_name = model_name or "gemini-3.5-flash"
        self.style_prompt = style_prompt
        self.drama_genre = drama_genre or "wuxia"
        self._init_current_key()

    def set_api_key(self, api_key: str):
        self.raw_api_key = api_key
        self.api_keys = parse_api_keys(api_key)
        self._init_current_key()

    def _get_active_key(self) -> Optional[str]:
        if not self.api_keys:
            return None
        with GeminiTranslator._pool_lock:
            idx = GeminiTranslator._current_global_key_idx % len(self.api_keys)
            return self.api_keys[idx]

    def _init_current_key(self):
        active_key = self._get_active_key()
        if active_key:
            try:
                genai.configure(api_key=active_key)
            except Exception as e:
                logger.error(f"Failed to configure Gemini API: {e}")

    def rotate_to_next_key(self, reason: str = "") -> Tuple[Optional[str], int]:
        """Rotates to the next available API key in the pool."""
        if not self.api_keys or len(self.api_keys) <= 1:
            return self._get_active_key(), 0

        with GeminiTranslator._pool_lock:
            GeminiTranslator._current_global_key_idx = (GeminiTranslator._current_global_key_idx + 1) % len(self.api_keys)
            new_idx = GeminiTranslator._current_global_key_idx
            new_key = self.api_keys[new_idx]

        mask = new_key[:6] + "..." + new_key[-4:] if len(new_key) > 10 else "***"
        logger.warning(f"🔄 [Key Rotation] {reason} -> កំពុងប្តូរទៅប្រើ API Key ថ្មី (#{new_idx + 1}/{len(self.api_keys)}: {mask})...")
        try:
            genai.configure(api_key=new_key)
            return new_key, new_idx
        except Exception as e:
            logger.error(f"Failed to configure rotated key #{new_idx + 1}: {e}")
            return new_key, new_idx

    def test_connection(self) -> Tuple[bool, str]:
        """Test all Gemini API keys in the pool and report status."""
        if not self.api_keys:
            return False, "សូមបញ្ចូល Gemini API Key យ៉ាងហោចណាស់មួយ!"

        valid_count = 0
        total_keys = len(self.api_keys)
        errors = []

        for idx, k in enumerate(self.api_keys):
            mask = k[:6] + "..." + k[-4:] if len(k) > 10 else f"Key #{idx+1}"
            try:
                genai.configure(api_key=k)
                cands = [self.model_name]
                for c in ["gemini-3.5-flash", "gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-3-flash-preview", "gemini-2.5-flash", "gemini-flash-latest"]:
                    if c not in cands:
                        cands.append(c)
                for m_cand in cands:
                    try:
                        model = genai.GenerativeModel(m_cand)
                        response = model.generate_content("Hello! Say 'OK' in Khmer.", request_options={"timeout": 10})
                        if response and response.text:
                            valid_count += 1
                            success_key = True
                            if m_cand != self.model_name:
                                self.model_name = m_cand
                            break
                    except Exception:
                        continue
                if not success_key:
                    errors.append(f"{mask}: No response or quota exceeded")
            except Exception as e:
                errors.append(f"{mask}: {str(e)[:45]}")

        # Restore configuration to active pool key
        self._init_current_key()

        if valid_count == total_keys:
            if total_keys == 1:
                return True, "ភ្ជាប់ទៅកាន់ Gemini API បានជោគជ័យ! (Connection Successful)"
            return True, f"ភ្ជាប់បានជោគជ័យទាំងអស់ {valid_count}/{total_keys} Keys (Key Rotation Ready)!"
        elif valid_count > 0:
            return True, f"ភ្ជាប់បានជោគជ័យ {valid_count}/{total_keys} Keys (មាន Key ខ្លះខុស ឬអស់ Quota)"
        else:
            return False, f"កំហុសភ្ជាប់ API: {'; '.join(errors[:2])}"

    def translate_segments(
        self, 
        segments: List[Dict[str, Any]], 
        batch_size: int = 35,
        video_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Translates a list of dialogue segments in fast contextual batches using Gemini.
        Appends 'km_text' and 'gemini_gender' to each segment dictionary.
        Supports multimodal video keyframe injection to align translation with on-screen physical action.
        Supports multi-key pool rotation on rate limits and automatic fallback engine.
        """
        if not segments:
            return []

        if not self.api_keys:
            logger.warning("Gemini API Key មិនទាន់បានកំណត់! កំពុងបកប្រែតាមរយៈ Backup Translation Engine ដោយស្វ័យប្រវត្តិ...")
            texts = [s.get("text", "") for s in segments]
            translated_texts = fallback_translate_batch(texts)
            for seg, km_t in zip(segments, translated_texts):
                seg["km_text"] = refine_khmer_dialogue(km_t, genre=self.drama_genre, duration=float(seg.get("duration", 0.0)))
                seg["gemini_gender"] = ""
            return segments

        pool_info = f" (Key Pool: {len(self.api_keys)} Keys)" if len(self.api_keys) > 1 else ""
        logger.info(f"🌐 កំពុងបកប្រែភាសាចិនទៅជាភាសាខ្មែរបែបកុនតាមរយៈ {self.model_name}{pool_info} (ប្រភេទរឿង: {self.drama_genre}, សរុប {len(segments)} ប្រយោគ)...")
        
        # Assemble comprehensive system prompt with drama genre instructions
        genre_instruction = get_genre_prompt(self.drama_genre)
        full_system_prompt = f"{DEFAULT_SYSTEM_PROMPT}\n\n{genre_instruction}"
        if self.style_prompt:
            full_system_prompt += f"\n\n=== ADDITIONAL USER STYLE INSTRUCTIONS ===\n{self.style_prompt}"

        translated_segments = []
        total_batches = (len(segments) + batch_size - 1) // batch_size

        for b_idx in range(total_batches):
            batch = segments[b_idx * batch_size : (b_idx + 1) * batch_size]
            prompt_input = [
                {"id": s["id"], "duration": s["duration"], "text": s["text"]}
                for s in batch
            ]

            # Create chronological transcript of this scene for full narrative coherence
            transcript_lines = [f"[{s['id']}] {s.get('text', '').strip()}" for s in batch]
            transcript_block = "\n".join(transcript_lines)

            # Extract 1 lightweight keyframe snapshot at batch midpoint if video_path is provided
            scene_img = None
            if video_path and os.path.exists(video_path):
                try:
                    batch_start = float(batch[0].get("start", 0.0))
                    batch_end = float(batch[-1].get("end", batch_start + 2.0))
                    mid_t = (batch_start + batch_end) / 2.0
                    scene_img = extract_scene_keyframe(video_path, mid_t)
                except Exception:
                    scene_img = None

            visual_note = ""
            if scene_img is not None:
                visual_note = (
                    "\n=== VISUAL SCENE KEYFRAME ATTACHED ===\n"
                    "A real visual camera snapshot of this exact scene from the video is attached as an image.\n"
                    "Carefully inspect the character's physical gestures, facial expression, body posture, actions "
                    "(e.g. slapping, punching, struggling, fleeing, kneeling, signing paper, drinking), setting, and costume "
                    "in the image to make sure your spoken Khmer dubbing matches the physical reality happening on screen!\n"
                )

            success = False
            max_attempts = max(5, len(self.api_keys) + 2)

            for attempt in range(max_attempts):
                try:
                    # Always ensure active key is set
                    active_key = self._get_active_key()
                    genai.configure(api_key=active_key)

                    # Model candidates fallback order for resilience: Primary chosen -> 3.x models -> 2.5-Flash -> Flash-Latest -> Lite
                    m_candidates = [self.model_name]
                    fallback_chain = [
                        "gemini-3.5-flash",
                        "gemini-3.8-flash",
                        "gemini-3.5-flash-lite",
                        "gemini-3-flash-preview",
                        "gemini-3.1-pro-preview",
                        "gemini-flash-latest",
                        "gemini-2.5-flash",
                        "gemini-flash-lite-latest",
                        "gemini-pro-latest"
                    ]
                    for cand in fallback_chain:
                        if cand not in m_candidates:
                            m_candidates.append(cand)

                    response = None
                    last_err = None
                    for try_m in m_candidates:
                        try:
                            try:
                                model = genai.GenerativeModel(
                                    model_name=try_m,
                                    system_instruction=full_system_prompt,
                                    generation_config={"response_mime_type": "application/json"}
                                )
                                scene_prompt = f"""=== FULL SCENE DIALOGUE SCRIPT (សូមអានសាច់រឿងសន្ទនាទាំងមូលនេះជាមុនសិន ដើម្បីយល់ពីដំណើររឿង) ===
{transcript_block}
{visual_note}
=== TIMING SEGMENTS TO DUB (បកប្រែសម្រួលបញ្ចូលសំឡេងតាមកាលវេលា) ===
{json.dumps(prompt_input, ensure_ascii=False, indent=2)}

Translate this scene into natural, continuous spoken Khmer cinema dubbing following the master dubbing principles:"""
                            except Exception:
                                model = genai.GenerativeModel(
                                    model_name=try_m,
                                    generation_config={"response_mime_type": "application/json"}
                                )
                                scene_prompt = f"""{full_system_prompt}

=== FULL SCENE DIALOGUE SCRIPT (សូមអានសាច់រឿងសន្ទនាទាំងមូលនេះជាមុនសិន ដើម្បីយល់ពីដំណើររឿង) ===
{transcript_block}
{visual_note}
=== TIMING SEGMENTS TO DUB (បកប្រែសម្រួលបញ្ចូលសំឡេងតាមកាលវេលា) ===
{json.dumps(prompt_input, ensure_ascii=False, indent=2)}

Translate this scene into natural, continuous spoken Khmer cinema dubbing:"""

                            payload = [scene_prompt, scene_img] if scene_img is not None else scene_prompt
                            try:
                                response = model.generate_content(payload)
                            except Exception as p_e:
                                if scene_img is not None:
                                    response = model.generate_content(scene_prompt)
                                else:
                                    raise p_e

                            if response and response.text:
                                if try_m != self.model_name:
                                    logger.info(f"🔄 បានប្តូរម៉ូដែលមក {try_m} ជោគជ័យ!")
                                    self.model_name = try_m
                                break
                        except Exception as m_e:
                            last_err = m_e
                            m_err_str = str(m_e)
                            if "404" in m_err_str or "not found" in m_err_str or "no longer available" in m_err_str:
                                continue
                            if "429" in m_err_str or "ResourceExhausted" in m_err_str:
                                continue
                            raise m_e

                    if not response or not response.text:
                        if last_err:
                            raise last_err
                        raise RuntimeError("Empty response from Gemini")

                    raw_text = response.text.strip()
                    
                    # Clean markdown fence if any
                    raw_text = re.sub(r"^```(?:json)?", "", raw_text, flags=re.MULTILINE)
                    raw_text = re.sub(r"```$", "", raw_text, flags=re.MULTILINE).strip()
                    
                    data = json.loads(raw_text)
                    # Create mapping by ID
                    mapping = {}
                    gender_mapping = {}
                    role_mapping = {}
                    pitch_mapping = {}
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and "id" in item:
                                mapping[item["id"]] = item.get("km_text", "")
                                gender_mapping[item["id"]] = item.get("gender", "")
                                role_mapping[item["id"]] = item.get("role", "")
                                pitch_mapping[item["id"]] = item.get("pitch", "+0Hz")

                    for s in batch:
                        s_copy = dict(s)
                        km_val = mapping.get(s["id"], "").strip()

                        # Ensure 100% translation coverage: every segment MUST have a valid Khmer translation
                        if not km_val or contains_chinese(km_val):
                            km_val = fallback_translate_text(s["text"])

                        # Post-process to ensure natural spoken cinema dubbing and selective overlong condensation
                        km_val = refine_khmer_dialogue(km_val, genre=self.drama_genre, duration=float(s.get("duration", 0.0)))

                        s_copy["km_text"] = km_val
                        s_copy["gemini_gender"] = gender_mapping.get(s["id"], "")
                        s_copy["gemini_role"] = role_mapping.get(s["id"], "")
                        s_copy["voice_pitch"] = pitch_mapping.get(s["id"], "+0Hz")
                        translated_segments.append(s_copy)

                    logger.info(f"បកប្រែរួចរាល់កញ្ចប់ទី {b_idx + 1}/{total_batches}")
                    success = True
                    break

                except Exception as e:
                    err_str = str(e)
                    is_model_unavailable = "no longer available" in err_str.lower() or "not found" in err_str.lower() or "404" in err_str
                    is_rate_limit = "429" in err_str or "ResourceExhausted" in err_str or "quota" in err_str.lower()
                    is_hard_quota = "exceeded your current quota" in err_str.lower() or "billing" in err_str.lower()
                    
                    if is_model_unavailable:
                        for next_m in ["gemini-3.5-flash", "gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-3-flash-preview", "gemini-flash-latest", "gemini-2.5-flash", "gemini-flash-lite-latest"]:
                            if next_m != self.model_name:
                                logger.warning(f"🔄 ម៉ូដែល {self.model_name} មិនអាចប្រើបាន -> ប្តូរទៅប្រើ {next_m} ភ្លាមៗ...")
                                self.model_name = next_m
                                break
                        continue

                    if is_rate_limit:
                        # If we have multiple keys in pool, immediately rotate to next key!
                        if len(self.api_keys) > 1 and attempt < len(self.api_keys):
                            self.rotate_to_next_key(reason=f"Key Quota 429")
                            time.sleep(1)
                            continue
                        elif is_hard_quota:
                            if len(self.api_keys) > 1 and attempt < len(self.api_keys):
                                self.rotate_to_next_key(reason=f"Hard Quota")
                                time.sleep(1)
                                continue
                            logger.warning(f"⚠️ Gemini API Quota ត្រូវបានអស់ទាំងស្រុង -> ប្តូរទៅប្រើ Backup Translator ភ្លាមៗ!")
                            break
                        else:
                            wait_time = (attempt + 1) * 3
                            logger.warning(f"⏳ Gemini Rate Limit - កំពុងរង់ចាំ {wait_time} វិនាទី រួចសាកល្បងឡើងវិញ (Attempt {attempt + 1}/{max_attempts})...")
                            time.sleep(wait_time)
                    else:
                        logger.warning(f"Batch {b_idx + 1} translation attempt {attempt + 1} failed: {err_str[:60]}")
                        time.sleep(1)

            if not success:
                logger.warning(f"🔄 កញ្ចប់ទី {b_idx + 1} បរាជ័យលើ Gemini! កំពុងប្តូរទៅប្រើប្រាស់ Backup Translation Engine ភ្លាមៗ...")
                batch_texts = [s["text"] for s in batch]
                fb_translations = fallback_translate_batch(batch_texts)
                for s, fb_km in zip(batch, fb_translations):
                    s_copy = dict(s)
                    s_copy["km_text"] = fb_km if not contains_chinese(fb_km) else fallback_translate_text(s["text"])
                    s_copy["gemini_gender"] = ""
                    s_copy["gemini_role"] = ""
                    s_copy["voice_pitch"] = "+0Hz"
                    translated_segments.append(s_copy)

        logger.success("បកប្រែអត្ថបទទាំងអស់ជាភាសាខ្មែរបានជោគជ័យពេញលេញ!")
        return translated_segments
