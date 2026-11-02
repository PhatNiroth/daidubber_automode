import os
import json
from pathlib import Path

DEFAULT_CONFIG = {
    "gemini_api_key": "",
    "gemini_model": "gemini-3.5-flash",
    "asr_engine": "whisper",
    "whisper_model": "large-v3-turbo",
    "device": "cuda",
    "vocal_mode": "separate_bgm",  # 'separate_bgm' (Keep BGM/FX, Remove Chinese Voice), 'mute_original' (Mute all original audio)
    "demucs_model": "htdemucs",
    "khmer_voice": "auto",  # 'km-KH-PisethNeural' (ប្រុស), 'km-KH-SreymomNeural' (ស្រី), 'auto' (ឆ្លាស់គ្នា)
    "voice_speed": "+10%",
    "voice_pitch": "+0Hz",
    "bgm_volume": 0.28,
    "voice_volume": 1.35,
    "ducking_factor": 0.25,
    "enable_time_stretch": True,
    "max_stretch_ratio": 1.30,
    "render_mode": "fast_copy",  # 'fast_copy' (-c:v copy for 3s render), 'nvenc' (h264_nvenc)
    "output_naming_pattern": "prefix_index",  # 'prefix_index' (EP001_KH.mp4) or 'original_suffix' (Name_KH.mp4)
    "naming_prefix": "EP",
    "last_input_folder": "",
    "last_output_folder": "",
    "khmer_prompt_style": "natural_movie",
    "drama_genre": "wuxia",  # 'wuxia' (បុរាណ ក្បាច់គុណ), 'modern' (សម័យទំនើប), 'palace' (រាជវាំង), 'auto' (ស្វ័យប្រវត្តិ)
    "enable_ai_review": True,  # Two-Pass Cinema Review & Audit Engine (ផ្ទៀងផ្ទាត់សកម្មភាព & អត្ថន័យស្វ័យប្រវត្តិ)
    "max_parallel_videos": 2  # 1 to 10 concurrent videos
}

CONFIG_FILE_NAME = "config.json"

class ConfigManager:
    """Manages application settings and persistent configuration."""
    def __init__(self, config_dir=None):
        if config_dir is None:
            self.config_dir = Path(__file__).resolve().parent.parent
        else:
            self.config_dir = Path(config_dir)
        self.config_path = self.config_dir / CONFIG_FILE_NAME
        self.config = self.load_config()

    def load_config(self):
        """Loads configuration from config.json or returns defaults."""
        config = DEFAULT_CONFIG.copy()
        # Also check environment variable for GEMINI_API_KEY
        env_key = os.environ.get("GEMINI_API_KEY", "")
        if env_key:
            config["gemini_api_key"] = env_key

        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                    config.update(saved)
            except Exception as e:
                print(f"[Warning] Failed to load config file: {e}")
        return config

    def save_config(self, new_config=None):
        """Saves current configuration to config.json."""
        if new_config:
            self.config.update(new_config)
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"[Error] Failed to save config: {e}")
            return False

    def get(self, key, default=None):
        return self.config.get(key, default)

    def set(self, key, value):
        self.config[key] = value
        self.save_config()
