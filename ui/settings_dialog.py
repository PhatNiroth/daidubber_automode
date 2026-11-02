import re
import tkinter as tk
import customtkinter as ctk
from typing import Callable, Optional
from core.config_manager import ConfigManager
from core.translator import GeminiTranslator
from ui.components import GlassCard
from ui.styles import (
    BG_DARK, CARD_BG, CARD_BORDER, ACCENT_CYAN, ACCENT_BLUE, ACCENT_PINK,
    ACCENT_GREEN, ACCENT_RED, TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED,
    INPUT_BG, INPUT_BORDER, FONT_TITLE, FONT_SUBTITLE, FONT_BODY, FONT_BODY_BOLD, FONT_SMALL
)

class SettingsDialog(ctk.CTkToplevel):
    """Modern Settings modal window for configuring API keys, AI models, and rendering parameters."""

    def __init__(self, parent, config_manager: ConfigManager, on_save_callback: Optional[Callable] = None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.on_save_callback = on_save_callback

        self.title("ការកំណត់ប្រព័ន្ធ (Settings) - JAI AUTO DUBBING")
        self.geometry("640x720")
        self.resizable(False, False)
        self.configure(fg_color=BG_DARK)

        # Make modal
        self.transient(parent)
        self.grab_set()

        self._build_ui()
        self._load_current_values()

    def _build_ui(self):
        # Header
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.pack(fill="x", padx=20, pady=(15, 10))

        title_lbl = ctk.CTkLabel(
            header_frame,
            text="⚙️ ការកំណត់ប្រព័ន្ធ (System Settings)",
            font=FONT_TITLE,
            text_color=ACCENT_CYAN
        )
        title_lbl.pack(side="left")

        # Scrollable container for settings sections
        scroll_frame = ctk.CTkScrollableFrame(self, fg_color="transparent")
        scroll_frame.pack(fill="both", expand=True, padx=20, pady=5)

        # ---------------- Section 1: Gemini AI API ----------------
        sec1 = GlassCard(scroll_frame)
        sec1.pack(fill="x", pady=8, padx=2)

        s1_title = ctk.CTkLabel(sec1, text="🔑 Google Gemini API Settings", font=FONT_SUBTITLE, text_color=TEXT_PRIMARY)
        s1_title.pack(anchor="w", padx=15, pady=(12, 6))

        # API Key entry
        key_lbl = ctk.CTkLabel(sec1, text="Gemini API Key (សម្រាប់បកប្រែចិន ➔ ខ្មែរ):", font=FONT_BODY, text_color=TEXT_SECONDARY)
        key_lbl.pack(anchor="w", padx=15, pady=(2, 2))

        key_row = ctk.CTkFrame(sec1, fg_color="transparent")
        key_row.pack(fill="x", padx=15, pady=(0, 6))

        self.key_entry = ctk.CTkEntry(
            key_row,
            show="*",
            fg_color=INPUT_BG,
            border_color=INPUT_BORDER,
            text_color=TEXT_PRIMARY,
            font=FONT_BODY
        )
        self.key_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))

        self.show_key_var = tk.BooleanVar(value=False)
        self.toggle_btn = ctk.CTkButton(
            key_row,
            text="👁️",
            width=40,
            command=self._toggle_show_key,
            fg_color="#1f2d4a",
            hover_color="#2b3e66"
        )
        self.toggle_btn.pack(side="left")

        # Test connection button & status
        test_row = ctk.CTkFrame(sec1, fg_color="transparent")
        test_row.pack(fill="x", padx=15, pady=(2, 12))

        self.test_btn = ctk.CTkButton(
            test_row,
            text="🔗 តេស្តភ្ជាប់ API (Test Connection)",
            command=self._test_gemini_connection,
            fg_color=ACCENT_BLUE,
            hover_color="#0052cc",
            font=FONT_BODY_BOLD
        )
        self.test_btn.pack(side="left", padx=(0, 10))

        self.test_status_lbl = ctk.CTkLabel(test_row, text="", font=FONT_SMALL, text_color=TEXT_MUTED)
        self.test_status_lbl.pack(side="left")

        # ---------------- Section 2: AI Models ----------------
        sec2 = GlassCard(scroll_frame)
        sec2.pack(fill="x", pady=8, padx=2)

        s2_title = ctk.CTkLabel(sec2, text="🧠 ម៉ូដែល AI (Models & Recognition)", font=FONT_SUBTITLE, text_color=TEXT_PRIMARY)
        s2_title.pack(anchor="w", padx=15, pady=(12, 6))

        # Gemini Model
        gm_lbl = ctk.CTkLabel(sec2, text="ម៉ូដែល Gemini AI:", font=FONT_BODY, text_color=TEXT_SECONDARY)
        gm_lbl.pack(anchor="w", padx=15, pady=(2, 2))

        self.gemini_model_cb = ctk.CTkComboBox(
            sec2,
            values=[
                "gemini-3.5-flash",
                "gemini-3.8-flash",
                "gemini-3.5-flash-lite",
                "gemini-3.1-pro-preview",
                "gemini-3-flash-preview",
                "gemini-flash-latest",
                "gemini-2.5-flash",
                "gemini-flash-lite-latest"
            ],
            fg_color=INPUT_BG,
            border_color=INPUT_BORDER,
            text_color=TEXT_PRIMARY
        )
        self.gemini_model_cb.pack(fill="x", padx=15, pady=(0, 8))

        # Whisper Model
        wm_lbl = ctk.CTkLabel(sec2, text="ម៉ូដែល Whisper ASR (សម្គាល់ពាក្យចិន):", font=FONT_BODY, text_color=TEXT_SECONDARY)
        wm_lbl.pack(anchor="w", padx=15, pady=(2, 2))

        self.whisper_model_cb = ctk.CTkComboBox(
            sec2,
            values=["large-v3-turbo", "base", "small", "medium", "large-v3", "tiny"],
            fg_color=INPUT_BG,
            border_color=INPUT_BORDER,
            text_color=TEXT_PRIMARY
        )
        self.whisper_model_cb.pack(fill="x", padx=15, pady=(0, 8))

        # Separation Engine
        sep_lbl = ctk.CTkLabel(sec2, text="ម៉ូដែលបំបែកសំឡេងចម្រៀង/BGM (Vocal Separation):", font=FONT_BODY, text_color=TEXT_SECONDARY)
        sep_lbl.pack(anchor="w", padx=15, pady=(2, 2))

        self.sep_engine_cb = ctk.CTkComboBox(
            sec2,
            values=[
                "roformer (BS-Roformer AI - លុបចម្រៀងស្អាត១០០% មិនដាច់ៗ ⭐)",
                "demucs (Demucs AI)"
            ],
            fg_color=INPUT_BG,
            border_color=INPUT_BORDER,
            text_color=TEXT_PRIMARY
        )
        self.sep_engine_cb.pack(fill="x", padx=15, pady=(0, 8))

        # Device selection (CUDA / CPU)
        dev_lbl = ctk.CTkLabel(sec2, text="Hardware Acceleration (ដំណើរការលើ):", font=FONT_BODY, text_color=TEXT_SECONDARY)
        dev_lbl.pack(anchor="w", padx=15, pady=(2, 2))

        self.device_cb = ctk.CTkComboBox(
            sec2,
            values=["cuda", "cpu"],
            fg_color=INPUT_BG,
            border_color=INPUT_BORDER,
            text_color=TEXT_PRIMARY
        )
        self.device_cb.pack(fill="x", padx=15, pady=(0, 12))

        # ---------------- Section 3: Audio Sync & Time-Stretch ----------------
        sec3 = GlassCard(scroll_frame)
        sec3.pack(fill="x", pady=8, padx=2)

        s3_title = ctk.CTkLabel(sec3, text="⏱️ Audio Sync & Time-Stretching (មិនដាច់សំឡេង)", font=FONT_SUBTITLE, text_color=TEXT_PRIMARY)
        s3_title.pack(anchor="w", padx=15, pady=(12, 6))

        self.stretch_switch = ctk.CTkSwitch(
            sec3,
            text="បើកមុខងារបត់បែនល្បឿននិយាយឲ្យត្រូវម៉ោងវីដេអូ (Dynamic Time-Stretch)",
            font=FONT_BODY,
            text_color=TEXT_PRIMARY,
            progress_color=ACCENT_CYAN
        )
        self.stretch_switch.pack(anchor="w", padx=15, pady=6)

        # Stretch ratio slider
        ratio_lbl_row = ctk.CTkFrame(sec3, fg_color="transparent")
        ratio_lbl_row.pack(fill="x", padx=15, pady=(6, 2))
        ctk.CTkLabel(ratio_lbl_row, text="កម្រិតពន្លឿនអតិបរមា (Max Stretch Ratio):", font=FONT_BODY, text_color=TEXT_SECONDARY).pack(side="left")
        self.ratio_val_lbl = ctk.CTkLabel(ratio_lbl_row, text="1.30x", font=FONT_BODY_BOLD, text_color=ACCENT_CYAN)
        self.ratio_val_lbl.pack(side="right")

        self.ratio_slider = ctk.CTkSlider(
            sec3,
            from_=1.00,
            to=2.50,
            number_of_steps=30,
            progress_color=ACCENT_CYAN,
            command=lambda v: self.ratio_val_lbl.configure(text=f"{v:.2f}x")
        )
        self.ratio_slider.pack(fill="x", padx=15, pady=(0, 10))

        # Voice base speed
        speed_lbl = ctk.CTkLabel(sec3, text="ល្បឿនសំឡេងដើម (TTS Base Speed):", font=FONT_BODY, text_color=TEXT_SECONDARY)
        speed_lbl.pack(anchor="w", padx=15, pady=(2, 2))

        self.voice_speed_cb = ctk.CTkComboBox(
            sec3,
            values=[
                "+0% (1.0x Normal)",
                "+15% (1.15x)",
                "+30% (1.30x)",
                "+50% (1.50x)",
                "+100% (2.00x)",
                "+150% (2.50x Max Speed ⚡)"
            ],
            fg_color=INPUT_BG,
            border_color=INPUT_BORDER,
            text_color=TEXT_PRIMARY
        )
        self.voice_speed_cb.pack(fill="x", padx=15, pady=(0, 12))

        # ---------------- Section 4: Fast Render & Export ----------------
        sec4 = GlassCard(scroll_frame)
        sec4.pack(fill="x", pady=8, padx=2)

        s4_title = ctk.CTkLabel(sec4, text="⚡ Fast Video Render (ការ Render លឿន)", font=FONT_SUBTITLE, text_color=TEXT_PRIMARY)
        s4_title.pack(anchor="w", padx=15, pady=(12, 6))

        render_lbl = ctk.CTkLabel(sec4, text="របៀប Render វីដេអូ:", font=FONT_BODY, text_color=TEXT_SECONDARY)
        render_lbl.pack(anchor="w", padx=15, pady=(2, 2))

        self.render_mode_cb = ctk.CTkComboBox(
            sec4,
            values=[
                "fast_copy (Direct Stream Copy - 3 វិនាទីរួចរាល់)",
                "nvenc (NVIDIA GPU Accelerated Re-encode)"
            ],
            fg_color=INPUT_BG,
            border_color=INPUT_BORDER,
            text_color=TEXT_PRIMARY
        )
        self.render_mode_cb.pack(fill="x", padx=15, pady=(0, 8))

        # Output naming
        naming_lbl = ctk.CTkLabel(sec4, text="ទម្រង់ឈ្មោះវីដេអូ Export:", font=FONT_BODY, text_color=TEXT_SECONDARY)
        naming_lbl.pack(anchor="w", padx=15, pady=(2, 2))

        self.naming_cb = ctk.CTkComboBox(
            sec4,
            values=[
                "EP001_KH.mp4 (តាមលំដាប់លេខភាគ)",
                "Original_KH.mp4 (រក្សាឈ្មោះដើម + _KH)"
            ],
            fg_color=INPUT_BG,
            border_color=INPUT_BORDER,
            text_color=TEXT_PRIMARY
        )
        self.naming_cb.pack(fill="x", padx=15, pady=(0, 8))

        # Concurrency
        par_lbl = ctk.CTkLabel(sec4, text="ចំនួនដំណើរការវីដេអូស្របគ្នា (Concurrent Parallel Videos):", font=FONT_BODY, text_color=TEXT_SECONDARY)
        par_lbl.pack(anchor="w", padx=15, pady=(2, 2))

        self.parallel_cb = ctk.CTkComboBox(
            sec4,
            values=[
                "1 វីដេអូ (ម្ដង ១ វីដេអូ - Sequential)",
                "2 វីដេអូស្របគ្នា (Recommended for RTX 3070)",
                "3 វីដេអូស្របគ្នា (Fast)",
                "4 វីដេអូស្របគ្នា (Very Fast)",
                "5 វីដេអូស្របគ្នា (High Performance)",
                "6 វីដេអូស្របគ្នា (Super High Performance)",
                "8 វីដេអូស្របគ្នា (Ultra Performance)",
                "10 វីដេអូស្របគ្នា (Maximum Beast Mode)"
            ],
            fg_color=INPUT_BG,
            border_color=INPUT_BORDER,
            text_color=TEXT_PRIMARY
        )
        self.parallel_cb.pack(fill="x", padx=15, pady=(0, 12))

        # ---------------- Footer Actions ----------------
        footer_frame = ctk.CTkFrame(self, fg_color="transparent")
        footer_frame.pack(fill="x", padx=20, pady=(10, 15))

        cancel_btn = ctk.CTkButton(
            footer_frame,
            text="បោះបង់ (Cancel)",
            command=self.destroy,
            fg_color="#1f2d4a",
            hover_color="#2b3e66",
            font=FONT_BODY
        )
        cancel_btn.pack(side="right", padx=(8, 0))

        save_btn = ctk.CTkButton(
            footer_frame,
            text="💾 រក្សាទុក (Save Settings)",
            command=self._save_settings,
            fg_color=ACCENT_CYAN,
            hover_color="#00b4d8",
            text_color="#000000",
            font=FONT_BODY_BOLD
        )
        save_btn.pack(side="right")

    def _load_current_values(self):
        cfg = self.config_manager.config
        self.key_entry.insert(0, cfg.get("gemini_api_key", ""))
        self.gemini_model_cb.set(cfg.get("gemini_model", "gemini-2.5-flash"))
        self.whisper_model_cb.set(cfg.get("whisper_model", "base"))
        engine = cfg.get("separation_engine", "roformer")
        if engine == "demucs":
            self.sep_engine_cb.set("demucs (Demucs AI)")
        else:
            self.sep_engine_cb.set("roformer (BS-Roformer AI - លុបចម្រៀងស្អាត១០០% មិនដាច់ៗ ⭐)")
        self.device_cb.set(cfg.get("device", "cuda"))
        
        self.stretch_switch.select() if cfg.get("enable_time_stretch", True) else self.stretch_switch.deselect()
        ratio = float(cfg.get("max_stretch_ratio", 2.50))
        self.ratio_slider.set(ratio)
        self.ratio_val_lbl.configure(text=f"{ratio:.2f}x")

        speed_val = cfg.get("voice_speed", "+0%")
        matched_speed = False
        for opt in self.voice_speed_cb.cget("values"):
            if speed_val in opt:
                self.voice_speed_cb.set(opt)
                matched_speed = True
                break
        if not matched_speed:
            self.voice_speed_cb.set("+0% (1.0x Normal)")

        render_mode = cfg.get("render_mode", "fast_copy")
        if render_mode == "nvenc":
            self.render_mode_cb.set("nvenc (NVIDIA GPU Accelerated Re-encode)")
        else:
            self.render_mode_cb.set("fast_copy (Direct Stream Copy - 3 វិនាទីរួចរាល់)")

        naming = cfg.get("output_naming_pattern", "prefix_index")
        if naming == "original_suffix":
            self.naming_cb.set("Original_KH.mp4 (រក្សាឈ្មោះដើម + _KH)")
        else:
            self.naming_cb.set("EP001_KH.mp4 (តាមលំដាប់លេខភាគ)")

        par_val = int(self.config_manager.get("max_parallel_videos", 2))
        found = False
        for val in self.parallel_cb.cget("values"):
            m = re.search(r'\d+', val)
            if m and int(m.group()) == par_val:
                self.parallel_cb.set(val)
                found = True
                break
        if not found:
            self.parallel_cb.set(f"{par_val} វីដេអូស្របគ្នា")

    def _toggle_show_key(self):
        if self.show_key_var.get():
            self.key_entry.configure(show="*")
            self.show_key_var.set(False)
            self.toggle_btn.configure(text="👁️")
        else:
            self.key_entry.configure(show="")
            self.show_key_var.set(True)
            self.toggle_btn.configure(text="🙈")

    def _test_gemini_connection(self):
        key = self.key_entry.get().strip()
        model_name = self.gemini_model_cb.get().strip()
        self.test_status_lbl.configure(text="កំពុងតេស្ត...", text_color=ACCENT_CYAN)
        self.update_idletasks()

        translator = GeminiTranslator(api_key=key, model_name=model_name)
        success, msg = translator.test_connection()
        if success:
            self.test_status_lbl.configure(text=f"✅ {msg}", text_color=ACCENT_GREEN)
        else:
            self.test_status_lbl.configure(text=f"❌ {msg}", text_color=ACCENT_RED)

    def _save_settings(self):
        key = self.key_entry.get().strip()
        gemini_model = self.gemini_model_cb.get().strip()
        whisper_model = self.whisper_model_cb.get().strip()
        device = self.device_cb.get().strip()
        enable_stretch = bool(self.stretch_switch.get())
        max_stretch = round(float(self.ratio_slider.get()), 2)

        speed_choice = self.voice_speed_cb.get()
        speed_code = "+0%"
        for opt in ["+150%", "+100%", "+50%", "+30%", "+15%", "+0%"]:
            if opt in speed_choice:
                speed_code = opt
                break

        render_choice = self.render_mode_cb.get()
        render_mode = "nvenc" if "nvenc" in render_choice else "fast_copy"

        naming_choice = self.naming_cb.get()
        naming_pattern = "original_suffix" if "Original" in naming_choice else "prefix_index"

        par_choice = self.parallel_cb.get()
        m = re.search(r'\d+', par_choice)
        par_num = int(m.group()) if m else 2

        self.config_manager.set("gemini_api_key", key)
        self.config_manager.set("gemini_model", gemini_model)
        self.config_manager.set("whisper_model", whisper_model)
        sep_choice = self.sep_engine_cb.get()
        separation_engine = "demucs" if "demucs" in sep_choice.lower() else "roformer"
        self.config_manager.set("separation_engine", separation_engine)
        self.config_manager.set("device", device)
        self.config_manager.set("enable_time_stretch", enable_stretch)
        self.config_manager.set("max_stretch_ratio", max_stretch)
        self.config_manager.set("voice_speed", speed_code)
        self.config_manager.set("render_mode", render_mode)
        self.config_manager.set("output_naming_pattern", naming_pattern)
        self.config_manager.set("max_parallel_videos", par_num)

        if self.on_save_callback:
            self.on_save_callback()

        self.destroy()
