import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
from typing import List, Dict, Any, Optional

import customtkinter as ctk

from core.config_manager import ConfigManager
from core.pipeline import DubbingPipeline, PIPELINE_STAGES
from core.tts_engine import KHMER_VOICES
from utils.logger import logger
from utils.file_utils import get_video_files, check_video_status
from ui.styles import (
    BG_DARK, CARD_BG, CARD_BORDER, CARD_HOVER, ACCENT_CYAN, ACCENT_BLUE, ACCENT_PINK,
    ACCENT_PURPLE, ACCENT_GREEN, ACCENT_RED, TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED,
    INPUT_BG, INPUT_BORDER, FONT_HEADER, FONT_TITLE, FONT_SUBTITLE, FONT_BODY, FONT_BODY_BOLD, FONT_SMALL
)
from ui.components import (
    GlassCard, PipelineProgressWidget, LogConsoleWidget,
    AudioVisualizerWidget, PulsingStatusDot, SmoothProgressBar
)
from ui.settings_dialog import SettingsDialog

class MainWindow(ctk.CTk):
    """Modern Cyberpunk MainWindow for JAI AUTO DUBBING with rich animations and smooth UI."""

    def __init__(self, config_manager: ConfigManager):
        super().__init__()
        self.config_manager = config_manager
        self.pipeline = DubbingPipeline(config_manager)

        # Set Window Properties
        self.title("JAI AUTO DUBBING - AI Chinese to Khmer Pro 🎬⚡")
        self.geometry("1020x910")
        self.minsize(920, 820)
        self.configure(fg_color=BG_DARK)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # State Variables
        self.video_files: List[Path] = []
        self.video_status_list: List[Dict[str, Any]] = []
        self.is_processing = False

        self._build_ui()
        self._setup_logger_listener()
        self._load_saved_paths()

    def _build_ui(self):
        # ---------------- TOP HEADER BANNER (ANIMATED) ----------------
        banner_card = GlassCard(self, glowing=True, glow_color="#2b3b63", corner_radius=14)
        banner_card.pack(fill="x", padx=18, pady=(14, 10))

        banner_inner = ctk.CTkFrame(banner_card, fg_color="transparent")
        banner_inner.pack(fill="x", padx=16, pady=12)

        # Glowing Logo Badge
        logo_badge = ctk.CTkFrame(
            banner_inner,
            width=52,
            height=52,
            corner_radius=12,
            fg_color="#800f2f",
            border_color=ACCENT_PINK,
            border_width=2
        )
        logo_badge.pack(side="left", padx=(0, 14))
        logo_badge.pack_propagate(False)

        logo_lbl = ctk.CTkLabel(
            logo_badge,
            text="🎬⚡",
            font=(FONT_HEADER[0], 20)
        )
        logo_lbl.pack(expand=True)

        # App Title & Subtitle
        title_box = ctk.CTkFrame(banner_inner, fg_color="transparent")
        title_box.pack(side="left")

        title_row = ctk.CTkFrame(title_box, fg_color="transparent")
        title_row.pack(anchor="w")

        app_title = ctk.CTkLabel(
            title_row,
            text="JAI AUTO DUBBING",
            font=(FONT_HEADER[0], 21, "bold"),
            text_color=ACCENT_PINK
        )
        app_title.pack(side="left", padx=(0, 10))

        # Version Pill Badge
        version_badge = ctk.CTkLabel(
            title_row,
            text="v2.5 PRO",
            fg_color="#0e2a47",
            text_color=ACCENT_CYAN,
            corner_radius=6,
            font=(FONT_SMALL[0], 9, "bold"),
            padx=8,
            pady=2
        )
        version_badge.pack(side="left")

        app_sub = ctk.CTkLabel(
            title_box,
            text="AI Video Dubbing Suite • Chinese ➔ Khmer • BGM Preserved • Fast Render",
            font=(FONT_SMALL[0], 10),
            text_color=ACCENT_CYAN
        )
        app_sub.pack(anchor="w", pady=(2, 0))

        # Right side: Visualizer & Settings Button
        right_header = ctk.CTkFrame(banner_inner, fg_color="transparent")
        right_header.pack(side="right")

        # Live Animated Equalizer Widget
        vis_container = ctk.CTkFrame(right_header, fg_color="#0b1120", corner_radius=8, border_color=CARD_BORDER, border_width=1)
        vis_container.pack(side="left", padx=(0, 12), pady=2)
        
        self.visualizer = AudioVisualizerWidget(vis_container, width=200, height=36, num_bars=20)
        self.visualizer.pack(padx=4, pady=4)

        # Settings Gear Button
        self.settings_btn = ctk.CTkButton(
            right_header,
            text="⚙️",
            width=46,
            height=46,
            corner_radius=12,
            fg_color="#201838",
            border_color="#5a2b82",
            border_width=2,
            hover_color="#3d1b66",
            font=(FONT_HEADER[0], 18),
            command=self._open_settings
        )
        self.settings_btn.pack(side="left")

        # ---------------- MAIN SCROLLABLE CONTAINER ----------------
        self.main_scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.main_scroll.pack(fill="both", expand=True, padx=18, pady=0)

        # ---------------- SECTION 1: CHOOSE FOLDER ----------------
        card_input = GlassCard(self.main_scroll)
        card_input.pack(fill="x", pady=6)

        lbl_input = ctk.CTkLabel(
            card_input,
            text="CHOOSE FOLDER (ជ្រើសរើស Folder វីដេអូចិន):",
            font=FONT_SUBTITLE,
            text_color=TEXT_PRIMARY
        )
        lbl_input.pack(anchor="w", padx=16, pady=(12, 4))

        row_input = ctk.CTkFrame(card_input, fg_color="transparent")
        row_input.pack(fill="x", padx=16, pady=(0, 6))

        self.input_entry = ctk.CTkEntry(
            row_input,
            placeholder_text="ជ្រើសរើស Folder ដែលមានវីដេអូចិន (ឧ. D:/video/origin/...)",
            fg_color=INPUT_BG,
            border_color=INPUT_BORDER,
            text_color=TEXT_PRIMARY,
            font=FONT_BODY,
            height=38
        )
        self.input_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))

        self.btn_browse_input = ctk.CTkButton(
            row_input,
            text="📁 Browse",
            width=115,
            height=38,
            command=self._browse_input_folder,
            fg_color="#182747",
            border_color=ACCENT_PINK,
            border_width=1.5,
            hover_color="#2a3d6d",
            font=FONT_BODY_BOLD
        )
        self.btn_browse_input.pack(side="left")

        # Status text for detected videos
        stats_row = ctk.CTkFrame(card_input, fg_color="transparent")
        stats_row.pack(fill="x", padx=16, pady=(0, 12))

        self.lbl_video_stats = ctk.CTkLabel(
            stats_row,
            text="រកឃើញវីដេអូ: 0 ឯកសារ (Dubbed រួច: 0 ភាគ, នៅសល់: 0 ភាគ)",
            font=FONT_BODY_BOLD,
            text_color=ACCENT_GREEN
        )
        self.lbl_video_stats.pack(side="left")

        # ---------------- SECTION 2: FOLDER SAVED ----------------
        card_output = GlassCard(self.main_scroll)
        card_output.pack(fill="x", pady=6)

        lbl_output = ctk.CTkLabel(
            card_output,
            text="FOLDER SAVED (Folder រក្សាទុកវីដេអូខ្មែរសម្រេច):",
            font=FONT_SUBTITLE,
            text_color=TEXT_PRIMARY
        )
        lbl_output.pack(anchor="w", padx=16, pady=(12, 4))

        row_output = ctk.CTkFrame(card_output, fg_color="transparent")
        row_output.pack(fill="x", padx=16, pady=(0, 6))

        self.output_entry = ctk.CTkEntry(
            row_output,
            placeholder_text="ជ្រើសរើស Folder សម្រាប់ Export វីដេអូ...",
            fg_color=INPUT_BG,
            border_color=INPUT_BORDER,
            text_color=TEXT_PRIMARY,
            font=FONT_BODY,
            height=38
        )
        self.output_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))

        self.btn_browse_output = ctk.CTkButton(
            row_output,
            text="📁 Browse",
            width=115,
            height=38,
            command=self._browse_output_folder,
            fg_color="#182747",
            border_color=ACCENT_CYAN,
            border_width=1.5,
            hover_color="#2a3d6d",
            font=FONT_BODY_BOLD
        )
        self.btn_browse_output.pack(side="left")

        self.lbl_naming_notice = ctk.CTkLabel(
            card_output,
            text="វីដេអូនឹងរក្សាទុកឈ្មោះ: EP001_KH.mp4 តាមលំដាប់លំដោយ • រក្សាទុកភ្លេងខាងក្រោយ (BGM) 100%",
            font=FONT_SMALL,
            text_color=TEXT_SECONDARY
        )
        self.lbl_naming_notice.pack(anchor="w", padx=16, pady=(0, 12))

        # ---------------- SECTION 3: VOCAL SEPARATION MODE ----------------
        card_vocal = GlassCard(self.main_scroll)
        card_vocal.pack(fill="x", pady=6)

        lbl_vocal = ctk.CTkLabel(
            card_vocal,
            text="🎚️ របៀបចាត់ចែងសំឡេងដើម (AUDIO VOCAL & BGM MODE):",
            font=FONT_SUBTITLE,
            text_color=TEXT_PRIMARY
        )
        lbl_vocal.pack(anchor="w", padx=16, pady=(12, 6))

        self.vocal_mode_var = ctk.StringVar(value=self.config_manager.get("vocal_mode", "separate_bgm"))

        vocal_radio_row = ctk.CTkFrame(card_vocal, fg_color="transparent")
        vocal_radio_row.pack(fill="x", padx=16, pady=(2, 14))

        # Mode 1 Card (Recommended)
        self.card_opt1 = ctk.CTkFrame(
            vocal_radio_row,
            fg_color="#0c1726",
            border_color=ACCENT_CYAN,
            border_width=1.5,
            corner_radius=10
        )
        self.card_opt1.pack(side="left", fill="x", expand=True, padx=(0, 8), pady=2)
        r1 = ctk.CTkRadioButton(
            self.card_opt1,
            text="រក្សា BGM & FX ស្អាត (លុបសំឡេងថ្ងូរ/ស្រែក ទុកសំឡេងកុន)",
            variable=self.vocal_mode_var,
            value="separate_bgm",
            font=FONT_BODY_BOLD,
            text_color=ACCENT_CYAN,
            command=self._on_vocal_mode_change
        )
        r1.pack(anchor="w", padx=12, pady=10)

        # Mode 2 Card
        self.card_opt2 = ctk.CTkFrame(
            vocal_radio_row,
            fg_color="#0b1120",
            border_color=CARD_BORDER,
            border_width=1,
            corner_radius=10
        )
        self.card_opt2.pack(side="left", fill="x", expand=True, padx=(4, 0), pady=2)
        r2 = ctk.CTkRadioButton(
            self.card_opt2,
            text="Mute សំឡេងដើមទាំងអស់ (១០០%)",
            variable=self.vocal_mode_var,
            value="mute_original",
            font=FONT_BODY,
            text_color=TEXT_PRIMARY,
            command=self._on_vocal_mode_change
        )
        r2.pack(anchor="w", padx=12, pady=10)

        # ---------------- SECTION 4: AI MODEL & VOICE ----------------
        card_ai = GlassCard(self.main_scroll)
        card_ai.pack(fill="x", pady=6)

        ai_grid = ctk.CTkFrame(card_ai, fg_color="transparent")
        ai_grid.pack(fill="x", padx=16, pady=12)

        # Gemini Model Dropdown
        col1 = ctk.CTkFrame(ai_grid, fg_color="transparent")
        col1.pack(side="left", fill="x", expand=True, padx=(0, 10))

        lbl_gemini = ctk.CTkLabel(
            col1,
            text="CHOOSE GEMINI MODEL (ជ្រើសរើសម៉ូដែល AI):",
            font=FONT_BODY_BOLD,
            text_color=TEXT_PRIMARY
        )
        lbl_gemini.pack(anchor="w", pady=(0, 6))

        self.gemini_model_cb = ctk.CTkComboBox(
            col1,
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
            command=self._on_gemini_model_change,
            fg_color=INPUT_BG,
            border_color=INPUT_BORDER,
            text_color=TEXT_PRIMARY,
            font=FONT_BODY,
            height=36
        )
        self.gemini_model_cb.set(self.config_manager.get("gemini_model", "gemini-3.5-flash"))
        self.gemini_model_cb.pack(fill="x")

        # Khmer Voice Dropdown
        col2 = ctk.CTkFrame(ai_grid, fg_color="transparent")
        col2.pack(side="left", fill="x", expand=True)

        lbl_voice = ctk.CTkLabel(
            col2,
            text="CHOOSE KHMER VOICE (សំឡេងនិយាយខ្មែរ):",
            font=FONT_BODY_BOLD,
            text_color=TEXT_PRIMARY
        )
        lbl_voice.pack(anchor="w", pady=(0, 6))

        voice_display_map = {k: f"{v} [{k}]" for k, v in KHMER_VOICES.items()}
        self.voice_map_reverse = {f"{v} [{k}]": k for k, v in KHMER_VOICES.items()}

        self.voice_cb = ctk.CTkComboBox(
            col2,
            values=list(voice_display_map.values()),
            command=self._on_voice_change,
            fg_color=INPUT_BG,
            border_color=INPUT_BORDER,
            text_color=TEXT_PRIMARY,
            font=FONT_BODY,
            height=36
        )
        current_voice = self.config_manager.get("khmer_voice", "km-KH-PisethNeural")
        self.voice_cb.set(voice_display_map.get(current_voice, list(voice_display_map.values())[0]))
        self.voice_cb.pack(fill="x")

        # ---------------- SECTION 5: PROGRESS & WORKFLOW ----------------
        self.card_progress = GlassCard(self.main_scroll)
        self.card_progress.pack(fill="x", pady=6)

        # Header of progress card with Pulsing Dot
        prog_header = ctk.CTkFrame(self.card_progress, fg_color="transparent")
        prog_header.pack(fill="x", padx=16, pady=(12, 6))

        counter_box = ctk.CTkFrame(prog_header, fg_color="transparent")
        counter_box.pack(side="left")

        # Pulsing Status Dot
        self.status_dot = PulsingStatusDot(counter_box, size=22)
        self.status_dot.pack(side="left", padx=(0, 8))

        self.lbl_progress_counter = ctk.CTkLabel(
            counter_box,
            text="[0/0] រង់ចាំ...",
            font=FONT_TITLE,
            text_color=ACCENT_PINK
        )
        self.lbl_progress_counter.pack(side="left")

        self.lbl_progress_stage = ctk.CTkLabel(
            prog_header,
            text="Ready (NVIDIA CUDA RTX 3070)",
            font=FONT_BODY_BOLD,
            text_color=ACCENT_GREEN
        )
        self.lbl_progress_stage.pack(side="right")

        # Smooth Interpolated Progress Bar
        self.progress_bar = SmoothProgressBar(
            self.card_progress,
            height=14,
            corner_radius=7,
            fg_color="#0b1120",
            progress_color=ACCENT_CYAN
        )
        self.progress_bar.pack(fill="x", padx=16, pady=(4, 8))
        self.progress_bar.set(0.0)

        # Current detailed status text
        self.lbl_status_detail = ctk.CTkLabel(
            self.card_progress,
            text="រង់ចាំចាប់ផ្តើម...",
            font=FONT_BODY,
            text_color=TEXT_SECONDARY
        )
        self.lbl_status_detail.pack(anchor="w", padx=16, pady=(0, 6))

        # Visual Pipeline Pills (With Pulsing Glow Animation)
        pipeline_title = ctk.CTkLabel(
            self.card_progress,
            text="ដំណាក់កាលដំណើរការស្វ័យប្រវត្តិ (Dubbing Pipeline Stages):",
            font=FONT_SMALL,
            text_color=TEXT_MUTED
        )
        pipeline_title.pack(anchor="w", padx=16, pady=(2, 2))

        self.pipeline_widget = PipelineProgressWidget(self.card_progress, stages=PIPELINE_STAGES)
        self.pipeline_widget.pack(fill="x", padx=16, pady=(2, 12))

        # ---------------- SECTION 6: LIVE LOG CONSOLE ----------------
        card_log = GlassCard(self.main_scroll)
        card_log.pack(fill="x", pady=6)

        log_header = ctk.CTkFrame(card_log, fg_color="transparent")
        log_header.pack(fill="x", padx=16, pady=(10, 4))

        ctk.CTkLabel(
            log_header,
            text="💻 ផ្ទាំងព័ត៌មានលម្អិត (Activity Logs):",
            font=FONT_SMALL,
            text_color=TEXT_SECONDARY
        ).pack(side="left")

        clear_btn = ctk.CTkButton(
            log_header,
            text="Clear",
            width=55,
            height=22,
            font=FONT_SMALL,
            fg_color="#1f2d4a",
            hover_color="#2b3e66",
            command=lambda: self.log_console.clear()
        )
        clear_btn.pack(side="right")

        self.log_console = LogConsoleWidget(card_log, height=135)
        self.log_console.pack(fill="x", padx=16, pady=(0, 12))

        # ---------------- BOTTOM FIXED ACTION BAR ----------------
        bottom_bar = ctk.CTkFrame(self, fg_color="#0c1220", corner_radius=0, height=72)
        bottom_bar.pack(fill="x", side="bottom")

        bar_inner = ctk.CTkFrame(bottom_bar, fg_color="transparent")
        bar_inner.pack(fill="both", expand=True, padx=22, pady=12)

        # Open Output Folder Button
        self.btn_open_folder = ctk.CTkButton(
            bar_inner,
            text="📂 បើក Folder សម្រេច",
            width=170,
            height=44,
            font=FONT_BODY_BOLD,
            fg_color="#18243b",
            border_color="#253556",
            border_width=1,
            hover_color="#2c3e66",
            command=self._open_output_folder
        )
        self.btn_open_folder.pack(side="left")

        # Stop Button
        self.btn_stop = ctk.CTkButton(
            bar_inner,
            text="⏹ Stop",
            width=115,
            height=44,
            font=FONT_BODY_BOLD,
            fg_color="#3d1424",
            border_color="#661430",
            border_width=1,
            hover_color=ACCENT_RED,
            state="disabled",
            command=self._stop_dubbing
        )
        self.btn_stop.pack(side="right", padx=(12, 0))

        # Big Animated Start Dubbing Button
        self.btn_start = ctk.CTkButton(
            bar_inner,
            text="▶ Start Dubbing (ចាប់ផ្តើម)",
            width=250,
            height=44,
            font=(FONT_HEADER[0], 13, "bold"),
            fg_color=ACCENT_CYAN,
            text_color="#000000",
            hover_color="#00e5ff",
            command=self._start_dubbing
        )
        self.btn_start.pack(side="right")

    def _setup_logger_listener(self):
        """Forward logs to the GUI console safely across threads."""
        def on_log(msg: str, level: str):
            self.after(0, lambda: self.log_console.append_log(msg, level))
        logger.add_listener(on_log)

    def _load_saved_paths(self):
        last_in = self.config_manager.get("last_input_folder", "")
        last_out = self.config_manager.get("last_output_folder", "")
        if last_in and os.path.isdir(last_in):
            self.input_entry.insert(0, last_in)
            self._refresh_video_counts(last_in, last_out)
        if last_out and os.path.isdir(last_out):
            self.output_entry.insert(0, last_out)

    def _browse_input_folder(self):
        folder = filedialog.askdirectory(title="ជ្រើសរើស Folder ដែលមានវីដេអូចិន")
        if folder:
            self.input_entry.delete(0, "end")
            self.input_entry.insert(0, folder)
            self.config_manager.set("last_input_folder", folder)
            
            # Auto-suggest output folder if empty
            if not self.output_entry.get().strip():
                auto_out = str(Path(folder).parent / f"{Path(folder).name}_Dubbed_KH")
                self.output_entry.delete(0, "end")
                self.output_entry.insert(0, auto_out)
                self.config_manager.set("last_output_folder", auto_out)

            self._refresh_video_counts(folder, self.output_entry.get().strip())

    def _browse_output_folder(self):
        folder = filedialog.askdirectory(title="ជ្រើសរើស Folder សម្រាប់ Export វីដេអូ")
        if folder:
            self.output_entry.delete(0, "end")
            self.output_entry.insert(0, folder)
            self.config_manager.set("last_output_folder", folder)
            self._refresh_video_counts(self.input_entry.get().strip(), folder)

    def _refresh_video_counts(self, input_folder: str, output_folder: str):
        if not input_folder or not os.path.isdir(input_folder):
            self.lbl_video_stats.configure(text="រកឃើញវីដេអូ: 0 ឯកសារ (Dubbed រួច: 0 ភាគ, នៅសល់: 0 ភាគ)")
            return

        self.video_files = get_video_files(input_folder)
        pattern = self.config_manager.get("output_naming_pattern", "prefix_index")
        prefix = self.config_manager.get("naming_prefix", "EP")

        dubbed_count, remaining_count, self.video_status_list = check_video_status(
            self.video_files, output_folder, pattern, prefix
        )

        total = len(self.video_files)
        self.lbl_video_stats.configure(
            text=f"រកឃើញវីដេអូ {total} ឯកសារ (Dubbed រួច: {dubbed_count} ភាគ, នៅសល់: {remaining_count} ភាគ)"
        )
        self.lbl_progress_counter.configure(text=f"[{dubbed_count}/{total}] រង់ចាំ...")

    def _on_vocal_mode_change(self):
        mode = self.vocal_mode_var.get()
        self.config_manager.set("vocal_mode", mode)
        
        # Update card borders
        self.card_opt1.configure(border_color=ACCENT_CYAN if mode == "separate_bgm" else CARD_BORDER, border_width=1.5 if mode == "separate_bgm" else 1)
        self.card_opt2.configure(border_color=ACCENT_CYAN if mode == "mute_original" else CARD_BORDER, border_width=1.5 if mode == "mute_original" else 1)

        if mode == "separate_bgm":
            self.lbl_naming_notice.configure(text="វីដេអូនឹងរក្សាទុកឈ្មោះ: EP001_KH.mp4 • រក្សាភ្លេង BGM/Sound FX 100% និងលុបតែសំឡេងចិន (Demucs AI)")
        else:
            self.lbl_naming_notice.configure(text="វីដេអូនឹងរក្សាទុកឈ្មោះ: EP001_KH.mp4 • សំឡេងដើមនឹងត្រូវបិទ (Mute) 100% ដាក់តែសំឡេងខ្មែរ")

    def _on_gemini_model_change(self, model_name: str):
        self.config_manager.set("gemini_model", model_name)

    def _on_voice_change(self, selected_display: str):
        voice_key = self.voice_map_reverse.get(selected_display, "km-KH-PisethNeural")
        self.config_manager.set("khmer_voice", voice_key)

    def _open_settings(self):
        SettingsDialog(self, self.config_manager, on_save_callback=self._on_settings_saved)

    def _on_settings_saved(self):
        self._refresh_video_counts(self.input_entry.get().strip(), self.output_entry.get().strip())
        logger.info("បានកែប្រែការកំណត់ប្រព័ន្ធរួចរាល់។")

    def _open_output_folder(self):
        out_dir = self.output_entry.get().strip()
        if out_dir and os.path.exists(out_dir):
            os.startfile(out_dir)
        else:
            messagebox.showinfo("Folder មិនទាន់មាន", "Folder នេះមិនទាន់ត្រូវបានបង្កើតនៅឡើយទេ។")

    def _start_dubbing(self):
        in_dir = self.input_entry.get().strip()
        out_dir = self.output_entry.get().strip()

        if not in_dir or not os.path.isdir(in_dir):
            messagebox.showwarning("ខ្វះទិន្នន័យ", "សូមជ្រើសរើស Input Folder ដែលមានវីដេអូចិនជាមុនសិន!")
            return

        if not out_dir:
            messagebox.showwarning("ខ្វះទិន្នន័យ", "សូមជ្រើសរើស Output Folder សម្រាប់រក្សាទុកវីដេអូ!")
            return

        # Check Gemini API Key
        api_key = self.config_manager.get("gemini_api_key", "").strip()
        if not api_key:
            res = messagebox.askyesno(
                "Gemini API Key មិនទាន់មាន",
                "អ្នកមិនទាន់បានបញ្ចូល Gemini API Key នៅក្នុង Settings ទេ។ តើអ្នកចង់បើក Settings ដើម្បីបញ្ចូល Key ឥឡូវនេះទេ?"
            )
            if res:
                self._open_settings()
            return

        self._refresh_video_counts(in_dir, out_dir)
        if not self.video_files:
            messagebox.showinfo("ពុំមានវីដេអូ", "រកមិនឃើញឯកសារវីដេអូណាមួយនៅក្នុង Folder នេះទេ។")
            return

        # Filter pending videos
        pending_items = [item for item in self.video_status_list if item["status"] == "pending"]
        if not pending_items:
            messagebox.showinfo("រួចរាល់ទាំងអស់", "វីដេអូទាំងអស់នៅក្នុង Folder នេះត្រូវបាន Dubbed រួចរាល់ហើយ!")
            return

        # Create output directory
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        # Switch UI state to ACTIVE & ANIMATED
        self.is_processing = True
        self.btn_start.configure(state="disabled", fg_color="#182238", text="⚡ កំពុងដំណើរការ...")
        self.btn_stop.configure(state="normal")
        self.settings_btn.configure(state="disabled")
        self.card_progress.set_glow(True, ACCENT_CYAN)

        # Trigger Visualizer & Status Dot animation
        self.visualizer.set_mode(True)
        self.status_dot.set_state("processing")

        # Launch processing in background thread
        thread = threading.Thread(target=self._run_batch_pipeline, args=(pending_items, out_dir), daemon=True)
        thread.start()

    def _stop_dubbing(self):
        if self.is_processing:
            self.pipeline.cancel()
            self.status_dot.set_state("stopped")
            self.lbl_status_detail.configure(text="🛑 កំពុងបញ្ឈប់ការងារ...", text_color=ACCENT_RED)

    def _run_batch_pipeline(self, pending_items: List[Dict[str, Any]], out_dir: str):
        total_videos = len(self.video_files)
        total_pending = len(pending_items)
        out_path = Path(out_dir)

        logger.info(f"🚀 ចាប់ផ្តើមដំណើរការ Dubbing ចំនួន {total_pending} វីដេអូ...")

        for idx, item in enumerate(pending_items):
            if self.pipeline.is_cancelled:
                break

            video_file = item["file"]
            out_name = item["out_name"]
            final_out_file = out_path / out_name
            current_video_num = item["index"]

            self.after(0, lambda num=current_video_num, tot=total_videos: 
                self.lbl_progress_counter.configure(text=f"[{num}/{tot}] កំពុងដំណើរការ..."))

            def on_progress(v_idx, v_total, stage_idx, stage_name, pct, detail):
                self.after(0, lambda: self._update_progress_ui(v_idx, v_total, stage_idx, stage_name, pct, detail))

            success = self.pipeline.process_single_video(
                video_path=str(video_file),
                output_path=str(final_out_file),
                video_index=current_video_num,
                total_videos=total_videos,
                progress_callback=on_progress
            )

            if success:
                logger.success(f"Dubbed សម្រេច: {out_name}")
            else:
                logger.error(f"បរាជ័យក្នុងការ Dubbed: {video_file.name}")

            # Refresh status count
            self.after(0, lambda: self._refresh_video_counts(self.input_entry.get().strip(), self.output_entry.get().strip()))

        # Finished all
        self.after(0, self._on_pipeline_finished)

    def _update_progress_ui(self, v_idx: int, v_total: int, stage_idx: int, stage_name: str, pct: float, detail: str):
        # Smooth animated interpolation
        self.progress_bar.set_target(pct)
        self.lbl_progress_stage.configure(text=f"⚡ {stage_name}")
        self.lbl_status_detail.configure(text=detail)
        self.pipeline_widget.set_active_stage(stage_idx)

    def _on_pipeline_finished(self):
        self.is_processing = False
        self.btn_start.configure(state="normal", fg_color=ACCENT_CYAN, text="▶ Start Dubbing (ចាប់ផ្តើម)")
        self.btn_stop.configure(state="disabled")
        self.settings_btn.configure(state="normal")
        self.card_progress.set_glow(False)
        self.pipeline_widget.reset()

        # Stop Visualizer active beat, return to ambient wave
        self.visualizer.set_mode(False)

        if self.pipeline.is_cancelled:
            self.status_dot.set_state("stopped")
            self.lbl_status_detail.configure(text="🛑 ការងារត្រូវបានបញ្ឈប់ដោយអ្នកប្រើប្រាស់។", text_color=ACCENT_RED)
            logger.warning("ការងារត្រូវបានបញ្ឈប់។")
        else:
            self.status_dot.set_state("completed")
            self.progress_bar.set_target(1.0)
            self.lbl_status_detail.configure(text="🎉 ដំណើរការ Dubbing វីដេអូទាំងអស់បានជោគជ័យពេញលេញ!", text_color=ACCENT_GREEN)
            logger.success("🎉 បញ្ចប់ការងារ Dubbing ទាំងអស់ដោយជោគជ័យ!")
            messagebox.showinfo("រួចរាល់", "ដំណើរការ Dubbing វីដេអូបានសម្រេចជោគជ័យ!")
