# -*- coding: utf-8 -*-
"""
Central Version & Release Metadata for JAI AUTO DUBBING.
Single source of truth for UI, Installer, and Build tools.
"""

__version__ = "2.5.3"
__release_date__ = "2026-11-02"
__build_code__ = "20261102.02"
__edition__ = "Imperial Pro Studio Review Edition"

CHANGELOG = [
    {
        "version": "2.5.3",
        "date": "2026-11-02",
        "title": "Two-Pass AI Cinema Review & Audit Engine",
        "highlights": [
            "Two-Pass Reflection Pipeline: automatically audits and refines draft dialogue against physical actions, screen gestures, and speech duration",
            "Action & Gesture Synchronization: guarantees precise cinema verbs for slapping, combat, fleeing, choking, and expulsions",
            "Duration & Timing Fitting: condenses overlong lines for short duration subtitles (<=1.8s) to prevent rushed TTS speech",
            "Pure Khmer Unicode Enforcement with automated regex filters eliminating stray Thai or Chinese Unicode leakage",
            "UI Settings toggle for AI Cinema Review Pass with live status indicator badge"
        ]
    },
    {
        "version": "2.5.2",
        "date": "2026-11-02",
        "title": "Next-Gen Gemini 3.x AI Series Integration",
        "highlights": [
            "Full integration of Gemini 3.x Series (Gemini 3.5 Flash, 3.8 Flash, 3.5 Flash Lite, 3.1 Pro Preview, 3-Flash Preview)",
            "Superior natural Khmer cinema dialogue translation with enhanced emotional depth and context retention",
            "Intelligent resilient fallback chain automatically navigating between 3.x Flash and Lite models",
            "Dynamic custom model recognition supporting future Gemini releases without code changes",
            "Enhanced API testing with automatic request timeouts preventing connection hangs"
        ]
    },
    {
        "version": "2.5.1",
        "date": "2026-10-22",
        "title": "Zero-Cut Audio & 3D Cinema Mastering Release",
        "highlights": [
            "100% Zero-Cut Dialogue Preservation (Whisper directly transcribes original audio)",
            "Zero-Drop TTS Emergency Recovery via gTTS fallback",
            "Foley Impact Gate: isolates battle impacts & horse hooves while dampening background songs",
            "Dynamic Character Role Cadence: custom pitch & rate for hero, elder, girl, and mother personas",
            "Studio Cinema Broadcast Mastering EQ (85Hz High-Pass, 220Hz Chest Warmth, 3.5kHz Clarity, Loudnorm)",
            "Web Audio API celebration chime sound and finish notification modal",
            "Official 3D Imperial Golden Dragon application & desktop icon"
        ]
    },
    {
        "version": "2.5.0",
        "date": "2026-10-15",
        "title": "Modern Animated WebView2 Architecture",
        "highlights": [
            "Hardware-accelerated modern web interface powered by Microsoft Edge WebView2",
            "Multi-video concurrent dubbing engine with individual thread workers",
            "Multi-Key Gemini API pool rotation with automatic quota failover",
            "Live audio spectrum equalizer visualizer"
        ]
    }
]

def get_version_string() -> str:
    return f"v{__version__}"

def get_full_title() -> str:
    return f"JAI AUTO DUBBING {get_version_string()} ({__edition__}) 🎬⚡"

def get_setup_filename() -> str:
    return f"AutoDaiDubber_Setup_v{__version__}.exe"
