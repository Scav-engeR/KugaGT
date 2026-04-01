#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   ██████╗██╗   ██╗██████╗ ███████╗██████╗ ████████╗██████╗  █████╗ ███╗   ██╗║
║  ██╔════╝╚██╗ ██╔╝██╔══██╗██╔════╝██╔══██╗╚══██╔══╝██╔══██╗██╔══██╗████╗  ██║║
║  ██║      ╚████╔╝ ██████╔╝█████╗  ██████╔╝   ██║   ██████╔╝███████║██╔██╗ ██║║
║  ██║       ╚██╔╝  ██╔══██╗██╔══╝  ██╔══██╗   ██║   ██╔══██╗██╔══██║██║╚██╗██║║
║  ╚██████╗   ██║   ██████╔╝███████╗██║  ██║   ██║   ██║  ██║██║  ██║██║ ╚████║║
║   ╚═════╝   ╚═╝   ╚═════╝ ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝║
║                                                                               ║
║                    NEURAL TRANSCRIPTION ENGINE - YEAR 3999                    ║
║                                                                               ║
║   Version: 5.0.0 V2 Enhanced Edition                                          ║
║   New: LLM Refinement • Hallucination Filter • Domain Presets • Multi-Format  ║
║        Glossary Engine • Audio Preprocessing • Confidence Scoring             ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import tempfile
import os
import io
import csv
import logging
from pathlib import Path
import gc
from datetime import datetime, timedelta
import time
from typing import List, Dict, Optional, Tuple, Any
import json
import random
import sys

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

# ============================================================================
# SAFE IMPORTS WITH ERROR HANDLING
# ============================================================================
IMPORTS_STATUS = {}


def safe_import(module_name, package_name=None):
    """Safely import a module and track status"""
    if package_name is None:
        package_name = module_name
    try:
        module = __import__(module_name, fromlist=[""])
        IMPORTS_STATUS[package_name] = "OK"
        return module
    except ImportError as e:
        IMPORTS_STATUS[package_name] = str(e)
        return None


# Core imports
torch = safe_import("torch")
WhisperModel = None
if torch:
    try:
        from faster_whisper import WhisperModel
        IMPORTS_STATUS["faster-whisper"] = "OK"
    except ImportError as e:
        IMPORTS_STATUS["faster-whisper"] = str(e)

Pipeline = None
try:
    from pyannote.audio import Pipeline
    IMPORTS_STATUS["pyannote.audio"] = "OK"
except ImportError as e:
    IMPORTS_STATUS["pyannote.audio"] = str(e)

yt_dlp = safe_import("yt_dlp", "yt-dlp")
ffmpeg = safe_import("ffmpeg", "ffmpeg-python")
psutil = safe_import("psutil")
srt = safe_import("srt")

# V2 audio preprocessing imports
nr_module = safe_import("noisereduce", "noisereduce")
sf_module = safe_import("soundfile", "soundfile")

try:
    import numpy as np
    IMPORTS_STATUS["numpy"] = "OK"
except ImportError:
    np = None
    IMPORTS_STATUS["numpy"] = "not installed"

# LLM imports
anthropic_module = safe_import("anthropic")
openai_module = safe_import("openai")

# Fine-tuning imports
datasets_module = None
transformers_module = None
peft_module = None

try:
    import datasets as datasets_module
    IMPORTS_STATUS["datasets"] = "OK"
except Exception as e:
    IMPORTS_STATUS["datasets"] = str(e)

try:
    import transformers as transformers_module
    IMPORTS_STATUS["transformers"] = "OK"
except Exception as e:
    IMPORTS_STATUS["transformers"] = str(e)

try:
    import peft as peft_module
    IMPORTS_STATUS["peft"] = "OK"
except Exception as e:
    IMPORTS_STATUS["peft"] = str(e)


# ============================================================================
# ENVIRONMENT VARIABLES
# ============================================================================
class EnvConfig:
    """Environment configuration from .env file"""
    HF_TOKEN = os.getenv("HF_TOKEN", "")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    GROK_API_KEY = os.getenv("GROK_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


# ============================================================================
# HALLUCINATION PATTERNS (common Whisper artifacts to filter)
# ============================================================================
HALLUCINATION_PATTERNS = [
    "thank you for watching",
    "thanks for watching",
    "please subscribe",
    "like and subscribe",
    "don't forget to subscribe",
    "hit the like button",
    "turn on notifications",
    "see you in the next video",
    "this video is brought to you",
    "subtitles by",
    "transcribed by",
    "amara.org",
    "www.youtube.com",
    "視聴ありがとうございました",  # Japanese "thanks for watching"
    "チャンネル登録",             # Japanese "subscribe to channel"
]


# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    """Application configuration"""

    APP_NAME = "CyberTranscribe 3999"
    APP_VERSION = "5.0.0"
    APP_SUBTITLE = "Neural Transcription Engine V2"

    # Domain presets for initial prompts — dramatically improve accuracy
    DOMAIN_PRESETS = {
        "general": {
            "name": "General",
            "prompt": "",
            "description": "General content, no specific domain",
        },
        "anime": {
            "name": "Anime / Manga",
            "prompt": (
                "Japanese anime dialogue. Character names, honorifics (san, kun, chan, "
                "senpai, sensei), anime-specific vocabulary, and emotional speech patterns."
            ),
            "description": "Optimized for anime/manga content",
        },
        "technical": {
            "name": "Technical / IT",
            "prompt": (
                "Technical documentation with precise terminology. Software development, "
                "programming, engineering concepts and acronyms."
            ),
            "description": "Technical and IT content",
        },
        "news": {
            "name": "News / Broadcast",
            "prompt": (
                "News broadcast with formal speech. Proper nouns, place names, "
                "political and economic terminology."
            ),
            "description": "News and formal broadcast",
        },
        "medical": {
            "name": "Medical",
            "prompt": (
                "Medical and clinical content with specialized terminology, "
                "drug names, anatomical terms, and diagnostic language."
            ),
            "description": "Medical and healthcare content",
        },
        "academic": {
            "name": "Academic / Research",
            "prompt": (
                "Academic lecture with technical vocabulary. Scientific terminology, "
                "research concepts, and citation-style speech."
            ),
            "description": "Academic and research content",
        },
        "business": {
            "name": "Business",
            "prompt": (
                "Business meeting with corporate terminology, financial terms, "
                "and formal Japanese keigo (honorific speech)."
            ),
            "description": "Business and corporate content",
        },
        "gaming": {
            "name": "Gaming / Esports",
            "prompt": (
                "Gaming commentary and esports content. Game titles, player names, "
                "gaming slang, and live-play reactions."
            ),
            "description": "Gaming and esports content",
        },
    }

    # Supported output formats
    OUTPUT_FORMATS = ["SRT", "VTT", "ASS", "TXT", "JSON", "CSV", "Dual-SRT"]

    # Comprehensive Whisper model list
    WHISPER_MODELS = {
        "kotoba-tech/kotoba-whisper-bilingual-v1.0-faster": {
            "name": "Kotoba Whisper Bilingual",
            "description": "Best for Japanese↔English translation",
            "size": "~800MB",
            "speed": "6.3x faster",
            "recommended": True,
            "language": "ja/en",
            "ct2_native": True,
        },
        "deepdml/faster-whisper-large-v3-turbo-ct2": {
            "name": "Large V3 Turbo",
            "description": "Fast & accurate, great all-rounder",
            "size": "~1.5GB",
            "speed": "8x faster",
            "recommended": True,
            "language": "multi",
            "ct2_native": True,
        },
        "large-v3": {
            "name": "Whisper Large V3",
            "description": "Highest accuracy, slowest",
            "size": "~3GB",
            "speed": "1x baseline",
            "recommended": False,
            "language": "multi",
            "ct2_native": True,
        },
        "large-v2": {
            "name": "Whisper Large V2",
            "description": "Previous best model",
            "size": "~3GB",
            "speed": "1x baseline",
            "recommended": False,
            "language": "multi",
            "ct2_native": True,
        },
        "medium": {
            "name": "Whisper Medium",
            "description": "Balanced speed/accuracy",
            "size": "~1.5GB",
            "speed": "3x faster",
            "recommended": True,
            "language": "multi",
            "ct2_native": True,
        },
        "small": {
            "name": "Whisper Small",
            "description": "Fast, good for real-time",
            "size": "~500MB",
            "speed": "5x faster",
            "recommended": False,
            "language": "multi",
            "ct2_native": True,
        },
        "base": {
            "name": "Whisper Base",
            "description": "Lightweight, quick tests",
            "size": "~150MB",
            "speed": "7x faster",
            "recommended": False,
            "language": "multi",
            "ct2_native": True,
        },
        "tiny": {
            "name": "Whisper Tiny",
            "description": "Smallest, fastest, lowest accuracy",
            "size": "~75MB",
            "speed": "10x faster",
            "recommended": False,
            "language": "multi",
            "ct2_native": True,
        },
        "Systran/faster-whisper-large-v3": {
            "name": "Systran Large V3",
            "description": "Optimized CTranslate2 version",
            "size": "~3GB",
            "speed": "2x faster",
            "recommended": False,
            "language": "multi",
            "ct2_native": True,
        },
        "Systran/faster-whisper-medium": {
            "name": "Systran Medium",
            "description": "Optimized medium model",
            "size": "~1.5GB",
            "speed": "4x faster",
            "recommended": False,
            "language": "multi",
            "ct2_native": True,
        },
        "openai/whisper-large-v3-turbo": {
            "name": "OpenAI V3 Turbo",
            "description": "Official turbo variant",
            "size": "~1.5GB",
            "speed": "8x faster",
            "recommended": False,
            "language": "multi",
            "ct2_native": False,
        },
        "distil-whisper/distil-large-v3": {
            "name": "Distil Large V3",
            "description": "Distilled for speed",
            "size": "~1GB",
            "speed": "6x faster",
            "recommended": False,
            "language": "en",
            "ct2_native": False,
        },
        "kotoba-tech/kotoba-whisper-v2.0": {
            "name": "Kotoba V2.0",
            "description": "Japanese ASR specialized",
            "size": "~800MB",
            "speed": "5x faster",
            "recommended": False,
            "language": "ja",
            "ct2_native": True,
        },
        "japanese-asr/distil-whisper-large-v3-ja-reazonspeech-all": {
            "name": "Distil Whisper JA ReazonSpeech",
            "description": "Distilled model trained on ReazonSpeech",
            "size": "~1GB",
            "speed": "6x faster",
            "recommended": True,
            "language": "ja",
            "ct2_native": False,
            "experimental": True,
        },
        "efwkjn/whisper-ja-anime-v0.3": {
            "name": "Whisper JA Anime v0.3",
            "description": "Fine-tuned for Japanese anime content",
            "size": "~1.5GB",
            "speed": "3x faster",
            "recommended": False,
            "language": "ja",
            "ct2_native": False,
            "experimental": True,
        },
        "hhim8826/whisper-large-v3-turbo-ja": {
            "name": "Whisper Large V3 Turbo JA",
            "description": "Turbo model fine-tuned for Japanese",
            "size": "~1.5GB",
            "speed": "6x faster",
            "recommended": False,
            "language": "ja",
            "ct2_native": False,
            "experimental": True,
        },
        "HuyHoang1977/ft-whisperl-ja": {
            "name": "FT Whisper-L Japanese",
            "description": "Fine-tuned Whisper Large for Japanese",
            "size": "~3GB",
            "speed": "1x baseline",
            "recommended": False,
            "language": "ja",
            "ct2_native": False,
            "experimental": True,
        },
    }

    DEFAULT_WHISPER_MODEL = "medium"
    DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"

    LANGUAGES = {
        "ja": "Japanese 日本語",
        "en": "English",
        "zh": "Chinese 中文",
        "ko": "Korean 한국어",
        "es": "Spanish Español",
        "fr": "French Français",
        "de": "German Deutsch",
        "it": "Italian Italiano",
        "pt": "Portuguese Português",
        "ru": "Russian Русский",
        "ar": "Arabic العربية",
        "hi": "Hindi हिन्दी",
        "auto": "Auto Detect",
    }

    LLM_PROVIDERS = {
        "claude": {
            "name": "Claude (Anthropic)",
            "models": [
                "claude-opus-4-6",
                "claude-sonnet-4-6",
                "claude-haiku-4-5-20251001",
            ],
            "default": "claude-sonnet-4-6",
        },
        "deepseek": {
            "name": "DeepSeek",
            "models": ["deepseek-chat", "deepseek-coder"],
            "default": "deepseek-chat",
        },
        "grok": {
            "name": "Grok (xAI)",
            "models": ["grok-beta", "grok-2-1212"],
            "default": "grok-beta",
        },
        "openai": {
            "name": "OpenAI",
            "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
            "default": "gpt-4o",
        },
    }

    MAX_FILE_SIZE_MB = 2500
    AUDIO_SAMPLE_RATE = 16000
    AUDIO_CHANNELS = 1
    TEMP_DIR = Path(tempfile.gettempdir()) / "cybertranscribe_v2"
    LOG_LEVEL = logging.INFO


# ============================================================================
# THEME DEFINITIONS
# ============================================================================
THEMES = {
    "cyberpunk_neon": {
        "name": "🌆 Cyberpunk Neon",
        "description": "Classic pink & cyan neon aesthetic",
        "primary": "#ff00ff",
        "secondary": "#00ffff",
        "accent": "#ff0080",
        "background": "#0a0e27",
        "surface": "#16003b",
        "text": "#ffffff",
        "success": "#00ff00",
        "error": "#ff0000",
        "warning": "#ffaa00",
    },
    "retro_terminal": {
        "name": "💻 Retro Terminal",
        "description": "Classic green phosphor CRT",
        "primary": "#00ff00",
        "secondary": "#00aa00",
        "accent": "#88ff88",
        "background": "#000000",
        "surface": "#001100",
        "text": "#00ff00",
        "success": "#00ff00",
        "error": "#ff0000",
        "warning": "#ffff00",
    },
    "synthwave_sunset": {
        "name": "🌅 Synthwave Sunset",
        "description": "Warm purple & orange gradient",
        "primary": "#ff6b35",
        "secondary": "#9b5de5",
        "accent": "#f15bb5",
        "background": "#10002b",
        "surface": "#240046",
        "text": "#ffffff",
        "success": "#00f5d4",
        "error": "#ff006e",
        "warning": "#fee440",
    },
    "matrix_code": {
        "name": "🟢 Matrix Code",
        "description": "Digital rain green theme",
        "primary": "#00ff41",
        "secondary": "#008f11",
        "accent": "#00ff41",
        "background": "#0d0208",
        "surface": "#003b00",
        "text": "#00ff41",
        "success": "#00ff41",
        "error": "#ff0000",
        "warning": "#ffff00",
    },
    "vaporwave": {
        "name": "🌸 Vaporwave",
        "description": "Aesthetic pastel pink & blue",
        "primary": "#ff71ce",
        "secondary": "#01cdfe",
        "accent": "#05ffa1",
        "background": "#1a1a2e",
        "surface": "#2d1b4e",
        "text": "#ffffff",
        "success": "#05ffa1",
        "error": "#ff2281",
        "warning": "#fffb96",
    },
    "blade_runner": {
        "name": "🌃 Blade Runner",
        "description": "Orange & teal neo-noir",
        "primary": "#ff9e00",
        "secondary": "#00b4d8",
        "accent": "#ff6b00",
        "background": "#0a1628",
        "surface": "#1b2838",
        "text": "#caf0f8",
        "success": "#06d6a0",
        "error": "#ef476f",
        "warning": "#ffd166",
    },
    "tokyo_night": {
        "name": "🗼 Tokyo Night",
        "description": "Soft purple & blue night theme",
        "primary": "#bb9af7",
        "secondary": "#7aa2f7",
        "accent": "#ff9e64",
        "background": "#1a1b26",
        "surface": "#24283b",
        "text": "#c0caf5",
        "success": "#9ece6a",
        "error": "#f7768e",
        "warning": "#e0af68",
    },
    "dracula": {
        "name": "🧛 Dracula",
        "description": "Popular dark purple theme",
        "primary": "#bd93f9",
        "secondary": "#8be9fd",
        "accent": "#ff79c6",
        "background": "#282a36",
        "surface": "#44475a",
        "text": "#f8f8f2",
        "success": "#50fa7b",
        "error": "#ff5555",
        "warning": "#f1fa8c",
    },
}

# ============================================================================
# LOGGING
# ============================================================================
logging.basicConfig(
    level=Config.LOG_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# THEME CSS GENERATOR
# ============================================================================
def generate_theme_css(theme_key: str) -> str:
    """Generate CSS for selected theme"""
    theme = THEMES.get(theme_key, THEMES["cyberpunk_neon"])

    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

    :root {{
        --primary: {theme['primary']};
        --secondary: {theme['secondary']};
        --accent: {theme['accent']};
        --background: {theme['background']};
        --surface: {theme['surface']};
        --text: {theme['text']};
        --success: {theme['success']};
        --error: {theme['error']};
        --warning: {theme['warning']};
    }}

    * {{ font-family: 'Rajdhani', 'Orbitron', 'Share Tech Mono', sans-serif !important; }}

    [data-testid="stAppViewContainer"] {{
        background: linear-gradient(135deg, var(--background) 0%, var(--surface) 50%, var(--background) 100%);
    }}

    [data-testid="stAppViewContainer"]::before {{
        content: "";
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        background: repeating-linear-gradient(0deg, rgba(255,255,255,0.02) 0px, transparent 1px, transparent 2px);
        pointer-events: none;
        z-index: 1000;
    }}

    [data-testid="stHeader"] {{ background: transparent !important; }}

    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, var(--surface) 0%, var(--background) 100%);
        border-right: 3px solid var(--primary);
    }}

    h1 {{
        color: var(--text) !important;
        font-weight: 900 !important;
        letter-spacing: 6px !important;
        text-transform: uppercase;
        text-shadow: 0 0 10px var(--primary), 0 0 20px var(--primary), 0 0 40px var(--primary);
    }}

    h2, h3 {{
        color: var(--secondary) !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 3px;
        text-shadow: 0 0 20px var(--secondary);
    }}

    p, span, div {{ color: var(--text); }}

    .stButton>button {{
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%) !important;
        color: var(--background) !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 15px 40px !important;
        font-weight: 900 !important;
        font-size: 16px !important;
        text-transform: uppercase !important;
        letter-spacing: 3px !important;
        box-shadow: 0 0 20px var(--primary);
        transition: all 0.3s ease;
    }}

    .stButton>button:hover {{
        transform: scale(1.05) translateY(-3px);
        box-shadow: 0 0 40px var(--primary), 0 0 80px var(--secondary);
    }}

    .stTextInput>div>div>input, .stTextArea textarea {{
        background: rgba(0,0,0,0.6) !important;
        border: 2px solid var(--secondary) !important;
        border-radius: 8px !important;
        color: var(--text) !important;
        font-family: 'Share Tech Mono', monospace !important;
    }}

    [data-testid="stFileUploader"] {{
        background: rgba(255,255,255,0.05) !important;
        border: 3px dashed var(--primary) !important;
        border-radius: 15px !important;
        padding: 30px !important;
    }}

    .stSelectbox>div>div {{
        background: rgba(0,0,0,0.6) !important;
        border: 2px solid var(--primary) !important;
        border-radius: 8px !important;
    }}

    .stProgress > div > div > div {{
        background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 50%, var(--accent) 100%) !important;
        box-shadow: 0 0 30px var(--primary);
        border-radius: 10px;
    }}

    [data-testid="stMetricValue"] {{
        color: var(--success) !important;
        font-size: 28px !important;
        font-weight: 900 !important;
        text-shadow: 0 0 20px var(--success);
    }}

    .stTabs [data-baseweb="tab"] {{
        background: rgba(255,255,255,0.05) !important;
        border: 2px solid var(--primary) !important;
        border-radius: 10px 10px 0 0;
        color: var(--primary) !important;
        font-weight: 700;
    }}

    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, var(--primary), var(--secondary)) !important;
        color: var(--background) !important;
    }}

    .stDownloadButton>button {{
        background: linear-gradient(135deg, var(--success) 0%, var(--secondary) 100%) !important;
        color: var(--background) !important;
    }}

    .stChatMessage {{
        background: rgba(0,0,0,0.3) !important;
        border: 1px solid var(--primary) !important;
        border-radius: 10px !important;
    }}

    @media (prefers-reduced-motion: reduce) {{
        *, *::before, *::after {{ animation-duration: 0.01ms !important; }}
    }}
    </style>
    """


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def get_system_stats() -> Dict:
    """Get current system statistics"""
    stats = {"memory_percent": 0, "cpu_percent": 0, "memory_used_gb": 0, "memory_total_gb": 0}
    if psutil:
        memory = psutil.virtual_memory()
        stats["memory_percent"] = memory.percent
        stats["memory_used_gb"] = memory.used / (1024 ** 3)
        stats["memory_total_gb"] = memory.total / (1024 ** 3)
        stats["cpu_percent"] = psutil.cpu_percent(interval=0.1)
    return stats


def format_timestamp_srt(seconds: float) -> str:
    """Format seconds to SRT timestamp HH:MM:SS,mmm"""
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    """Format seconds to WebVTT timestamp HH:MM:SS.mmm"""
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def format_timestamp_ass(seconds: float) -> str:
    """Format seconds to ASS timestamp H:MM:SS.cc"""
    total_cs = int(seconds * 100)
    hours = total_cs // 360000
    minutes = (total_cs % 360000) // 6000
    secs = (total_cs % 6000) // 100
    cs = total_cs % 100
    return f"{hours}:{minutes:02d}:{secs:02d}.{cs:02d}"


def generate_session_id() -> str:
    """Generate unique session identifier"""
    return f"SESSION-{datetime.now().strftime('%Y%m%d%H%M%S')}-{random.randint(1000, 9999)}"


def clear_gpu_memory():
    """Clear GPU memory"""
    if torch and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def safe_model_cleanup():
    """Safely cleanup model resources"""
    clear_gpu_memory()
    gc.collect()
    time.sleep(0.5)


def parse_glossary_text(text: str) -> Dict[str, str]:
    """Parse glossary text (one 'term = replacement' per line) into a dict"""
    glossary = {}
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            parts = line.split("=", 1)
            term = parts[0].strip()
            replacement = parts[1].strip()
            if term:
                glossary[term] = replacement
    return glossary


def calculate_confidence_stats(segments: List[Dict]) -> Dict:
    """Calculate confidence statistics across segments"""
    if not segments:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "low_confidence_count": 0}

    probs = [s.get("avg_logprob", 0.0) for s in segments]
    mean_prob = sum(probs) / len(probs)
    low_conf = sum(1 for p in probs if p < -0.5)

    return {
        "mean": mean_prob,
        "min": min(probs),
        "max": max(probs),
        "low_confidence_count": low_conf,
        "total": len(segments),
    }


# ============================================================================
# V2 TRANSLATION QUALITY FUNCTIONS
# ============================================================================
def detect_and_filter_hallucinations(
    segments: List[Dict],
    patterns: Optional[List[str]] = None,
    min_confidence: float = -2.0,
    max_no_speech_prob: float = 0.9,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Filter hallucinated segments.
    Returns (clean_segments, filtered_segments).
    """
    if patterns is None:
        patterns = HALLUCINATION_PATTERNS

    patterns_lower = [p.lower() for p in patterns]
    clean: List[Dict] = []
    filtered: List[Dict] = []

    for seg in segments:
        text_lower = seg["text"].lower().strip()

        # Filter by log-prob confidence
        if seg.get("avg_logprob", 0.0) < min_confidence:
            filtered.append(seg)
            continue

        # Filter high no-speech probability
        if seg.get("no_speech_prob", 0.0) > max_no_speech_prob:
            filtered.append(seg)
            continue

        # Filter empty or trivially short segments
        if len(text_lower) < 2:
            filtered.append(seg)
            continue

        # Filter known hallucination strings
        if any(pattern in text_lower for pattern in patterns_lower):
            filtered.append(seg)
            continue

        clean.append(seg)

    return clean, filtered


def merge_short_segments(
    segments: List[Dict],
    min_duration: float = 1.5,
    max_gap: float = 0.5,
    max_merged_duration: float = 8.0,
) -> List[Dict]:
    """
    Merge short/adjacent segments for better subtitle readability.
    Segments shorter than min_duration or with gap < max_gap are merged,
    as long as the result stays under max_merged_duration.
    """
    if not segments:
        return segments

    merged: List[Dict] = []
    current = segments[0].copy()

    for next_seg in segments[1:]:
        duration = current["end"] - current["start"]
        gap = next_seg["start"] - current["end"]
        merged_duration = next_seg["end"] - current["start"]

        should_merge = (
            (duration < min_duration or gap <= max_gap)
            and merged_duration <= max_merged_duration
        )

        if should_merge:
            current["end"] = next_seg["end"]
            current["text"] = current["text"].rstrip() + " " + next_seg["text"].lstrip()
            # Propagate lowest confidence
            if "avg_logprob" in current and "avg_logprob" in next_seg:
                current["avg_logprob"] = min(
                    current["avg_logprob"], next_seg["avg_logprob"]
                )
        else:
            merged.append(current)
            current = next_seg.copy()

    merged.append(current)
    return merged


def apply_glossary(segments: List[Dict], glossary: Dict[str, str]) -> List[Dict]:
    """Apply terminology glossary replacements to segment text"""
    if not glossary:
        return segments

    result = []
    for seg in segments:
        text = seg["text"]
        for term, replacement in glossary.items():
            text = text.replace(term, replacement)
        new_seg = seg.copy()
        new_seg["text"] = text
        result.append(new_seg)

    return result


# ============================================================================
# MODEL LOADING
# ============================================================================
def clear_model_cache():
    """Clear the cached model"""
    load_whisper_model.clear()
    clear_gpu_memory()
    logger.info("Model cache cleared")


@st.cache_resource(show_spinner=False)
def load_whisper_model(model_name: str, device: str, compute_type: str):
    """Load and cache Whisper model"""
    if not WhisperModel:
        return None

    clear_gpu_memory()

    model_info = Config.WHISPER_MODELS.get(model_name, {})
    is_experimental = model_info.get("experimental", False)
    is_ct2_native = model_info.get("ct2_native", True)

    try:
        if is_experimental:
            logger.warning(f"Loading experimental model: {model_name}")
            st.warning(f"⚠️ {model_name} is experimental")

        if not is_ct2_native:
            try:
                model = WhisperModel(
                    model_name,
                    device=device,
                    compute_type=compute_type,
                    download_root=str(Config.TEMP_DIR / "models"),
                )
            except Exception as e:
                logger.warning(f"Failed with {compute_type}, trying float32: {e}")
                if compute_type != "float32":
                    model = WhisperModel(
                        model_name,
                        device="cpu",
                        compute_type="float32",
                        download_root=str(Config.TEMP_DIR / "models"),
                    )
                else:
                    raise
        else:
            model = WhisperModel(
                model_name,
                device=device,
                compute_type=compute_type,
                download_root=str(Config.TEMP_DIR / "models"),
            )

        logger.info(f"Successfully loaded model: {model_name}")
        return model

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Model loading failed for {model_name}: {error_msg}")
        if "CUDA" in error_msg or "cuda" in error_msg:
            st.error("❌ GPU error. Try switching to CPU.")
        elif "memory" in error_msg.lower():
            st.error("❌ Out of memory. Try a smaller model.")
        else:
            st.error(f"❌ Failed to load model: {error_msg}")
        clear_gpu_memory()
        return None


@st.cache_resource(show_spinner=False)
def load_diarization_model(hf_token: Optional[str] = None):
    """Load speaker diarization model"""
    if not Pipeline:
        return None
    try:
        token = hf_token or EnvConfig.HF_TOKEN
        pipeline = Pipeline.from_pretrained(
            Config.DIARIZATION_MODEL, use_auth_token=token
        )
        if torch and torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
        return pipeline
    except Exception as e:
        logger.error(f"Diarization loading failed: {e}")
        return None


# ============================================================================
# LLM INTEGRATION
# ============================================================================
def get_llm_response(
    provider: str, model: str, messages: list, system_prompt: str = None
) -> str:
    """Get response from LLM provider"""
    try:
        if provider == "claude":
            if not anthropic_module:
                return "❌ Anthropic library not installed"
            import anthropic
            client = anthropic.Anthropic(api_key=EnvConfig.ANTHROPIC_API_KEY)
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                system=system_prompt or "You are a helpful AI assistant.",
                messages=messages,
            )
            return response.content[0].text

        elif provider in ("deepseek", "grok", "openai"):
            if not openai_module:
                return "❌ OpenAI library not installed"
            from openai import OpenAI

            base_urls = {
                "deepseek": "https://api.deepseek.com",
                "grok": "https://api.x.ai/v1",
                "openai": None,
            }
            api_keys = {
                "deepseek": EnvConfig.DEEPSEEK_API_KEY,
                "grok": EnvConfig.GROK_API_KEY,
                "openai": EnvConfig.OPENAI_API_KEY,
            }

            kwargs = {"api_key": api_keys[provider]}
            if base_urls[provider]:
                kwargs["base_url"] = base_urls[provider]

            client = OpenAI(**kwargs)
            formatted_messages = messages.copy()
            if system_prompt:
                formatted_messages.insert(0, {"role": "system", "content": system_prompt})

            response = client.chat.completions.create(
                model=model, messages=formatted_messages
            )
            return response.choices[0].message.content

        return f"❌ Unknown provider: {provider}"

    except Exception as e:
        logger.error(f"LLM error: {e}")
        return f"❌ Error: {str(e)}"


def refine_translation_with_llm(
    segments: List[Dict],
    provider: str,
    model: str,
    target_language: str = "English",
    domain: str = "general",
    chunk_size: int = 40,
) -> List[Dict]:
    """
    Refine transcription/translation using an LLM in batched chunks.
    Sends segments in groups to stay within token limits.
    Returns segments with 'original_text' set to the pre-refinement text.
    """
    if not segments:
        return segments

    domain_info = Config.DOMAIN_PRESETS.get(domain, {})
    domain_context = domain_info.get("prompt", "")

    system_prompt = (
        f"You are an expert translator and transcription editor specialising in {target_language}.\n"
        + (f"Domain: {domain_context}\n" if domain_context else "")
        + "Improve accuracy and naturalness of transcribed segments.\n"
        "Return ONLY a valid JSON array of improved strings — same count as input, same order.\n"
        "Example: [\"improved text 1\", \"improved text 2\"]"
    )

    all_refined: List[Dict] = []

    total_chunks = (len(segments) + chunk_size - 1) // chunk_size
    progress = st.progress(0, text="Refining with LLM...")

    for chunk_idx in range(0, len(segments), chunk_size):
        chunk = segments[chunk_idx: chunk_idx + chunk_size]

        numbered = "\n".join(
            f"{i + 1}. {seg['text']}" for i, seg in enumerate(chunk)
        )

        user_message = (
            f"Improve these {len(chunk)} segments to {target_language}:\n\n"
            f"{numbered}\n\n"
            f"Return a JSON array with exactly {len(chunk)} improved strings."
        )

        response = get_llm_response(
            provider, model, [{"role": "user", "content": user_message}], system_prompt
        )

        # Try to extract JSON from the response
        improved_texts = None
        try:
            # Strip markdown code fences if present
            clean_response = response.strip()
            if clean_response.startswith("```"):
                lines = clean_response.splitlines()
                clean_response = "\n".join(
                    l for l in lines if not l.startswith("```")
                )
            improved_texts = json.loads(clean_response)
            if not isinstance(improved_texts, list) or len(improved_texts) != len(chunk):
                improved_texts = None
        except (json.JSONDecodeError, ValueError):
            improved_texts = None

        for i, seg in enumerate(chunk):
            new_seg = seg.copy()
            new_seg["original_text"] = seg["text"]
            if improved_texts and i < len(improved_texts):
                new_seg["text"] = str(improved_texts[i]).strip()
            all_refined.append(new_seg)

        pct = int(((chunk_idx + len(chunk)) / len(segments)) * 100)
        progress.progress(pct, text=f"Refining chunk {chunk_idx // chunk_size + 1}/{total_chunks}...")

        # Polite delay between chunks
        if chunk_idx + chunk_size < len(segments):
            time.sleep(0.3)

    progress.empty()
    return all_refined


# ============================================================================
# AUDIO / VIDEO PROCESSING
# ============================================================================
def download_video_from_url(url: str, output_path: Path) -> Optional[Path]:
    """Download video/audio using yt-dlp"""
    if not yt_dlp:
        return None
    try:
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": str(output_path),
            "quiet": True,
            "no_warnings": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return output_path
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return None


def extract_audio(video_path: Path, output_path: Path) -> Optional[Path]:
    """Extract audio from video file using ffmpeg"""
    if not ffmpeg:
        return None
    try:
        (
            ffmpeg.input(str(video_path))
            .output(
                str(output_path),
                acodec="pcm_s16le",
                ac=Config.AUDIO_CHANNELS,
                ar=str(Config.AUDIO_SAMPLE_RATE),
            )
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True, quiet=True)
        )
        return output_path
    except Exception as e:
        logger.error(f"Audio extraction failed: {e}")
        return None


def preprocess_audio_file(
    audio_path: Path,
    output_path: Path,
    reduce_noise: bool = True,
    normalize: bool = True,
) -> Path:
    """
    Apply noise reduction and amplitude normalisation.
    Falls back to the original path if prerequisites are missing.
    """
    if not (nr_module and sf_module and np is not None):
        logger.info("Preprocessing skipped — noisereduce/soundfile/numpy not available")
        return audio_path

    try:
        data, rate = sf_module.read(str(audio_path))

        # Ensure mono float32
        if data.ndim > 1:
            data = data.mean(axis=1)
        data = data.astype("float32")

        if reduce_noise:
            data = nr_module.reduce_noise(y=data, sr=rate, stationary=False, prop_decrease=0.75)

        if normalize:
            peak = np.abs(data).max()
            if peak > 0:
                data = data / peak * 0.9

        sf_module.write(str(output_path), data, rate, subtype="PCM_16")
        logger.info(f"Audio preprocessed → {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Audio preprocessing failed: {e}")
        return audio_path


# ============================================================================
# TRANSCRIPTION  (enhanced for V2)
# ============================================================================
def transcribe_audio(
    model,
    audio_path: Path,
    language: str = "ja",
    task: str = "translate",
    beam_size: int = 5,
    initial_prompt: str = "",
    temperature: float = 0.0,
    condition_on_previous_text: bool = True,
    no_speech_threshold: float = 0.6,
    log_prob_threshold: float = -1.0,
    compression_ratio_threshold: float = 2.4,
    repetition_penalty: float = 1.0,
    word_timestamps: bool = True,
) -> Tuple[Optional[List[Dict]], Optional[Any]]:
    """
    Transcribe audio with enhanced accuracy parameters.
    Returns (segment_list, transcription_info).
    """
    try:
        progress_bar = st.progress(0, text="Initializing transcription...")

        lang_param = None if language == "auto" else language

        kwargs: Dict[str, Any] = {
            "language": lang_param,
            "task": task,
            "beam_size": beam_size,
            "vad_filter": True,
            "word_timestamps": word_timestamps,
            "condition_on_previous_text": condition_on_previous_text,
            "no_speech_threshold": no_speech_threshold,
            "log_prob_threshold": log_prob_threshold,
            "compression_ratio_threshold": compression_ratio_threshold,
        }

        if initial_prompt:
            kwargs["initial_prompt"] = initial_prompt

        if temperature > 0.0:
            kwargs["temperature"] = temperature

        if repetition_penalty != 1.0:
            kwargs["repetition_penalty"] = repetition_penalty

        segments_gen, info = model.transcribe(str(audio_path), **kwargs)

        progress_bar.progress(
            15,
            text=f"Detected language: {info.language} ({info.language_probability:.0%} confidence)",
        )

        segment_list: List[Dict] = []
        segments_temp = list(segments_gen)
        total = len(segments_temp)

        for idx, seg in enumerate(segments_temp):
            segment_list.append(
                {
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text.strip(),
                    "avg_logprob": seg.avg_logprob,
                    "no_speech_prob": seg.no_speech_prob,
                    "compression_ratio": seg.compression_ratio,
                }
            )
            pct = 15 + int((idx / max(total, 1)) * 80)
            progress_bar.progress(pct, text=f"Transcribing segment {idx + 1}/{total}...")

        progress_bar.progress(100, text="✅ Transcription complete!")
        time.sleep(0.4)
        progress_bar.empty()

        return segment_list, info

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        st.error(f"Transcription failed: {e}")
        return None, None
    finally:
        gc.collect()


# ============================================================================
# SPEAKER DIARIZATION
# ============================================================================
def perform_diarization(
    pipeline, audio_path: Path, num_speakers: Optional[int] = None
) -> Optional[List[Dict]]:
    """Perform speaker diarization"""
    try:
        kwargs = {}
        if num_speakers:
            kwargs["num_speakers"] = num_speakers
        diarization = pipeline(str(audio_path), **kwargs)
        return [
            {"start": turn.start, "end": turn.end, "speaker": speaker}
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]
    except Exception as e:
        logger.error(f"Diarization failed: {e}")
        return None


def align_transcription_with_speakers(
    transcription: List[Dict], diarization: List[Dict]
) -> List[Dict]:
    """Align transcription segments with speaker labels via overlap"""
    aligned = []
    for trans_seg in transcription:
        max_overlap = 0.0
        best_speaker = "UNKNOWN"
        for diar_seg in diarization:
            overlap = max(
                0.0,
                min(trans_seg["end"], diar_seg["end"])
                - max(trans_seg["start"], diar_seg["start"]),
            )
            if overlap > max_overlap:
                max_overlap = overlap
                best_speaker = diar_seg["speaker"]
        aligned.append({**trans_seg, "speaker": best_speaker})
    return aligned


# ============================================================================
# OUTPUT FORMAT GENERATORS
# ============================================================================
def generate_srt(segments: List[Dict], include_speakers: bool = False) -> str:
    """Generate SRT subtitle file"""
    if srt:
        subtitles = []
        for i, seg in enumerate(segments, 1):
            text = seg["text"]
            if include_speakers and "speaker" in seg:
                text = f"[{seg['speaker'].replace('SPEAKER_', 'Person ')}] {text}"
            subtitles.append(
                srt.Subtitle(
                    index=i,
                    start=timedelta(seconds=seg["start"]),
                    end=timedelta(seconds=seg["end"]),
                    content=text,
                )
            )
        return srt.compose(subtitles)
    else:
        output = []
        for i, seg in enumerate(segments, 1):
            text = seg["text"]
            if include_speakers and "speaker" in seg:
                text = f"[{seg['speaker'].replace('SPEAKER_', 'Person ')}] {text}"
            output.extend(
                [str(i), f"{format_timestamp_srt(seg['start'])} --> {format_timestamp_srt(seg['end'])}", text, ""]
            )
        return "\n".join(output)


def generate_vtt(segments: List[Dict], include_speakers: bool = False) -> str:
    """Generate WebVTT subtitle file"""
    output = ["WEBVTT", ""]
    for i, seg in enumerate(segments, 1):
        text = seg["text"]
        if include_speakers and "speaker" in seg:
            speaker = seg["speaker"].replace("SPEAKER_", "Person ")
            text = f"<v {speaker}>{text}"
        start = format_timestamp_vtt(seg["start"])
        end = format_timestamp_vtt(seg["end"])
        output.extend([str(i), f"{start} --> {end}", text, ""])
    return "\n".join(output)


def generate_ass(segments: List[Dict], include_speakers: bool = False) -> str:
    """Generate Advanced SubStation Alpha (.ass) subtitle file"""
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "PlayResX: 384\n"
        "PlayResY: 288\n"
        "ScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,"
        "0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    lines = [header]
    for seg in segments:
        start = format_timestamp_ass(seg["start"])
        end = format_timestamp_ass(seg["end"])
        text = seg["text"].replace("\n", "\\N")
        name = ""
        if include_speakers and "speaker" in seg:
            name = seg["speaker"].replace("SPEAKER_", "Person ")
        lines.append(f"Dialogue: 0,{start},{end},Default,{name},0,0,0,,{text}")
    return "\n".join(lines)


def generate_csv_export(segments: List[Dict]) -> str:
    """Generate CSV export with all segment metadata"""
    output = io.StringIO()
    fieldnames = ["index", "start", "end", "duration", "speaker", "text",
                  "original_text", "avg_logprob", "no_speech_prob"]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for i, seg in enumerate(segments, 1):
        writer.writerow({
            "index": i,
            "start": f"{seg['start']:.3f}",
            "end": f"{seg['end']:.3f}",
            "duration": f"{seg['end'] - seg['start']:.3f}",
            "speaker": seg.get("speaker", ""),
            "text": seg["text"],
            "original_text": seg.get("original_text", ""),
            "avg_logprob": f"{seg.get('avg_logprob', 0):.4f}",
            "no_speech_prob": f"{seg.get('no_speech_prob', 0):.4f}",
        })
    return output.getvalue()


def generate_dual_srt(segments: List[Dict]) -> str:
    """Generate bilingual SRT showing original_text + refined text"""
    output = []
    for i, seg in enumerate(segments, 1):
        start = format_timestamp_srt(seg["start"])
        end = format_timestamp_srt(seg["end"])
        original = seg.get("original_text", "")
        translated = seg["text"]
        if original and original != translated:
            body = f"{original}\n{translated}"
        else:
            body = translated
        output.extend([str(i), f"{start} --> {end}", body, ""])
    return "\n".join(output)


# ============================================================================
# SHARED PIPELINE LOGIC
# ============================================================================
def _run_pipeline(
    audio_path: Path,
    whisper_model_name: str,
    device: str,
    compute_type: str,
    source_lang: str,
    task: str,
    beam_size: int,
    enable_diarization: bool,
    hf_token: Optional[str],
    num_speakers: int,
    # V2 quality params
    initial_prompt: str = "",
    temperature: float = 0.0,
    condition_on_previous_text: bool = True,
    no_speech_threshold: float = 0.6,
    log_prob_threshold: float = -1.0,
    compression_ratio_threshold: float = 2.4,
    repetition_penalty: float = 1.0,
    enable_preprocessing: bool = False,
    enable_hallucination_filter: bool = True,
    min_confidence: float = -2.0,
    enable_segment_merge: bool = False,
    min_segment_duration: float = 1.5,
    glossary: Optional[Dict[str, str]] = None,
) -> Optional[Dict]:
    """
    Core transcription pipeline shared by file and URL processing.
    Returns results dict or None on failure.
    """
    start_time = time.time()

    # --- Audio preprocessing ---
    if enable_preprocessing:
        st.write("🎛️ Preprocessing audio (noise reduction)...")
        preprocessed_path = audio_path.parent / ("preprocessed_" + audio_path.name)
        audio_path = preprocess_audio_file(audio_path, preprocessed_path)

    # --- Model loading ---
    st.write("🤖 Loading Whisper model...")
    model = load_whisper_model(whisper_model_name, device, compute_type)
    if not model:
        return None

    # --- Transcription ---
    st.write("🎙️ Transcribing audio...")
    segments, trans_info = transcribe_audio(
        model,
        audio_path,
        language=source_lang,
        task=task,
        beam_size=beam_size,
        initial_prompt=initial_prompt,
        temperature=temperature,
        condition_on_previous_text=condition_on_previous_text,
        no_speech_threshold=no_speech_threshold,
        log_prob_threshold=log_prob_threshold,
        compression_ratio_threshold=compression_ratio_threshold,
        repetition_penalty=repetition_penalty,
    )
    if not segments:
        return None

    # --- Hallucination filtering ---
    filtered_count = 0
    if enable_hallucination_filter:
        st.write("🔍 Filtering hallucinations...")
        segments, filtered = detect_and_filter_hallucinations(
            segments, min_confidence=min_confidence
        )
        filtered_count = len(filtered)
        if filtered_count:
            st.info(f"ℹ️ Removed {filtered_count} hallucinated/low-confidence segment(s)")

    # --- Segment merging ---
    if enable_segment_merge:
        st.write("🔗 Merging short segments...")
        before = len(segments)
        segments = merge_short_segments(segments, min_duration=min_segment_duration)
        st.info(f"ℹ️ Merged {before - len(segments)} segment(s)")

    # --- Glossary ---
    if glossary:
        st.write("📖 Applying glossary...")
        segments = apply_glossary(segments, glossary)

    # --- Speaker diarization ---
    num_speakers_found: Any = "N/A"
    if enable_diarization and hf_token:
        st.write("👥 Identifying speakers...")
        diar_pipeline = load_diarization_model(hf_token)
        if diar_pipeline:
            diar_segments = perform_diarization(diar_pipeline, audio_path, num_speakers)
            if diar_segments:
                segments = align_transcription_with_speakers(segments, diar_segments)
                num_speakers_found = len({s.get("speaker", "UNKNOWN") for s in segments})

    # --- Outputs ---
    st.write("📝 Generating output files...")
    full_text = " ".join(seg["text"] for seg in segments)
    srt_content = generate_srt(segments, include_speakers=enable_diarization)
    vtt_content = generate_vtt(segments, include_speakers=enable_diarization)
    ass_content = generate_ass(segments, include_speakers=enable_diarization)
    csv_content = generate_csv_export(segments)

    processing_time = time.time() - start_time
    duration = segments[-1]["end"] if segments else 0.0
    conf_stats = calculate_confidence_stats(segments)

    detected_lang = getattr(trans_info, "language", source_lang) if trans_info else source_lang
    lang_prob = getattr(trans_info, "language_probability", 0.0) if trans_info else 0.0

    return {
        "num_segments": len(segments),
        "duration": duration,
        "processing_time": processing_time,
        "num_speakers": num_speakers_found,
        "full_text": full_text,
        "srt_content": srt_content,
        "vtt_content": vtt_content,
        "ass_content": ass_content,
        "csv_content": csv_content,
        "segments": segments,
        "filtered_count": filtered_count,
        "confidence_stats": conf_stats,
        "detected_language": detected_lang,
        "language_probability": lang_prob,
    }


def process_file(
    uploaded_file,
    whisper_model_name: str,
    device: str,
    compute_type: str,
    source_lang: str,
    task: str,
    beam_size: int,
    enable_diarization: bool,
    hf_token: Optional[str],
    num_speakers: int,
    **pipeline_kwargs,
):
    """Process an uploaded file through the transcription pipeline"""
    Config.TEMP_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with st.status("🚀 Processing file...", expanded=True) as status:
            st.write("💾 Saving uploaded file...")
            temp_video = Config.TEMP_DIR / uploaded_file.name
            with open(temp_video, "wb") as f:
                f.write(uploaded_file.getbuffer())

            audio_path = temp_video
            if not uploaded_file.name.lower().endswith((".wav", ".mp3", ".flac", ".m4a")):
                st.write("🔊 Extracting audio track...")
                wav_path = Config.TEMP_DIR / "audio.wav"
                audio_path = extract_audio(temp_video, wav_path)
                if not audio_path:
                    status.update(label="❌ Audio extraction failed", state="error")
                    return

            results = _run_pipeline(
                audio_path, whisper_model_name, device, compute_type,
                source_lang, task, beam_size, enable_diarization,
                hf_token, num_speakers, **pipeline_kwargs,
            )

            if results:
                st.session_state.results = results
                status.update(label="✅ Complete!", state="complete")
                st.success(f"✅ Done in {results['processing_time']:.1f}s!")
            else:
                status.update(label="❌ Processing failed", state="error")

    except Exception as e:
        logger.error(f"File processing failed: {e}")
        st.error(f"❌ Failed: {e}")
    finally:
        try:
            for f in Config.TEMP_DIR.glob("*"):
                if f.is_file():
                    f.unlink()
        except Exception:
            pass
        safe_model_cleanup()


def process_url(
    video_url: str,
    whisper_model_name: str,
    device: str,
    compute_type: str,
    source_lang: str,
    task: str,
    beam_size: int,
    enable_diarization: bool,
    hf_token: Optional[str],
    num_speakers: int,
    **pipeline_kwargs,
):
    """Download from URL then process through the transcription pipeline"""
    Config.TEMP_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with st.status("🚀 Processing URL...", expanded=True) as status:
            st.write("⬇️ Downloading media...")
            temp_video = Config.TEMP_DIR / "downloaded_media"
            video_path = download_video_from_url(video_url, temp_video)
            if not video_path:
                status.update(label="❌ Download failed", state="error")
                return

            st.write("🔊 Extracting audio track...")
            audio_path = Config.TEMP_DIR / "audio.wav"
            audio_path = extract_audio(video_path, audio_path)
            if not audio_path:
                status.update(label="❌ Audio extraction failed", state="error")
                return

            results = _run_pipeline(
                audio_path, whisper_model_name, device, compute_type,
                source_lang, task, beam_size, enable_diarization,
                hf_token, num_speakers, **pipeline_kwargs,
            )

            if results:
                st.session_state.results = results
                status.update(label="✅ Complete!", state="complete")
                st.success(f"✅ Done in {results['processing_time']:.1f}s!")
            else:
                status.update(label="❌ Processing failed", state="error")

    except Exception as e:
        logger.error(f"URL processing failed: {e}")
        st.error(f"❌ Failed: {e}")
    finally:
        try:
            for f in Config.TEMP_DIR.glob("*"):
                if f.is_file():
                    f.unlink()
        except Exception:
            pass
        safe_model_cleanup()


# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    st.set_page_config(
        page_title=f"🌆 {Config.APP_NAME} V2",
        page_icon="🌆",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Session state initialisation
    defaults = {
        "theme": "cyberpunk_neon",
        "results": None,
        "session_id": generate_session_id(),
        "chat_messages": [],
        "chat_provider": "claude",
        "glossary_text": "# Add term replacements below (one per line):\n# Example: AI = Artificial Intelligence\n",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    st.markdown(generate_theme_css(st.session_state.theme), unsafe_allow_html=True)

    theme = THEMES[st.session_state.theme]
    stats = get_system_stats()

    # Header
    st.markdown(
        f"""
        <div style='text-align:center;padding:20px 0;'>
            <h1 style='margin-bottom:10px;'>🌆 {Config.APP_NAME.upper()} V2</h1>
            <p style='color:{theme['secondary']};font-size:18px;font-weight:700;letter-spacing:4px;'>
                ⚡ {Config.APP_SUBTITLE.upper()} ⚡
            </p>
            <p style='color:{theme['text']};opacity:0.7;font-size:12px;letter-spacing:2px;'>
                VERSION {Config.APP_VERSION} • SESSION: {st.session_state.session_id}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # System status bar
    gpu_status = "GPU ONLINE" if (torch and torch.cuda.is_available()) else "CPU MODE"
    st.markdown(
        f"""
        <div style='background:rgba(0,0,0,0.3);border:2px solid {theme['primary']};
                    border-radius:10px;padding:12px 20px;margin-bottom:20px;
                    display:flex;justify-content:space-between;align-items:center;'>
            <div style='color:{theme['success']};font-weight:700;'>● SYSTEM ONLINE</div>
            <div style='color:{theme['secondary']};font-size:12px;'>
                MEM: {stats['memory_percent']:.1f}% | CPU: {stats['cpu_percent']:.1f}% | {gpu_status}
            </div>
            <div style='color:{theme['primary']};font-weight:700;'>YEAR 3999</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ========================================================================
    # SIDEBAR
    # ========================================================================
    with st.sidebar:
        st.markdown(
            f"""
            <div style='background:linear-gradient(135deg,{theme['primary']},{theme['secondary']});
                        padding:15px;border-radius:10px;text-align:center;margin-bottom:20px;'>
                <h3 style='color:{theme['background']};margin:0;font-weight:900;'>⚙️ CONTROL PANEL</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Theme
        st.markdown("### 🎨 THEME")
        theme_options = {k: v["name"] for k, v in THEMES.items()}
        selected_theme = st.selectbox(
            "Select Theme",
            options=list(theme_options.keys()),
            format_func=lambda x: theme_options[x],
            index=list(theme_options.keys()).index(st.session_state.theme),
            label_visibility="collapsed",
        )
        if selected_theme != st.session_state.theme:
            st.session_state.theme = selected_theme
            st.rerun()

        st.markdown("---")

        # Model
        st.markdown("### 🤖 NEURAL ENGINE")
        recommended = [k for k, v in Config.WHISPER_MODELS.items() if v.get("recommended")]
        experimental = [k for k, v in Config.WHISPER_MODELS.items() if v.get("experimental")]
        others = [
            k for k in Config.WHISPER_MODELS
            if not Config.WHISPER_MODELS[k].get("recommended")
            and not Config.WHISPER_MODELS[k].get("experimental")
        ]
        model_options = recommended + others + experimental

        whisper_model = st.selectbox(
            "Whisper Model",
            model_options,
            index=(
                model_options.index(Config.DEFAULT_WHISPER_MODEL)
                if Config.DEFAULT_WHISPER_MODEL in model_options
                else 0
            ),
            format_func=lambda x: (
                f"{'⭐ ' if Config.WHISPER_MODELS[x].get('recommended') else ''}"
                f"{'🧪 ' if Config.WHISPER_MODELS[x].get('experimental') else ''}"
                f"{Config.WHISPER_MODELS[x]['name']}"
            ),
        )
        model_info = Config.WHISPER_MODELS[whisper_model]
        st.caption(f"Speed: {model_info['speed']} | {model_info['description']}")

        st.markdown("---")

        # Hardware
        st.markdown("### 💻 HARDWARE")
        device_options = ["cpu"]
        if torch and torch.cuda.is_available():
            device_options.insert(0, "cuda")
        device = st.selectbox("Device", device_options)
        compute_options = ["int8_float16", "float16", "int8"] if device == "cuda" else ["int8", "float32"]
        compute_type = st.selectbox("Compute Type", compute_options)

        if st.button("🗑️ Clear Model Cache"):
            clear_model_cache()
            st.success("✅ Cleared!")

        st.markdown("---")

        # Language & Task
        st.markdown("### 🌐 LANGUAGE")
        source_lang = st.selectbox(
            "Source Language",
            options=list(Config.LANGUAGES.keys()),
            format_func=lambda x: Config.LANGUAGES[x],
        )
        task = st.radio("Task", ["translate", "transcribe"], horizontal=True)

        st.markdown("---")

        # Domain Preset — key for accuracy
        st.markdown("### 🎯 DOMAIN PRESET")
        domain = st.selectbox(
            "Content Domain",
            options=list(Config.DOMAIN_PRESETS.keys()),
            format_func=lambda x: Config.DOMAIN_PRESETS[x]["name"],
            help="Providing domain context significantly improves transcription accuracy",
        )
        st.caption(Config.DOMAIN_PRESETS[domain]["description"])

        # Allow overriding the auto-generated initial prompt
        domain_prompt = Config.DOMAIN_PRESETS[domain]["prompt"]
        custom_prompt = st.text_area(
            "Initial Prompt (editable)",
            value=domain_prompt,
            height=80,
            help="This text primes the model — add character names, technical terms, etc.",
        )

        st.markdown("---")

        # Advanced Whisper Settings
        with st.expander("⚡ ADVANCED SETTINGS"):
            beam_size = st.slider("Beam Size", 1, 10, 5,
                                  help="Higher = more accurate but slower")
            temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.05,
                                    help="0 = greedy (fastest); >0 adds diversity")
            condition_prev = st.checkbox("Condition on Previous Text", value=True,
                                         help="Use prior segments as context")
            no_speech_thresh = st.slider("No-Speech Threshold", 0.1, 1.0, 0.6, 0.05,
                                          help="Segments with higher silence probability are dropped")
            logprob_thresh = st.slider("Log-Prob Threshold", -3.0, 0.0, -1.0, 0.1,
                                        help="Drop segments below this confidence")
            comp_ratio_thresh = st.slider("Compression Ratio Threshold", 1.0, 4.0, 2.4, 0.1,
                                           help="Drop over-compressed (repetitive) segments")
            repetition_penalty = st.slider("Repetition Penalty", 1.0, 2.0, 1.0, 0.05,
                                            help="Penalises repeated phrases")

            st.markdown("**Speaker Diarization**")
            enable_diarization = st.checkbox("🎤 Identify Speakers")
            num_speakers = 2
            if enable_diarization:
                num_speakers = st.number_input("Expected Speakers", 1, 10, 2)

        st.markdown("---")

        # Post-Processing
        with st.expander("🔧 POST-PROCESSING"):
            enable_preprocessing = st.checkbox(
                "🎛️ Noise Reduction",
                help="Reduce background noise before transcription (requires noisereduce)",
            )
            enable_hallucination_filter = st.checkbox(
                "🚫 Filter Hallucinations", value=True,
                help="Remove common Whisper artifacts (music notes, subscribe prompts, etc.)",
            )
            min_confidence = st.slider(
                "Min Confidence (log-prob)", -3.0, 0.0, -2.0, 0.1,
                help="Segments below this threshold are filtered when hallucination filter is on",
            )
            enable_merge = st.checkbox(
                "🔗 Merge Short Segments",
                help="Combine very short segments for better subtitle readability",
            )
            min_seg_dur = 1.5
            if enable_merge:
                min_seg_dur = st.slider("Min Segment Duration (s)", 0.5, 5.0, 1.5, 0.5)

        st.markdown("---")

        # Glossary
        with st.expander("📖 GLOSSARY"):
            st.caption("Format: `original = replacement` (one per line, # for comments)")
            st.session_state.glossary_text = st.text_area(
                "Glossary",
                value=st.session_state.glossary_text,
                height=120,
                label_visibility="collapsed",
            )

        st.markdown("---")

        col1, col2 = st.columns(2)
        col1.metric("MEM", f"{stats['memory_percent']:.0f}%")
        col2.metric("CPU", f"{stats['cpu_percent']:.0f}%")
        st.info(f"v{Config.APP_VERSION}")

    # Parse glossary once
    glossary = parse_glossary_text(st.session_state.glossary_text)

    # Pipeline kwargs passed to both process_file and process_url
    pipeline_kwargs = dict(
        initial_prompt=custom_prompt,
        temperature=temperature,
        condition_on_previous_text=condition_prev,
        no_speech_threshold=no_speech_thresh,
        log_prob_threshold=logprob_thresh,
        compression_ratio_threshold=comp_ratio_thresh,
        repetition_penalty=repetition_penalty,
        enable_preprocessing=enable_preprocessing,
        enable_hallucination_filter=enable_hallucination_filter,
        min_confidence=min_confidence,
        enable_segment_merge=enable_merge,
        min_segment_duration=min_seg_dur,
        glossary=glossary if glossary else None,
    )

    # ========================================================================
    # TABS
    # ========================================================================
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        ["📤 Upload", "🔗 URL", "📊 Results", "✨ Translation Studio",
         "🔧 Fine-Tune", "🤖 AI Chat", "ℹ️ Info"]
    )

    # ------------------------------------------------------------------
    # TAB 1 — File Upload
    # ------------------------------------------------------------------
    with tab1:
        st.markdown("### 📤 Upload Video or Audio File")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["mp4", "avi", "mov", "mkv", "flv", "mp3", "wav", "flac", "m4a", "webm"],
        )
        if uploaded_file:
            file_size_mb = uploaded_file.size / (1024 * 1024)
            c1, c2, c3 = st.columns(3)
            c1.metric("📁 File", uploaded_file.name[:18] + ("..." if len(uploaded_file.name) > 18 else ""))
            c2.metric("📊 Size", f"{file_size_mb:.1f} MB")
            c3.metric("🎯 Type", uploaded_file.type.split("/")[-1].upper())

            if file_size_mb > Config.MAX_FILE_SIZE_MB:
                st.error(f"❌ File too large! Max: {Config.MAX_FILE_SIZE_MB} MB")
            else:
                if st.button("🚀 PROCESS FILE", type="primary", use_container_width=True):
                    process_file(
                        uploaded_file, whisper_model, device, compute_type,
                        source_lang, task, beam_size, enable_diarization,
                        EnvConfig.HF_TOKEN, num_speakers, **pipeline_kwargs,
                    )

    # ------------------------------------------------------------------
    # TAB 2 — URL Download
    # ------------------------------------------------------------------
    with tab2:
        st.markdown("### 🔗 Download & Process from URL")
        video_url = st.text_input("Video URL", placeholder="https://www.youtube.com/watch?v=...")
        if video_url:
            if st.button("🚀 PROCESS URL", type="primary", use_container_width=True):
                process_url(
                    video_url, whisper_model, device, compute_type,
                    source_lang, task, beam_size, enable_diarization,
                    EnvConfig.HF_TOKEN, num_speakers, **pipeline_kwargs,
                )

    # ------------------------------------------------------------------
    # TAB 3 — Results
    # ------------------------------------------------------------------
    with tab3:
        st.markdown("### 📊 Processing Results")
        if st.session_state.results:
            r = st.session_state.results
            conf = r.get("confidence_stats", {})

            # Metrics row
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("📝 Segments", r.get("num_segments", 0))
            c2.metric("⏱️ Duration", f"{r.get('duration', 0):.1f}s")
            c3.metric("⚡ Time", f"{r.get('processing_time', 0):.1f}s")
            c4.metric("🎤 Speakers", r.get("num_speakers", "N/A"))
            c5.metric("🗑️ Filtered", r.get("filtered_count", 0))

            # Language detection info
            det_lang = r.get("detected_language", "")
            lang_prob = r.get("language_probability", 0.0)
            if det_lang:
                st.info(
                    f"🌐 Detected language: **{Config.LANGUAGES.get(det_lang, det_lang)}** "
                    f"({lang_prob:.0%} confidence) | "
                    f"Avg confidence: {conf.get('mean', 0):.3f} | "
                    f"Low-confidence segments: {conf.get('low_confidence_count', 0)}"
                )

            st.markdown("---")
            st.markdown("### 📝 Full Text")
            st.text_area("", value=r.get("full_text", ""), height=250, label_visibility="collapsed")

            st.markdown("---")
            st.markdown("### 💾 Downloads")

            # Format downloads
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            dl_cols = st.columns(7)
            format_data = [
                ("📄 TXT", r.get("full_text", ""), f"transcription_{ts}.txt", "text/plain"),
                ("📺 SRT", r.get("srt_content", ""), f"subtitles_{ts}.srt", "text/plain"),
                ("🌐 VTT", r.get("vtt_content", ""), f"subtitles_{ts}.vtt", "text/plain"),
                ("🎬 ASS", r.get("ass_content", ""), f"subtitles_{ts}.ass", "text/plain"),
                ("📊 CSV", r.get("csv_content", ""), f"segments_{ts}.csv", "text/csv"),
                ("🔁 Dual-SRT", generate_dual_srt(r.get("segments", [])), f"dual_{ts}.srt", "text/plain"),
                ("📋 JSON", json.dumps(r.get("segments", []), indent=2, ensure_ascii=False),
                 f"segments_{ts}.json", "application/json"),
            ]
            for col, (label, data, fname, mime) in zip(dl_cols, format_data):
                if data:
                    col.download_button(label, data=data, file_name=fname,
                                        mime=mime, use_container_width=True)

            st.markdown("---")
            # Confidence heat map
            with st.expander("📈 Segment Confidence Details"):
                for i, seg in enumerate(r.get("segments", []), 1):
                    lp = seg.get("avg_logprob", 0.0)
                    ns = seg.get("no_speech_prob", 0.0)
                    conf_color = "🟢" if lp > -0.3 else ("🟡" if lp > -0.6 else "🔴")
                    speaker = f"[{seg['speaker'].replace('SPEAKER_', 'P')}] " if seg.get("speaker") else ""
                    orig = f"\n  ↳ *Original:* {seg['original_text']}" if seg.get("original_text") else ""
                    st.markdown(
                        f"**#{i}** {conf_color} `{seg['start']:.2f}s → {seg['end']:.2f}s` "
                        f"logprob={lp:.3f} nspeech={ns:.2f} {speaker}\n"
                        f"  {seg['text']}{orig}"
                    )
                    st.markdown("---")
        else:
            st.info("ℹ️ No results yet. Process a file or URL first.")

    # ------------------------------------------------------------------
    # TAB 4 — Translation Studio (NEW in V2)
    # ------------------------------------------------------------------
    with tab4:
        st.markdown("### ✨ Translation Studio")
        st.caption(
            "Use an LLM to post-process and refine your transcription for higher accuracy "
            "and more natural language. Works best after initial Whisper transcription."
        )

        if not st.session_state.results:
            st.info("ℹ️ Run a transcription first (Upload or URL tab), then come back here.")
        else:
            r = st.session_state.results
            segs = r.get("segments", [])

            col1, col2 = st.columns(2)
            with col1:
                refine_provider = st.selectbox(
                    "LLM Provider",
                    options=list(Config.LLM_PROVIDERS.keys()),
                    format_func=lambda x: Config.LLM_PROVIDERS[x]["name"],
                    key="refine_provider",
                )
            with col2:
                pconf = Config.LLM_PROVIDERS[refine_provider]
                refine_model = st.selectbox(
                    "Model",
                    options=pconf["models"],
                    index=pconf["models"].index(pconf["default"]),
                    key="refine_model",
                )

            api_keys = {
                "claude": EnvConfig.ANTHROPIC_API_KEY,
                "deepseek": EnvConfig.DEEPSEEK_API_KEY,
                "grok": EnvConfig.GROK_API_KEY,
                "openai": EnvConfig.OPENAI_API_KEY,
            }
            if not api_keys.get(refine_provider):
                st.warning(
                    f"⚠️ No API key found for {Config.LLM_PROVIDERS[refine_provider]['name']}. "
                    "Set it in your .env file."
                )

            col1, col2 = st.columns(2)
            with col1:
                target_language = st.text_input(
                    "Target Language", value="English",
                    help="Language you want the refined output in"
                )
            with col2:
                refine_domain = st.selectbox(
                    "Domain",
                    options=list(Config.DOMAIN_PRESETS.keys()),
                    format_func=lambda x: Config.DOMAIN_PRESETS[x]["name"],
                    key="refine_domain",
                )

            chunk_size = st.slider(
                "Segments per LLM call", 10, 80, 40,
                help="Smaller = more API calls but lower risk of token limit errors"
            )

            st.markdown("**Current transcription preview** (first 5 segments):")
            for seg in segs[:5]:
                st.markdown(f"- `{seg['start']:.1f}s` {seg['text']}")
            if len(segs) > 5:
                st.caption(f"... and {len(segs) - 5} more segments")

            st.markdown("---")

            if st.button("✨ REFINE WITH LLM", type="primary", use_container_width=True):
                with st.spinner(f"Sending {len(segs)} segments to {Config.LLM_PROVIDERS[refine_provider]['name']}..."):
                    refined_segs = refine_translation_with_llm(
                        segs,
                        provider=refine_provider,
                        model=refine_model,
                        target_language=target_language,
                        domain=refine_domain,
                        chunk_size=chunk_size,
                    )

                # Store refined version back into results
                st.session_state.results["segments"] = refined_segs
                st.session_state.results["full_text"] = " ".join(s["text"] for s in refined_segs)
                st.session_state.results["srt_content"] = generate_srt(refined_segs)
                st.session_state.results["vtt_content"] = generate_vtt(refined_segs)
                st.session_state.results["ass_content"] = generate_ass(refined_segs)
                st.session_state.results["csv_content"] = generate_csv_export(refined_segs)

                st.success("✅ Refinement complete! Check the Results tab for updated outputs.")

                # Side-by-side preview
                st.markdown("### 🔍 Before / After Preview")
                refined_with_orig = [s for s in refined_segs if s.get("original_text")]
                if refined_with_orig:
                    c1, c2 = st.columns(2)
                    c1.markdown("**Before (Whisper)**")
                    c2.markdown("**After (LLM refined)**")
                    for seg in refined_with_orig[:10]:
                        c1.markdown(f"> {seg['original_text']}")
                        c2.markdown(f"> {seg['text']}")
                else:
                    st.info("No changes detected from LLM refinement.")

            # Standalone glossary preview
            st.markdown("---")
            st.markdown("### 📖 Active Glossary")
            if glossary:
                for term, rep in glossary.items():
                    st.markdown(f"- `{term}` → `{rep}`")
            else:
                st.caption("No glossary terms defined. Add them in the sidebar Glossary section.")

    # ------------------------------------------------------------------
    # TAB 5 — Fine-Tuning
    # ------------------------------------------------------------------
    with tab5:
        st.markdown("### 🔧 Model Fine-Tuning Studio")

        if not datasets_module or not transformers_module:
            st.error("❌ Fine-tuning requires: `pip install datasets transformers peft accelerate`")
            st.stop()

        ft_tab1, ft_tab2, ft_tab3 = st.tabs(["📊 Dataset", "⚙️ Training", "📈 Evaluation"])

        with ft_tab1:
            st.markdown("#### Dataset Configuration")
            dataset_source = st.selectbox(
                "Dataset Source",
                ["HuggingFace Hub", "Common Voice", "ReazonSpeech", "Local Files"],
            )

            if dataset_source == "HuggingFace Hub":
                c1, c2 = st.columns(2)
                with c1:
                    dataset_name = st.text_input("Dataset Name",
                                                  value="mozilla-foundation/common_voice_11_0")
                with c2:
                    dataset_subset = st.text_input("Subset/Language", value="ja")

                dataset_split = st.selectbox("Split", ["train", "validation", "test"])

                if st.button("📥 Load Dataset Preview"):
                    with st.spinner("Loading..."):
                        try:
                            from datasets import load_dataset
                            ds = load_dataset(
                                dataset_name, dataset_subset,
                                split=f"{dataset_split}[:10]",
                                trust_remote_code=True,
                            )
                            st.success(f"✅ Loaded {len(ds)} samples")
                            st.dataframe(ds.to_pandas())
                        except Exception as e:
                            st.error(f"❌ Failed: {e}")

            elif dataset_source == "Local Files":
                audio_files = st.file_uploader(
                    "Upload Audio Files", type=["wav", "mp3", "flac"], accept_multiple_files=True
                )
                transcript_file = st.file_uploader("Upload Transcripts (CSV/JSON)", type=["csv", "json"])
                if audio_files and transcript_file:
                    st.success(f"✅ {len(audio_files)} audio files uploaded")

        with ft_tab2:
            st.markdown("#### Training Configuration")
            c1, c2 = st.columns(2)
            with c1:
                base_model = st.selectbox(
                    "Base Model",
                    ["openai/whisper-small", "openai/whisper-medium", "openai/whisper-large-v3"],
                )
                learning_rate = st.number_input("Learning Rate", value=1e-5, format="%.1e", step=1e-6)
                batch_size = st.slider("Batch Size", 1, 32, 8)
            with c2:
                epochs = st.slider("Epochs", 1, 50, 3)
                warmup_steps = st.number_input("Warmup Steps", value=500)
                gradient_accumulation = st.slider("Gradient Accumulation", 1, 16, 1)

            with st.expander("🔬 Advanced"):
                use_lora = st.checkbox("Use LoRA", value=True)
                if use_lora:
                    lc1, lc2 = st.columns(2)
                    lora_r = lc1.slider("LoRA Rank", 8, 64, 16)
                    lora_alpha = lc2.slider("LoRA Alpha", 8, 64, 32)
                fp16 = st.checkbox("FP16 Training", value=True)
                gradient_checkpointing = st.checkbox("Gradient Checkpointing", value=True)

            output_dir = st.text_input("Output Directory", value="./fine-tuned-model")

            if st.button("🚀 Start Training", type="primary", use_container_width=True):
                st.warning("⚠️ Training requires significant GPU compute. Code skeleton shown below.")
                st.code("""
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from peft import LoraConfig, get_peft_model

model = WhisperForConditionalGeneration.from_pretrained(base_model)
processor = WhisperProcessor.from_pretrained(base_model)

if use_lora:
    lora_config = LoraConfig(r=lora_r, lora_alpha=lora_alpha,
                              target_modules=["q_proj", "v_proj"])
    model = get_peft_model(model, lora_config)

trainer = Seq2SeqTrainer(model=model, args=training_args, ...)
trainer.train()
""", language="python")

        with ft_tab3:
            st.markdown("#### Model Evaluation")
            test_audio = st.file_uploader("Test Audio File", type=["wav", "mp3"])
            if test_audio:
                st.audio(test_audio)
                if st.button("🔍 Evaluate"):
                    st.info("Evaluation would compare base vs fine-tuned model.")
            c1, c2, c3 = st.columns(3)
            c1.metric("WER", "N/A", help="Word Error Rate")
            c2.metric("CER", "N/A", help="Character Error Rate")
            c3.metric("BLEU", "N/A", help="BLEU Score")

    # ------------------------------------------------------------------
    # TAB 6 — AI Chat
    # ------------------------------------------------------------------
    with tab6:
        st.markdown("### 🤖 AI Assistant")

        c1, c2 = st.columns([1, 2])
        with c1:
            provider = st.selectbox(
                "Provider",
                options=list(Config.LLM_PROVIDERS.keys()),
                format_func=lambda x: Config.LLM_PROVIDERS[x]["name"],
                key="llm_provider",
            )
        with c2:
            pconf = Config.LLM_PROVIDERS[provider]
            chat_model = st.selectbox(
                "Model",
                options=pconf["models"],
                index=pconf["models"].index(pconf["default"]),
            )

        api_keys = {
            "claude": EnvConfig.ANTHROPIC_API_KEY,
            "deepseek": EnvConfig.DEEPSEEK_API_KEY,
            "grok": EnvConfig.GROK_API_KEY,
            "openai": EnvConfig.OPENAI_API_KEY,
        }
        if not api_keys.get(provider):
            st.warning(f"⚠️ No API key for {Config.LLM_PROVIDERS[provider]['name']}. Add to .env file.")

        # Quick actions
        st.markdown("#### Quick Actions")
        qa1, qa2, qa3, qa4 = st.columns(4)
        with qa1:
            if st.button("📝 Summarize", use_container_width=True):
                if st.session_state.results:
                    st.session_state.chat_messages.append({
                        "role": "user",
                        "content": f"Summarize:\n\n{st.session_state.results.get('full_text', '')}",
                    })
                    st.rerun()
        with qa2:
            if st.button("🌐 Translate", use_container_width=True):
                if st.session_state.results:
                    st.session_state.chat_messages.append({
                        "role": "user",
                        "content": f"Translate to English:\n\n{st.session_state.results.get('full_text', '')}",
                    })
                    st.rerun()
        with qa3:
            if st.button("❓ Q&A", use_container_width=True):
                if st.session_state.results:
                    st.session_state.chat_messages.append({
                        "role": "user",
                        "content": f"Generate 5 questions about:\n\n{st.session_state.results.get('full_text', '')}",
                    })
                    st.rerun()
        with qa4:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_messages = []
                st.rerun()

        st.markdown("---")

        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        if prompt := st.chat_input("Ask about your transcription..."):
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    system_prompt = (
                        "You are a helpful AI assistant specialising in audio transcription and translation."
                    )
                    if st.session_state.results:
                        system_prompt += (
                            f"\n\nTranscription context:\n"
                            f"Duration: {st.session_state.results.get('duration', 0):.1f}s | "
                            f"Segments: {st.session_state.results.get('num_segments', 0)}\n"
                            f"Text: {st.session_state.results.get('full_text', '')[:2000]}..."
                        )
                    response = get_llm_response(
                        provider, chat_model, st.session_state.chat_messages, system_prompt
                    )
                    st.write(response)
                    st.session_state.chat_messages.append({"role": "assistant", "content": response})

        if st.session_state.chat_messages:
            st.markdown("---")
            st.download_button(
                "💾 Export Chat",
                data=json.dumps(st.session_state.chat_messages, indent=2),
                file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )

    # ------------------------------------------------------------------
    # TAB 7 — Info
    # ------------------------------------------------------------------
    with tab7:
        st.markdown("### ℹ️ Application Information")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 🎨 Themes")
            for td in THEMES.values():
                st.markdown(f"- **{td['name']}** — {td['description']}")

        with c2:
            st.markdown("#### 🤖 Models")
            for mk, md in list(Config.WHISPER_MODELS.items())[:10]:
                rec = "⭐" if md.get("recommended") else ""
                exp = "🧪" if md.get("experimental") else ""
                st.markdown(f"- {rec}{exp} **{md['name']}** ({md['size']})")

        st.markdown("---")
        st.markdown("#### 🔑 API Key Status")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"- HuggingFace: {'✅' if EnvConfig.HF_TOKEN else '❌'}")
            st.markdown(f"- Anthropic: {'✅' if EnvConfig.ANTHROPIC_API_KEY else '❌'}")
            st.markdown(f"- DeepSeek: {'✅' if EnvConfig.DEEPSEEK_API_KEY else '❌'}")
        with c2:
            st.markdown(f"- Grok: {'✅' if EnvConfig.GROK_API_KEY else '❌'}")
            st.markdown(f"- OpenAI: {'✅' if EnvConfig.OPENAI_API_KEY else '❌'}")

        st.markdown("---")
        st.markdown("#### ✨ V2 New Features")
        st.markdown("""
- **Domain Presets** — prime Whisper with domain-specific context for better accuracy
- **Editable Initial Prompt** — customise the model's starting context per file
- **Hallucination Filter** — automatically removes common Whisper artifacts
- **Segment Merging** — combines short segments for more readable subtitles
- **Confidence Scoring** — per-segment log-probability and no-speech probability
- **Noise Reduction** — optional audio preprocessing before transcription
- **Glossary Engine** — persist term replacements across all sessions
- **LLM Translation Studio** — post-process with Claude/GPT/DeepSeek/Grok in batches
- **Multi-Format Export** — SRT, VTT, ASS, CSV, Dual-SRT, JSON, TXT
- **Repetition Penalty** — reduces repetitive hallucinations in the model output
- **Temperature Control** — tune between deterministic and diverse outputs
        """)

        st.markdown("---")
        st.markdown("#### 📦 Dependency Status")
        cols = st.columns(4)
        for i, (dep, status) in enumerate(IMPORTS_STATUS.items()):
            col = cols[i % 4]
            if status == "OK":
                col.success(f"✅ {dep}")
            else:
                col.error(f"❌ {dep}")


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    Config.TEMP_DIR.mkdir(parents=True, exist_ok=True)
    try:
        main()
    except Exception as e:
        st.error(f"💥 Fatal error: {e}")
        logger.exception("Application crashed")
