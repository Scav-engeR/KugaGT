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
║   Version: 4.0.0 Ultimate All-in-One Edition                                  ║
║   Features: Transcription • Fine-Tuning • AI Chat • 20+ Models • 8 Themes     ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import tempfile
import os
import logging
from pathlib import Path
import gc
from datetime import datetime, timedelta
import time
from typing import List, Dict, Optional, Tuple, Any
import json
import random
import signal
import sys

# Load environment variables
try:
    from dotenv import load_dotenv

    load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    print(
        "Warning: python-dotenv not installed. Install with: pip install python-dotenv"
    )

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

# LLM imports
anthropic_module = safe_import("anthropic")
openai_module = safe_import("openai")

# Fine-tuning imports - with error handling for packaging issues
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
# SEGMENTATION FAULT HANDLER
# ============================================================================
# def segfault_handler(signum, frame):
#    """Handle segmentation faults gracefully"""
#    logger.error("Segmentation fault detected! This usually indicates a model compatibility issue.")
#    sys.exit(1)

# Register signal handler only if in main thread (Streamlit runs in separate thread)
# try:
#    import threading
#    if threading.current_thread() is threading.main_thread():
#        if hasattr(signal, 'SIGSEGV'):
#            signal.signal(signal.SIGSEGV, segfault_handler)
# except Exception as e:
#    # Signal handling not available in this context - continue without it
#    pass


# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    """Application configuration"""

    APP_NAME = "CyberTranscribe 3999"
    APP_VERSION = "4.0.0"
    APP_SUBTITLE = "Neural Transcription Engine"

    # Comprehensive model list with descriptions
    WHISPER_MODELS = {
        # ===== RECOMMENDED MODELS (CTranslate2 Native) =====
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
        # Standard OpenAI Models
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
        # Systran Optimized Models
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
        # Other models
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
        # ===== NEW JAPANESE SPECIALIZED MODELS =====
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
        "bob80333/speechbrain_ja2en_st_63M_yt600h": {
            "name": "SpeechBrain JA→EN 63M",
            "description": "Japanese to English speech translation",
            "size": "~250MB",
            "speed": "8x faster",
            "recommended": False,
            "language": "ja→en",
            "ct2_native": False,
            "experimental": True,
            "note": "SpeechBrain model - different architecture",
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
        "developerkyimage/whisper-large-v3-ja-finetune": {
            "name": "Whisper Large V3 JA Finetune",
            "description": "Fine-tuned Large V3 for Japanese",
            "size": "~3GB",
            "speed": "1x baseline",
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
        "Nikolajvestergaard/Japanese_Fine_Tuned_Whisper_Model": {
            "name": "Japanese Fine-Tuned Whisper",
            "description": "Community fine-tuned for Japanese",
            "size": "~1.5GB",
            "speed": "3x faster",
            "recommended": False,
            "language": "ja",
            "ct2_native": False,
            "experimental": True,
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
    }

    DEFAULT_WHISPER_MODEL = "medium"
    DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"

    # Supported languages
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

    # LLM Providers
    LLM_PROVIDERS = {
        "claude": {
            "name": "Claude (Anthropic)",
            "models": [
                "claude-sonnet-4-20250514",
                "claude-3-5-sonnet-20241022",
                "claude-3-haiku-20240307",
            ],
            "default": "claude-sonnet-4-20250514",
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

    # Processing settings
    MAX_FILE_SIZE_MB = 2500
    AUDIO_SAMPLE_RATE = 16000
    AUDIO_CHANNELS = 1

    TEMP_DIR = Path(tempfile.gettempdir()) / "cybertranscribe_3999"
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
    
    * {{
        font-family: 'Rajdhani', 'Orbitron', 'Share Tech Mono', sans-serif !important;
    }}
    
    [data-testid="stAppViewContainer"] {{
        background: linear-gradient(135deg, var(--background) 0%, var(--surface) 50%, var(--background) 100%);
    }}
    
    @keyframes gradientShift {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}
    
    [data-testid="stAppViewContainer"]::before {{
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: repeating-linear-gradient(0deg, rgba(255, 255, 255, 0.02) 0px, transparent 1px, transparent 2px);
        pointer-events: none;
        z-index: 1000;
    }}
    
    [data-testid="stHeader"] {{
        background: transparent !important;
    }}
    
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
    
    @keyframes neonPulse {{
        from {{ text-shadow: 0 0 10px var(--primary), 0 0 20px var(--primary), 0 0 40px var(--primary); }}
        to {{ text-shadow: 0 0 20px var(--primary), 0 0 40px var(--primary), 0 0 80px var(--primary); }}
    }}
    
    h2, h3 {{
        color: var(--secondary) !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 3px;
        text-shadow: 0 0 20px var(--secondary);
    }}
    
    p, span, div {{
        color: var(--text);
    }}
    
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
        background: rgba(0, 0, 0, 0.6) !important;
        border: 2px solid var(--secondary) !important;
        border-radius: 8px !important;
        color: var(--text) !important;
        font-family: 'Share Tech Mono', monospace !important;
    }}
    
    [data-testid="stFileUploader"] {{
        background: rgba(255, 255, 255, 0.05) !important;
        border: 3px dashed var(--primary) !important;
        border-radius: 15px !important;
        padding: 30px !important;
    }}
    
    .stSelectbox>div>div {{
        background: rgba(0, 0, 0, 0.6) !important;
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
        background: rgba(255, 255, 255, 0.05) !important;
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
    
    /* Chat styling */
    .stChatMessage {{
        background: rgba(0, 0, 0, 0.3) !important;
        border: 1px solid var(--primary) !important;
        border-radius: 10px !important;
    }}
    
    @media (prefers-reduced-motion: reduce) {{
        *, *::before, *::after {{
            animation-duration: 0.01ms !important;
        }}
    }}
    </style>
    """


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def get_system_stats() -> Dict:
    """Get current system statistics"""
    stats = {
        "memory_percent": 0,
        "cpu_percent": 0,
        "memory_used_gb": 0,
        "memory_total_gb": 0,
    }

    if psutil:
        memory = psutil.virtual_memory()
        stats["memory_percent"] = memory.percent
        stats["memory_used_gb"] = memory.used / (1024**3)
        stats["memory_total_gb"] = memory.total / (1024**3)
        stats["cpu_percent"] = psutil.cpu_percent(interval=0.1)

    return stats


def format_timestamp_srt(seconds: float) -> str:
    """Format seconds to SRT timestamp"""
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def generate_session_id() -> str:
    """Generate unique session identifier"""
    return f"SESSION-{datetime.now().strftime('%Y%m%d%H%M%S')}-{random.randint(1000, 9999)}"


def clear_gpu_memory():
    """Clear GPU memory"""
    if torch:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    gc.collect()


def safe_model_cleanup():
    """Safely cleanup model resources"""
    clear_gpu_memory()
    gc.collect()
    time.sleep(0.5)


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
            logger.info(f"Model {model_name} is not CT2 native")
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
            st.error(f"❌ GPU error. Try switching to CPU.")
        elif "memory" in error_msg.lower():
            st.error(f"❌ Out of memory. Try a smaller model.")
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
                system=system_prompt
                or "You are a helpful AI assistant specializing in audio transcription and translation.",
                messages=messages,
            )
            return response.content[0].text

        elif provider == "deepseek":
            if not openai_module:
                return "❌ OpenAI library not installed"

            from openai import OpenAI

            client = OpenAI(
                api_key=EnvConfig.DEEPSEEK_API_KEY, base_url="https://api.deepseek.com"
            )

            formatted_messages = messages.copy()
            if system_prompt:
                formatted_messages.insert(
                    0, {"role": "system", "content": system_prompt}
                )

            response = client.chat.completions.create(
                model=model, messages=formatted_messages
            )
            return response.choices[0].message.content

        elif provider == "grok":
            if not openai_module:
                return "❌ OpenAI library not installed"

            from openai import OpenAI

            client = OpenAI(
                api_key=EnvConfig.GROK_API_KEY, base_url="https://api.x.ai/v1"
            )

            formatted_messages = messages.copy()
            if system_prompt:
                formatted_messages.insert(
                    0, {"role": "system", "content": system_prompt}
                )

            response = client.chat.completions.create(
                model=model, messages=formatted_messages
            )
            return response.choices[0].message.content

        elif provider == "openai":
            if not openai_module:
                return "❌ OpenAI library not installed"

            from openai import OpenAI

            client = OpenAI(api_key=EnvConfig.OPENAI_API_KEY)

            formatted_messages = messages.copy()
            if system_prompt:
                formatted_messages.insert(
                    0, {"role": "system", "content": system_prompt}
                )

            response = client.chat.completions.create(
                model=model, messages=formatted_messages
            )
            return response.choices[0].message.content

        else:
            return f"❌ Unknown provider: {provider}"

    except Exception as e:
        logger.error(f"LLM error: {e}")
        return f"❌ Error: {str(e)}"


# ============================================================================
# VIDEO/AUDIO PROCESSING
# ============================================================================
def download_video_from_url(url: str, output_path: Path) -> Optional[Path]:
    """Download video using yt-dlp"""
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
    """Extract audio from video file"""
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


# ============================================================================
# TRANSCRIPTION
# ============================================================================
def transcribe_audio(
    model,
    audio_path: Path,
    language: str = "ja",
    task: str = "translate",
    beam_size: int = 5,
) -> Optional[List[Dict]]:
    """Transcribe audio with progress tracking"""
    try:
        progress_bar = st.progress(0, text="Initializing...")

        lang_param = None if language == "auto" else language

        segments, info = model.transcribe(
            str(audio_path),
            language=lang_param,
            task=task,
            beam_size=beam_size,
            vad_filter=True,
            word_timestamps=True,
        )

        progress_bar.progress(20, text=f"Processing (detected: {info.language})...")

        segment_list = []
        segments_temp = list(segments)
        total = len(segments_temp)

        for idx, segment in enumerate(segments_temp):
            segment_list.append(
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                }
            )

            progress = 20 + int((idx / max(total, 1)) * 75)
            progress_bar.progress(progress, text=f"Transcribing {idx+1}/{total}...")

        progress_bar.progress(100, text="✅ Complete!")
        time.sleep(0.5)
        progress_bar.empty()

        return segment_list

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        st.error(f"Transcription failed: {e}")
        return None
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

        diar_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            diar_segments.append(
                {"start": turn.start, "end": turn.end, "speaker": speaker}
            )

        return diar_segments

    except Exception as e:
        logger.error(f"Diarization failed: {e}")
        return None


def align_transcription_with_speakers(
    transcription: List[Dict], diarization: List[Dict]
) -> List[Dict]:
    """Align transcription with speaker labels"""
    aligned = []

    for trans_seg in transcription:
        max_overlap = 0
        best_speaker = "UNKNOWN"

        for diar_seg in diarization:
            overlap_start = max(trans_seg["start"], diar_seg["start"])
            overlap_end = min(trans_seg["end"], diar_seg["end"])
            overlap = max(0, overlap_end - overlap_start)

            if overlap > max_overlap:
                max_overlap = overlap
                best_speaker = diar_seg["speaker"]

        aligned.append(
            {
                "start": trans_seg["start"],
                "end": trans_seg["end"],
                "speaker": best_speaker,
                "text": trans_seg["text"],
            }
        )

    return aligned


# ============================================================================
# SRT GENERATION
# ============================================================================
def generate_srt(segments: List[Dict], include_speakers: bool = False) -> str:
    """Generate SRT subtitle file"""
    if srt:
        subtitles = []
        for i, seg in enumerate(segments, start=1):
            text = seg["text"]
            if include_speakers and "speaker" in seg:
                speaker_label = seg["speaker"].replace("SPEAKER_", "Person ")
                text = f"[{speaker_label}] {text}"

            subtitle = srt.Subtitle(
                index=i,
                start=timedelta(seconds=seg["start"]),
                end=timedelta(seconds=seg["end"]),
                content=text,
            )
            subtitles.append(subtitle)

        return srt.compose(subtitles)
    else:
        output = []
        for i, seg in enumerate(segments, start=1):
            text = seg["text"]
            if include_speakers and "speaker" in seg:
                text = f"[{seg['speaker'].replace('SPEAKER_', 'Person ')}] {text}"

            start = format_timestamp_srt(seg["start"])
            end = format_timestamp_srt(seg["end"])
            output.extend([str(i), f"{start} --> {end}", text, ""])

        return "\n".join(output)


# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    """Main application entry point"""

    # Page configuration
    st.set_page_config(
        page_title=f"🌆 {Config.APP_NAME}",
        page_icon="🌆",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize session state
    if "theme" not in st.session_state:
        st.session_state.theme = "cyberpunk_neon"
    if "results" not in st.session_state:
        st.session_state.results = None
    if "session_id" not in st.session_state:
        st.session_state.session_id = generate_session_id()
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "chat_provider" not in st.session_state:
        st.session_state.chat_provider = "claude"

    # Apply theme
    st.markdown(generate_theme_css(st.session_state.theme), unsafe_allow_html=True)

    # Header
    theme = THEMES[st.session_state.theme]

    st.markdown(
        f"""
        <div style='text-align: center; padding: 20px 0;'>
            <h1 style='margin-bottom: 10px;'>🌆 {Config.APP_NAME.upper()}</h1>
            <p style='color: {theme['secondary']}; font-size: 18px; font-weight: 700; letter-spacing: 4px;'>
                ⚡ {Config.APP_SUBTITLE.upper()} ⚡
            </p>
            <p style='color: {theme['text']}; opacity: 0.7; font-size: 12px; letter-spacing: 2px;'>
                VERSION {Config.APP_VERSION} • SESSION: {st.session_state.session_id}
            </p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # System status bar
    stats = get_system_stats()
    st.markdown(
        f"""
        <div style='background: rgba(0,0,0,0.3); border: 2px solid {theme['primary']}; 
                    border-radius: 10px; padding: 12px 20px; margin-bottom: 20px;
                    display: flex; justify-content: space-between; align-items: center;'>
            <div style='color: {theme['success']}; font-weight: 700;'>● SYSTEM ONLINE</div>
            <div style='color: {theme['secondary']}; font-size: 12px;'>
                MEM: {stats['memory_percent']:.1f}% | CPU: {stats['cpu_percent']:.1f}%
            </div>
            <div style='color: {theme['primary']}; font-weight: 700;'>YEAR 3999</div>
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
            <div style='background: linear-gradient(135deg, {theme['primary']}, {theme['secondary']}); 
                        padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 20px;'>
                <h3 style='color: {theme['background']}; margin: 0; font-weight: 900;'>⚙️ CONTROL PANEL</h3>
            </div>
        """,
            unsafe_allow_html=True,
        )

        # Theme selection
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

        # Model selection
        st.markdown("### 🤖 NEURAL ENGINE")

        recommended_models = [
            k for k, v in Config.WHISPER_MODELS.items() if v.get("recommended")
        ]
        experimental_models = [
            k for k, v in Config.WHISPER_MODELS.items() if v.get("experimental")
        ]
        other_models = [
            k
            for k, v in Config.WHISPER_MODELS.items()
            if not v.get("recommended") and not v.get("experimental")
        ]

        model_options = recommended_models + other_models + experimental_models

        whisper_model = st.selectbox(
            "Whisper Model",
            model_options,
            index=(
                model_options.index(Config.DEFAULT_WHISPER_MODEL)
                if Config.DEFAULT_WHISPER_MODEL in model_options
                else 0
            ),
            format_func=lambda x: f"{'⭐ ' if Config.WHISPER_MODELS[x].get('recommended') else ''}{'🧪 ' if Config.WHISPER_MODELS[x].get('experimental') else ''}{Config.WHISPER_MODELS[x]['name']}",
        )

        model_info = Config.WHISPER_MODELS[whisper_model]
        st.caption(f"Speed: {model_info['speed']} | {model_info['description']}")

        st.markdown("---")

        # Device settings
        st.markdown("### 💻 HARDWARE")

        device_options = ["cpu"]
        if torch and torch.cuda.is_available():
            device_options.insert(0, "cuda")

        device = st.selectbox("Device", device_options)

        if device == "cuda":
            compute_options = ["int8_float16", "float16", "int8"]
        else:
            compute_options = ["int8", "float32"]

        compute_type = st.selectbox("Compute Type", compute_options)

        if st.button("🗑️ Clear Cache"):
            clear_model_cache()
            st.success("✅ Cleared!")

        st.markdown("---")

        # Language and task
        st.markdown("### 🌐 LANGUAGE")

        source_lang = st.selectbox(
            "Source Language",
            options=list(Config.LANGUAGES.keys()),
            format_func=lambda x: Config.LANGUAGES[x],
        )

        task = st.radio("Task", ["translate", "transcribe"], horizontal=True)

        st.markdown("---")

        # Advanced settings
        with st.expander("⚡ ADVANCED"):
            beam_size = st.slider("Beam Size", 1, 10, 5)

            enable_diarization = st.checkbox("🎤 Speaker Diarization")

            num_speakers = 2
            if enable_diarization:
                num_speakers = st.number_input("Expected Speakers", 1, 10, 2)

        st.markdown("---")

        # Stats
        col1, col2 = st.columns(2)
        col1.metric("MEM", f"{stats['memory_percent']:.0f}%")
        col2.metric("CPU", f"{stats['cpu_percent']:.0f}%")

        st.info(f"v{Config.APP_VERSION}")

    # ========================================================================
    # MAIN CONTENT TABS
    # ========================================================================
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["📤 Upload", "🔗 URL", "📊 Results", "🔧 Fine-Tune", "🤖 AI Chat", "ℹ️ Info"]
    )

    # ------------ TAB 1: File Upload ------------
    with tab1:
        st.markdown("### 📤 Upload Video or Audio File")

        uploaded_file = st.file_uploader(
            "Choose a file",
            type=[
                "mp4",
                "avi",
                "mov",
                "mkv",
                "flv",
                "mp3",
                "wav",
                "flac",
                "m4a",
                "webm",
            ],
        )

        if uploaded_file:
            file_size_mb = uploaded_file.size / (1024 * 1024)

            col1, col2, col3 = st.columns(3)
            col1.metric("📁 File", uploaded_file.name[:15] + "...")
            col2.metric("📊 Size", f"{file_size_mb:.1f} MB")
            col3.metric("🎯 Type", uploaded_file.type.split("/")[-1].upper())

            if file_size_mb > Config.MAX_FILE_SIZE_MB:
                st.error(f"❌ File too large! Max: {Config.MAX_FILE_SIZE_MB}MB")
            else:
                if st.button(
                    "🚀 PROCESS FILE", type="primary", use_container_width=True
                ):
                    process_file(
                        uploaded_file,
                        whisper_model,
                        device,
                        compute_type,
                        source_lang,
                        task,
                        beam_size,
                        enable_diarization,
                        EnvConfig.HF_TOKEN,
                        num_speakers,
                    )

    # ------------ TAB 2: URL Download ------------
    with tab2:
        st.markdown("### 🔗 Download from URL")

        video_url = st.text_input(
            "Video URL", placeholder="https://www.youtube.com/watch?v=..."
        )

        if video_url:
            if st.button("🚀 PROCESS URL", type="primary", use_container_width=True):
                process_url(
                    video_url,
                    whisper_model,
                    device,
                    compute_type,
                    source_lang,
                    task,
                    beam_size,
                    enable_diarization,
                    EnvConfig.HF_TOKEN,
                    num_speakers,
                )

    # ------------ TAB 3: Results ------------
    with tab3:
        st.markdown("### 📊 Processing Results")

        if st.session_state.results:
            results = st.session_state.results

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("📝 Segments", results.get("num_segments", 0))
            col2.metric("⏱️ Duration", f"{results.get('duration', 0):.1f}s")
            col3.metric("⚡ Time", f"{results.get('processing_time', 0):.1f}s")
            col4.metric("🎤 Speakers", results.get("num_speakers", "N/A"))

            st.markdown("---")

            st.markdown("### 📝 Transcription")
            st.text_area(
                "Output",
                value=results.get("full_text", ""),
                height=300,
                label_visibility="collapsed",
            )

            st.markdown("---")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.download_button(
                    "📄 TXT",
                    data=results.get("full_text", ""),
                    file_name=f"transcription_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    use_container_width=True,
                )

            with col2:
                if results.get("srt_content"):
                    st.download_button(
                        "📺 SRT",
                        data=results["srt_content"],
                        file_name=f"subtitles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.srt",
                        use_container_width=True,
                    )

            with col3:
                st.download_button(
                    "📊 JSON",
                    data=json.dumps(
                        results.get("segments", []), indent=2, ensure_ascii=False
                    ),
                    file_name=f"transcription_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    use_container_width=True,
                )

            with st.expander("📋 Detailed Segments"):
                for i, seg in enumerate(results.get("segments", []), 1):
                    speaker = seg.get("speaker", "")
                    speaker_label = (
                        f"[{speaker.replace('SPEAKER_', 'P')}]" if speaker else ""
                    )
                    st.markdown(
                        f"**#{i}** `{seg['start']:.2f}s → {seg['end']:.2f}s` {speaker_label}\n{seg['text']}"
                    )
                    st.markdown("---")
        else:
            st.info("ℹ️ No results yet. Process a file to see results.")

    # ------------ TAB 4: Fine-Tuning ------------
    with tab4:
        st.markdown("### 🔧 Model Fine-Tuning Studio")

        if not datasets_module or not transformers_module:
            st.error(
                "❌ Fine-tuning requires: `pip install datasets transformers peft accelerate`"
            )
            st.stop()

        ft_tab1, ft_tab2, ft_tab3 = st.tabs(
            ["📊 Dataset", "⚙️ Training", "📈 Evaluation"]
        )

        with ft_tab1:
            st.markdown("#### Dataset Configuration")

            dataset_source = st.selectbox(
                "Dataset Source",
                ["HuggingFace Hub", "Common Voice", "ReazonSpeech", "Local Files"],
            )

            if dataset_source == "HuggingFace Hub":
                col1, col2 = st.columns(2)
                with col1:
                    dataset_name = st.text_input(
                        "Dataset Name",
                        value="mozilla-foundation/common_voice_11_0",
                        placeholder="username/dataset-name",
                    )
                with col2:
                    dataset_subset = st.text_input("Subset/Language", value="ja")

                dataset_split = st.selectbox("Split", ["train", "validation", "test"])

                if st.button("📥 Load Dataset Preview"):
                    with st.spinner("Loading dataset..."):
                        try:
                            from datasets import load_dataset

                            ds = load_dataset(
                                dataset_name,
                                dataset_subset,
                                split=f"{dataset_split}[:10]",
                                trust_remote_code=True,
                            )
                            st.success(f"✅ Loaded {len(ds)} samples")
                            st.dataframe(ds.to_pandas())
                        except Exception as e:
                            st.error(f"❌ Failed to load: {e}")

            elif dataset_source == "Local Files":
                audio_files = st.file_uploader(
                    "Upload Audio Files",
                    type=["wav", "mp3", "flac"],
                    accept_multiple_files=True,
                )

                transcript_file = st.file_uploader(
                    "Upload Transcripts (CSV/JSON)", type=["csv", "json"]
                )

                if audio_files and transcript_file:
                    st.success(f"✅ {len(audio_files)} audio files uploaded")

        with ft_tab2:
            st.markdown("#### Training Configuration")

            col1, col2 = st.columns(2)

            with col1:
                base_model = st.selectbox(
                    "Base Model",
                    [
                        "openai/whisper-small",
                        "openai/whisper-medium",
                        "openai/whisper-large-v3",
                    ],
                )

                learning_rate = st.number_input(
                    "Learning Rate", value=1e-5, format="%.1e", step=1e-6
                )

                batch_size = st.slider("Batch Size", 1, 32, 8)

            with col2:
                epochs = st.slider("Epochs", 1, 50, 3)
                warmup_steps = st.number_input("Warmup Steps", value=500)
                gradient_accumulation = st.slider("Gradient Accumulation", 1, 16, 1)

            with st.expander("🔬 Advanced Options"):
                use_lora = st.checkbox("Use LoRA", value=True)
                if use_lora:
                    col1, col2 = st.columns(2)
                    with col1:
                        lora_r = st.slider("LoRA Rank", 8, 64, 16)
                    with col2:
                        lora_alpha = st.slider("LoRA Alpha", 8, 64, 32)

                fp16 = st.checkbox("FP16 Training", value=True)
                gradient_checkpointing = st.checkbox(
                    "Gradient Checkpointing", value=True
                )

            output_dir = st.text_input("Output Directory", value="./fine-tuned-model")

            if st.button("🚀 Start Training", type="primary", use_container_width=True):
                st.warning(
                    "⚠️ Training would start here. Full implementation requires significant compute resources."
                )

                # Training code skeleton
                st.code(
                    """
# Training would use:
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from peft import LoraConfig, get_peft_model

# Load model
model = WhisperForConditionalGeneration.from_pretrained(base_model)
processor = WhisperProcessor.from_pretrained(base_model)

# Apply LoRA
if use_lora:
    lora_config = LoraConfig(r=lora_r, lora_alpha=lora_alpha, ...)
    model = get_peft_model(model, lora_config)

# Train
trainer = Seq2SeqTrainer(model=model, args=training_args, ...)
trainer.train()
                """,
                    language="python",
                )

        with ft_tab3:
            st.markdown("#### Model Evaluation")

            test_audio = st.file_uploader("Test Audio File", type=["wav", "mp3"])

            if test_audio:
                st.audio(test_audio)

                if st.button("🔍 Evaluate"):
                    st.info(
                        "Evaluation would compare base model vs fine-tuned model here."
                    )

            st.markdown("#### Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("WER", "N/A", help="Word Error Rate")
            col2.metric("CER", "N/A", help="Character Error Rate")
            col3.metric("BLEU", "N/A", help="BLEU Score")

    # ------------ TAB 5: AI Chat ------------
    with tab5:
        st.markdown("### 🤖 AI Assistant")

        # Provider selection
        col1, col2 = st.columns([1, 2])

        with col1:
            provider = st.selectbox(
                "Provider",
                options=list(Config.LLM_PROVIDERS.keys()),
                format_func=lambda x: Config.LLM_PROVIDERS[x]["name"],
                key="llm_provider",
            )

        with col2:
            provider_config = Config.LLM_PROVIDERS[provider]
            model = st.selectbox(
                "Model",
                options=provider_config["models"],
                index=provider_config["models"].index(provider_config["default"]),
            )

        # Check API key
        api_keys = {
            "claude": EnvConfig.ANTHROPIC_API_KEY,
            "deepseek": EnvConfig.DEEPSEEK_API_KEY,
            "grok": EnvConfig.GROK_API_KEY,
            "openai": EnvConfig.OPENAI_API_KEY,
        }

        if not api_keys.get(provider):
            st.warning(
                f"⚠️ No API key found for {Config.LLM_PROVIDERS[provider]['name']}. Add it to your .env file."
            )

        # Quick actions
        st.markdown("#### Quick Actions")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button("📝 Summarize", use_container_width=True):
                if st.session_state.results:
                    st.session_state.chat_messages.append(
                        {
                            "role": "user",
                            "content": f"Please summarize this transcription:\n\n{st.session_state.results.get('full_text', '')}",
                        }
                    )
                    st.rerun()

        with col2:
            if st.button("🌐 Translate", use_container_width=True):
                if st.session_state.results:
                    st.session_state.chat_messages.append(
                        {
                            "role": "user",
                            "content": f"Please translate this to English:\n\n{st.session_state.results.get('full_text', '')}",
                        }
                    )
                    st.rerun()

        with col3:
            if st.button("❓ Q&A", use_container_width=True):
                if st.session_state.results:
                    st.session_state.chat_messages.append(
                        {
                            "role": "user",
                            "content": f"Generate 5 questions about this content:\n\n{st.session_state.results.get('full_text', '')}",
                        }
                    )
                    st.rerun()

        with col4:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_messages = []
                st.rerun()

        st.markdown("---")

        # Chat history
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        # Chat input
        if prompt := st.chat_input("Ask about your transcription..."):
            st.session_state.chat_messages.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.write(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Build context
                    system_prompt = "You are a helpful AI assistant specializing in audio transcription and translation."

                    if st.session_state.results:
                        context = f"""
Current transcription context:
- Duration: {st.session_state.results.get('duration', 0):.1f}s
- Segments: {st.session_state.results.get('num_segments', 0)}
- Text: {st.session_state.results.get('full_text', '')[:2000]}...
"""
                        system_prompt += f"\n\n{context}"

                    response = get_llm_response(
                        provider, model, st.session_state.chat_messages, system_prompt
                    )

                    st.write(response)

                    st.session_state.chat_messages.append(
                        {"role": "assistant", "content": response}
                    )

        # Export chat
        if st.session_state.chat_messages:
            st.markdown("---")
            st.download_button(
                "💾 Export Chat",
                data=json.dumps(st.session_state.chat_messages, indent=2),
                file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )

    # ------------ TAB 6: Info ------------
    with tab6:
        st.markdown("### ℹ️ Application Information")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🎨 Themes")
            for key, theme_data in THEMES.items():
                st.markdown(f"- **{theme_data['name']}**")

        with col2:
            st.markdown("#### 🤖 Models")
            for model_key, model_data in list(Config.WHISPER_MODELS.items())[:8]:
                rec = "⭐" if model_data.get("recommended") else ""
                exp = "🧪" if model_data.get("experimental") else ""
                st.markdown(f"- {rec}{exp} **{model_data['name']}**")

        st.markdown("---")

        # API Key Status
        st.markdown("#### 🔑 API Key Status")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"- HuggingFace: {'✅' if EnvConfig.HF_TOKEN else '❌'}")
            st.markdown(f"- Anthropic: {'✅' if EnvConfig.ANTHROPIC_API_KEY else '❌'}")
            st.markdown(f"- DeepSeek: {'✅' if EnvConfig.DEEPSEEK_API_KEY else '❌'}")

        with col2:
            st.markdown(f"- Grok: {'✅' if EnvConfig.GROK_API_KEY else '❌'}")
            st.markdown(f"- OpenAI: {'✅' if EnvConfig.OPENAI_API_KEY else '❌'}")

        st.markdown("---")

        # Dependencies
        st.markdown("#### 📦 Dependencies")
        cols = st.columns(4)
        for i, (dep, status) in enumerate(IMPORTS_STATUS.items()):
            col = cols[i % 4]
            if status == "OK":
                col.success(f"✅ {dep}")
            else:
                col.error(f"❌ {dep}")


# ============================================================================
# PROCESSING FUNCTIONS
# ============================================================================
def process_file(
    uploaded_file,
    whisper_model: str,
    device: str,
    compute_type: str,
    source_lang: str,
    task: str,
    beam_size: int,
    enable_diarization: bool,
    hf_token: Optional[str],
    num_speakers: int,
):
    """Process uploaded file"""

    Config.TEMP_DIR.mkdir(parents=True, exist_ok=True)

    try:
        with st.status("🚀 Processing...", expanded=True) as status:
            start_time = time.time()

            st.write("💾 Saving file...")
            temp_video = Config.TEMP_DIR / uploaded_file.name
            with open(temp_video, "wb") as f:
                f.write(uploaded_file.getbuffer())

            audio_path = temp_video
            audio_extensions = (".wav", ".mp3", ".flac", ".m4a")

            if not uploaded_file.name.lower().endswith(audio_extensions):
                st.write("🔊 Extracting audio...")
                audio_path = Config.TEMP_DIR / "audio.wav"
                audio_path = extract_audio(temp_video, audio_path)

                if not audio_path:
                    status.update(label="❌ Failed", state="error")
                    return

            st.write(f"🤖 Loading model...")
            model = load_whisper_model(whisper_model, device, compute_type)

            if not model:
                status.update(label="❌ Failed", state="error")
                return

            st.write("🎙️ Transcribing...")
            segments = transcribe_audio(model, audio_path, source_lang, task, beam_size)

            if not segments:
                status.update(label="❌ Failed", state="error")
                return

            num_speakers_found = "N/A"
            if enable_diarization and hf_token:
                st.write("👥 Identifying speakers...")
                diar_pipeline = load_diarization_model(hf_token)

                if diar_pipeline:
                    diar_segments = perform_diarization(
                        diar_pipeline, audio_path, num_speakers
                    )

                    if diar_segments:
                        segments = align_transcription_with_speakers(
                            segments, diar_segments
                        )
                        num_speakers_found = len(
                            set([s.get("speaker", "UNKNOWN") for s in segments])
                        )

            st.write("📝 Generating outputs...")
            full_text = " ".join([seg["text"] for seg in segments])
            srt_content = generate_srt(segments, include_speakers=enable_diarization)

            processing_time = time.time() - start_time
            duration = segments[-1]["end"] if segments else 0

            st.session_state.results = {
                "num_segments": len(segments),
                "duration": duration,
                "processing_time": processing_time,
                "num_speakers": num_speakers_found,
                "full_text": full_text,
                "srt_content": srt_content,
                "segments": segments,
            }

            status.update(label="✅ Complete!", state="complete")
            st.success(f"✅ Done in {processing_time:.1f}s!")

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        st.error(f"❌ Failed: {e}")
    finally:
        try:
            for file in Config.TEMP_DIR.glob("*"):
                if file.is_file():
                    file.unlink()
        except:
            pass
        safe_model_cleanup()


def process_url(
    video_url: str,
    whisper_model: str,
    device: str,
    compute_type: str,
    source_lang: str,
    task: str,
    beam_size: int,
    enable_diarization: bool,
    hf_token: Optional[str],
    num_speakers: int,
):
    """Process video from URL"""

    Config.TEMP_DIR.mkdir(parents=True, exist_ok=True)

    try:
        with st.status("🚀 Processing URL...", expanded=True) as status:
            start_time = time.time()

            st.write("⬇️ Downloading...")
            temp_video = Config.TEMP_DIR / "downloaded_video.mp4"
            video_path = download_video_from_url(video_url, temp_video)

            if not video_path:
                status.update(label="❌ Failed", state="error")
                return

            st.write("🔊 Extracting audio...")
            audio_path = Config.TEMP_DIR / "audio.wav"
            audio_path = extract_audio(video_path, audio_path)

            if not audio_path:
                status.update(label="❌ Failed", state="error")
                return

            st.write(f"🤖 Loading model...")
            model = load_whisper_model(whisper_model, device, compute_type)

            if not model:
                status.update(label="❌ Failed", state="error")
                return

            st.write("🎙️ Transcribing...")
            segments = transcribe_audio(model, audio_path, source_lang, task, beam_size)

            if not segments:
                status.update(label="❌ Failed", state="error")
                return

            num_speakers_found = "N/A"
            if enable_diarization and hf_token:
                st.write("👥 Identifying speakers...")
                diar_pipeline = load_diarization_model(hf_token)

                if diar_pipeline:
                    diar_segments = perform_diarization(
                        diar_pipeline, audio_path, num_speakers
                    )

                    if diar_segments:
                        segments = align_transcription_with_speakers(
                            segments, diar_segments
                        )
                        num_speakers_found = len(
                            set([s.get("speaker", "UNKNOWN") for s in segments])
                        )

            st.write("📝 Generating outputs...")
            full_text = " ".join([seg["text"] for seg in segments])
            srt_content = generate_srt(segments, include_speakers=enable_diarization)

            processing_time = time.time() - start_time
            duration = segments[-1]["end"] if segments else 0

            st.session_state.results = {
                "num_segments": len(segments),
                "duration": duration,
                "processing_time": processing_time,
                "num_speakers": num_speakers_found,
                "full_text": full_text,
                "srt_content": srt_content,
                "segments": segments,
            }

            status.update(label="✅ Complete!", state="complete")
            st.success(f"✅ Done in {processing_time:.1f}s!")

    except Exception as e:
        logger.error(f"URL processing failed: {e}")
        st.error(f"❌ Failed: {e}")
    finally:
        try:
            for file in Config.TEMP_DIR.glob("*"):
                if file.is_file():
                    file.unlink()
        except:
            pass
        safe_model_cleanup()


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
