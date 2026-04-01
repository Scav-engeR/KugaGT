# CyberTranscribe 3999 — Kuga V2

A Streamlit-based neural transcription and translation engine powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper), with LLM post-processing, hallucination filtering, domain presets, and multi-format subtitle export.

---

## Features

### Core (V1)
- 20+ Whisper model variants (standard, Japanese-specialised, distilled)
- GPU/CPU inference with selectable compute types
- Speaker diarization via pyannote.audio
- YouTube/URL download via yt-dlp
- AI Chat with Claude, OpenAI, DeepSeek, Grok
- 8 visual themes

### New in V2
| Feature | Description |
|---|---|
| **Domain Presets** | Primes Whisper with domain-specific context (anime, medical, news, etc.) for significantly better accuracy |
| **Editable Initial Prompt** | Fine-tune the model's starting context per file — add character names, technical terms |
| **Hallucination Filter** | Removes common Whisper artifacts (subscribe prompts, music notes, low-confidence segments) |
| **Segment Merging** | Combines short/adjacent segments for more readable subtitles |
| **Confidence Scoring** | Per-segment log-probability and no-speech probability displayed in results |
| **Noise Reduction** | Optional audio preprocessing (noisereduce) before transcription |
| **Glossary Engine** | Persistent term replacements applied across all outputs |
| **LLM Translation Studio** | Batch post-process all segments with Claude/GPT/DeepSeek/Grok |
| **Multi-Format Export** | SRT, WebVTT, ASS/SSA, CSV, Dual-SRT, JSON, TXT |
| **Repetition Penalty** | Reduces repetitive hallucinations |
| **Temperature Control** | Tune between deterministic and diverse outputs |

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/your-username/kuga-transcribe.git
cd kuga-transcribe
pip install -r requirements.txt
```

For GPU support install the CUDA version of PyTorch first:
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2. Configure environment

Copy the example env file and fill in your API keys:
```bash
cp .env.example .env
```

`.env` contents:
```env
HF_TOKEN=hf_...           # HuggingFace (required for speaker diarization)
ANTHROPIC_API_KEY=sk-...  # Claude
OPENAI_API_KEY=sk-...     # OpenAI / GPT
DEEPSEEK_API_KEY=...      # DeepSeek
GROK_API_KEY=...          # Grok (xAI)
```

### 3. Run

```bash
streamlit run Kuga-V2.py
```

---

## Usage Guide

### Basic Transcription

1. Open the **Upload** or **URL** tab
2. Select a **Whisper model** in the sidebar (⭐ recommended models are highlighted)
3. Choose your **Source Language** and **Task** (`translate` → English, `transcribe` → keep original)
4. Pick a **Domain Preset** matching your content type
5. Press **PROCESS**

### Improving Accuracy

| Setting | When to use |
|---|---|
| **Domain Preset + Initial Prompt** | Always — biggest single accuracy improvement |
| **Beam Size 7–10** | When accuracy matters more than speed |
| **Noise Reduction** | Noisy recordings, outdoor audio, phone calls |
| **Hallucination Filter** | YouTube videos, content with music/silence |
| **Segment Merge** | When subtitles feel too fragmented |

### LLM Translation Studio

After transcription, open the **Translation Studio** tab:

1. Select your LLM provider and model
2. Set the target language and domain
3. Click **REFINE WITH LLM**

The LLM processes all segments in batches, improving naturalness and fixing Whisper errors. A before/after preview is shown. Refined outputs are automatically updated in the Results tab.

### Glossary

Add term replacements in the sidebar **Glossary** section:
```
# Lines starting with # are comments
Naruto = Naruto (ナルト)
AI = Artificial Intelligence
sensei = teacher
```

---

## Models

| Model | Language | Size | Best For |
|---|---|---|---|
| Kotoba Whisper Bilingual ⭐ | JA/EN | ~800 MB | Japanese↔English |
| Large V3 Turbo ⭐ | Multi | ~1.5 GB | General purpose, fast |
| Whisper Medium ⭐ | Multi | ~1.5 GB | Balanced speed/accuracy |
| Distil Whisper JA ReazonSpeech ⭐🧪 | JA | ~1 GB | Japanese content |
| Whisper Large V3 | Multi | ~3 GB | Maximum accuracy |

---

## Architecture

```
Kuga-V2.py
├── Audio preprocessing (noisereduce + soundfile)
├── faster-whisper transcription
│   ├── Domain-primed initial prompt
│   ├── Configurable beam size, temperature, thresholds
│   └── Per-segment confidence scores
├── Hallucination detection & filtering
├── Segment merging
├── Glossary application
├── Speaker diarization (pyannote.audio)
└── Multi-format export
    ├── SRT / VTT / ASS
    ├── CSV (with metadata)
    ├── Dual-SRT (original + refined)
    └── JSON
```

---

## Requirements

- Python 3.9+
- 4 GB RAM minimum (8 GB+ recommended)
- GPU optional but recommended for large models
- ffmpeg installed on system PATH

Install ffmpeg:
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
winget install ffmpeg
```

---

## Troubleshooting

**`CUDA out of memory`** — switch to a smaller model or CPU in the Hardware sidebar section.

**`Model loading failed`** — try clearing the model cache (sidebar button) and reload.

**`Diarization not working`** — ensure your HuggingFace token has accepted the pyannote model license at https://huggingface.co/pyannote/speaker-diarization-3.1

**`noisereduce not available`** — noise reduction is optional; install with `pip install noisereduce soundfile`.

**`LLM refinement returns unchanged text`** — check your API key in `.env` and verify the provider is reachable.

---

## License

MIT
