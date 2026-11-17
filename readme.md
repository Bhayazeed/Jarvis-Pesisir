# 🤖 MARK XL — Local AI Assistant

> **Modified version** of [FatihMakes/JARVIS Mark-XL](https://github.com/FatihMakes/Mark-XL)  
> Adapted for Indonesian/Malay language support, SI-Lab LLM backend, and Wav2Vec 2.0 Melayu-Riau STT.

---

## What's Different from the Original

| Area | Original (FatihMakes) | This Fork |
|------|----------------------|-----------|
| **LLM Backend** | Ollama (local only) | Ollama + **SI-Lab API** (`silab` provider via OpenAI-compatible endpoint) |
| **STT** | Whisper / Vosk | Whisper / Vosk + **Wav2Vec 2.0 Melayu-Riau** (local & remote) |
| **Default Language** | English / Turkish | **Indonesian / Malay** (Riau dialect phonetic decoding) |
| **TTS Voice** | `en-US-GuyNeural` | `id-ID-ArdiNeural` |
| **System Prompt** | English-centric | Bilingual — auto-responds in Malay/Indonesian, deciphers Riau dialect phonetics |
| **Model Weights** | Downloaded at runtime | Includes bundled `wav2vec-melayu-riau` tokenizer |

---

## Overview

MARK XL is a real-time voice and visual AI agent. It combines speech recognition, an LLM (via Ollama or SI-Lab API), and text-to-speech to deliver a personal assistant with OS-level control capabilities.

### Architecture

```
Microphone → STT (Whisper / Vosk / Wav2Vec)
                  ↓
           LLM (Ollama or SI-Lab API — streaming + tool calling)
                  ↓
         Tool Execution (OS, Browser, Files …)
                  ↓
           TTS (EdgeTTS / Kokoro / ElevenLabs)
                  ↓
              Speaker
```

### Core Components

| Layer | Technology | Notes |
|-------|-----------|-------|
| **STT** | faster-whisper / Vosk / Wav2Vec 2.0 | Fully offline. Auto language detection or forced locale. |
| **LLM** | Ollama / SI-Lab API | Streaming + tool calling. Supports `ollama`, `silab`, `openai` providers. |
| **TTS** | EdgeTTS / Kokoro / ElevenLabs | EdgeTTS = free + internet. Kokoro = fully offline. |
| **UI** | PyQt6 | HUD overlay with system monitor, log panel, file drop zone. |
| **Agent** | Custom task queue | Multi-step planner + executor + error recovery. |
| **Memory** | JSON long-term memory | Silently saves personal facts; recalled in every conversation. |

---

## Features

- **Streaming responses** — TTS starts speaking on the first sentence, not after the full response
- **Tool calling** — 18 built-in tools: browser control, file management, weather, YouTube, messaging, screen analysis, code helper, game updater, flight finder, and more
- **Long-term memory** — Silently saves personal facts; recalled in every conversation
- **Live configuration** — Change LLM model, STT engine, TTS voice without restarting (⚙ Configure button)
- **Ollama auto-start** — Automatically launches `ollama serve` if not running
- **Model warmup** — Pre-loads the LLM into memory so the first message is as fast as subsequent ones
- **Multi-language STT** — Set `stt_language` to `auto` (Whisper detects) or a specific locale
- **Malay/Indonesian dialect support** — Deciphers phonetic Riau dialect transcriptions automatically
- **Cross-platform** — Windows, macOS, Linux (OS detected automatically at runtime)
- **File drop zone** — Drag and drop images, PDFs, Word docs, CSV, audio, video for AI processing

---

## Requirements

- Python 3.11 or 3.12
- [Ollama](https://ollama.com) installed and a model pulled — **or** an SI-Lab API key
- A microphone

---

## Quick Start

```bash
# 1. Clone the project
git clone <your-repo-url>
cd Mark-XL

# 2. Set up environment variables
copy .env.example .env
# Edit .env and add your HF_TOKEN and/or SILAB_KEY

# 3. Launch (option A: batch file)
run.bat

# 3. Launch (option B: manual)
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

On first run MARK XL:
1. Auto-installs base packages (PyQt6, numpy, etc.) and restarts once
2. Opens the **Initialisation** overlay — choose STT engine, LLM model, TTS engine
3. Click **INITIALISE SYSTEMS** — engine packages install in the background (progress shown in log)
4. JARVIS comes online

After setup, use the **⚙ CONFIGURE** button in the right panel to change any setting at any time without restarting.

---

## Environment Variables (`.env`)

```bash
# Hugging Face token — required for private Wav2Vec model
HF_TOKEN=hf_your_token_here

# SI-Lab API key — only if using "silab" as LLM provider
SILAB_KEY=your_silab_key_here
```

> **Note:** Copy `.env.example` to `.env` and fill in your real keys. Never commit `.env` to git.

---

## Configuration (`config/api_keys.json`)

Created automatically on first run, or configure via the ⚙ button in the UI.

```json
{
    "stt_engine":         "remote",
    "stt_model":          "",
    "stt_url":            "https://api-wav2vec-melayu.deeptalk.id/transcribe",
    "stt_language":       "auto",
    "llm_provider":       "silab",
    "llm_url":            "https://llms.si-lab.org/api",
    "llm_model":          "gemma4:e4b",
    "llm_api_key":        "",
    "tts_engine":         "edgetts",
    "tts_voice":          "id-ID-ArdiNeural",
    "tts_speed":          "1.0",
    "elevenlabs_api_key": ""
}
```

| Key | Values | Default |
|-----|--------|---------| 
| `stt_engine` | `whisper` / `vosk` / `wav2vec` / `remote` | `remote` |
| `stt_url` | Remote STT API endpoint (for `remote` engine) | DeepTalk Wav2Vec API |
| `stt_language` | `auto` or ISO code (`id`, `ms`, `en`, `tr` …) | `auto` |
| `llm_provider` | `ollama` / `silab` / `openai` / `lmstudio` | `silab` |
| `llm_url` | LLM API base URL | `https://llms.si-lab.org/api` |
| `llm_model` | Model name | `gemma4:e4b` |
| `tts_engine` | `edgetts` / `kokoro` / `elevenlabs` | `edgetts` |
| `tts_voice` | Voice name / ID depending on engine | `id-ID-ArdiNeural` |

---

## Project Structure

```
Mark-XL/
├── main.py              # Entry point — bootstraps packages, runs the assistant
├── ui.py                # PyQt6 HUD overlay UI
├── run.bat              # Windows one-click launcher
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variable template
│
├── core/                # Engine modules
│   ├── stt.py           # Speech-to-text (Whisper / Vosk / Wav2Vec)
│   ├── tts.py           # Text-to-speech (EdgeTTS / Kokoro / ElevenLabs)
│   ├── llm_client.py    # LLM client (Ollama / OpenAI-compatible)
│   ├── installer.py     # Dynamic package installer
│   └── prompt.txt       # JARVIS system prompt
│
├── actions/             # 17 built-in tool modules
│   ├── browser_control.py
│   ├── code_helper.py
│   ├── computer_control.py
│   ├── computer_settings.py
│   ├── desktop.py
│   ├── dev_agent.py
│   ├── file_controller.py
│   ├── file_processor.py
│   ├── flight_finder.py
│   ├── game_updater.py
│   ├── open_app.py
│   ├── reminder.py
│   ├── screen_processor.py
│   ├── send_message.py
│   ├── weather_report.py
│   ├── web_search.py
│   └── youtube_video.py
│
├── agent/               # Multi-step autonomous agent
│   ├── planner.py
│   ├── executor.py
│   ├── task_queue.py
│   └── error_handler.py
│
├── memory/              # Persistent memory system
│   ├── memory_manager.py
│   ├── config_manager.py
│   └── long_term.json
│
├── models/              # Bundled model weights
│   └── wav2vec-melayu-riau/
│
└── config/              # Runtime configuration
    └── api_keys.json
```

---

## Built-in Tools

| Tool | Description |
|------|-------------|
| `open_app` | Opens any application or website |
| `web_search` | Web search and compare mode |
| `weather_report` | Current weather for any city |
| `send_message` | WhatsApp / Telegram messaging |
| `reminder` | Timed reminders via Task Scheduler |
| `youtube_video` | Play, summarize, trending videos |
| `screen_process` | Screen capture + vision model analysis |
| `computer_settings` | Volume, brightness, window management, shortcuts |
| `browser_control` | Full Playwright browser automation |
| `file_controller` | File/folder CRUD, search, disk usage |
| `desktop_control` | Wallpaper, organize, clean desktop |
| `code_helper` | Write, edit, explain, run code |
| `dev_agent` | Build complete multi-file projects |
| `agent_task` | Multi-step autonomous task execution |
| `computer_control` | Direct mouse/keyboard control |
| `game_updater` | Steam / Epic Games install & update |
| `flight_finder` | Google Flights search |
| `file_processor` | Process images, PDFs, CSV, audio, video |

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `F4` | Mute / unmute microphone |
| `F11` | Toggle fullscreen |

---

## TTS Engine Comparison

| Engine | Internet | Quality | Cost |
|--------|----------|---------|------|
| EdgeTTS | Required | Good | Free |
| Kokoro | No | Excellent | Free (local model ~100 MB) |
| ElevenLabs | Required | Best | Paid API |

---

## Credits

- **Original project:** [FatihMakes/Mark-XL](https://github.com/FatihMakes/Mark-XL) — JARVIS Mark-XL by FatihMakes Industries
- **Modifications:** Adapted for Indonesian/Malay language, SI-Lab LLM backend, Wav2Vec 2.0 Melayu-Riau STT

---

## License

MIT — Based on [FatihMakes Industries](https://github.com/FatihMakes) original work
