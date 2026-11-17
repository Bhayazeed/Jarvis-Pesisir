# JARVIS PESISIR — Coastal Riau Malay Voice Assistant
### Fork of FatihMakes' Mark LVII, adapted for Indonesian and Coastal Riau Malay

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

## ✨ Overview

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

Switch language mid-conversation and the voice switches with you. No detection, no dependency, no guess — just information the pipeline was already holding.

| Layer | Technology | Notes |
|-------|-----------|-------|
| **STT** | faster-whisper / Vosk / Wav2Vec 2.0 | Fully offline. Auto language detection or forced locale. |
| **LLM** | Ollama / SI-Lab API | Streaming + tool calling. Supports `ollama`, `silab`, `openai` providers. |
| **TTS** | EdgeTTS / Kokoro / ElevenLabs | EdgeTTS = free + internet. Kokoro = fully offline. |
| **UI** | PyQt6 | HUD overlay with system monitor, log panel, file drop zone. |
| **Agent** | Custom task queue | Multi-step planner + executor + error recovery. |
| **Memory** | JSON long-term memory | Silently saves personal facts; recalled in every conversation. |

---

## 📦 Previously, in Mark LV

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

## 🔄 The Foundation Update — in every Mark from LII

These four landed across **Mark LII, LIII, LIV and LV at the same time**, after each of those releases had already shipped. They are not what any one of those versions originally introduced; they are the floor all of them now stand on, so moving up a Mark never costs you something the one below it had.

No new dependencies. No bundled asset files. No hardcoded language, and nothing that assumes one operating system.

### 🧠 A memory that actually remembers

The store was capped at **2,200 characters — the whole memory, not per entry** — because all of it was pasted into the system prompt on every connect, so growing the memory grew every request. When it filled, the oldest entries were deleted and one line was printed to a console nobody reads. An assistant advertised as remembering "projects, preferences and personal context" was in practice a two-page notepad that quietly forgot your sister's name after a few weeks.

Storage and prompt budget are now separate problems:

* **Nothing is deleted.** The cap is a runaway guard normal use never approaches, and if it is ever hit it says so in the activity log instead of on stdout.
* **The prompt carries a core, not a dump.** Identity in full, then the most recently updated facts, budgeted — measured at **971 characters on a memory holding 62 stored facts.** That is *smaller* than the old whole-store cap, so sessions now connect with fewer tokens than before.
* **The rest is fetched on demand.** A `recall_memory` tool searches the full store locally — no network, no second model, well under a millisecond.

The part that is easy to get wrong: **a model cannot look something up if it doesn't know the thing exists.** So the prompt also carries an **index of the keys** it had no room for. Without it, "who is Ayşe?" gets "I don't know" while `ayse_sister` sits on disk unread. That index interleaves categories rather than sorting by recency — sorted like the core, a memory with forty preferences pushed the one entry the index existed for off the end.

⚙ → **🧠 MEMORY** shows every stored fact, when it was learned, and a ✕ to forget it. Everything stays in `memory/long_term.json` on your machine.

### ↩️ Undo — it can take back what it did

JARVIS moves files, renames them, writes to them and changes your settings. None of that had a way back; if it misheard you, the only remedy was to fix it by hand.

Say **"undo"** — in any language — and it reverses its own last action:

| | |
|---|---|
| **Files** | move · rename · create · copy · write · delete · organize desktop |
| **Settings** | volume · brightness · dark mode |

Three things it deliberately does *not* do:

* **It does not guess.** Settings undo reads the current value *before* changing it. Where a platform won't report that value, nothing is registered — an undo that restores a guess is worse than no undo.
* **It does not hoard.** Undoing a write means keeping the old contents in memory, so files over 1 MB are excluded and it says so rather than holding a 200 MB log for the session.
* **It does not delete your files to undo a copy.** The reverse of a copy is removing the copy; the reverse of "create a folder" is removing it *only while it's still empty*.

`organize_desktop` gets special treatment — one command that moves dozens of files, which made it the least reversible thing the assistant could do. It journals every move and puts all of them back in one go, cleaning up the folders it created if they're still empty.

**Undo costs nothing at runtime.** It appends a closure to a list; nothing in it runs unless you ask.

### ⚠️ A confirmation the model can't forge

The old gate read like this:

```python
if action in _DANGEROUS_ACTIONS:            # {"restart", "shutdown"}
    confirmed = str(params.get("confirmed", "")).lower()
```

`confirmed` is a **tool parameter, which means the model fills it in.** Nothing stopped it sending `confirmed=yes` on the first call and nothing checked that a human was ever involved. It was a convention, not a gate. And its coverage was two actions — so `toggle_wifi`, which cuts the assistant's own connection to the Live API and therefore *cannot be asked to undo itself*, went through with no gate at all.

The token is now issued by the interface. Shutdown, restart and WiFi put a banner on the HUD and **return immediately**; the action runs only if you press CONFIRM. Nothing blocks — JARVIS keeps talking while the banner is up — so this is **cheaper than the old gate**, which burned two tool round trips on every power command.

> The split between the two mechanisms is about reversibility, not about how alarming a word sounds. Anything undoable is done at once; only the genuinely irreversible asks. An assistant that checks with you before turning the volume down is one you stop talking to.

### 🎧 It finally asks which microphone

Both audio streams opened with no device argument at all, so they always took whatever the OS called "default" — and on Windows that *moves on its own* the moment you plug a headset in. "JARVIS can't hear me" almost always meant "JARVIS is listening to the webcam".

⚙ → **🎧 AUDIO DEVICES** lets you pick the microphone and the speakers by name. Two things matter more than the dropdown:

**The list is short.** `query_devices()` returns one entry per *device × host API*, not per device — measured on an ordinary Windows machine, **41 entries for what the sound settings show as 4 microphones and 4 speakers.** The same microphone appears four times, under MME, DirectSound, WASAPI and WDM-KS, with nothing to say which is which. That is not a choice, it's a quiz. The picker takes one host API per direction, drops the "Sound Mapper" and "Primary Sound Driver" pseudo-devices that just mean "default", and deduplicates. **41 → 8.**

**Every entry has been measured, not assumed.** The obvious approach is to pick the host API with the nicest names — WASAPI on Windows, which in shared mode **doesn't resample**, so with 16 kHz in and 24 kHz out against 48 kHz hardware every open failed. Adding a rate check and moving to DirectSound passes that test on both sides, and PortAudio's DirectSound **output is a silent sink**: the stream opens, every write returns success in ~0 ms, and not one sample reaches the speakers.

| | write(2.0 s) took | |
|---|---|---|
| MME | **2.02 s** | consumed in real time |
| DirectSound | **0.00 s** | swallowed instantly |

No capability flag reports that. So the app measures it — once per host API per direction, on a background thread at startup, using silence. Two consequences worth stating plainly:

* **Each direction picks its own host API.** On Windows this lands on DirectSound for the microphone and MME for the speakers — a split no amount of reasoning would have produced.
* **The probe runs in the mode the app actually ships.** DirectSound input passes a callback stream and fails a blocking read; probing the wrong mode rejected a microphone that works perfectly.

Your choice is stored **by name, not by index** — indices shift whenever something is plugged in. If the saved device is gone, it falls back to the system default and says so in the log rather than failing to start.

### 🔗 It stops forgetting the conversation when the connection drops

`session_resumption` was switched on in the config and the handle the server sent back was **never read** — so every reconnect started an empty session. A dropped packet, or simply changing the voice, wiped the conversation. "Unlimited sessions" leaked through exactly this hole.

The handle is captured and replayed now. A network blip, or switching your microphone, keeps the conversation intact.

It is held in memory only, deliberately: writing it to disk would make a fresh launch continue yesterday's chat, which sounds appealing but breaks the session-summary flow — a conversation that never ends never produces a summary, and the "yesterday we talked about…" line in the morning briefing silently disappears. Changing the **voice** also starts clean on purpose, since resuming restores the server's session state and would likely bring the old voice back with it.

### 🩹 Fixes that came with it

* **The assistant could die on a log line.** Status lines carry emoji and arrows (`📤 file_controller → Moved: a.txt → Documents/`). On a non-UTF-8 console — cp1254 on a Turkish Windows, cp1251 on a Russian one, cp932 on a Japanese one — printing one raises `UnicodeEncodeError`, and because that print sits *after* the tool's own `try/except`, it escaped into the receive loop and took the session down.
* **Every computer command paid for two model round trips.** `computer_settings` made an *entire second Gemini call, inside the tool*, purely to translate the request into one of its own action names — because the declaration only said "The action to perform", so the model rarely filled it in. When that second call failed, the fallback was `description.lower().replace(" ", "_")`, which turns the Turkish for "turn it down" into `sesi_kis` and straight into "Unknown action". The declaration now names all 56 actions and the rest is spelling tolerance handled locally by `difflib` in microseconds. When nothing matches it suggests real action names instead of dead-ending.
* An unresolvable saved audio device, or one the driver refuses to open, falls back to the system default and says so — on both the microphone and the speakers.
* A rejected session-resumption handle is dropped after one attempt, so an expired handle can never be replayed on every retry and prevent the reconnect it exists to protect.


- Python 3.11 or 3.12
- [Ollama](https://ollama.com) installed and a model pulled — **or** an SI-Lab API key
- A microphone

---

## 🗺️ Mark Roadmap

| Mark | Focus |
|---|---|
| **XLIX** | Auto-start · clipboard intelligence · assistant customization |
| **L** | Session memory · background monitoring · proactive 2.0 · instant vision |
| **LI** | Plugin system · affective dialog · proactive audio · unlimited sessions |
| **LII** | Voice picker · live theming · reactive HUD · recallable memory · undo · real confirmation · audio device picker · session continuity |
| **LIII** | Wake word · Gemini 3.1 Flash Live · instant acknowledgment · self-describing action/plugin architecture |
| **LIV** | Holographic avatar · viseme lip-sync · facial acting · face-as-status · push-to-talk · self-echo guard · runtime self-knowledge & limits |
| **LV** | Video on the HUD · model ladder with measured fallback · split settings drawers · trimmed bundled skill list |
| **LVI** | Choose your engine · local & third-party models · Whisper + Piper in ~50 languages · language-following voice · ElevenLabs · tool budget for small models · verified outcomes instead of claimed ones · close an app by name · untrusted-content boundary · platform honesty on Wayland and macOS · hardened phone dashboard |
| *shared* | The last five above also shipped to LIII, LIV and LV at the same time — moving up a Mark never loses them |
| **LVII** | Background tasks · agents as files · deep research with quotes checked in code · one routine engine for briefing, check-ins, news watch and reminders · quiet hours · command-center HUD · 3-D reactor core · re-measured model ladder |
| **LVIII+** | MCP tools · an approval policy for actions · event-driven proactivity · a knowledge folder · conversation history · interrupt by voice |

---

## ⚡ Quick Start

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

> ⚠️ **Installation Note:** If you hit a `ModuleNotFoundError` for an OS-specific package, install it with `pip install <module_name>`. The optional **wake word** engine is *not* installed here — grab it in one click from **⚙ → WAKE WORD** inside the app.

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

## 🔒 Your Data

Everything stays on your machine. There is no MARK server, no telemetry and no account.

| What | Where | Notes |
|---|---|---|
| Gemini API key, plugin and provider credentials | `config/api_keys.json` | **Plaintext.** Anyone with your user account can read it. Treat it like a password file. |
| Dashboard TLS certificate + private key | `config/certs/` | Generated locally, self-signed, never leaves the machine. |
| What the assistant remembers about you | `memory/long_term.json` | Delete the file to make it forget everything. |
| Background tasks, their checkpoints and saved pages | `memory/tasks/` | Delete a task's files to remove it; finished tasks beyond the latest 100 are pruned. |
| Task reports | `reports/` | Plain Markdown. |
| Your routines and quiet hours | `memory/routines.json` | Delete it and the built-in routines are recreated from your settings. |

All of them are listed in `.gitignore`, so a fork or a pull request cannot leak them by accident. **If you have already committed `config/api_keys.json` anywhere public, revoke that key** at [aistudio.google.com](https://aistudio.google.com/app/apikey) and generate a new one — removing the file in a later commit does not remove it from the history.

Your voice is streamed to Google's Gemini Live API while a session is open; that is the one thing that leaves your computer, and it stops when you mute or close the app.

**Mark LVI changes what that sentence can say about you.** Choose a local engine — Whisper on your CPU, a model served by Ollama or LM Studio, your system's own voice — and nothing leaves the machine at all: no audio, no transcript, no key, no account. Choose a hosted provider instead and your audio and text go to that vendor rather than to Google. The panel says which is which, and the assistant never quietly switches between them.

---

## ⚠️ License

Personal and non-commercial use only.
Licensed under **[Creative Commons BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)**.

---

## 👤 Connect with the Creator

Engineered by a developer building a real-world JARVIS-style assistant.
⭐ **Star the repository to support the journey to Mark 100.**

---

## Credits

- **Original project:** [FatihMakes/Mark-XL](https://github.com/FatihMakes/Mark-XL) — JARVIS Mark-XL by FatihMakes Industries
- **Modifications:** Adapted for Indonesian/Malay language, SI-Lab LLM backend, Wav2Vec 2.0 Melayu-Riau STT

---

## License

MIT — Based on [FatihMakes Industries](https://github.com/FatihMakes) original work
