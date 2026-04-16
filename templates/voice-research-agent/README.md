# voice-research-agent

Voice-driven research agent — ask questions by voice, get researched answers.

Part of [AgentKit](https://github.com/samgit17/agent-kit).

## Features

- **Voice input**: Push-to-talk microphone or audio file
- **Research pipeline**: LangGraph multi-step research (planner → searcher → synthesiser → verifier)
- **Voice output**: Optional TTS to read answers aloud
- **Dual mode**: OpenAI APIs (default) or fully local (faster-whisper + Piper)

## Quick Start

```bash
# Clone and setup
cd templates/voice-research-agent
python -m venv .venv

# Activate (Linux/macOS)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate

pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env — set OPENAI_API_KEY for default mode

# Run with text query
python run.py "What are the latest developments in autonomous vehicles?"

# Run with voice input (push-to-talk)
python run.py

# Run with audio file
python run.py --audio question.wav

# Speak the answer
python run.py --speak "What is quantum computing?"

# Fully local (no API calls for STT/TTS)
python run.py --local --speak
```

## CLI Reference

```
usage: voice-research [-h] [--audio PATH] [--speak] [--local] [query]

positional arguments:
  query          Research question (text). If omitted, uses microphone.

options:
  --audio PATH   Path to audio file (WAV/MP3) instead of microphone.
  --speak        Read the answer aloud via TTS.
  --local        Use local models (faster-whisper + Piper) instead of OpenAI.
```

## Modes

### Default (OpenAI APIs)

Uses OpenAI Whisper API for STT and OpenAI TTS for speech output. Requires `OPENAI_API_KEY`.

### Local (`--local`)

Uses faster-whisper for STT and Piper for TTS. Models auto-download on first run (~500MB total).

```bash
# Install local dependencies
pip install faster-whisper piper-tts

# Run locally
python run.py --local --speak "Explain transformer architecture"
```

## Pipeline

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│   Planner   │────▶│   Searcher   │────▶│  Synthesiser │
│ (3 queries) │     │ (web search) │     │   (draft)    │
└─────────────┘     └──────────────┘     └──────┬───────┘
                                                │
                    ┌──────────────┐     ┌──────▼───────┐
                    │  Formatter   │◀────│   Verifier   │
                    │   (report)   │     │ (confidence) │
                    └──────────────┘     └──────────────┘
                           │                    │
                           │              retry if < 60%
                           ▼
                    ┌──────────────┐
                    │  TTS Output  │ (optional)
                    └──────────────┘
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `ollama` | `ollama` or `openai` |
| `OLLAMA_MODEL` | `qwen3.5:27b` | Ollama model name |
| `OPENAI_API_KEY` | — | Required for OpenAI LLM/STT/TTS |
| `SEARCH_PROVIDER` | `duckduckgo` | `duckduckgo` or `tavily` |
| `TAVILY_API_KEY` | — | Required if using Tavily |
| `WHISPER_DEVICE` | `cpu` | `cpu` or `cuda` (for `--local` STT) |
| `TRACING_ENABLED` | — | Set to `true` for Phoenix tracing |

## Requirements

- Python 3.10+
- ffmpeg (for MP3/M4A file support)
- Microphone (for push-to-talk)

## Platform Notes

### Windows

- Push-to-talk works without admin privileges
- If using `--local` with CUDA, install: `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12`
- If CUDA fails, set `WHISPER_DEVICE=cpu` in `.env`

### Linux

- `keyboard` module requires root: run with `sudo python run.py --local`
- Alternative: add user to `input` group

### macOS

- `keyboard` module may require accessibility permissions

## Tests

```bash
# Install dev dependencies
pip install pytest

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_stt.py -v
```

## License

MIT
