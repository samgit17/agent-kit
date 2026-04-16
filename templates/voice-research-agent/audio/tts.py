"""
audio/tts.py — Text-to-speech abstraction.

Default: OpenAI TTS API
Local:   Piper TTS (with --local flag)
"""

from __future__ import annotations
import os
import subprocess
import numpy as np
from pathlib import Path
from functools import lru_cache

from log import log

# Piper model config
PIPER_MODEL_NAME = "en_US-amy-medium"
PIPER_MODEL_URL = f"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/{PIPER_MODEL_NAME}.onnx"
PIPER_CONFIG_URL = f"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/{PIPER_MODEL_NAME}.onnx.json"


def speak(text: str, local: bool = False) -> bytes:
    """Convert text to speech audio bytes.
    
    Args:
        text: Text to synthesize
        local: If True, use Piper; else use OpenAI API
    
    Returns:
        Raw PCM audio bytes
    """
    if not text.strip():
        return b""
    
    if local:
        return _speak_local(text)
    else:
        return _speak_openai(text)


def _speak_openai(text: str) -> bytes:
    """Synthesize using OpenAI TTS API."""
    from openai import OpenAI
    
    client = OpenAI()
    
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text,
        response_format="pcm",
    )
    
    log("[tts]", "Synthesized via OpenAI TTS API")
    return response.content


def _sanitize_text_for_tts(text: str) -> str:
    """Remove problematic characters for TTS.
    
    Piper's espeak phonemizer can choke on Unicode surrogates,
    special characters, and various non-ASCII symbols.
    Safest approach: convert to ASCII-only.
    """
    # First pass: replace common Unicode with ASCII equivalents
    replacements = {
        "\u202f": " ",   # narrow no-break space
        "\u00a0": " ",   # non-breaking space
        "\u2013": "-",   # en dash
        "\u2014": "-",   # em dash
        "\u2018": "'",   # left single quote
        "\u2019": "'",   # right single quote
        "\u201c": '"',   # left double quote
        "\u201d": '"',   # right double quote
        "\u2026": "...", # ellipsis
        "\u2022": ",",   # bullet point -> pause
        "\u2023": ",",   # triangular bullet
        "\u2043": "-",   # hyphen bullet
        "\u00b7": ",",   # middle dot
        "\ufffd": "",    # replacement character
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    
    # Second pass: encode to ASCII, dropping anything that doesn't fit
    text = text.encode("ascii", errors="ignore").decode("ascii")
    
    # Clean up any resulting double spaces
    while "  " in text:
        text = text.replace("  ", " ")
    
    return text.strip()


def _speak_local(text: str) -> bytes:
    """Synthesize using local Piper TTS."""
    _ensure_piper_model()
    model_path = _get_piper_model_path()
    
    sanitized_text = _sanitize_text_for_tts(text)
    
    if not sanitized_text:
        return b""
    
    # Run piper CLI with bytes input (avoids Windows encoding issues)
    result = subprocess.run(
        ["piper", "--model", str(model_path), "--output-raw"],
        input=sanitized_text.encode("utf-8"),
        capture_output=True,
    )
    
    if result.returncode != 0:
        log("[tts]", f"Piper error: {result.stderr.decode()}", style="red")
        return b""
    
    log("[tts]", "Synthesized via Piper TTS")
    return result.stdout


def play_audio(audio_bytes: bytes, sample_rate: int = 22050) -> None:
    """Play audio bytes through speakers. Press Escape to stop."""
    import threading
    import sounddevice as sd
    import keyboard
    
    if not audio_bytes:
        return
    
    # Convert bytes to numpy array (16-bit PCM)
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
    audio_float = audio_np.astype(np.float32) / 32768.0
    
    stop_event = threading.Event()
    
    def wait_for_escape():
        keyboard.wait("escape")
        stop_event.set()
        sd.stop()
    
    listener = threading.Thread(target=wait_for_escape, daemon=True)
    listener.start()
    
    log("[tts]", "Playing audio (press Escape to stop)...")
    sd.play(audio_float, samplerate=sample_rate)
    
    # Poll instead of sd.wait() so escape can interrupt
    while sd.get_stream().active and not stop_event.is_set():
        sd.sleep(100)
    
    sd.stop()


def _get_cache_dir() -> Path:
    """Get cache directory for Piper models."""
    cache_home = os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")
    return Path(cache_home) / "voice-research-agent" / "piper"


def _get_piper_model_path() -> Path:
    """Get path to Piper model file."""
    return _get_cache_dir() / f"{PIPER_MODEL_NAME}.onnx"


def _ensure_piper_model() -> None:
    """Download Piper model if not present."""
    model_path = _get_piper_model_path()
    config_path = model_path.with_suffix(".onnx.json")
    
    if model_path.exists() and config_path.exists():
        return
    
    _download_piper_model()


def _download_piper_model() -> None:
    """Download Piper voice model from Hugging Face."""
    import urllib.request
    
    cache_dir = _get_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = cache_dir / f"{PIPER_MODEL_NAME}.onnx"
    config_path = cache_dir / f"{PIPER_MODEL_NAME}.onnx.json"
    
    log("[tts]", f"Downloading Piper model ({PIPER_MODEL_NAME})...")
    
    urllib.request.urlretrieve(PIPER_MODEL_URL, model_path)
    urllib.request.urlretrieve(PIPER_CONFIG_URL, config_path)
    
    log("[tts]", "Piper model downloaded")
