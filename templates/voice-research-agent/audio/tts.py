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
        WAV audio bytes
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
        response_format="wav",
    )
    
    log("[tts]", "Synthesized via OpenAI TTS API")
    return response.content


def _speak_local(text: str) -> bytes:
    """Synthesize using local Piper TTS."""
    _ensure_piper_model()
    model_path = _get_piper_model_path()
    
    # Clean text of special Unicode characters that cause encoding issues on Windows
    text = _sanitize_text_for_tts(text)
    
    # Run piper CLI
    result = subprocess.run(
        ["piper", "--model", str(model_path), "--output-raw"],
        input=text.encode("utf-8"),
        capture_output=True,
    )
    
    if result.returncode != 0:
        log("[tts]", f"Piper error: {result.stderr.decode('utf-8', errors='ignore')}", style="red")
        return b""
    
    log("[tts]", "Synthesized via Piper TTS")
    return result.stdout


def _sanitize_text_for_tts(text: str) -> str:
    """Remove/replace characters that cause TTS issues."""
    import re
    
    # Replace narrow no-break space and other special spaces with regular space
    text = text.replace('\u202f', ' ')  # narrow no-break space
    text = text.replace('\u00a0', ' ')  # non-breaking space
    text = text.replace('\u2009', ' ')  # thin space
    
    # Replace special dashes with regular dash
    text = text.replace('–', '-')  # en-dash
    text = text.replace('—', '-')  # em-dash
    
    # Remove other problematic Unicode
    text = re.sub(r'[^\x00-\x7F]+', lambda m: m.group(0) if m.group(0) in ' -.,!?\'"' else ' ', text)
    
    return text


def play_audio(audio_bytes: bytes, sample_rate: int = 22050) -> None:
    """Play audio bytes through speakers."""
    import sounddevice as sd
    
    if not audio_bytes:
        return
    
    # Convert bytes to numpy array (16-bit PCM)
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
    audio_float = audio_np.astype(np.float32) / 32768.0
    
    sd.play(audio_float, samplerate=sample_rate)
    sd.wait()


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
