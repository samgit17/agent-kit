"""
audio/stt.py — Speech-to-text abstraction.

Default: OpenAI Whisper API
Local:   faster-whisper (with --local flag)
"""

from __future__ import annotations
import io
import numpy as np
from functools import lru_cache

from log import log


def transcribe(audio_bytes: bytes, local: bool = False) -> str:
    """Transcribe audio bytes to text.
    
    Args:
        audio_bytes: Raw PCM audio (16-bit, mono, 16kHz) or WAV/MP3 bytes
        local: If True, use faster-whisper; else use OpenAI API
    
    Returns:
        Transcribed text
    """
    if not audio_bytes:
        return ""
    
    if local:
        return _transcribe_local(audio_bytes)
    else:
        return _transcribe_openai(audio_bytes)


def _transcribe_openai(audio_bytes: bytes) -> str:
    """Transcribe using OpenAI Whisper API."""
    from openai import OpenAI
    
    client = OpenAI()
    
    # Wrap bytes in a file-like object with .wav extension
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "audio.wav"
    
    response = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        response_format="text",
    )
    
    log("[stt]", "Transcribed via OpenAI Whisper API")
    return response.strip() if isinstance(response, str) else response.text.strip()


def _transcribe_local(audio_bytes: bytes) -> str:
    """Transcribe using local faster-whisper model."""
    model = _get_local_model()
    
    # Convert bytes to numpy array
    audio_np = _bytes_to_numpy(audio_bytes)
    
    # Transcribe
    segments, info = model.transcribe(audio_np, beam_size=5)
    text = " ".join(segment.text for segment in segments)
    
    log("[stt]", f"Transcribed via faster-whisper (lang: {info.language})")
    return text.strip()


@lru_cache(maxsize=1)
def _get_local_model():
    """Load and cache the faster-whisper model."""
    from faster_whisper import WhisperModel
    import os
    
    log("[stt]", "Loading faster-whisper model (small.en)...")
    
    # Use DEVICE env var to override, default to CPU (more reliable cross-platform)
    device = os.getenv("WHISPER_DEVICE", "cpu")
    compute_type = "float16" if device == "cuda" else "int8"
    
    model = WhisperModel("small.en", device=device, compute_type=compute_type)
    log("[stt]", f"Model loaded ({device.upper()})")
    
    return model




def _bytes_to_numpy(audio_bytes: bytes) -> np.ndarray:
    """Convert raw PCM bytes (16-bit mono 16kHz) to float32 numpy array."""
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
    return audio_np.astype(np.float32) / 32768.0
