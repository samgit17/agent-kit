"""
audio/recorder.py — Push-to-talk microphone recording.

Hold Space to record, release to stop.
"""

from __future__ import annotations
import numpy as np
import sounddevice as sd
from log import log

SAMPLE_RATE = 16000  # 16kHz for Whisper
CHANNELS = 1


def record_push_to_talk() -> bytes:
    """Record audio while Space is held, return PCM bytes on release.
    
    Returns:
        Raw PCM audio bytes (16-bit, mono, 16kHz)
    """
    try:
        import keyboard
    except ImportError:
        log("[recorder]", "keyboard module not installed — run: pip install keyboard", style="red")
        return b""
    
    log("[recorder]", "Hold [Space] to speak, release when done...")
    
    # Wait for Space press
    keyboard.wait("space")
    log("[recorder]", "Recording...", style="green")
    
    # Start recording
    frames = []
    
    def callback(indata, frame_count, time_info, status):
        if status:
            log("[recorder]", f"Status: {status}", style="yellow")
        frames.append(indata.copy())
    
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=np.int16, callback=callback):
        # Record until Space is released
        keyboard.wait("space", suppress=True, trigger_on_release=True)
    
    log("[recorder]", "Recording stopped")
    
    if not frames:
        return b""
    
    # Concatenate all frames
    audio_np = np.concatenate(frames, axis=0)
    return audio_np.tobytes()


def record_from_file(path: str) -> bytes:
    """Read audio from a file and return as PCM bytes.
    
    Args:
        path: Path to WAV or MP3 file
    
    Returns:
        Raw PCM audio bytes (16-bit, mono, 16kHz)
    """
    import wave
    from pathlib import Path
    
    file_path = Path(path)
    
    if not file_path.exists():
        log("[recorder]", f"File not found: {path}", style="red")
        return b""
    
    suffix = file_path.suffix.lower()
    
    if suffix == ".wav":
        return _read_wav(file_path)
    elif suffix in (".mp3", ".m4a", ".ogg", ".flac"):
        return _read_with_pydub(file_path)
    else:
        # Return raw bytes for unknown formats — let Whisper handle it
        return file_path.read_bytes()


def _read_wav(path) -> bytes:
    """Read WAV file to PCM bytes."""
    import wave
    
    with wave.open(str(path), "rb") as wf:
        # Resample if needed
        if wf.getframerate() != SAMPLE_RATE:
            return _resample_wav(path)
        
        return wf.readframes(wf.getnframes())


def _read_with_pydub(path) -> bytes:
    """Read audio file using pydub, convert to PCM."""
    try:
        from pydub import AudioSegment
    except ImportError:
        log("[recorder]", "pydub not installed — run: pip install pydub", style="red")
        return b""
    
    audio = AudioSegment.from_file(str(path))
    audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(CHANNELS).set_sample_width(2)
    return audio.raw_data


def _resample_wav(path) -> bytes:
    """Resample WAV to 16kHz using scipy."""
    try:
        from scipy.io import wavfile
        from scipy import signal
    except ImportError:
        log("[recorder]", "scipy not installed — run: pip install scipy", style="red")
        return b""
    
    rate, data = wavfile.read(str(path))
    
    # Resample to 16kHz
    num_samples = int(len(data) * SAMPLE_RATE / rate)
    resampled = signal.resample(data, num_samples)
    
    # Convert to int16
    if resampled.dtype != np.int16:
        resampled = (resampled * 32767).astype(np.int16)
    
    return resampled.tobytes()
