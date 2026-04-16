"""
tests/test_stt.py — Unit tests for STT module.
"""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np


class TestTranscribeOpenAI:
    """Tests for OpenAI Whisper API transcription."""

    def test_transcribe_returns_text(self):
        """transcribe() should return transcribed text from audio bytes."""
        from audio.stt import transcribe

        mock_response = MagicMock()
        mock_response.text = "What are the latest AI developments?"

        with patch("audio.stt._transcribe_openai", return_value="What are the latest AI developments?"):
            result = transcribe(b"fake_audio_bytes", local=False)

        assert result == "What are the latest AI developments?"

    def test_transcribe_openai_calls_api(self):
        """_transcribe_openai should call OpenAI API with correct params."""
        from audio.stt import _transcribe_openai

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = MagicMock(text="Hello world")

        with patch("openai.OpenAI", return_value=mock_client):
            result = _transcribe_openai(b"audio_bytes")

        assert result == "Hello world"
        mock_client.audio.transcriptions.create.assert_called_once()

    def test_transcribe_empty_audio_returns_empty_string(self):
        """Empty audio should return empty string, not error."""
        from audio.stt import transcribe

        with patch("audio.stt._transcribe_openai", return_value=""):
            result = transcribe(b"", local=False)

        assert result == ""


class TestTranscribeLocal:
    """Tests for local faster-whisper transcription."""

    def test_transcribe_local_returns_text(self):
        """transcribe(local=True) should use faster-whisper."""
        from audio.stt import transcribe

        with patch("audio.stt._transcribe_local", return_value="Local transcription"):
            result = transcribe(b"audio_bytes", local=True)

        assert result == "Local transcription"

    def test_transcribe_local_loads_model_once(self):
        """Model should be cached, not reloaded on each call."""
        # Clear the lru_cache first
        from audio.stt import _get_local_model
        _get_local_model.cache_clear()

        mock_model = MagicMock()
        with patch("faster_whisper.WhisperModel", return_value=mock_model) as mock_cls:
            model1 = _get_local_model()
            model2 = _get_local_model()

        # Should only instantiate once due to caching
        assert mock_cls.call_count == 1
        assert model1 is model2
        
        # Clear cache for other tests
        _get_local_model.cache_clear()


class TestAudioConversion:
    """Tests for audio format conversion helpers."""

    def test_bytes_to_numpy_correct_shape(self):
        """Audio bytes should convert to numpy array correctly."""
        from audio.stt import _bytes_to_numpy

        # 16-bit PCM, mono, 16kHz — 1 second = 32000 bytes
        fake_audio = np.zeros(16000, dtype=np.int16).tobytes()
        result = _bytes_to_numpy(fake_audio)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert len(result) == 16000
