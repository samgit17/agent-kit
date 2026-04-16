"""
tests/test_tts.py — Unit tests for TTS module.
"""

import pytest
from unittest.mock import patch, MagicMock


class TestSpeakOpenAI:
    """Tests for OpenAI TTS."""

    def test_speak_returns_audio_bytes(self):
        """speak() should return audio bytes."""
        from audio.tts import speak

        with patch("audio.tts._speak_openai", return_value=b"fake_audio"):
            result = speak("Hello world", local=False)

        assert result == b"fake_audio"

    def test_speak_openai_calls_api(self):
        """_speak_openai should call OpenAI TTS API."""
        from audio.tts import _speak_openai

        mock_response = MagicMock()
        mock_response.content = b"audio_bytes"

        mock_client = MagicMock()
        mock_client.audio.speech.create.return_value = mock_response

        with patch("openai.OpenAI", return_value=mock_client):
            result = _speak_openai("Test text")

        assert result == b"audio_bytes"
        mock_client.audio.speech.create.assert_called_once()

    def test_speak_empty_text_returns_empty(self):
        """Empty text should return empty bytes."""
        from audio.tts import speak

        result = speak("", local=False)

        assert result == b""


class TestSpeakLocal:
    """Tests for local Piper TTS."""

    def test_speak_local_returns_audio(self):
        """speak(local=True) should use Piper."""
        from audio.tts import speak

        with patch("audio.tts._speak_local", return_value=b"piper_audio"):
            result = speak("Hello", local=True)

        assert result == b"piper_audio"

    def test_speak_local_downloads_model_if_missing(self):
        """Piper should auto-download voice model on first use."""
        from audio.tts import _ensure_piper_model
        from pathlib import Path
        
        # Mock the path to return non-existent file
        mock_model_path = MagicMock()
        mock_model_path.exists.return_value = False
        mock_config_path = MagicMock()
        mock_config_path.exists.return_value = False
        mock_model_path.with_suffix.return_value = mock_config_path

        with patch("audio.tts._get_piper_model_path", return_value=mock_model_path):
            with patch("audio.tts._download_piper_model") as mock_download:
                _ensure_piper_model()
                mock_download.assert_called_once()


class TestPlayAudio:
    """Tests for audio playback."""

    def test_play_audio_calls_sounddevice(self):
        """play_audio should use sounddevice to play."""
        from audio.tts import play_audio
        import numpy as np

        # Create valid 16-bit PCM audio bytes
        audio_bytes = np.zeros(1000, dtype=np.int16).tobytes()

        with patch("sounddevice.play") as mock_play:
            with patch("sounddevice.wait") as mock_wait:
                play_audio(audio_bytes, sample_rate=22050)

        mock_play.assert_called_once()
        mock_wait.assert_called_once()
