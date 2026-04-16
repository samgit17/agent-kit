"""
cli.py — Command-line interface for voice research agent.

Usage:
    voice-research "What are the latest AI developments?"
    voice-research --audio question.wav
    voice-research --speak
    voice-research --local --speak
"""

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="voice-research",
        description="Voice-driven research agent — ask questions by voice, get researched answers.",
    )
    
    parser.add_argument(
        "query",
        nargs="?",
        help="Research question (text). If omitted, uses microphone input.",
    )
    
    parser.add_argument(
        "--audio",
        metavar="PATH",
        help="Path to audio file (WAV/MP3) instead of microphone.",
    )
    
    parser.add_argument(
        "--speak",
        action="store_true",
        help="Read the answer aloud via TTS.",
    )
    
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local models (faster-whisper + Piper) instead of OpenAI APIs.",
    )
    
    return parser.parse_args()
