#!/usr/bin/env python3
"""
run.py — Entry point for voice research agent.

Flow:
1. Get input (mic / file / text arg)
2. Transcribe if audio
3. Run research pipeline
4. Optionally speak the answer
"""

import sys
from dotenv import load_dotenv
load_dotenv()

from cli import parse_args
from log import console, log
from tracing import init_tracing


def main():
    init_tracing()
    args = parse_args()
    
    # Step 1: Get the query
    if args.query:
        # Text query provided directly
        query = args.query
        log("[input]", f"Query: {query}")
    
    elif args.audio:
        # Audio file provided
        from audio.recorder import record_from_file
        from audio.stt import transcribe
        
        log("[input]", f"Reading audio from: {args.audio}")
        audio_bytes = record_from_file(args.audio)
        
        if not audio_bytes:
            console.print("[red]Failed to read audio file.[/red]")
            sys.exit(1)
        
        query = transcribe(audio_bytes, local=args.local)
        log("[input]", f"Transcribed: {query}")
    
    else:
        # Record from microphone
        from audio.recorder import record_push_to_talk
        from audio.stt import transcribe
        
        audio_bytes = record_push_to_talk()
        
        if not audio_bytes:
            console.print("[red]No audio recorded.[/red]")
            sys.exit(1)
        
        query = transcribe(audio_bytes, local=args.local)
        log("[input]", f"Transcribed: {query}")
    
    if not query.strip():
        console.print("[red]No query provided.[/red]")
        sys.exit(1)
    
    # Step 2: Run research
    from graph import run_research
    
    console.print()
    report = run_research(query)
    
    # Step 3: Output
    console.print()
    console.print(report)
    
    # Step 4: Optionally speak
    if args.speak:
        from audio.tts import speak, play_audio
        
        # Strip markdown formatting for cleaner speech
        clean_text = _strip_markdown(report)
        
        audio_bytes = speak(clean_text, local=args.local)
        
        if audio_bytes:
            sample_rate = 22050 if args.local else 24000  # Piper vs OpenAI
            play_audio(audio_bytes, sample_rate=sample_rate)


def _strip_markdown(text: str) -> str:
    """Remove markdown formatting for TTS."""
    import re
    
    # Remove horizontal rules
    text = re.sub(r"^---+$", "", text, flags=re.MULTILINE)
    
    # Remove italics markers
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    
    # Remove bold markers
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    
    # Remove citation numbers like [1]
    text = re.sub(r"\[\d+\]", "", text)
    
    # Clean up extra whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    return text.strip()


if __name__ == "__main__":
    main()
