"""
tracing.py — Optional Phoenix tracing.

Set TRACING_ENABLED=true to activate.
"""

import os


def init_tracing(project: str = "voice-research-agent"):
    if not os.getenv("TRACING_ENABLED"):
        return
    try:
        from phoenix.otel import register
        from openinference.instrumentation.langchain import LangChainInstrumentor
        tracer = register(
            project_name=project,
            endpoint=os.getenv("PHOENIX_ENDPOINT", "http://localhost:6006/v1/traces")
        )
        LangChainInstrumentor().instrument(tracer_provider=tracer)
    except ImportError:
        pass  # tracing deps not installed
