import os
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor

def init_tracing(project: str):
    tracer = register(
        project_name=project,
        endpoint=os.getenv("PHOENIX_ENDPOINT", "http://localhost:6006/v1/traces")
    )
    LangChainInstrumentor().instrument(tracer_provider=tracer)