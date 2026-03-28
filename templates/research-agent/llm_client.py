"""
llm_client.py — Provider-agnostic LLM factory.
Reads LLM_PROVIDER from environment and returns the appropriate LangChain chat model.
Supports: openai | ollama
"""

import os
from langchain_core.language_models import BaseChatModel


def get_llm() -> BaseChatModel:
    provider = os.getenv("LLM_PROVIDER", "openai").lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.2,
        )

    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            #format="json",       # constrains token sampler to valid JSON only —
                                 # also suppresses <think> blocks since they aren't valid JSON
            temperature=0.2,
            num_ctx=8192,
            num_predict=int(os.getenv("OLLAMA_MAX_TOKENS", "2048")),
            timeout=int(os.getenv("OLLAMA_TIMEOUT", "120")),
        )

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER='{provider}'. Set to 'openai' or 'ollama' in .env"
        )
