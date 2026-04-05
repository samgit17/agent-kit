import os

REQUIRED_ENV_VARS = ["LLM_API_KEY", "LLM_MODEL"]
MAX_ITERATIONS = int(os.getenv("WIKI_MAX_ITERATIONS", "10"))
WIKI_DIR = os.getenv("WIKI_DIR", "wiki")
SOURCES_DIR = os.getenv("SOURCES_DIR", "sources")


def validate_env() -> None:
    missing = [v for v in REQUIRED_ENV_VARS if not os.getenv(v)]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            f"Copy .env.example to .env and fill in the values."
        )
