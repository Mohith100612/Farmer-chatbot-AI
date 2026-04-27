import os
from dotenv import load_dotenv

load_dotenv()


def configure_langsmith() -> None:
    """Set LangSmith env vars. Called once at app startup."""
    os.environ.setdefault("LANGCHAIN_TRACING_V2", os.getenv("LANGCHAIN_TRACING_V2", "true"))
    os.environ.setdefault("LANGCHAIN_ENDPOINT", os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"))
    os.environ.setdefault("LANGCHAIN_PROJECT", os.getenv("LANGCHAIN_PROJECT", "farm-ai"))

    api_key = os.getenv("LANGCHAIN_API_KEY", "")
    if api_key:
        os.environ["LANGCHAIN_API_KEY"] = api_key


GROQ_MODEL  = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
