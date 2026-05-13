import os
from pathlib import Path

from pydantic import BaseModel


class Settings(BaseModel):
    app_name: str = "Estatia"
    llm_provider: str = "openai"
    openai_api_key: str | None = None
    nvidia_api_key: str | None = None
    nvidia_base_url: str | None = None
    tavily_api_key: str | None = None
    fast_model: str = "gpt-5-nano"
    quality_model: str = "gpt-5-mini"
    log_level: str = "INFO"
    listing_mode: str = "playwright"
    browser_headless: bool = True
    scrape_timeout_ms: int = 20000
    search_results_limit: int = 5
    evaluation_threshold: float = 0.72
    max_retries: int = 1
    enable_news_agent: bool = True
    news_results_limit: int = 5
    enable_whatsapp_agent: bool = False


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _load_dotenv() -> None:
    env_path = Path(".env")
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        os.environ.setdefault(key, value)


_load_dotenv()

_provider = os.getenv("ESTATIA_LLM_PROVIDER", "openai").lower()
_default_fast_model = (
    os.getenv("NVIDIA_FAST_MODEL", "nvidia/llama-3.1-nemotron-nano-8b-v1")
    if _provider == "nvidia"
    else "gpt-5-nano"
)
_default_quality_model = (
    os.getenv("NVIDIA_QUALITY_MODEL", "nvidia/llama-3.3-nemotron-super-49b-v1.5")
    if _provider == "nvidia"
    else "gpt-5-mini"
)


settings = Settings(
    app_name=os.getenv("ESTATIA_APP_NAME", "Estatia"),
    llm_provider=_provider,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    nvidia_api_key=os.getenv("NVIDIA_API_KEY"),
    nvidia_base_url=os.getenv("NVIDIA_BASE_URL"),
    tavily_api_key=os.getenv("TAVILY_API_KEY"),
    fast_model=os.getenv("ESTATIA_FAST_MODEL", _default_fast_model),
    quality_model=os.getenv("ESTATIA_QUALITY_MODEL", _default_quality_model),
    log_level=os.getenv("ESTATIA_LOG_LEVEL", "INFO").upper(),
    listing_mode=os.getenv("ESTATIA_LISTING_MODE", "playwright"),
    browser_headless=_env_bool("ESTATIA_BROWSER_HEADLESS", True),
    scrape_timeout_ms=int(os.getenv("ESTATIA_SCRAPE_TIMEOUT_MS", "20000")),
    search_results_limit=int(os.getenv("ESTATIA_SEARCH_RESULTS_LIMIT", "5")),
    evaluation_threshold=float(os.getenv("ESTATIA_EVALUATION_THRESHOLD", "0.72")),
    max_retries=int(os.getenv("ESTATIA_MAX_RETRIES", "1")),
    enable_news_agent=_env_bool("ESTATIA_ENABLE_NEWS_AGENT", True),
    news_results_limit=int(os.getenv("ESTATIA_NEWS_RESULTS_LIMIT", "5")),
    enable_whatsapp_agent=_env_bool("ESTATIA_ENABLE_WHATSAPP_AGENT", False),
)
