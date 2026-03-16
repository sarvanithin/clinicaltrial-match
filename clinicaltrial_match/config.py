"""
clinicaltrial-match configuration management.

Config is loaded from ~/.clinicaltrial_match/config.json and can be overridden via
environment variables with the CTM_ prefix and __ as nested delimiter.

Example env overrides:
    CTM_TRIALS__PAGE_SIZE=100
    CTM_CLAUDE__FAST_MODEL=claude-haiku-4-5-20251001
    CTM_EMBEDDING__USE_FAISS=true
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TrialsConfig(BaseModel):
    base_url: str = "https://clinicaltrials.gov/api/v2"
    page_size: int = 100
    cache_ttl_seconds: int = 86400
    parse_concurrency: int = 5
    max_trials_per_sync: int = 1000
    request_timeout_seconds: float = 30.0


class EmbeddingConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 64
    use_faiss: bool = False
    faiss_rebuild_threshold: int = 5000


class ClaudeConfig(BaseModel):
    api_key_env: str = "ANTHROPIC_API_KEY"
    fast_model: str = "claude-haiku-4-5-20251001"
    reasoning_model: str = "claude-sonnet-4-6-20251101"
    timeout_seconds: float = 30.0
    max_retries: int = 3
    max_tokens: int = 2048


class DatabaseConfig(BaseModel):
    path: str = "~/.clinicaltrial_match/data.db"
    diagnosis_cache_ttl_seconds: int = 604800  # 7 days


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = ["*"]
    log_level: str = "INFO"
    log_requests: bool = True


class CTMConfig(BaseSettings):
    trials: TrialsConfig = Field(default_factory=TrialsConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    claude: ClaudeConfig = Field(default_factory=ClaudeConfig)
    db: DatabaseConfig = Field(default_factory=DatabaseConfig)
    api: APIConfig = Field(default_factory=APIConfig)

    model_config = SettingsConfigDict(
        env_prefix="CTM_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    @classmethod
    def load(cls, path: Path | None = None) -> CTMConfig:
        config_path = path or _default_config_path()
        if config_path.exists():
            raw = json.loads(config_path.read_text())
            return cls.model_validate(raw)
        return cls()

    def save(self, path: Path | None = None) -> None:
        config_path = path or _default_config_path()
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(self.model_dump_json(indent=2, exclude_none=True))


def _default_config_path() -> Path:
    import os
    env_path = os.environ.get("CTM_CONFIG")
    if env_path:
        return Path(env_path)
    return Path.home() / ".clinicaltrial_match" / "config.json"


def _ensure_data_dir() -> None:
    data_dir = Path.home() / ".clinicaltrial_match"
    data_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_config() -> CTMConfig:
    _ensure_data_dir()
    return CTMConfig.load()


def reset_config_cache() -> None:
    get_config.cache_clear()
