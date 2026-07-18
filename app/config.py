from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    environment: str = "development"
    redis_url: str = "redis://localhost:6379/0"
    api_service_url: str = "http://localhost:8000"

    # LLM
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-5"

    # Market data
    news_api_key: str = ""

    # Signal thresholds
    confidence_threshold: float = 0.65
    min_reward_risk: float = 1.5

    # FinBERT
    finbert_model: str = "ProsusAI/finbert"

    # OANDA Forex (demo account)
    oanda_api_key: str = ""
    oanda_account_id: str = ""
    oanda_env: str = "practice"  # "practice" = demo, "live" = live

    class Config:
        env_file = ".env"


settings = Settings()
