from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    environment: str = "development"

    # AWS
    aws_region: str = "ap-south-1"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""

    # SQS — replaces Redis pub/sub
    sqs_signals_queue_url: str = ""          # signals:new equivalent
    sqs_tasks_queue_url: str = ""            # on-demand generate_single tasks

    # Bedrock Mantle — OpenAI-compatible endpoint (ap-south-1)
    # Available: deepseek.v3.2 | mistral.mistral-large-3-675b-instruct | qwen.qwen3-235b-a22b-2507
    bedrock_api_key: str = ""
    bedrock_model_id: str = "deepseek.v3.2"
    bedrock_base_url: str = "https://bedrock-mantle.ap-south-1.api.aws/v1"

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
    oanda_env: str = "practice"

    class Config:
        env_file = ".env"


settings = Settings()
