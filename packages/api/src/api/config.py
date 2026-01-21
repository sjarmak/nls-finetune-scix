"""Application configuration from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # API
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    cors_origins: str = "http://localhost:5173"

    # Modal
    modal_token_id: str = ""
    modal_token_secret: str = ""
    # Default to deployed endpoint for Playground functionality
    modal_inference_endpoint: str = (
        "https://sourcegraph--nls-finetune-serve-nlsquerymodel-query.modal.run"
    )

    # OpenAI
    openai_api_key: str = ""

    # Anthropic
    anthropic_api_key: str = ""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
