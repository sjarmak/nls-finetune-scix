"""Model configuration API routes."""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/models", tags=["models"])


class ModelConfig(BaseModel):
    """Configuration for a model."""

    id: str
    name: str
    provider: str  # "local", "openai", "anthropic"
    description: str
    is_available: bool = True


# Model configurations - can be expanded via config file later
AVAILABLE_MODELS = [
    ModelConfig(
        id="fine-tuned",
        name="Fine-tuned Qwen3-1.7B",
        provider="local",
        description="Fine-tuned model for Sourcegraph queries",
    ),
    ModelConfig(
        id="gpt-4o-mini",
        name="GPT-4o Mini",
        provider="openai",
        description="OpenAI GPT-4o Mini baseline",
    ),
    ModelConfig(
        id="base-qwen",
        name="Base Qwen3-1.7B",
        provider="local",
        description="Base model without fine-tuning (for comparison)",
    ),
]


@router.get("/", response_model=list[ModelConfig])
async def list_models() -> list[ModelConfig]:
    """List available models for inference."""
    return AVAILABLE_MODELS


@router.get("/{model_id}", response_model=ModelConfig)
async def get_model(model_id: str) -> ModelConfig:
    """Get a specific model configuration."""
    for model in AVAILABLE_MODELS:
        if model.id == model_id:
            return model
    return ModelConfig(
        id=model_id,
        name=model_id,
        provider="unknown",
        description="Unknown model",
        is_available=False,
    )
