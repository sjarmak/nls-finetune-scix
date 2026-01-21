# NLS Query API

FastAPI backend for the NLS Query Fine-tuning Playground.

## Development

```bash
# Install dependencies
uv sync

# Run server
uv run uvicorn api.main:app --reload
```

## API Endpoints

- `GET /api/health` - Health check
- `POST /api/inference/generate` - Generate a Sourcegraph query
- `POST /api/inference/compare` - Compare multiple models
- `GET /api/models` - List available models
- `GET /api/datasets/stats` - Get dataset statistics
- `GET /api/datasets/examples` - List examples
- `POST /api/evaluation/run` - Run evaluation
