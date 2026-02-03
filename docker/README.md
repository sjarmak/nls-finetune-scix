# NLS Server - Local Development & Beta Deployment

Run the NLS (Natural Language Search) model locally.

## Quick Start

### Option 1: Local Python (Recommended for Mac)

```bash
# Install dependencies
pip install torch transformers accelerate fastapi uvicorn

# Run server (auto-detects MPS/CUDA/CPU)
./docker/run_local.sh

# Or specify device
./docker/run_local.sh mps   # Apple Silicon
./docker/run_local.sh cuda  # NVIDIA GPU
./docker/run_local.sh cpu   # CPU only
```

### Option 2: Docker (GPU)

```bash
# Build
docker build -t nls-server -f docker/Dockerfile .

# Run with GPU
docker run --gpus all -p 8000:8000 nls-server
```

### Option 3: Docker Compose

```bash
# GPU version
docker-compose -f docker/docker-compose.yml up nls-server

# CPU version (slower)
docker-compose -f docker/docker-compose.yml up nls-server-cpu
```

## Configure Nectar

Update `~/ads-dev/nectar/.env.local`:

```bash
# Enable NL Search
NEXT_PUBLIC_NL_SEARCH=enabled

# Point to local server
NL_SEARCH_PIPELINE_ENDPOINT=http://localhost:8000
NL_SEARCH_VLLM_ENDPOINT=http://localhost:8000/v1/chat/completions
```

Then restart nectar:
```bash
cd ~/ads-dev/nectar && pnpm dev
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/v1/models` | GET | List models (OpenAI-compatible) |
| `/v1/chat/completions` | POST | vLLM-compatible chat endpoint |
| `/pipeline` | POST | Hybrid NER pipeline endpoint |

### Example Requests

**Health check:**
```bash
curl http://localhost:8000/health
```

**Generate query (vLLM style):**
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llm",
    "messages": [
      {"role": "user", "content": "Query: papers about exoplanets from 2023\nDate: 2026-01-23"}
    ],
    "max_tokens": 128
  }'
```

**Generate query (pipeline):**
```bash
curl -X POST http://localhost:8000/pipeline \
  -H "Content-Type: application/json" \
  -d '{
    "model": "pipeline",
    "messages": [
      {"role": "system", "content": "Convert natural language to ADS query."},
      {"role": "user", "content": "Query: highly cited dark matter papers\nDate: 2026-01-23"}
    ]
  }'
```

## Test Server

```bash
# Start server first, then in another terminal:
pip install requests
python docker/test_server.py
```

## Beta Deployment

For beta testing with other users:

1. **Build and push Docker image:**
   ```bash
   docker build -t your-registry/nls-server:beta -f docker/Dockerfile .
   docker push your-registry/nls-server:beta
   ```

2. **Deploy on any server with Docker:**
   ```bash
   docker run -d --gpus all -p 8000:8000 --name nls-server your-registry/nls-server:beta
   ```

3. **Update nectar environment:**
   ```bash
   NL_SEARCH_PIPELINE_ENDPOINT=http://your-server:8000
   NL_SEARCH_VLLM_ENDPOINT=http://your-server:8000/v1/chat/completions
   ```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `adsabs/scix-nls-translator` | HuggingFace model to load |
| `DEVICE` | auto-detect | `cuda`, `mps`, or `cpu` |
| `PORT` | `8000` | Server port |

## Performance

| Device | Latency | Notes |
|--------|---------|-------|
| A10G (vLLM) | ~50ms | Production target |
| M4 Max (MPS) | ~500ms | Local development |
| T4 (CUDA) | ~200ms | Colab/cloud GPU |
| CPU | ~2000ms | Fallback only |
