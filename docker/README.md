# NLS Server - Local Development & Beta Deployment

Run the NLS (Natural Language Search) server locally: the hybrid pipeline
(deterministic intent extraction, retrieval and query assembly) with the
fine-tuned model as the low-confidence fallback.

## Quick Start

### Option 1: Local Python (Recommended for Mac)

```bash
# Install dependencies (torch/transformers/accelerate are needed only when the
# model loads; ROUTING_MODE=pipeline runs without them)
pip install -r docker/requirements.txt

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

Compose passes the server settings through from the host environment (see
[Environment Variables](#environment-variables)); unset ones take the server
defaults. `TYPESAFE_API_KEY` is read from the host shell and never written
into the compose file. Telemetry goes to the `nls-telemetry` volume mounted at
`/app/telemetry`, so set `TELEMETRY_LOG` to a file under that directory for
the log to survive the container.

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

If `pipeline_available` is false, `pipeline_import_error` says why (for
example a missing dependency). In `hybrid` mode the server then sends every
request to the model, so check this field after a deploy.

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

`docker/server.py` reads these at startup; its module docstring is the
authoritative list. An invalid combination fails at startup with a message
naming the variable.

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `adsabs/scix-nls-translator` | HuggingFace model to load |
| `DEVICE` | auto-detect | `cuda`, `mps`, or `cpu`. Unset means detect when the model loads (`cpu` when torch is not installed); `/health` shows `auto` until then |
| `PORT` | `8000` | Server port |
| `ROUTING_MODE` | `hybrid` | `hybrid`: pipeline first, model fallback on low confidence. `pipeline`: pipeline only, the model is never loaded. `model`: model only |
| `PIPELINE_CONFIDENCE_THRESHOLD` | `0.5` | Below this routing confidence the request goes to the model (in `hybrid`, when the model is loaded) |
| `TELEMETRY_LOG` | unset | JSONL file; one `request` row per request, plus one `intent_shadow` row per shadow run |
| `INTENT_BACKEND` | `regex` | Intent stage: `regex`, `jev` or `jev_gated` (see [Intent backends](#intent-backends)) |
| `TYPESAFE_API_KEY` | unset | TypeSafe System One key; required when `INTENT_BACKEND` is `jev`/`jev_gated` or `SHADOW_INTENT_BACKEND` is set |
| `JEV_TIMEOUT_S` | `2.0` | Wall-clock bound on each System One call, in seconds, including any wait for a free connection slot; must be positive |
| `JEV_CACHE_PATH` | `data/cache/jev_systemone.jsonl`; empty in the image and compose | Append-only JSONL cache of System One responses, relative to the working directory. Empty disables the file |
| `SHADOW_INTENT_BACKEND` | unset | `jev` or `jev_gated`: serve regex and run this backend off the request path. Needs `INTENT_BACKEND=regex`, `TELEMETRY_LOG`, `TYPESAFE_API_KEY`, and a `ROUTING_MODE` other than `model` |
| `GOLD_EXAMPLES_PATH` | repo `data/datasets/raw/gold_examples.json`; `/app/data/gold_examples.json` in the image | Few-shot examples for retrieval (read by `retrieval.py`) |

The JSONL cache is off in containers because it grows by one row per distinct
query without bound and is lost when the container is replaced. With the cache
off the Jev client keeps no in-process copy either, so memory stays flat. When
the cache is on and its file cannot be written, the failure is logged and the
answer is still served.

## Intent backends

The pipeline's first stage builds an IntentSpec (operator, enum filters,
names, years, topics). `INTENT_BACKEND` picks who makes the judgement calls:

- `regex`: the rules in `ner.py`, as shipped. No network calls.
- `jev`: Jev typed classifiers (TypeSafe System One, model `jev-1.13.0`)
  answer one request per query. They decide the operator, doctype, bibgroup,
  collection and the refereed, openaccess and eprint properties, plus:
  - **Recency.** "recent" or "latest" without an explicit year becomes a
    window (this year, last 2, 3, 5 or 10 years) ending at the request's
    reference year.
  - **Highly cited.** Adds `citation_count:[100 TO *]`. "Popular" and
    "trending" do not count.
  - **First author.** When names were found, puts the `^` on the first one
    only if the query asks for first-author papers.
  - **Topic.** When the regex found one topic phrase of at most six words,
    Jev picks which contiguous span of it is the subject (or none), so
    "recent asteroids" becomes `asteroids`.
  - **Facilities.** Only the facilities named in the query are offered as
    bibgroup options; with none named the question is not asked.
  - **Authors.** The regex names plus up to four capitalized words are
    offered, and Jev says for each whether it is a person, so "papers that
    cite Jarmak" becomes `citations(author:"Jarmak")`. Lowercase names not
    after "by" ("smith j") are not offered yet.

  Code still finds the candidate names, facilities, years and words; Jev
  chooses among them. This is the backend to serve.
  [reports/jev-intent-classifier-eval.md](../reports/jev-intent-classifier-eval.md)
  has the evidence (round 5 for authors and cost).
- `jev_gated`: regex first; Jev is called only when the regex finds no
  operator or its structural confidence is below 0.5. Because the gate
  skips Jev whenever the regex finds an operator, every Jev decision is
  skipped on those queries ("papers that cite Jarmak" keeps the regex
  reading, `citations(abs:jarmak)`). It saves little: on val and the
  held-out set the gate opens for 93% to 99% of queries.

**Reference year.** Relative years ("recent", "last 5 years") end at the year
on the prompt's `Date: YYYY-MM-DD` line, which Nectar sends. A malformed date
returns 400; a prompt with no `Date:` line uses the server's current year.

**Fallback.** A failed Jev call (HTTP error, timeout after `JEV_TIMEOUT_S`,
network error, or a response that breaks the contract) never fails the
request: the pipeline serves the regex intent and records the reason in
`debug_info.classifier_error` on `/pipeline` and in the `classifier_error`
telemetry field. `classifier_called` is true whenever Jev was consulted,
answered or not; `classifier_cached` is true when the answer came from the
cache, so billed calls are those with `classifier_called` true and
`classifier_cached` false.

**Timeout.** `JEV_TIMEOUT_S` bounds the whole call by wall clock. httpx's own
timeout applies per phase (connect, write, each socket read), so a response
that trickles in could run past it; the client stops waiting at the deadline
and serves the regex intent. The request handlers run in FastAPI's
threadpool, so a slow Jev or model call does not block `/health`.

**Routing confidence.** Without a Jev answer the routing confidence is the
structural score (0.9 when authors, an operator or years were extracted, 0.7
for constraints or three or more topic words, 0.5 for two, 0.3 otherwise).
When Jev answered it is `min(structural confidence, Jev operator
confidence)`, so the pipeline serves only when both clear
`PIPELINE_CONFIDENCE_THRESHOLD`. Both inputs appear in `debug_info` and
telemetry as `structural_confidence` and `classifier_operator_confidence`.

## Rolling out the Jev intent backend

1. **Shadow.** Keep serving regex and log what Jev would have decided:

   ```bash
   export TYPESAFE_API_KEY=...   # from your secret store, not a file in the repo
   INTENT_BACKEND=regex SHADOW_INTENT_BACKEND=jev \
     TELEMETRY_LOG=/app/telemetry/nls.jsonl \
     docker compose -f docker/docker-compose.yml up nls-server
   ```

   Shadow runs execute on a two-thread pool off the request path, so they
   add no latency to the response. At most 32 may be pending; further ones are
   skipped with a log line. Shadow failures are logged, never raised.

2. **Review.** Copy the log out of the volume and summarize it:

   ```bash
   docker compose -f docker/docker-compose.yml cp nls-server:/app/telemetry/nls.jsonl .
   uv run python scripts/summarize_intent_shadow.py nls.jsonl
   ```

   The summary counts rows, Jev calls (and how many were billed rather than
   cache answers), errors, and disagreements per field, and lists each
   disagreeing query with both values. Disagreements are counted only for
   requests the pipeline served; when a request fell back to the model the
   regex intent never reached the user, so those rows are counted separately. It does not say which
   side is right; read the disagreements.

3. **Serve.** Switch the served intent and drop the shadow:
   `INTENT_BACKEND=jev`, `SHADOW_INTENT_BACKEND` unset. Watch
   `classifier_error` in the request rows; a Jev outage shows up there while
   requests keep being served from the regex intent.

## Telemetry rows

With `TELEMETRY_LOG` set, each line is one JSON object. Rows of both types
from the same request share `request_id`.

`record_type: "request"`, one per request:

| Field | Meaning |
|-------|---------|
| `timestamp`, `request_id` | UTC ISO time; random hex id |
| `nl_query`, `generated_query` | Input text and the served ADS query |
| `path` | `pipeline` or `model` |
| `confidence` | Routing confidence of the pipeline result; 0.0 when the model served |
| `pipeline_confidence` | The pipeline's routing confidence whenever it ran, including the value that fell below the threshold on a model fallback |
| `fallback_reason` | Why the model served it, or why a low-confidence pipeline result was served anyway |
| `latency_ms` | Time spent in the path that served |
| `routing_mode`, `intent_backend` | Server settings at the time |
| `classifier_called`, `classifier_cached`, `classifier_error` | Whether Jev was consulted, whether the answer came from the cache (not billed), and the failure reason |
| `structural_confidence`, `classifier_operator_confidence` | The two inputs to the routing confidence |

`record_type: "intent_shadow"`, one per shadow run:

| Field | Meaning |
|-------|---------|
| `timestamp`, `request_id` | As above; `request_id` matches the request row |
| `nl_query` | Input text |
| `served_backend`, `shadow_backend` | `regex` and the shadow backend |
| `served_path` | `pipeline` when the regex intent was served, `model` when the request fell back to the model |
| `served_intent`, `shadow_intent` | Both IntentSpecs as dicts |
| `disagreements`, `disagree` | Compared fields that differ (`operator`, `doctype`, `bibgroup`, `collection`, `property`, `year_from`, `year_to`, `free_text_terms`, `first_author`, `min_citations`) and whether any do |
| `classifier_called`, `classifier_cached`, `classifier_error`, `classifier_operator_confidence` | As in request rows, for the shadow run |
| `shadow_latency_ms` | Duration of the shadow intent run |

## Performance

| Device | Latency | Notes |
|--------|---------|-------|
| A10G (vLLM) | ~50ms | Production target |
| M4 Max (MPS) | ~500ms | Local development |
| T4 (CUDA) | ~200ms | Colab/cloud GPU |
| CPU | ~2000ms | Fallback only |
