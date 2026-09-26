# Architecture diagram (LikeC4)

Architecture-as-code model of **NLS Fine-tune (SciX)** — natural language →
ADS/SciX search query — rendered with [LikeC4](https://likec4.dev). The model is
the source of truth across [`spec.c4`](spec.c4) (element kinds, tags, deployment
node kinds), [`model.c4`](model.c4) (the system), and [`views.c4`](views.c4)
(structure, walkthrough, and risk views), with the deployment model in
[`deployment.c4`](deployment.c4). The narrative companions are the repo-root
[`README.md`](../README.md) and [`docs/current-state.md`](../docs/current-state.md).

The system translates a user's natural-language search into valid ADS query
syntax. Translation is **hybrid**: a deterministic NER + retrieval + template-
assembly pipeline serves queries first (~5ms, guaranteed-valid syntax), and a
fine-tuned **Qwen3-1.7B + LoRA** model
([adsabs/scix-nls-translator](https://huggingface.co/adsabs/scix-nls-translator))
is the fallback for the low-confidence tail. The repo also contains the data-
prep, training, and evaluation machinery that produces the model and the
gold/benchmark datasets it is measured against.

Every element `link`s to its source (`docker/…`, `packages/…`, `scripts/…`,
`data/…`) and, where helpful, to the doc that explains the rationale
([`docs/HYBRID_PIPELINE.md`](../docs/HYBRID_PIPELINE.md),
[`docs/current-state.md`](../docs/current-state.md)) — so any box in the
explorer is one click from the code and the reasoning behind it.

## Delivery state is tagged, not guessed

Every element carries a tag so **planned and research work renders distinctly
from what is already built** (legend in `spec.c4`):

| Tag | Meaning | Render |
|---|---|---|
| `#built` | code path exists and is exercised | solid |
| `#evolving` | built, but the contract / model / labels are still moving | solid |
| `#planned` | designed; not yet implemented (or v1 is a stub/heuristic) | **dashed, dimmed** |
| `#research` | speculative / exploratory track | **dashed, indigo** |

Planned / research items in the model: constrained decoding for the model
fallback (planned), the enrichment (SciBERT NER) label track, and the NER
annotation-review dashboard (research).

## Views

**Structure** — the static map:

| View | Scope |
|---|---|
| `index` | system landscape — `nls` in context of Nectar, ADS, Anthropic, OpenAI, HuggingFace, Colab, vocabulary sources |
| `nlsSystem` | the `nls` system decomposed into containers (built vs planned) |
| `serverContainer` | the inference server — routing layer, model runtime, telemetry log |
| `pipelineContainer` | the hybrid NER pipeline internals (NER → retrieval → assembly → constrain → resolver) |
| `dataPrepContainer` | the dataset generation agent — the 10-stage pipeline |
| `evalContainer` | the evaluation harness — benchmark & semantic-overlap |
| `apiContainer` | the review API — services & routers |
| `planned` | planned + research work, with built dependencies dimmed |
| `deployment` | where each piece runs — process & data boundaries (server :8001, API :8000, web :5173, Colab, external APIs) |

**Walkthrough flows** (dynamic / numbered-step views) — the narrative spine for
a design-review walkthrough:

| View | Flow |
|---|---|
| `translateFlow` | a translation request at runtime (Nectar → pipeline-first → model fallback → telemetry) |
| `buildDataset` | building the training dataset (synthetic + query-log + gold → validate → 90/10 split) |
| `trainFlow` | fine-tune & publish the LoRA adapter (Colab → HuggingFace → server loads it) |
| `evalFlow` | evaluating translation quality (benchmark + semantic-overlap via ADS → eval viewer) |

**Risk lens:**

| View | Scope |
|---|---|
| `risks` | the `#risk`-flagged elements with each open question stated in-box (model-conflation history, unpublished semantic-overlap numbers, "Sourcegraph" leftovers in the review app, dataset-service edit stubs) |

### Running the walkthrough

For a design review, present in this order: `index` → `nlsSystem` (orient on
structure) → the four walkthrough flows in sequence (what actually happens) →
`deployment` (where it runs) → `risks` (what to probe) → `planned` (what's next).
In `npx likec4 start`, the dynamic views animate step-by-step and each view's
notes panel carries the gotchas (the pipeline-first / model-fallback contract,
the off-machine Colab training boundary, the ADS-API dependency for the
target metric).

## Viewing & regenerating

```bash
# Interactive, hot-reloading explorer (recommended)
npx likec4 start architecture

# Re-export static PNGs (needs a one-time browser download:
#   npx playwright install chromium-headless-shell)
npx likec4 export png architecture -o architecture/exports

# Validate the model (strict — the source of truth for correctness)
npx likec4 validate architecture
```

### Viewing the interactive explorer over SSH (headless remote)

`likec4 start` serves a Vite dev server on `localhost:5173`. From a headless
remote, forward that port to your laptop and open it locally — three options,
easiest first:

1. **VS Code / Cursor Remote-SSH** — run `npx likec4 start architecture` in the
   integrated terminal; the editor auto-forwards 5173 and offers "Open in
   Browser". Nothing else to configure.
2. **SSH local port-forward** — on your laptop:
   ```bash
   ssh -N -L 5173:localhost:5173 user@remote   # leave running
   ```
   then on the remote `npx likec4 start architecture` and open
   <http://localhost:5173> locally. (Already in an SSH session? Add the tunnel
   without reconnecting: press `~C` then type `-L 5173:localhost:5173`.)
3. **Bind + reach directly** — `npx likec4 start architecture --listen 0.0.0.0`
   and browse to `http://<remote-ip>:5173` (only if that port is reachable /
   firewall-open; the tunnel in option 2 is safer).

No browser at all? Export the PNGs with `npx likec4 export png` — they need no
display, so `scp` them down or view inline if your terminal supports images.
