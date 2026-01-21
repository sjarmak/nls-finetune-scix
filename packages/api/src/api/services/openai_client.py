"""OpenAI inference client for GPT-4o-mini comparison."""

import json
import time

from openai import AsyncOpenAI

from api.config import settings
from api.models.inference import InferenceResponse

# System prompt for GPT-4o-mini (condensed version)
SYSTEM_PROMPT = """Convert natural language to Sourcegraph query. Output JSON: {"query": "..."}

Rules:
- Repo: repo:^github.com/org/repo$ (no escaping dots in domain)
- Time: Convert "last week" to after:YYYY-MM-DD
- Language: Use lang:python, lang:go, etc.
- type:commit for history, type:diff for added/removed, type:symbol for definitions
- Don't repeat repo name in search terms if query matches repo"""


async def openai_inference(
    query: str,
    date: str,
) -> InferenceResponse:
    """Call OpenAI GPT-4o-mini for comparison."""
    if not settings.openai_api_key:
        return InferenceResponse(
            sourcegraph_query="[OPENAI_API_KEY not configured]",
            model_id="gpt-4o-mini",
            latency_ms=0,
        )

    client = AsyncOpenAI(api_key=settings.openai_api_key)

    user_content = f"Query: {query}\nDate: {date}"

    start = time.time()

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.1,
        max_tokens=256,
    )

    latency_ms = (time.time() - start) * 1000

    content = response.choices[0].message.content or ""

    # Parse JSON from response
    try:
        data = json.loads(content)
        sourcegraph_query = data.get("query", content)
    except json.JSONDecodeError:
        sourcegraph_query = content

    return InferenceResponse(
        sourcegraph_query=sourcegraph_query,
        model_id="gpt-4o-mini",
        latency_ms=latency_ms,
    )
