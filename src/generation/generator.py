"""
DocuRAG — LLM answer generation with response caching.

Builds the prompt from retrieved evidence chunks, calls the LLM with an
explicit max_tokens cap, and caches responses keyed by a hash of
(query + chunk IDs) so repeated identical requests never cost API money.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import yaml

from src.config import LLM_MODEL, LLM_API_KEY, LLM_BASE_URL, LLM_MAX_TOKENS, PROJECT_ROOT

logger = logging.getLogger(__name__)

# ── Prompt template ──────────────────────────────────────────────────────
_PROMPT_PATH = PROJECT_ROOT / "prompts" / "answer_prompt_v1.yaml"

def _load_prompt_template() -> dict[str, str]:
    with open(_PROMPT_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)

# ── Local response cache ────────────────────────────────────────────────
_CACHE_DIR = PROJECT_ROOT / ".cache" / "llm_responses"

def _cache_key(query: str, chunk_ids: list[str]) -> str:
    """Deterministic hash of (query + sorted chunk IDs)."""
    blob = json.dumps({"q": query, "ids": sorted(chunk_ids)}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def _get_cached(key: str) -> str | None:
    path = _CACHE_DIR / f"{key}.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("answer")
    return None


def _set_cache(key: str, query: str, chunk_ids: list[str], answer: str) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _CACHE_DIR / f"{key}.json"
    path.write_text(
        json.dumps({
            "query": query,
            "chunk_ids": chunk_ids,
            "answer": answer,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ── Evidence formatting ─────────────────────────────────────────────────
def _format_evidence(chunks: list[dict]) -> str:
    """Format a list of chunk dicts into the evidence block for the prompt."""
    parts: list[str] = []
    for c in chunks:
        chunk_id = c["chunk_id"]
        title = c.get("title", "")
        section = c.get("section_heading", "")
        url = c.get("url", "")
        content = c["content"]
        header = f"[{chunk_id}]  {title}"
        if section:
            header += f" > {section}"
        header += f"\nURL: {url}"
        parts.append(f"{header}\n\n{content}")
    return "\n\n---\n\n".join(parts)


# ── LLM call ────────────────────────────────────────────────────────────
def _call_openai(system_msg: str, user_msg: str) -> str:
    """Call OpenAI-compatible API."""
    import openai

    client = openai.OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=LLM_MAX_TOKENS,
        temperature=0.1,
    )
    return response.choices[0].message.content


def generate_answer(
    query: str,
    retrieved_chunks: list[dict],
    *,
    use_cache: bool = True,
) -> tuple[str, str]:
    """Generate an answer for *query* using the top retrieved chunks.

    Parameters
    ----------
    query : str
        The user question.
    retrieved_chunks : list[dict]
        Top-k chunks (each must have chunk_id, content, title, section_heading, url).
    use_cache : bool
        If True (default), check / update the local response cache.

    Returns
    -------
    tuple[str, str]
        The LLM-generated answer text with inline [chunk_id] citations, and the prompt version used.
    """
    chunk_ids = [c["chunk_id"] for c in retrieved_chunks]
    prompt_version = _PROMPT_PATH.stem

    # ── Cache check ──────────────────────────────────────────────────
    if use_cache:
        key = _cache_key(query, chunk_ids)
        cached = _get_cached(key)
        if cached is not None:
            logger.info("Cache hit for query: %s", query[:60])
            return cached, prompt_version

    # ── Build prompt ─────────────────────────────────────────────────
    tmpl = _load_prompt_template()
    evidence_text = _format_evidence(retrieved_chunks)
    system_msg = tmpl["system"]
    user_msg = tmpl["user"].format(evidence=evidence_text, question=query)

    # ── Call LLM ─────────────────────────────────────────────────────
    logger.info("Calling LLM (%s) for query: %s", LLM_MODEL, query[:60])
    answer = _call_openai(system_msg, user_msg)

    # ── Cache store ──────────────────────────────────────────────────
    if use_cache:
        _set_cache(key, query, chunk_ids, answer)
        logger.info("Response cached.")

    return answer, prompt_version
