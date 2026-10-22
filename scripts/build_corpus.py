#!/usr/bin/env python
"""
scripts/build_corpus.py — crawl, clean, parse, and persist documentation.

Usage
-----
    # Crawl FastAPI only (max 40 pages, the default)
    python scripts/build_corpus.py --sources FastAPI

    # Crawl all 6 sources
    python scripts/build_corpus.py

    # Quick test with 5 pages
    python scripts/build_corpus.py --sources FastAPI --max-pages 5
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

# Ensure project root is on sys.path so ``from src…`` works.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR, RAW_DIR, PROCESSED_DIR, METADATA_PATH, MAX_PAGES_PER_SOURCE
from src.ingestion.sources import SOURCES, get_source_by_name
from src.ingestion.crawler import crawl_source
from src.ingestion.parser import parse_page

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("build_corpus")


# ── Helpers ──────────────────────────────────────────────────────────────

def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _ensure_dirs(source_slug: str) -> tuple[Path, Path]:
    """Create raw/ and processed/ sub-dirs for *source_slug*; return paths."""
    raw_dir = RAW_DIR / source_slug
    proc_dir = PROCESSED_DIR / source_slug
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir, proc_dir


def _save_raw_html(html: str, raw_dir: Path, doc_id: str) -> Path:
    path = raw_dir / f"{doc_id}.html"
    path.write_text(html, encoding="utf-8")
    return path


def _save_processed(page_dict: dict, proc_dir: Path, doc_id: str) -> Path:
    path = proc_dir / f"{doc_id}.json"
    path.write_text(json.dumps(page_dict, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    return path


def _page_to_metadata(page) -> dict:
    """Extract the metadata-only fields for metadata.jsonl."""
    return {
        "doc_id": page.doc_id,
        "source_name": page.source_name,
        "url": page.url,
        "canonical_url": page.canonical_url,
        "title": page.title,
        "sections": [s.heading for s in page.sections if s.heading],
        "fetch_timestamp": page.fetch_timestamp,
        "content_hash": page.content_hash,
    }


def _page_to_full_dict(page) -> dict:
    """Serialise a ParsedPage to a dict for JSON storage."""
    return {
        "doc_id": page.doc_id,
        "source_name": page.source_name,
        "url": page.url,
        "canonical_url": page.canonical_url,
        "title": page.title,
        "sections": [
            {"heading": s.heading, "heading_level": s.heading_level,
             "content": s.content}
            for s in page.sections
        ],
        "full_text": page.full_text,
        "fetch_timestamp": page.fetch_timestamp,
        "content_hash": page.content_hash,
    }


# ── Main logic ───────────────────────────────────────────────────────────

def build_corpus(
    source_names: list[str] | None = None,
    max_pages: int | None = None,
) -> list[dict]:
    """Crawl, parse, and persist documentation.  Returns metadata dicts."""
    if source_names:
        sources = []
        for name in source_names:
            src = get_source_by_name(name)
            if src is None:
                logger.error("Unknown source: %s", name)
                sys.exit(1)
            sources.append(src)
    else:
        sources = list(SOURCES)

    all_metadata: list[dict] = []

    for source in sources:
        slug = _slugify(source.name)
        raw_dir, proc_dir = _ensure_dirs(slug)

        logger.info("=" * 60)
        logger.info("CRAWLING: %s", source.name)
        logger.info("=" * 60)

        results = crawl_source(source, max_pages=max_pages)

        if not results:
            logger.warning("No pages fetched for %s — skipping.", source.name)
            continue

        logger.info("Parsing %d pages from %s …", len(results), source.name)
        for cr in results:
            page = parse_page(
                url=cr.url,
                html=cr.html,
                source_name=cr.source_name,
            )

            # Persist raw HTML
            _save_raw_html(cr.html, raw_dir, page.doc_id)

            # Persist processed JSON
            _save_processed(_page_to_full_dict(page), proc_dir, page.doc_id)

            meta = _page_to_metadata(page)
            all_metadata.append(meta)

            logger.info(
                "  ✓ %s  — %d sections, %d chars",
                page.title[:60], len(page.sections), len(page.full_text),
            )

    # Write metadata.jsonl (overwrite — caller controls which sources to include)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with METADATA_PATH.open("w", encoding="utf-8") as f:
        for meta in all_metadata:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    logger.info("=" * 60)
    logger.info("DONE — %d pages across %d source(s).", len(all_metadata),
                len(sources))
    logger.info("metadata → %s", METADATA_PATH)
    logger.info("=" * 60)

    return all_metadata


# ── CLI ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Build the DocuRAG corpus.")
    parser.add_argument(
        "--sources", nargs="+", default=None,
        help="Source names to crawl (default: all). E.g. --sources FastAPI PyTorch",
    )
    parser.add_argument(
        "--max-pages", type=int, default=None,
        help=f"Max pages per source (default: {MAX_PAGES_PER_SOURCE}).",
    )
    args = parser.parse_args()
    build_corpus(source_names=args.sources, max_pages=args.max_pages)


if __name__ == "__main__":
    main()
