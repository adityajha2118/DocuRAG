"""
DocuRAG — Build Vector Index (Milestone 4).

Reads processed JSON documents, chunks them, computes local embeddings
with sentence-transformers, and upserts into a persistent ChromaDB index.

Idempotent: chunks already present in the collection are skipped on
subsequent runs.  Use --force to rebuild from scratch.
"""

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

import chromadb

from src.config import (
    PROCESSED_DIR,
    CHROMA_PATH,
    CHROMA_COLLECTION_NAME,
    CHUNKS_PATH,
    EMBEDDING_BATCH_SIZE,
)
from src.chunking.chunker import process_page_into_chunks, DocumentChunk
from src.embeddings.embedder import embed_texts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("build_index")


# ── helpers ──────────────────────────────────────────────────────────────
def load_processed_pages() -> list[dict]:
    """Load all processed JSON files from data/processed/."""
    pages: list[dict] = []
    if not PROCESSED_DIR.exists():
        logger.warning("Processed dir not found: %s", PROCESSED_DIR)
        return pages

    for json_file in PROCESSED_DIR.rglob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as fh:
                pages.append(json.load(fh))
        except Exception as exc:
            logger.error("Error reading %s: %s", json_file, exc)

    return pages


# ── main entry point ─────────────────────────────────────────────────────
def build_index(force: bool = False) -> None:
    """Chunk all processed documents and build/update the ChromaDB index."""

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    if force:
        # Drop and recreate
        try:
            client.delete_collection(name=CHROMA_COLLECTION_NAME)
            logger.info("Force rebuild: deleted existing collection.")
        except Exception:
            pass
        collection = client.create_collection(name=CHROMA_COLLECTION_NAME)
    else:
        collection = client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

    existing_count = collection.count()
    if existing_count > 0 and not force:
        logger.info(
            "Collection '%s' already has %d items. "
            "Will skip already-indexed chunks (idempotent upsert).",
            CHROMA_COLLECTION_NAME,
            existing_count,
        )

    # ── Load & chunk ─────────────────────────────────────────────────────
    logger.info("Loading processed pages …")
    pages = load_processed_pages()
    if not pages:
        logger.error("No pages to index.")
        return

    logger.info("Loaded %d pages. Chunking …", len(pages))
    all_chunks: list[DocumentChunk] = []
    for page in pages:
        all_chunks.extend(process_page_into_chunks(page))

    logger.info("Generated %d chunks total.", len(all_chunks))

    # ── Determine which chunks are new ───────────────────────────────────
    all_ids = [c.chunk_id for c in all_chunks]

    if existing_count > 0 and not force:
        # Query existing IDs in batches (Chroma get() limit)
        existing_ids: set[str] = set()
        for i in range(0, len(all_ids), 1000):
            batch_ids = all_ids[i : i + 1000]
            result = collection.get(ids=batch_ids, include=[])
            existing_ids.update(result["ids"])

        new_chunks = [c for c in all_chunks if c.chunk_id not in existing_ids]
        logger.info(
            "Skipping %d already-indexed chunks; %d new chunks to embed.",
            len(all_chunks) - len(new_chunks),
            len(new_chunks),
        )
    else:
        new_chunks = all_chunks

    # ── Save ALL chunks to chunks.jsonl (always the full set) ────────────
    logger.info("Saving chunks to %s …", CHUNKS_PATH)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as fh:
        for chunk in all_chunks:
            chunk_dict = {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "source": chunk.source,
                "url": chunk.url,
                "title": chunk.title,
                "section_heading": chunk.section_heading,
                "content": chunk.content,
                "token_count": chunk.token_count,
            }
            fh.write(json.dumps(chunk_dict, ensure_ascii=False) + "\n")

    if not new_chunks:
        logger.info("Nothing new to embed — index is up to date.")
        return

    # ── Embed & upsert in batches ────────────────────────────────────────
    logger.info("Embedding %d new chunks and upserting into ChromaDB …", len(new_chunks))

    for i in range(0, len(new_chunks), EMBEDDING_BATCH_SIZE):
        batch = new_chunks[i : i + EMBEDDING_BATCH_SIZE]
        texts = [c.content for c in batch]
        ids = [c.chunk_id for c in batch]
        metadatas = [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "source": c.source,
                "url": c.url,
                "title": c.title,
                "section_heading": c.section_heading,
            }
            for c in batch
        ]

        batch_num = i // EMBEDDING_BATCH_SIZE + 1
        logger.info("  Batch %d  (%d chunks) …", batch_num, len(batch))
        embeddings = embed_texts(texts, batch_size=EMBEDDING_BATCH_SIZE)

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=texts,
        )

    final_count = collection.count()
    logger.info(
        "Index built — %d items in collection '%s' at %s",
        final_count,
        CHROMA_COLLECTION_NAME,
        CHROMA_PATH,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build the DocuRAG vector index.")
    ap.add_argument(
        "--force",
        action="store_true",
        help="Drop and rebuild the index from scratch.",
    )
    args = ap.parse_args()
    build_index(force=args.force)
