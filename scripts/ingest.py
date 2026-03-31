"""Ingestion pipeline: EDGAR search → download → extract → filter → chunk → embed → index.

Usage:
    python scripts/ingest.py                          # defaults: 50 agreements, 2020-2025
    python scripts/ingest.py --max-results 20         # fewer for quick test
    python scripts/ingest.py --start-date 2023-01-01  # narrow date range
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import date
from typing import Any

from sec_rag.config import Settings
from sec_rag.embedding import ChunkEmbedder, SparseEncoder
from sec_rag.ingestion.chunker import chunk_document
from sec_rag.ingestion.edgar_client import fetch_agreements
from sec_rag.ingestion.filter import is_full_agreement
from sec_rag.ingestion.html_extractor import extract_text_from_html
from sec_rag.ingestion.indexer import QdrantIndexer
from sec_rag.models.documents import Document

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


async def run_ingestion(
    max_results: int = 50,
    start_date: str = "2020-01-01",
    end_date: str = "2025-12-31",
) -> dict[str, Any]:
    """Run the full ingestion pipeline. Returns stats dict."""

    settings = Settings()

    # --- Step 1: Fetch agreements from EDGAR ---
    logger.info("Searching EDGAR for up to %d employment agreements...", max_results)
    raw_agreements = await fetch_agreements(
        start_date=start_date,
        end_date=end_date,
        max_results=max_results,
        user_agent=settings.edgar_user_agent,
    )
    logger.info("Downloaded %d exhibits from EDGAR", len(raw_agreements))

    # --- Step 2: Extract text + filter ---
    documents: list[Document] = []
    skipped_not_full = 0
    skipped_extract_fail = 0

    total_raw = len(raw_agreements)
    for idx, agreement in enumerate(raw_agreements):
        if (idx + 1) % 10 == 0 or idx == 0:
            logger.info("Processing %d/%d documents...", idx + 1, total_raw)
        extracted_text = extract_text_from_html(agreement.raw_html)
        if not extracted_text:
            skipped_extract_fail += 1
            continue

        try:
            filing_date = date.fromisoformat(agreement.filing_date_raw)
        except (ValueError, TypeError):
            filing_date = date(2000, 1, 1)

        full = is_full_agreement(
            extracted_text,
            min_chars=settings.min_agreement_chars,
            min_keywords=settings.min_section_keywords,
        )
        if not full:
            skipped_not_full += 1
            continue

        doc = Document(
            accession_number=agreement.accession_number,
            exhibit_number=agreement.filename or "unknown",
            company_name=agreement.company_name,
            cik=agreement.cik,
            filing_date=filing_date,
            source_url=agreement.exhibit_url,
            raw_html=agreement.raw_html,
            extracted_text=extracted_text,
            is_full_agreement=True,
        )
        documents.append(doc)

    logger.info(
        "Filtered: %d full agreements, %d not full agreements skipped, %d extraction failures",
        len(documents),
        skipped_not_full,
        skipped_extract_fail,
    )

    if not documents:
        logger.warning("No documents to index. Try increasing --max-results.")
        return {"downloaded": len(raw_agreements), "indexed_chunks": 0}

    # --- Step 3: Chunk all documents ---
    all_chunks = []
    for doc in documents:
        chunks = chunk_document(doc, max_chunk_chars=settings.max_chunk_chars)
        all_chunks.extend(chunks)
        logger.info(
            "  %s: %d chunks",
            doc.company_name[:40],
            len(chunks),
        )

    logger.info("Total chunks: %d from %d documents", len(all_chunks), len(documents))

    # --- Step 4: Embed ---
    logger.info("Loading embedding model: %s", settings.embedding_model)
    embedder = ChunkEmbedder(model_name=settings.embedding_model)
    sparse_encoder = SparseEncoder(model_name=settings.sparse_model)

    logger.info("Generating dense + sparse embeddings in parallel...")
    texts = [c.text for c in all_chunks]

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=2) as pool:
        dense_future = pool.submit(embedder.embed_texts, texts)
        sparse_future = pool.submit(sparse_encoder.encode, texts)
        dense_vectors = dense_future.result()
        sparse_vectors = sparse_future.result()

    logger.info("Dense embeddings shape: %s", dense_vectors.shape)
    logger.info("Sparse vectors: %d", len(sparse_vectors))

    # --- Step 5: Index into Qdrant ---
    logger.info("Connecting to Qdrant at %s...", settings.qdrant_url)
    indexer = QdrantIndexer.from_url(
        url=settings.qdrant_url,
        collection_name=settings.qdrant_collection,
        dense_dim=embedder.dimension,
        api_key=settings.qdrant_api_key.get_secret_value(),
    )
    indexer.ensure_collection()

    logger.info("Upserting %d chunks...", len(all_chunks))
    try:
        count = indexer.upsert_chunks(all_chunks, dense_vectors, sparse_vectors)
        info = indexer.collection_info()
        logger.info("Upserted %d points. Collection: %s", count, info)
    finally:
        indexer.close()

    # --- Summary ---
    section_types: dict[str, int] = {}
    for c in all_chunks:
        st = c.metadata.section_type.value
        section_types[st] = section_types.get(st, 0) + 1

    logger.info("Section type distribution:")
    for st, cnt in sorted(section_types.items(), key=lambda x: -x[1]):
        logger.info("  %s: %d", st, cnt)

    return {
        "downloaded": len(raw_agreements),
        "full_agreements": len(documents),
        "skipped_not_full": skipped_not_full,
        "skipped_extract_fail": skipped_extract_fail,
        "total_chunks": len(all_chunks),
        "indexed_points": count,
        "section_distribution": section_types,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest employment agreements from SEC EDGAR")
    parser.add_argument("--max-results", type=int, default=50, help="Max exhibits to download")
    parser.add_argument(
        "--start-date", type=str, default="2020-01-01", help="Start date YYYY-MM-DD"
    )
    parser.add_argument("--end-date", type=str, default="2025-12-31", help="End date YYYY-MM-DD")
    args = parser.parse_args()

    stats = asyncio.run(
        run_ingestion(
            max_results=args.max_results,
            start_date=args.start_date,
            end_date=args.end_date,
        )
    )
    logger.info("Ingestion complete: %s", stats)


if __name__ == "__main__":
    main()
