#!/usr/bin/env python3
"""
Knowledge Corpus Ingestion Script for RAG Service

Reads sustainability_knowledge.jsonl and ingests into Qdrant vector database
via the RAG service's API, or directly if Qdrant is accessible.

Usage:
    # Via RAG service API (service must be running):
    python scripts/ingest_knowledge.py --mode api --rag-url http://localhost:8003

    # Direct to Qdrant (Qdrant must be running):
    python scripts/ingest_knowledge.py --mode direct --qdrant-url http://localhost:6333

    # Dry run (validate only):
    python scripts/ingest_knowledge.py --mode dry-run
"""

import json
import sys
import os
import time
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CORPUS_PATH = Path(__file__).parent.parent / "data" / "knowledge_corpus" / "sustainability_knowledge.jsonl"
DEFAULT_COLLECTION = "sustainability_knowledge"


def load_corpus(path: Path) -> List[Dict[str, Any]]:
    """Load and validate the knowledge corpus."""
    if not path.exists():
        logger.error(f"Corpus file not found: {path}")
        sys.exit(1)

    docs = []
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
                assert "id" in doc and "content" in doc and "title" in doc
                docs.append(doc)
            except (json.JSONDecodeError, AssertionError) as e:
                logger.warning(f"Skipping invalid line {line_num}: {e}")

    logger.info(f"Loaded {len(docs)} documents from {path}")
    return docs


def ingest_via_api(docs: List[Dict], rag_url: str, collection: str):
    """Ingest documents via the RAG service REST API."""
    import httpx

    logger.info(f"Ingesting {len(docs)} documents via RAG API at {rag_url}")
    client = httpx.Client(timeout=60.0)

    # Check RAG service health
    try:
        resp = client.get(f"{rag_url}/health")
        resp.raise_for_status()
        logger.info(f"RAG service is healthy: {resp.json().get('status', 'unknown')}")
    except Exception as e:
        logger.error(f"RAG service not reachable at {rag_url}: {e}")
        sys.exit(1)

    success = 0
    failed = 0
    for i, doc in enumerate(docs, 1):
        try:
            payload = {
                "collection": collection,
                "documents": [{
                    "id": doc["id"],
                    "content": doc["content"],
                    "metadata": {
                        "title": doc["title"],
                        "category": doc.get("category", "general"),
                        "source": doc.get("source", ""),
                    }
                }]
            }
            resp = client.post(f"{rag_url}/ingest", json=payload)
            resp.raise_for_status()
            success += 1
            if i % 10 == 0:
                logger.info(f"  Progress: {i}/{len(docs)} ingested")
        except Exception as e:
            failed += 1
            logger.warning(f"  Failed to ingest '{doc['title']}': {e}")

    logger.info(f"Ingestion complete: {success} success, {failed} failed")
    client.close()


def ingest_direct(docs: List[Dict], qdrant_url: str, collection: str):
    """Ingest documents directly into Qdrant with sentence-transformers embeddings."""
    try:
        from sentence_transformers import SentenceTransformer
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct
    except ImportError as e:
        logger.error(f"Required packages not available: {e}")
        logger.error("pip install sentence-transformers qdrant-client")
        sys.exit(1)

    model_name = "all-MiniLM-L6-v2"
    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    vector_size = model.get_sentence_embedding_dimension()

    logger.info(f"Connecting to Qdrant at {qdrant_url}")
    client = QdrantClient(url=qdrant_url)

    # Create or recreate collection
    try:
        client.delete_collection(collection)
        logger.info(f"Deleted existing collection: {collection}")
    except Exception:
        pass

    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    logger.info(f"Created collection: {collection} (dim={vector_size})")

    # Embed and upsert
    texts = [f"{d['title']}\n\n{d['content']}" for d in docs]
    logger.info(f"Embedding {len(texts)} documents...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=16)

    points = [
        PointStruct(
            id=i,
            vector=embeddings[i].tolist(),
            payload={
                "doc_id": doc["id"],
                "title": doc["title"],
                "content": doc["content"],
                "category": doc.get("category", "general"),
                "source": doc.get("source", ""),
            },
        )
        for i, doc in enumerate(docs)
    ]

    client.upsert(collection_name=collection, points=points)
    logger.info(f"✅ Ingested {len(points)} documents into Qdrant collection '{collection}'")

    # Verify
    info = client.get_collection(collection)
    logger.info(f"Collection info: {info.points_count} points, vector_size={info.config.params.vectors.size}")


def dry_run(docs: List[Dict]):
    """Validate corpus without ingesting."""
    logger.info(f"DRY RUN: Validating {len(docs)} documents")
    categories = {}
    for doc in docs:
        cat = doc.get("category", "uncategorized")
        categories[cat] = categories.get(cat, 0) + 1

    for cat, count in sorted(categories.items()):
        logger.info(f"  {cat}: {count} documents")

    total_chars = sum(len(d["content"]) for d in docs)
    logger.info(f"  Total: {len(docs)} docs, {total_chars:,} chars, avg {total_chars//len(docs)} chars/doc")
    logger.info("✅ Dry run complete — all documents valid")


def main():
    parser = argparse.ArgumentParser(description="Ingest sustainability knowledge corpus into RAG")
    parser.add_argument("--mode", choices=["api", "direct", "dry-run"], default="dry-run")
    parser.add_argument("--rag-url", default="http://localhost:8003")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--corpus", default=str(CORPUS_PATH))
    args = parser.parse_args()

    docs = load_corpus(Path(args.corpus))

    if args.mode == "dry-run":
        dry_run(docs)
    elif args.mode == "api":
        ingest_via_api(docs, args.rag_url, args.collection)
    elif args.mode == "direct":
        ingest_direct(docs, args.qdrant_url, args.collection)


if __name__ == "__main__":
    main()
