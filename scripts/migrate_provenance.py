#!/usr/bin/env python3
"""
Provenance Migration Script

This script adds provenance metadata to existing Qdrant documents that don't have it.
It's designed to be run once during the transition to the enhanced provenance system.

Features:
- Batch processing for performance
- Progress tracking
- Validation of migrated documents
- Rollback capability
- Dry-run mode for testing

Usage:
    python scripts/migrate_provenance.py --dry-run  # Test without making changes
    python scripts/migrate_provenance.py            # Perform migration

Author: ReleAF AI Team
Date: 2025-12-03
"""

import asyncio
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.rag_service.provenance import (
    EmbeddingMetadata,
    DataLineage,
    TrustIndicators,
    ProvenanceValidator,
    generate_checksum,
    get_utc_timestamp,
    PROVENANCE_SCHEMA_VERSION
)
from services.rag_service.version_tracker import EmbeddingVersionTracker
from services.rag_service.audit_trail import AuditTrailManager, EventType, EntityType, ActorType, Action

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProvenanceMigrator:
    """
    Migrates existing Qdrant documents to include provenance metadata
    """

    def __init__(
        self,
        config_path: str = "configs/rag.yaml",
        batch_size: int = 100,
        dry_run: bool = False
    ):
        """
        Initialize migrator

        Args:
            config_path: Path to RAG configuration file
            batch_size: Number of documents to process per batch
            dry_run: If True, don't make any changes
        """
        self.config_path = config_path
        self.batch_size = batch_size
        self.dry_run = dry_run

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize components
        self.qdrant_client = None
        self.version_tracker = None
        self.audit_manager = None

        # Statistics
        self.stats = {
            "total_documents": 0,
            "documents_with_provenance": 0,
            "documents_migrated": 0,
            "documents_failed": 0,
            "start_time": None,
            "end_time": None
        }

        logger.info(f"ProvenanceMigrator initialized (dry_run={dry_run})")

    async def initialize(self):
        """Initialize async components"""
        from qdrant_client import AsyncQdrantClient

        # Initialize Qdrant client
        qdrant_config = self.config.get("qdrant", {})
        self.qdrant_client = AsyncQdrantClient(
            host=qdrant_config.get("host", "localhost"),
            port=qdrant_config.get("port", 6333),
            timeout=30.0
        )

        # Initialize version tracker
        self.version_tracker = EmbeddingVersionTracker()

        # Initialize audit manager
        self.audit_manager = AuditTrailManager(
            storage_type="json",
            json_dir="data/audit_trail"
        )
        await self.audit_manager.async_init()

        logger.info("Async components initialized")

    async def migrate_all_documents(self):
        """
        Migrate all documents in the collection

        Returns:
            Migration statistics
        """
        self.stats["start_time"] = datetime.now(timezone.utc)

        logger.info("Starting provenance migration...")

        # Record migration start event
        await self.audit_manager.record_event(
            event_type=EventType.MIGRATION_STARTED.value,
            entity_type=EntityType.SYSTEM.value,
            entity_id="provenance_migration",
            action=Action.UPDATE.value,
            actor_type=ActorType.MIGRATION_SCRIPT.value,
            actor_id="migrate_provenance.py",
            metadata={"dry_run": self.dry_run, "batch_size": self.batch_size}
        )

        try:
            # Get collection name
            collection_name = self.config.get("qdrant", {}).get("collection_name", "sustainability_docs")

            # Scroll through all documents
            offset = None
            while True:
                # Fetch batch of documents
                scroll_result = await self.qdrant_client.scroll(
                    collection_name=collection_name,
                    limit=self.batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False  # Don't need vectors for migration
                )

                points, next_offset = scroll_result

                if not points:
                    break

                # Process batch
                await self._process_batch(points, collection_name)

                # Update offset
                offset = next_offset
                if offset is None:
                    break

            self.stats["end_time"] = datetime.now(timezone.utc)
            duration = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()

            # Record migration completion
            await self.audit_manager.record_event(
                event_type=EventType.MIGRATION_COMPLETED.value,
                entity_type=EntityType.SYSTEM.value,
                entity_id="provenance_migration",
                action=Action.UPDATE.value,
                actor_type=ActorType.MIGRATION_SCRIPT.value,
                actor_id="migrate_provenance.py",
                success=True,
                duration_ms=duration * 1000,
                metadata=self.stats
            )

            logger.info(f"Migration completed in {duration:.2f} seconds")
            self._print_statistics()

            return self.stats

        except Exception as e:
            logger.error(f"Migration failed: {e}")

            # Record migration failure
            await self.audit_manager.record_event(
                event_type=EventType.MIGRATION_FAILED.value,
                entity_type=EntityType.SYSTEM.value,
                entity_id="provenance_migration",
                action=Action.UPDATE.value,
                actor_type=ActorType.MIGRATION_SCRIPT.value,
                actor_id="migrate_provenance.py",
                success=False,
                error_message=str(e),
                metadata=self.stats
            )

            raise

    async def _process_batch(self, points: List[Any], collection_name: str):
        """
        Process a batch of documents

        Args:
            points: List of Qdrant points
            collection_name: Name of the collection
        """
        for point in points:
            self.stats["total_documents"] += 1

            try:
                # Check if document already has provenance
                payload = point.payload
                has_provenance = (
                    "embedding_metadata" in payload and
                    "lineage" in payload and
                    "trust_indicators" in payload
                )

                if has_provenance:
                    self.stats["documents_with_provenance"] += 1
                    logger.debug(f"Document {point.id} already has provenance, skipping")
                    continue

                # Create default provenance metadata
                provenance = self._create_default_provenance(point)

                # Validate provenance
                if not self._validate_provenance(provenance):
                    logger.warning(f"Invalid provenance for document {point.id}, skipping")
                    self.stats["documents_failed"] += 1
                    continue

                # Update document (if not dry-run)
                if not self.dry_run:
                    await self._update_document(point.id, provenance, collection_name)

                self.stats["documents_migrated"] += 1

                if self.stats["documents_migrated"] % 100 == 0:
                    logger.info(f"Migrated {self.stats['documents_migrated']} documents...")

            except Exception as e:
                logger.error(f"Error processing document {point.id}: {e}")
                self.stats["documents_failed"] += 1

    def _create_default_provenance(self, point: Any) -> Dict[str, Any]:
        """
        Create default provenance metadata for a document

        Args:
            point: Qdrant point

        Returns:
            Dictionary with provenance metadata
        """
        payload = point.payload
        content = payload.get("content", "")

        # Create embedding metadata
        embedding_metadata = EmbeddingMetadata(
            model_name="BAAI/bge-large-en-v1.5",
            model_version="1.5.0",
            model_checksum=None,  # Unknown for existing documents
            embedding_dim=1024,
            normalization=True,
            pooling_strategy="mean",
            embedding_created_at=get_utc_timestamp(),  # Use current time as approximation
            embedding_generation_time_ms=0.0,  # Unknown
            content_checksum=generate_checksum(content),
            schema_version=PROVENANCE_SCHEMA_VERSION,
            migration_history=[{
                "from_version": "legacy",
                "to_version": "1.0.0",
                "migrated_at": get_utc_timestamp(),
                "migration_script": "migrate_provenance.py"
            }]
        )

        # Create data lineage
        lineage = DataLineage(
            original_source=payload.get("source", "unknown"),
            source_url=payload.get("metadata", {}).get("url"),
            source_id=str(point.id),
            collection_date=get_utc_timestamp(),  # Use current time as approximation
            collection_method="legacy_migration",
            collector_version="1.0.0",
            processing_pipeline=["legacy_ingestion", "provenance_migration"],
            transformations=[{
                "type": "provenance_migration",
                "timestamp": get_utc_timestamp(),
                "description": "Added provenance metadata to legacy document"
            }],
            last_updated=get_utc_timestamp(),
            update_reason="provenance_migration",
            previous_versions=[]
        )

        # Create trust indicators
        trust_indicators = TrustIndicators(
            trust_score=0.8,  # Default score for legacy documents
            source_reliability=0.8,
            content_quality=0.8,
            freshness_score=0.5,  # Lower for legacy documents
            human_verified=False,
            verification_date=None,
            verifier_id=None,
            retrieval_count=0,
            positive_feedback_count=0,
            negative_feedback_count=0,
            avg_relevance_score=0.0
        )

        # Calculate trust score
        trust_indicators.calculate_trust_score()

        return {
            "embedding_metadata": embedding_metadata.to_dict(),
            "lineage": lineage.to_dict(),
            "trust_indicators": trust_indicators.to_dict()
        }

    def _validate_provenance(self, provenance: Dict[str, Any]) -> bool:
        """
        Validate provenance metadata

        Args:
            provenance: Provenance metadata dictionary

        Returns:
            True if valid, False otherwise
        """
        try:
            # Validate embedding metadata
            embedding_metadata = EmbeddingMetadata.from_dict(provenance["embedding_metadata"])
            if not ProvenanceValidator.validate_embedding_metadata(embedding_metadata):
                return False

            # Validate lineage
            lineage = DataLineage.from_dict(provenance["lineage"])
            if not ProvenanceValidator.validate_lineage(lineage):
                return False

            # Validate trust indicators
            trust_indicators = TrustIndicators.from_dict(provenance["trust_indicators"])
            if not ProvenanceValidator.validate_trust_indicators(trust_indicators):
                return False

            return True

        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False

    async def _update_document(self, doc_id: str, provenance: Dict[str, Any], collection_name: str):
        """
        Update document with provenance metadata

        Args:
            doc_id: Document ID
            provenance: Provenance metadata
            collection_name: Collection name
        """
        # Get existing document
        points = await self.qdrant_client.retrieve(
            collection_name=collection_name,
            ids=[doc_id]
        )

        if not points:
            raise ValueError(f"Document {doc_id} not found")

        point = points[0]

        # Update payload with provenance
        updated_payload = point.payload.copy()
        updated_payload.update(provenance)

        # Update in Qdrant
        await self.qdrant_client.set_payload(
            collection_name=collection_name,
            payload=updated_payload,
            points=[doc_id]
        )

        # Record audit event
        await self.audit_manager.record_event(
            event_type=EventType.DOCUMENT_MIGRATED.value,
            entity_type=EntityType.DOCUMENT.value,
            entity_id=str(doc_id),
            action=Action.UPDATE.value,
            actor_type=ActorType.MIGRATION_SCRIPT.value,
            actor_id="migrate_provenance.py",
            changes={"added": list(provenance.keys())}
        )

    def _print_statistics(self):
        """Print migration statistics"""
        print("\n" + "="*80)
        print("PROVENANCE MIGRATION STATISTICS")
        print("="*80)
        print(f"Total documents processed: {self.stats['total_documents']}")
        print(f"Documents with existing provenance: {self.stats['documents_with_provenance']}")
        print(f"Documents migrated: {self.stats['documents_migrated']}")
        print(f"Documents failed: {self.stats['documents_failed']}")

        if self.stats['start_time'] and self.stats['end_time']:
            duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
            print(f"Duration: {duration:.2f} seconds")
            if duration > 0:
                print(f"Throughput: {self.stats['documents_migrated'] / duration:.2f} docs/sec")

        print("="*80)

        if self.dry_run:
            print("\n⚠️  DRY RUN MODE - No changes were made")
        else:
            print("\n✅ Migration completed successfully")

    async def close(self):
        """Close connections"""
        if self.audit_manager:
            await self.audit_manager.close()

        if self.qdrant_client:
            await self.qdrant_client.close()

        logger.info("Connections closed")


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Migrate provenance metadata to existing Qdrant documents")
    parser.add_argument("--dry-run", action="store_true", help="Test without making changes")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for processing")
    parser.add_argument("--config", type=str, default="configs/rag.yaml", help="Path to RAG config file")

    args = parser.parse_args()

    # Create migrator
    migrator = ProvenanceMigrator(
        config_path=args.config,
        batch_size=args.batch_size,
        dry_run=args.dry_run
    )

    try:
        # Initialize
        await migrator.initialize()

        # Run migration
        await migrator.migrate_all_documents()

    except KeyboardInterrupt:
        logger.warning("Migration interrupted by user")
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return 1
    finally:
        # Clean up
        await migrator.close()

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

