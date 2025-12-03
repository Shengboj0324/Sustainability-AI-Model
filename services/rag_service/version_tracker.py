"""
Embedding Version Tracker - Track embedding model versions and migrations

Provides comprehensive version control for embedding models including:
- Model version tracking and comparison
- Model checksum generation and validation
- Migration history tracking
- Version compatibility checking

Thread-safe, production-ready implementation.
"""

import hashlib
import logging
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class EmbeddingVersionTracker:
    """
    Track embedding model versions, checksums, and migrations
    
    Maintains a version history of all embedding models used,
    tracks migrations between versions, and validates model integrity.
    
    Thread-safe with asyncio.Lock for concurrent access.
    """
    
    def __init__(self, version_file: Optional[str] = None):
        """
        Initialize version tracker
        
        Args:
            version_file: Path to version history JSON file
        """
        self.version_file = version_file or "data/embedding_versions.json"
        self.version_history: Dict[str, Dict[str, Any]] = {}
        self.current_version: Optional[str] = None
        self.lock = asyncio.Lock()
        
        # Load version history if file exists
        self._load_version_history()
    
    def _load_version_history(self):
        """Load version history from file"""
        try:
            version_path = Path(self.version_file)
            if version_path.exists():
                with open(version_path, 'r') as f:
                    data = json.load(f)
                    self.version_history = data.get('versions', {})
                    self.current_version = data.get('current_version')
                    logger.info(f"Loaded {len(self.version_history)} version(s) from {self.version_file}")
            else:
                logger.info(f"No version history file found at {self.version_file}, starting fresh")
                self._initialize_default_version()
        except Exception as e:
            logger.error(f"Failed to load version history: {e}")
            self._initialize_default_version()
    
    def _initialize_default_version(self):
        """Initialize with default version"""
        self.current_version = "1.0.0"
        self.version_history = {
            "1.0.0": {
                "model_name": "BAAI/bge-large-en-v1.5",
                "model_version": "1.5.0",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "status": "active",
                "num_documents": 0,
                "embedding_dim": 1024,
                "normalization": True,
                "pooling_strategy": "mean"
            }
        }
    
    async def _save_version_history_unlocked(self):
        """
        Save version history to file (INTERNAL - assumes lock is already held)

        CRITICAL: This method must only be called when self.lock is already acquired!
        """
        try:
            version_path = Path(self.version_file)

            # Create parent directories (synchronous, but fast)
            version_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "current_version": self.current_version,
                "versions": self.version_history,
                "last_updated": datetime.now(timezone.utc).isoformat()
            }

            # Use asyncio.to_thread for file I/O to avoid blocking
            def _write_json():
                with open(version_path, 'w') as f:
                    json.dump(data, f, indent=2)

            await asyncio.to_thread(_write_json)

            logger.info(f"Saved version history to {self.version_file}")

        except Exception as e:
            logger.error(f"Failed to save version history: {e}")

    async def save_version_history(self):
        """Save version history to file (thread-safe)"""
        async with self.lock:
            await self._save_version_history_unlocked()
    
    async def register_version(
        self,
        version: str,
        model_name: str,
        model_version: str,
        embedding_dim: int,
        normalization: bool = True,
        pooling_strategy: str = "mean",
        model_checksum: Optional[str] = None
    ) -> bool:
        """
        Register a new embedding model version (thread-safe)
        
        Args:
            version: Version identifier (e.g., "1.0.0")
            model_name: Model name (e.g., "BAAI/bge-large-en-v1.5")
            model_version: Model version (e.g., "1.5.0")
            embedding_dim: Embedding dimension
            normalization: Whether embeddings are normalized
            pooling_strategy: Pooling strategy used
            model_checksum: Optional model checksum
            
        Returns:
            True if registered successfully, False otherwise
        """
        async with self.lock:
            try:
                if version in self.version_history:
                    logger.warning(f"Version {version} already exists")
                    return False
                
                self.version_history[version] = {
                    "model_name": model_name,
                    "model_version": model_version,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "status": "active",
                    "num_documents": 0,
                    "embedding_dim": embedding_dim,
                    "normalization": normalization,
                    "pooling_strategy": pooling_strategy,
                    "model_checksum": model_checksum
                }
                
                logger.info(f"Registered new version: {version} ({model_name})")
                await self._save_version_history_unlocked()
                return True
                
            except Exception as e:
                logger.error(f"Failed to register version: {e}")
                return False
    
    async def set_current_version(self, version: str) -> bool:
        """
        Set the current active version (thread-safe)
        
        Args:
            version: Version to set as current
            
        Returns:
            True if set successfully, False otherwise
        """
        async with self.lock:
            if version not in self.version_history:
                logger.error(f"Version {version} not found in history")
                return False
            
            self.current_version = version
            logger.info(f"Set current version to: {version}")
            await self._save_version_history_unlocked()
            return True

    async def get_current_version_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the current version (thread-safe)

        Returns:
            Dictionary with version information or None
        """
        async with self.lock:
            if not self.current_version:
                return None
            return self.version_history.get(self.current_version)

    async def get_version_info(self, version: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific version (thread-safe)

        Args:
            version: Version identifier

        Returns:
            Dictionary with version information or None
        """
        async with self.lock:
            return self.version_history.get(version)

    async def increment_document_count(self, version: Optional[str] = None):
        """
        Increment document count for a version (thread-safe)

        Args:
            version: Version to increment (defaults to current)
        """
        async with self.lock:
            target_version = version or self.current_version
            if target_version and target_version in self.version_history:
                self.version_history[target_version]['num_documents'] += 1

    async def record_migration(
        self,
        from_version: str,
        to_version: str,
        num_documents: int,
        migration_time_seconds: float
    ):
        """
        Record a migration between versions (thread-safe)

        Args:
            from_version: Source version
            to_version: Target version
            num_documents: Number of documents migrated
            migration_time_seconds: Time taken for migration
        """
        async with self.lock:
            try:
                # Add migration record to target version
                if to_version in self.version_history:
                    if 'migrations' not in self.version_history[to_version]:
                        self.version_history[to_version]['migrations'] = []

                    migration_record = {
                        "from_version": from_version,
                        "num_documents": num_documents,
                        "migration_time_seconds": migration_time_seconds,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }

                    self.version_history[to_version]['migrations'].append(migration_record)
                    logger.info(f"Recorded migration: {from_version} -> {to_version} ({num_documents} docs)")

                    await self._save_version_history_unlocked()

            except Exception as e:
                logger.error(f"Failed to record migration: {e}")

    async def deprecate_version(self, version: str):
        """
        Mark a version as deprecated (thread-safe)

        Args:
            version: Version to deprecate
        """
        async with self.lock:
            if version in self.version_history:
                self.version_history[version]['status'] = 'deprecated'
                self.version_history[version]['deprecated_at'] = datetime.now(timezone.utc).isoformat()
                logger.info(f"Deprecated version: {version}")
                await self._save_version_history_unlocked()

    @staticmethod
    def generate_model_checksum(model_path: str) -> str:
        """
        Generate checksum for a model file

        Args:
            model_path: Path to model file

        Returns:
            SHA-256 checksum as hex string
        """
        try:
            sha256_hash = hashlib.sha256()
            with open(model_path, "rb") as f:
                # Read in chunks to handle large files
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"Failed to generate model checksum: {e}")
            return ""

    async def validate_version_compatibility(
        self,
        version1: str,
        version2: str
    ) -> bool:
        """
        Check if two versions are compatible (same embedding dim)

        Args:
            version1: First version
            version2: Second version

        Returns:
            True if compatible, False otherwise
        """
        async with self.lock:
            v1_info = self.version_history.get(version1)
            v2_info = self.version_history.get(version2)

            if not v1_info or not v2_info:
                return False

            # Check embedding dimension compatibility
            return v1_info.get('embedding_dim') == v2_info.get('embedding_dim')

    async def get_all_versions(self) -> List[str]:
        """
        Get list of all registered versions (thread-safe)

        Returns:
            List of version identifiers
        """
        async with self.lock:
            return list(self.version_history.keys())

    async def get_active_versions(self) -> List[str]:
        """
        Get list of active (non-deprecated) versions (thread-safe)

        Returns:
            List of active version identifiers
        """
        async with self.lock:
            return [
                version for version, info in self.version_history.items()
                if info.get('status') == 'active'
            ]

