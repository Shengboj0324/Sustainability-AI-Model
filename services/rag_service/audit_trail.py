"""
Audit Trail System for Embedding Provenance

This module provides comprehensive audit logging for all provenance-related operations.
Tracks document access, embedding generation, version changes, and provenance updates.

Features:
- Immutable audit records
- Async, non-blocking recording
- Dual storage: PostgreSQL (primary) + JSON (fallback)
- Tamper detection with checksums
- GDPR-compliant data handling
- High-performance batch writes

Author: ReleAF AI Team
Date: 2025-12-03
"""

import asyncio
import json
import hashlib
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Event Type Enums
# ============================================================================

class EventType(str, Enum):
    """Audit event types"""
    # Document events
    DOCUMENT_CREATED = "DOCUMENT_CREATED"
    DOCUMENT_UPDATED = "DOCUMENT_UPDATED"
    DOCUMENT_DELETED = "DOCUMENT_DELETED"
    DOCUMENT_ACCESSED = "DOCUMENT_ACCESSED"
    DOCUMENT_MIGRATED = "DOCUMENT_MIGRATED"

    # Embedding events
    EMBEDDING_GENERATED = "EMBEDDING_GENERATED"
    EMBEDDING_REGENERATED = "EMBEDDING_REGENERATED"
    EMBEDDING_VALIDATED = "EMBEDDING_VALIDATED"
    EMBEDDING_CHECKSUM_VERIFIED = "EMBEDDING_CHECKSUM_VERIFIED"

    # Version events
    VERSION_REGISTERED = "VERSION_REGISTERED"
    VERSION_ACTIVATED = "VERSION_ACTIVATED"
    VERSION_DEPRECATED = "VERSION_DEPRECATED"
    VERSION_COMPATIBILITY_CHECKED = "VERSION_COMPATIBILITY_CHECKED"

    # Provenance events
    PROVENANCE_CREATED = "PROVENANCE_CREATED"
    PROVENANCE_UPDATED = "PROVENANCE_UPDATED"
    PROVENANCE_VALIDATED = "PROVENANCE_VALIDATED"
    PROVENANCE_ACCESSED = "PROVENANCE_ACCESSED"

    # Trust events
    TRUST_SCORE_CALCULATED = "TRUST_SCORE_CALCULATED"
    TRUST_SCORE_UPDATED = "TRUST_SCORE_UPDATED"
    HUMAN_VERIFICATION = "HUMAN_VERIFICATION"
    FEEDBACK_RECEIVED = "FEEDBACK_RECEIVED"

    # System events
    MIGRATION_STARTED = "MIGRATION_STARTED"
    MIGRATION_COMPLETED = "MIGRATION_COMPLETED"
    MIGRATION_FAILED = "MIGRATION_FAILED"
    AUDIT_TRAIL_ACCESSED = "AUDIT_TRAIL_ACCESSED"


class EntityType(str, Enum):
    """Entity types for audit records"""
    DOCUMENT = "document"
    EMBEDDING = "embedding"
    VERSION = "version"
    PROVENANCE = "provenance"
    TRUST = "trust"
    SYSTEM = "system"


class ActorType(str, Enum):
    """Actor types for audit records"""
    SYSTEM = "system"
    USER = "user"
    ADMIN = "admin"
    MIGRATION_SCRIPT = "migration_script"
    API = "api"


class Action(str, Enum):
    """CRUD actions"""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    VALIDATE = "validate"


# ============================================================================
# Audit Record Data Structure
# ============================================================================

@dataclass
class AuditRecord:
    """
    Immutable audit record for provenance operations

    All audit records are immutable and tamper-proof with checksums.
    """
    # Core identification
    audit_id: str
    event_type: str
    timestamp: str

    # Entity information
    entity_type: str
    entity_id: str

    # Actor information
    actor_type: str
    actor_id: Optional[str] = None

    # Change details
    action: str = Action.READ.value
    changes: Dict[str, Any] = field(default_factory=dict)

    # Context
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

    # Metadata
    success: bool = True
    error_message: Optional[str] = None
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Integrity
    checksum: Optional[str] = None




    def calculate_checksum(self) -> str:
        """
        Calculate SHA-256 checksum for tamper detection

        Returns:
            Hex digest of record checksum
        """
        # Create deterministic string representation (exclude checksum field)
        data = {
            "audit_id": self.audit_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "actor_type": self.actor_type,
            "actor_id": self.actor_id,
            "action": self.action,
            "changes": self.changes,
            "success": self.success,
            "duration_ms": self.duration_ms
        }

        # Sort keys for deterministic serialization
        json_str = json.dumps(data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(json_str.encode('utf-8')).hexdigest()

    def verify_checksum(self) -> bool:
        """
        Verify record integrity by checking checksum

        Returns:
            True if checksum is valid, False otherwise
        """
        if self.checksum is None:
            return False

        expected_checksum = self.calculate_checksum()
        return self.checksum == expected_checksum

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditRecord':
        """Create from dictionary"""
        return cls(**data)


# ============================================================================
# Utility Functions
# ============================================================================

def get_utc_timestamp() -> str:
    """Get current UTC timestamp in ISO 8601 format"""
    return datetime.now(timezone.utc).isoformat()


# ============================================================================
# Audit Trail Manager
# ============================================================================

class AuditTrailManager:
    """
    Manages audit trail recording and querying

    Features:
    - Async, non-blocking recording
    - Dual storage: PostgreSQL (primary) + JSON (fallback)
    - Batch writes for performance
    - Thread-safe with asyncio.Lock
    - Tamper detection with checksums
    - GDPR-compliant data handling

    Storage Strategy:
    1. Try PostgreSQL first (if configured)
    2. Fallback to JSON files if PostgreSQL unavailable
    3. JSON files are append-only, rotated daily
    """

    def __init__(
        self,
        storage_type: str = "json",  # "postgresql" or "json"
        json_dir: Optional[str] = None,
        pg_connection_string: Optional[str] = None,
        batch_size: int = 100,
        flush_interval_seconds: float = 5.0
    ):
        """
        Initialize audit trail manager

        Args:
            storage_type: "postgresql" or "json"
            json_dir: Directory for JSON audit files
            pg_connection_string: PostgreSQL connection string
            batch_size: Number of records to batch before writing
            flush_interval_seconds: Max time to wait before flushing batch
        """
        self.storage_type = storage_type
        self.json_dir = Path(json_dir or "data/audit_trail")
        self.pg_connection_string = pg_connection_string
        self.batch_size = batch_size
        self.flush_interval_seconds = flush_interval_seconds

        # Thread safety
        self.lock = asyncio.Lock()

        # Batch buffer
        self.batch_buffer: List[AuditRecord] = []
        self.last_flush_time = datetime.now(timezone.utc)

        # PostgreSQL connection pool (if using PostgreSQL)
        self.pg_pool = None

        # Initialize storage
        self._initialize_storage()

        logger.info(f"AuditTrailManager initialized with storage_type={storage_type}")

    def _initialize_storage(self):
        """Initialize storage backend (synchronous)"""
        if self.storage_type == "json":
            # Create JSON directory
            self.json_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"JSON audit trail directory: {self.json_dir}")
        elif self.storage_type == "postgresql":
            # PostgreSQL initialization will be done async in async_init()
            logger.info("PostgreSQL audit trail will be initialized asynchronously")
        else:
            raise ValueError(f"Invalid storage_type: {self.storage_type}")

    async def async_init(self):
        """
        Async initialization (call this after creating instance)

        For PostgreSQL, this creates the connection pool and tables.
        """
        if self.storage_type == "postgresql" and self.pg_connection_string:
            try:
                import asyncpg

                # Create connection pool
                self.pg_pool = await asyncpg.create_pool(
                    self.pg_connection_string,
                    min_size=2,
                    max_size=10,
                    command_timeout=10.0
                )

                # Create audit_trail table if not exists
                await self._create_pg_table()

                logger.info("PostgreSQL audit trail initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize PostgreSQL audit trail: {e}")
                logger.warning("Falling back to JSON storage")
                self.storage_type = "json"
                self._initialize_storage()


    async def _create_pg_table(self):
        """Create PostgreSQL audit_trail table if not exists"""
        if not self.pg_pool:
            return

        create_table_sql = """
        CREATE TABLE IF NOT EXISTS audit_trail (
            audit_id VARCHAR(36) PRIMARY KEY,
            event_type VARCHAR(50) NOT NULL,
            timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
            entity_type VARCHAR(20) NOT NULL,
            entity_id VARCHAR(100) NOT NULL,
            actor_type VARCHAR(20) NOT NULL,
            actor_id VARCHAR(100),
            action VARCHAR(20) NOT NULL,
            changes JSONB,
            request_id VARCHAR(36),
            session_id VARCHAR(36),
            ip_address VARCHAR(45),
            user_agent TEXT,
            success BOOLEAN NOT NULL DEFAULT TRUE,
            error_message TEXT,
            duration_ms FLOAT NOT NULL DEFAULT 0.0,
            metadata JSONB,
            checksum VARCHAR(64) NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_audit_entity ON audit_trail(entity_id, entity_type);
        CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_trail(timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_audit_event_type ON audit_trail(event_type);
        CREATE INDEX IF NOT EXISTS idx_audit_actor ON audit_trail(actor_id, actor_type);
        """

        async with self.pg_pool.acquire() as conn:
            await conn.execute(create_table_sql)

        logger.info("PostgreSQL audit_trail table created/verified")

    async def record_event(
        self,
        event_type: str,
        entity_type: str,
        entity_id: str,
        action: str,
        actor_type: str = ActorType.SYSTEM.value,
        actor_id: Optional[str] = None,
        changes: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        duration_ms: float = 0.0
    ) -> str:
        """
        Record an audit event (async, non-blocking)

        Args:
            event_type: Type of event (from EventType enum)
            entity_type: Type of entity (from EntityType enum)
            entity_id: ID of the affected entity
            action: Action performed (from Action enum)
            actor_type: Type of actor (from ActorType enum)
            actor_id: ID of the actor (user ID, API key, etc.)
            changes: Before/after values for updates
            metadata: Additional context
            request_id: Request ID for tracing
            session_id: Session ID for user sessions
            ip_address: Source IP address
            user_agent: Client user agent
            success: Whether operation succeeded
            error_message: Error message if failed
            duration_ms: Operation duration in milliseconds

        Returns:
            audit_id: UUID of the created audit record
        """
        # Generate audit ID
        audit_id = str(uuid.uuid4())

        # Create audit record
        record = AuditRecord(
            audit_id=audit_id,
            event_type=event_type,
            timestamp=get_utc_timestamp(),
            entity_type=entity_type,
            entity_id=entity_id,
            actor_type=actor_type,
            actor_id=actor_id,
            action=action,
            changes=changes or {},
            request_id=request_id,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            error_message=error_message,
            duration_ms=duration_ms,
            metadata=metadata or {}
        )

        # Calculate checksum for tamper detection
        record.checksum = record.calculate_checksum()

        # Add to batch buffer (thread-safe)
        async with self.lock:
            self.batch_buffer.append(record)

            # Check if we should flush
            should_flush = (
                len(self.batch_buffer) >= self.batch_size or
                (datetime.now(timezone.utc) - self.last_flush_time).total_seconds() >= self.flush_interval_seconds
            )

        # Flush if needed (non-blocking)
        if should_flush:
            # Don't await - fire and forget for performance
            asyncio.create_task(self._flush_batch())

        logger.debug(f"Recorded audit event: {event_type} for {entity_type}:{entity_id}")
        return audit_id

    async def _flush_batch(self):
        """Flush batch buffer to storage (internal, thread-safe)"""
        async with self.lock:
            if not self.batch_buffer:
                return

            # Get records to flush
            records_to_flush = self.batch_buffer.copy()
            self.batch_buffer.clear()
            self.last_flush_time = datetime.now(timezone.utc)

        # Write to storage (outside lock for performance)
        try:
            if self.storage_type == "postgresql" and self.pg_pool:
                await self._write_to_postgresql(records_to_flush)
            else:
                await self._write_to_json(records_to_flush)

            logger.debug(f"Flushed {len(records_to_flush)} audit records to {self.storage_type}")
        except Exception as e:
            logger.error(f"Failed to flush audit records: {e}")
            # Re-add to buffer for retry
            async with self.lock:
                self.batch_buffer.extend(records_to_flush)


    async def _write_to_postgresql(self, records: List[AuditRecord]):
        """Write records to PostgreSQL (internal)"""
        if not self.pg_pool:
            raise RuntimeError("PostgreSQL pool not initialized")

        insert_sql = """
        INSERT INTO audit_trail (
            audit_id, event_type, timestamp, entity_type, entity_id,
            actor_type, actor_id, action, changes, request_id, session_id,
            ip_address, user_agent, success, error_message, duration_ms,
            metadata, checksum
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18)
        """

        async with self.pg_pool.acquire() as conn:
            # Use executemany for batch insert
            await conn.executemany(
                insert_sql,
                [
                    (
                        r.audit_id, r.event_type, r.timestamp, r.entity_type, r.entity_id,
                        r.actor_type, r.actor_id, r.action, json.dumps(r.changes),
                        r.request_id, r.session_id, r.ip_address, r.user_agent,
                        r.success, r.error_message, r.duration_ms,
                        json.dumps(r.metadata), r.checksum
                    )
                    for r in records
                ]
            )

    async def _write_to_json(self, records: List[AuditRecord]):
        """Write records to JSON file (internal, append-only)"""
        # Get today's date for file rotation
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        json_file = self.json_dir / f"audit_trail_{today}.jsonl"

        # Prepare JSONL data (one JSON object per line)
        lines = [json.dumps(r.to_dict()) + "\n" for r in records]

        # Write to file (use asyncio.to_thread for non-blocking I/O)
        def _write():
            with open(json_file, 'a') as f:
                f.writelines(lines)

        await asyncio.to_thread(_write)

    async def get_entity_history(
        self,
        entity_id: str,
        entity_type: Optional[str] = None,
        limit: int = 100
    ) -> List[AuditRecord]:
        """
        Get audit history for a specific entity

        Args:
            entity_id: ID of the entity
            entity_type: Optional entity type filter
            limit: Maximum number of records to return

        Returns:
            List of audit records, sorted by timestamp (newest first)
        """
        # Flush any pending records first
        await self._flush_batch()

        if self.storage_type == "postgresql" and self.pg_pool:
            return await self._query_postgresql_entity_history(entity_id, entity_type, limit)
        else:
            return await self._query_json_entity_history(entity_id, entity_type, limit)

    async def _query_postgresql_entity_history(
        self,
        entity_id: str,
        entity_type: Optional[str],
        limit: int
    ) -> List[AuditRecord]:
        """Query PostgreSQL for entity history"""
        if not self.pg_pool:
            return []

        if entity_type:
            query = """
            SELECT * FROM audit_trail
            WHERE entity_id = $1 AND entity_type = $2
            ORDER BY timestamp DESC
            LIMIT $3
            """
            params = (entity_id, entity_type, limit)
        else:
            query = """
            SELECT * FROM audit_trail
            WHERE entity_id = $1
            ORDER BY timestamp DESC
            LIMIT $2
            """
            params = (entity_id, limit)

        async with self.pg_pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        # Convert rows to AuditRecord objects
        records = []
        for row in rows:
            record_dict = dict(row)
            # Parse JSON fields
            record_dict['changes'] = json.loads(record_dict['changes']) if record_dict['changes'] else {}
            record_dict['metadata'] = json.loads(record_dict['metadata']) if record_dict['metadata'] else {}
            records.append(AuditRecord.from_dict(record_dict))

        return records

    async def _query_json_entity_history(
        self,
        entity_id: str,
        entity_type: Optional[str],
        limit: int
    ) -> List[AuditRecord]:
        """Query JSON files for entity history"""
        records = []

        # Read all JSON files (use asyncio.to_thread for non-blocking I/O)
        def _read_files():
            all_records = []
            for json_file in sorted(self.json_dir.glob("audit_trail_*.jsonl"), reverse=True):
                try:
                    with open(json_file, 'r') as f:
                        for line in f:
                            record_dict = json.loads(line.strip())
                            if record_dict['entity_id'] == entity_id:
                                if entity_type is None or record_dict['entity_type'] == entity_type:
                                    all_records.append(AuditRecord.from_dict(record_dict))
                                    if len(all_records) >= limit:
                                        return all_records
                except Exception as e:
                    logger.error(f"Error reading {json_file}: {e}")
            return all_records

        records = await asyncio.to_thread(_read_files)

        # Sort by timestamp (newest first)
        records.sort(key=lambda r: r.timestamp, reverse=True)
        return records[:limit]

    async def close(self):
        """Close audit trail manager and flush pending records"""
        # Flush any pending records
        await self._flush_batch()

        # Close PostgreSQL pool if exists
        if self.pg_pool:
            await self.pg_pool.close()
            logger.info("PostgreSQL audit trail connection pool closed")
