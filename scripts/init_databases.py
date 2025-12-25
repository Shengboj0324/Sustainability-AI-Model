#!/usr/bin/env python3
"""
Database Initialization Script

Initializes PostgreSQL, Neo4j, and Qdrant databases with required schemas and indexes.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncpg
from neo4j import AsyncGraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

print("🔧 Initializing databases...")

async def init_postgres():
    """Initialize PostgreSQL database"""
    print("\n📊 Initializing PostgreSQL...")
    
    try:
        # Connect to PostgreSQL
        conn = await asyncpg.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            user=os.getenv("POSTGRES_USER", "releaf_user"),
            password=os.getenv("POSTGRES_PASSWORD", "releaf_password"),
            database=os.getenv("POSTGRES_DB", "releaf")
        )
        
        # Create feedback table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id SERIAL PRIMARY KEY,
                feedback_id VARCHAR(64) UNIQUE NOT NULL,
                feedback_type VARCHAR(50) NOT NULL,
                service VARCHAR(50) NOT NULL,
                rating INTEGER,
                comment TEXT,
                query TEXT,
                response TEXT,
                session_id VARCHAR(128),
                user_id VARCHAR(128),
                metadata JSONB,
                created_at TIMESTAMP DEFAULT NOW(),
                processed BOOLEAN DEFAULT FALSE,
                processed_at TIMESTAMP
            )
        """)
        
        # Create indexes
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_feedback_service ON feedback(service);
            CREATE INDEX IF NOT EXISTS idx_feedback_created ON feedback(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_feedback_processed ON feedback(processed);
        """)
        
        # Create retraining triggers table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS retraining_triggers (
                id SERIAL PRIMARY KEY,
                service VARCHAR(50) NOT NULL,
                trigger_reason TEXT NOT NULL,
                feedback_count INTEGER,
                satisfaction_score FLOAT,
                negative_feedback_count INTEGER,
                triggered_at TIMESTAMP DEFAULT NOW(),
                status VARCHAR(50) DEFAULT 'pending',
                completed_at TIMESTAMP
            )
        """)
        
        # Create audit trail table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_trail (
                id SERIAL PRIMARY KEY,
                event_id VARCHAR(64) UNIQUE NOT NULL,
                event_type VARCHAR(50) NOT NULL,
                entity_type VARCHAR(50) NOT NULL,
                entity_id VARCHAR(128) NOT NULL,
                actor_type VARCHAR(50),
                actor_id VARCHAR(128),
                changes JSONB,
                metadata JSONB,
                timestamp TIMESTAMP DEFAULT NOW()
            )
        """)
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_entity ON audit_trail(entity_id, entity_type);
            CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_trail(timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_audit_event_type ON audit_trail(event_type);
        """)
        
        await conn.close()
        print("✅ PostgreSQL initialized successfully")
        
    except Exception as e:
        print(f"❌ PostgreSQL initialization failed: {e}")
        raise


async def init_neo4j():
    """Initialize Neo4j database"""
    print("\n🕸️  Initializing Neo4j...")
    
    try:
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "releaf_password")
        
        driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        
        async with driver.session() as session:
            # Create constraints
            await session.run("""
                CREATE CONSTRAINT material_id IF NOT EXISTS
                FOR (m:Material) REQUIRE m.id IS UNIQUE
            """)
            
            await session.run("""
                CREATE CONSTRAINT project_id IF NOT EXISTS
                FOR (p:Project) REQUIRE p.id IS UNIQUE
            """)
            
            # Create indexes
            await session.run("""
                CREATE INDEX material_name IF NOT EXISTS
                FOR (m:Material) ON (m.name)
            """)
            
        await driver.close()
        print("✅ Neo4j initialized successfully")
        
    except Exception as e:
        print(f"❌ Neo4j initialization failed: {e}")
        print("   (This is OK if Neo4j is not running)")


def init_qdrant():
    """Initialize Qdrant vector database"""
    print("\n🔍 Initializing Qdrant...")
    
    try:
        client = QdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", "6333"))
        )
        
        # Create collections if they don't exist
        collections = ["sustainability_docs", "upcycling_knowledge"]
        
        for collection_name in collections:
            try:
                client.get_collection(collection_name)
                print(f"   Collection '{collection_name}' already exists")
            except:
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
                )
                print(f"   Created collection '{collection_name}'")
        
        print("✅ Qdrant initialized successfully")
        
    except Exception as e:
        print(f"❌ Qdrant initialization failed: {e}")
        print("   (This is OK if Qdrant is not running)")


async def main():
    """Main initialization"""
    print("="*60)
    print("  DATABASE INITIALIZATION")
    print("="*60)
    
    # Initialize databases
    await init_postgres()
    await init_neo4j()
    init_qdrant()
    
    print("\n" + "="*60)
    print("✅ Database initialization complete!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())

