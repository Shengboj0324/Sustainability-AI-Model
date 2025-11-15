"""
Knowledge Graph Service - Material relationships and upcycling paths

This service provides:
- Material property queries
- Upcycling path discovery
- Relationship traversal
- Graph-based reasoning
"""

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
import logging
import yaml
from pathlib import Path
from datetime import datetime
import asyncio

# Neo4j imports
try:
    from neo4j import AsyncGraphDatabase, AsyncDriver
    from neo4j.exceptions import ServiceUnavailable, AuthError
except ImportError as e:
    logging.error(f"Missing neo4j dependency: {e}. Install with: pip install neo4j")
    raise

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ReleAF AI Knowledge Graph Service",
    description="Material relationships and upcycling knowledge graph",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


class QueryType(str, Enum):
    """Knowledge graph query types"""
    MATERIAL_PROPERTIES = "material_properties"
    UPCYCLING_PATHS = "upcycling_paths"
    SIMILAR_MATERIALS = "similar_materials"
    RECYCLING_PROCESS = "recycling_process"
    COMPATIBILITY = "compatibility"


class MaterialQuery(BaseModel):
    """Material property query"""
    material_name: str = Field(..., min_length=1, max_length=100)
    include_properties: bool = Field(default=True)
    include_relationships: bool = Field(default=True)


class UpcyclingPathQuery(BaseModel):
    """Upcycling path discovery query"""
    source_material: str = Field(..., min_length=1, max_length=100)
    target_product: Optional[str] = Field(default=None, max_length=100)
    max_depth: int = Field(default=3, ge=1, le=5)
    difficulty_level: Optional[str] = Field(default=None, regex="^(easy|medium|hard)$")


class RelationshipQuery(BaseModel):
    """General relationship query"""
    entity: str = Field(..., min_length=1)
    relationship_type: Optional[str] = None
    max_hops: int = Field(default=2, ge=1, le=4)


class KGResponse(BaseModel):
    """Knowledge graph response"""
    results: List[Dict[str, Any]]
    query_type: str
    num_results: int
    query_time_ms: float
    metadata: Dict[str, Any]


class KnowledgeGraphService:
    """
    Production-grade Knowledge Graph service
    
    Manages connections to Neo4j and provides high-level
    query interfaces for material knowledge and upcycling paths.
    """
    
    def __init__(self, config_path: str = "configs/gnn.yaml"):
        """Initialize KG service"""
        self.config = self._load_config(config_path)
        self.driver: Optional[AsyncDriver] = None
        self.database = self.config.get("neo4j", {}).get("database", "neo4j")
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration with validation"""
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                logger.warning(f"Config file not found: {config_path}, using defaults")
                return self._get_default_config()
            
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {config_path}")
                return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "neo4j": {
                "uri": "bolt://localhost:7687",
                "user": "neo4j",
                "password": "releaf_password",
                "database": "neo4j"
            },
            "query": {
                "timeout": 30,
                "max_results": 100
            }
        }
    
    async def initialize(self):
        """Initialize Neo4j connection"""
        try:
            logger.info("Initializing Knowledge Graph service...")
            
            neo4j_config = self.config["neo4j"]
            uri = neo4j_config.get("uri", "bolt://localhost:7687")
            user = neo4j_config.get("user", "neo4j")
            password = neo4j_config.get("password", "releaf_password")
            
            logger.info(f"Connecting to Neo4j at {uri}")
            
            self.driver = AsyncGraphDatabase.driver(
                uri,
                auth=(user, password),
                max_connection_lifetime=3600,
                max_connection_pool_size=50,
                connection_acquisition_timeout=60
            )
            
            # Verify connectivity
            await self.verify_connectivity()
            
            logger.info("Knowledge Graph service initialized successfully")
            
        except AuthError as e:
            logger.error(f"Neo4j authentication failed: {e}")
            raise
        except ServiceUnavailable as e:
            logger.error(f"Neo4j service unavailable: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize KG service: {e}", exc_info=True)
            raise

    async def verify_connectivity(self):
        """Verify Neo4j connection"""
        try:
            async with self.driver.session(database=self.database) as session:
                result = await session.run("RETURN 1 AS num")
                record = await result.single()
                if record["num"] != 1:
                    raise RuntimeError("Connectivity test failed")
                logger.info("Neo4j connectivity verified")
        except Exception as e:
            logger.error(f"Connectivity verification failed: {e}")
            raise

    async def close(self):
        """Close Neo4j connection"""
        if self.driver:
            await self.driver.close()
            logger.info("Neo4j connection closed")

    async def query_material_properties(
        self,
        material_name: str,
        include_properties: bool = True,
        include_relationships: bool = True
    ) -> Dict[str, Any]:
        """
        Query material properties and relationships

        Args:
            material_name: Name of the material
            include_properties: Include material properties
            include_relationships: Include related materials

        Returns:
            Material data with properties and relationships
        """
        try:
            async with self.driver.session(database=self.database) as session:
                # Build Cypher query
                query = """
                MATCH (m:Material {name: $material_name})
                OPTIONAL MATCH (m)-[r]->(related)
                RETURN m, collect({type: type(r), node: related}) as relationships
                """

                result = await session.run(query, material_name=material_name)
                record = await result.single()

                if not record:
                    return {
                        "material": material_name,
                        "found": False,
                        "properties": {},
                        "relationships": []
                    }

                material_node = record["m"]
                relationships = record["relationships"]

                # Extract properties
                properties = dict(material_node) if include_properties else {}

                # Extract relationships
                rel_list = []
                if include_relationships:
                    for rel in relationships:
                        if rel["node"]:
                            rel_list.append({
                                "type": rel["type"],
                                "target": dict(rel["node"])
                            })

                return {
                    "material": material_name,
                    "found": True,
                    "properties": properties,
                    "relationships": rel_list
                }

        except Exception as e:
            logger.error(f"Material query failed: {e}", exc_info=True)
            raise

    async def find_upcycling_paths(
        self,
        source_material: str,
        target_product: Optional[str] = None,
        max_depth: int = 3,
        difficulty_level: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Find upcycling paths from source material to products

        Args:
            source_material: Starting material
            target_product: Optional target product
            max_depth: Maximum path length
            difficulty_level: Filter by difficulty (easy, medium, hard)

        Returns:
            List of upcycling paths with steps and metadata
        """
        try:
            async with self.driver.session(database=self.database) as session:
                # Build Cypher query based on parameters
                if target_product:
                    # Find specific path
                    query = """
                    MATCH path = (m:Material {name: $source})-[:CAN_BECOME*1..%d]->(p:Product {name: $target})
                    WHERE ALL(r IN relationships(path) WHERE
                        CASE WHEN $difficulty IS NOT NULL
                        THEN r.difficulty = $difficulty
                        ELSE true END
                    )
                    RETURN path,
                           [r IN relationships(path) | r.difficulty] as difficulties,
                           [r IN relationships(path) | r.tools_required] as tools,
                           length(path) as steps
                    LIMIT 10
                    """ % max_depth

                    result = await session.run(
                        query,
                        source=source_material,
                        target=target_product,
                        difficulty=difficulty_level
                    )
                else:
                    # Find all possible products
                    query = """
                    MATCH path = (m:Material {name: $source})-[:CAN_BECOME*1..%d]->(p:Product)
                    WHERE ALL(r IN relationships(path) WHERE
                        CASE WHEN $difficulty IS NOT NULL
                        THEN r.difficulty = $difficulty
                        ELSE true END
                    )
                    RETURN path,
                           p.name as product,
                           [r IN relationships(path) | r.difficulty] as difficulties,
                           [r IN relationships(path) | r.tools_required] as tools,
                           length(path) as steps
                    ORDER BY steps ASC
                    LIMIT 20
                    """ % max_depth

                    result = await session.run(
                        query,
                        source=source_material,
                        difficulty=difficulty_level
                    )

                # Process results
                paths = []
                async for record in result:
                    path_data = {
                        "source": source_material,
                        "target": target_product or record.get("product"),
                        "steps": record["steps"],
                        "difficulties": record["difficulties"],
                        "tools_required": record["tools"],
                        "path_nodes": self._extract_path_nodes(record["path"])
                    }
                    paths.append(path_data)

                return paths

        except Exception as e:
            logger.error(f"Upcycling path query failed: {e}", exc_info=True)
            raise

    def _extract_path_nodes(self, path) -> List[Dict[str, Any]]:
        """Extract nodes from Neo4j path object"""
        try:
            nodes = []
            for node in path.nodes:
                nodes.append({
                    "id": node.id,
                    "labels": list(node.labels),
                    "properties": dict(node)
                })
            return nodes
        except Exception as e:
            logger.warning(f"Failed to extract path nodes: {e}")
            return []

    async def query_relationships(
        self,
        entity: str,
        relationship_type: Optional[str] = None,
        max_hops: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Query general relationships from an entity

        Args:
            entity: Starting entity name
            relationship_type: Optional relationship type filter
            max_hops: Maximum number of hops

        Returns:
            List of related entities with relationship info
        """
        try:
            async with self.driver.session(database=self.database) as session:
                # Build query
                if relationship_type:
                    query = f"""
                    MATCH path = (start)-[r:{relationship_type}*1..{max_hops}]->(end)
                    WHERE start.name = $entity
                    RETURN path, length(path) as hops
                    LIMIT 50
                    """
                else:
                    query = f"""
                    MATCH path = (start)-[*1..{max_hops}]->(end)
                    WHERE start.name = $entity
                    RETURN path, length(path) as hops
                    LIMIT 50
                    """

                result = await session.run(query, entity=entity)

                relationships = []
                async for record in result:
                    relationships.append({
                        "hops": record["hops"],
                        "path": self._extract_path_nodes(record["path"])
                    })

                return relationships

        except Exception as e:
            logger.error(f"Relationship query failed: {e}", exc_info=True)
            raise


# Initialize service
kg_service = KnowledgeGraphService()


@app.on_event("startup")
async def startup():
    """Initialize service on startup"""
    await kg_service.initialize()


@app.on_event("shutdown")
async def shutdown():
    """Close connections on shutdown"""
    await kg_service.close()


@app.post("/material/properties", response_model=KGResponse)
async def get_material_properties(query: MaterialQuery):
    """
    Get material properties and relationships

    Returns comprehensive information about a material including
    its properties, recycling process, and related materials.
    """
    try:
        start_time = datetime.now()

        result = await kg_service.query_material_properties(
            material_name=query.material_name,
            include_properties=query.include_properties,
            include_relationships=query.include_relationships
        )

        query_time = (datetime.now() - start_time).total_seconds() * 1000

        return KGResponse(
            results=[result],
            query_type="material_properties",
            num_results=1 if result["found"] else 0,
            query_time_ms=query_time,
            metadata={
                "material": query.material_name,
                "found": result["found"]
            }
        )

    except Exception as e:
        logger.error(f"Material properties request failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}"
        )


@app.post("/upcycling/paths", response_model=KGResponse)
async def find_upcycling_paths_endpoint(query: UpcyclingPathQuery):
    """
    Find upcycling paths from material to products

    Discovers creative ways to transform waste materials into
    useful products, with step-by-step guidance.
    """
    try:
        start_time = datetime.now()

        paths = await kg_service.find_upcycling_paths(
            source_material=query.source_material,
            target_product=query.target_product,
            max_depth=query.max_depth,
            difficulty_level=query.difficulty_level
        )

        query_time = (datetime.now() - start_time).total_seconds() * 1000

        return KGResponse(
            results=paths,
            query_type="upcycling_paths",
            num_results=len(paths),
            query_time_ms=query_time,
            metadata={
                "source": query.source_material,
                "target": query.target_product,
                "max_depth": query.max_depth,
                "difficulty": query.difficulty_level
            }
        )

    except Exception as e:
        logger.error(f"Upcycling paths request failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}"
        )


@app.post("/relationships", response_model=KGResponse)
async def query_relationships_endpoint(query: RelationshipQuery):
    """
    Query general relationships from an entity

    Explores connections between materials, products, and processes
    in the knowledge graph.
    """
    try:
        start_time = datetime.now()

        relationships = await kg_service.query_relationships(
            entity=query.entity,
            relationship_type=query.relationship_type,
            max_hops=query.max_hops
        )

        query_time = (datetime.now() - start_time).total_seconds() * 1000

        return KGResponse(
            results=relationships,
            query_type="relationships",
            num_results=len(relationships),
            query_time_ms=query_time,
            metadata={
                "entity": query.entity,
                "relationship_type": query.relationship_type,
                "max_hops": query.max_hops
            }
        )

    except Exception as e:
        logger.error(f"Relationships request failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}"
        )


@app.get("/health")
async def health():
    """Health check endpoint"""
    is_healthy = kg_service.driver is not None

    # Try a simple query to verify connection
    if is_healthy:
        try:
            async with kg_service.driver.session(database=kg_service.database) as session:
                await session.run("RETURN 1")
        except Exception as e:
            logger.warning(f"Health check query failed: {e}")
            is_healthy = False

    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "service": "knowledge_graph",
        "neo4j_connected": is_healthy,
        "database": kg_service.database
    }


@app.get("/stats")
async def get_stats():
    """Get knowledge graph statistics"""
    try:
        if kg_service.driver is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Neo4j not connected"
            )

        async with kg_service.driver.session(database=kg_service.database) as session:
            # Count nodes by label
            node_counts = {}
            for label in ["Material", "Product", "Process"]:
                result = await session.run(f"MATCH (n:{label}) RETURN count(n) as count")
                record = await result.single()
                node_counts[label.lower()] = record["count"] if record else 0

            # Count relationships
            result = await session.run("MATCH ()-[r]->() RETURN count(r) as count")
            record = await result.single()
            relationship_count = record["count"] if record else 0

            return {
                "database": kg_service.database,
                "nodes": node_counts,
                "relationships": relationship_count,
                "total_nodes": sum(node_counts.values())
            }

    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8004,
        reload=False,
        log_level="info"
    )

