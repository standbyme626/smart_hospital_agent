import os
import asyncio
from neo4j import AsyncGraphDatabase
import structlog

logger = structlog.get_logger(__name__)

class Neo4jClient:
    def __init__(self):
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "password")
        self.driver = None

    async def connect(self):
        if not self.driver:
            try:
                self.driver = AsyncGraphDatabase.driver(
                    self.uri, 
                    auth=(self.user, self.password)
                )
                await self.driver.verify_connectivity()
                logger.info("Connected to Neo4j", uri=self.uri)
            except Exception as e:
                logger.error("Failed to connect to Neo4j", error=str(e))
                # We don't raise here to allow the app to start without Neo4j (soft failure)
                self.driver = None

    async def close(self):
        if self.driver:
            await self.driver.close()
            self.driver = None

    async def init_schema(self):
        """
        Initialize schema constraints and indexes.
        """
        if not self.driver:
            return

        queries = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Symptom) REQUIRE s.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Disease) REQUIRE d.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Medication) REQUIRE m.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (dept:Department) REQUIRE dept.name IS UNIQUE",
            # Add fulltext index for fuzzy search if needed (Enterprise or recent Community versions)
            # "CREATE FULLTEXT INDEX symptomNames IF NOT EXISTS FOR (n:Symptom) ON EACH [n.name]"
        ]
        
        async with self.driver.session() as session:
            for q in queries:
                try:
                    await session.run(q)
                except Exception as e:
                    logger.warning("Schema init warning", query=q, error=str(e))
        logger.info("Neo4j schema initialized")

    async def get_related_entities(self, entity_name: str, entity_type: str = "Symptom", relation_type: str = "HAS_SYMPTOM", direction: str = "INCOMING", limit: int = 5):
        """
        Find entities related to the given entity.
        e.g. Find diseases (INCOMING) that have this symptom.
        """
        if not self.driver:
            return []

        # Cypher query construction based on direction
        # INCOMING: (other)-[:REL]->(start)
        # OUTGOING: (start)-[:REL]->(other)
        
        if direction == "INCOMING":
            match_clause = f"MATCH (n)-[r:{relation_type}]->(s:{entity_type} {{name: $name}})"
        else:
            match_clause = f"MATCH (s:{entity_type} {{name: $name}})-[r:{relation_type}]->(n)"
            
        query = f"""
        {match_clause}
        RETURN n.name AS name, labels(n) AS labels, type(r) AS relation
        LIMIT $limit
        """
        
        try:
            async with self.driver.session() as session:
                result = await session.run(query, name=entity_name, limit=limit)
                records = [record.data() for record in await result.data()]
                return records
        except Exception as e:
            logger.error("Neo4j query failed", error=str(e))
            return []

# Singleton instance
neo4j_client = Neo4jClient()
