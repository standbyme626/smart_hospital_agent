import asyncio
import os
import sys
from neo4j import AsyncGraphDatabase

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from backend.app.db.neo4j_client import neo4j_client

async def init_data():
    print("Starting Neo4j data ingestion...")
    
    # Ensure connection
    await neo4j_client.connect()
    if not neo4j_client.driver:
        print("Could not connect to Neo4j. Is it running?")
        return

    # Initialize schema first
    await neo4j_client.init_schema()

    # Data to ingest
    departments = ["Neurology", "Cardiology", "Gastroenterology", "General Internal Medicine"]
    
    diseases = [
        {"name": "Migraine", "dept": "Neurology"},
        {"name": "Tension Headache", "dept": "Neurology"},
        {"name": "Hypertension", "dept": "Cardiology"},
        {"name": "Gastritis", "dept": "Gastroenterology"},
        {"name": "Upper Respiratory Infection", "dept": "General Internal Medicine"},
    ]
    
    symptoms = [
        "Headache", "Nausea", "Dizziness", "Stomach Pain", "Cough", "Fever", "Photophobia"
    ]
    
    medications = [
        {"name": "Ibuprofen", "treats": ["Headache", "Fever", "Tension Headache"]},
        {"name": "Sumatriptan", "treats": ["Migraine"]},
        {"name": "Lisinopril", "treats": ["Hypertension"]},
        {"name": "Omeprazole", "treats": ["Gastritis"]},
    ]
    
    # Relationships: Disease -> HAS_SYMPTOM -> Symptom
    disease_symptoms = [
        ("Migraine", ["Headache", "Nausea", "Photophobia", "Dizziness"]),
        ("Tension Headache", ["Headache"]),
        ("Hypertension", ["Headache", "Dizziness"]),
        ("Gastritis", ["Stomach Pain", "Nausea"]),
        ("Upper Respiratory Infection", ["Cough", "Fever", "Headache"]),
    ]

    async with neo4j_client.driver.session() as session:
        print("Clearing existing data (optional)...")
        # await session.run("MATCH (n) DETACH DELETE n") # Uncomment to clear DB

        print("Creating Departments...")
        for dept in departments:
            await session.run("MERGE (d:Department {name: $name})", name=dept)
            
        print("Creating Symptoms...")
        for sym in symptoms:
            await session.run("MERGE (s:Symptom {name: $name})", name=sym)
            
        print("Creating Diseases and linking to Departments...")
        for disease in diseases:
            query = """
            MERGE (d:Disease {name: $d_name})
            MERGE (dept:Department {name: $dept_name})
            MERGE (d)-[:BELONGS_TO]->(dept)
            """
            await session.run(query, d_name=disease["name"], dept_name=disease["dept"])
            
        print("Linking Diseases to Symptoms...")
        for d_name, sym_list in disease_symptoms:
            for sym in sym_list:
                query = """
                MATCH (d:Disease {name: $d_name})
                MATCH (s:Symptom {name: $s_name})
                MERGE (d)-[:HAS_SYMPTOM]->(s)
                """
                await session.run(query, d_name=d_name, s_name=sym)
                
        print("Creating Medications and linking to Treatments...")
        for med in medications:
            await session.run("MERGE (m:Medication {name: $name})", name=med["name"])
            for condition in med["treats"]:
                # Try to find if it's a symptom or disease
                query = """
                MATCH (m:Medication {name: $m_name})
                MATCH (t) WHERE (t:Symptom OR t:Disease) AND t.name = $t_name
                MERGE (m)-[:TREATS]->(t)
                """
                await session.run(query, m_name=med["name"], t_name=condition)

    print("Data ingestion complete.")
    await neo4j_client.close()

if __name__ == "__main__":
    asyncio.run(init_data())
