"""
This Module consists of code for database connection:
1. MongoDB connection
2. QdrantDB connection
"""
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.models import Distance,VectorParams
import os

os.environ.pop("SSL_CERT_FILE", None)
# Database configuration
DATABASE_URL = "mongodb://localhost:27017"
DATABASE_NAME = "surveillance"

# Qdrant Configuration
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_COLLECTION_NAME = "faces"
VECTOR_SIZE = 512
DISTANCE = Distance.COSINE

# Connect to MongoDB
client = AsyncIOMotorClient(DATABASE_URL)
db = client[DATABASE_NAME]

# Collections
People = db.get_collection("people")
Tracking = db.get_collection("tracking")

# Connect to Qdrant Database
q_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


async def initialize_qdrant():
    try:
        collection_info = q_client.get_collection(QDRANT_COLLECTION_NAME)
        print(f"Collection exists: {collection_info}")
    except Exception as e:
        print(f"Collection not found. Creating new collection: {e}")
        q_client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,  # Replace with your vector size
                distance="Cosine"  # Or use "Euclidean" or another metric
            ),
        )


async def initialize_mongodb():
    """
    Ensure MongoDB is initialized with default data.
    """
    # Check if a person document exists, if not, insert one
    person = await People.find_one()
    if not person:
        person_document = {
            "_id": ObjectId(),
            "person_id": 0,
            "first_name": "Test",
            "last_name": "Case",
            "phone_number": "9818615071",
            "role": "test subject",
            "birth_date": "2000-01-01T00:00:00Z",
        }
        await People.insert_one(person_document)

    # Check if a tracking document exists, if not, insert one
    tracking = await Tracking.find_one()
    if not tracking:
        event_document = {
            "_id": ObjectId(),
            "person_id": "unprocessed",
            "time_entered": datetime(2024, 12, 25, 12, 0, 0),
            "time_exited": datetime(2024, 12, 25, 12, 15, 0),
            "face_images": [],
            "frame_images":[],
            "track_id":100000
        }
        await Tracking.insert_one(event_document)

    print("MongoDB initialized successfully.")


async def initialize_db():
    """
    Initialize both Qdrant and MongoDB.
    """
    await initialize_qdrant()
    await initialize_mongodb()
