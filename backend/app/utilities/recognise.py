import asyncio
import logging
from typing import List, Dict, Any
from app.database import q_client, QDRANT_COLLECTION_NAME, Tracking
from app.utilities.embedding import get_image_embedding
from app.models import tracking_serializer

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

EMBEDDING_DIM = 512  
MATCH_THRESHOLD = 0.2360
CHECK_INTERVAL = 30  # seconds

async def process_single_tracking_entry(tracking: Dict[str, Any]) -> None:
    """
    Process a single tracking entry for face recognition.
    """
    track_id = tracking.get("track_id")
    face_images = tracking.get("face_images", [])

    if not face_images:
        logger.warning(f"No face images for track_id: {track_id}")
        return

    recognized_person_id = "unknown"
    face_embedding = None

    for image_path in face_images:
        try:
            embedding = get_image_embedding(image_path)

            search_result = q_client.search(
                collection_name=QDRANT_COLLECTION_NAME,
                query_vector=embedding,
                query_filter=None,
                limit=1
            )

            if search_result and search_result[0].score >= MATCH_THRESHOLD:
                best_match = search_result[0]
                recognized_person_id = best_match.payload.get("person_id", "unknown")
                face_embedding = embedding
                logger.info(f"Match found for track_id {track_id}: {recognized_person_id}")
                break

        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")

    # Always update person_id, defaulting to "unknown" if no match
    update_data = {
        "person_id": recognized_person_id
    }

    if face_embedding is not None:
        update_data["face_embeddings"] = face_embedding.tolist()

    try:
        update_result = await Tracking.update_one(
            {"track_id": track_id},
            {"$set": update_data}
        )

        if update_result.modified_count > 0:
            logger.info(f"Updated track_id {track_id} with person_id: {recognized_person_id}")
        else:
            logger.warning(f"No update performed for track_id {track_id}")

    except Exception as e:
        logger.error(f"Database update error for track_id {track_id}: {e}")

async def recognize_faces():
    """
    Background task for recognizing faces in unprocessed tracking data.
    """
    while True:
        try:
            # Find unprocessed entries
            unprocessed_entries = await Tracking.find({"person_id": "unprocessed"}).to_list()
            
            if not unprocessed_entries:
                logger.info("No unprocessed entries found.")
                await asyncio.sleep(CHECK_INTERVAL)
                continue

            # Process entries concurrently
            tasks = [process_single_tracking_entry(entry) for entry in unprocessed_entries]
            await asyncio.gather(*tasks)

        except Exception as e:
            logger.error(f"Global face recognition task error: {e}")

        # Sleep before next iteration
        logger.info(f"Sleeping for {CHECK_INTERVAL} seconds...")
        await asyncio.sleep(CHECK_INTERVAL)

async def start_recognition_task():
    """
    Starts the face recognition background task.
    """
    logger.info("Initializing face recognition background task...")
    asyncio.create_task(recognize_faces())