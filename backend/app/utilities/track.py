from bson import ObjectId
import cv2
from ultralytics import YOLO
from datetime import datetime,timezone
import time
import os
import asyncio
import logging
from typing import Optional
from collections import defaultdict
from app.models import TrackingModel, tracking_serializer
from app.database import Tracking

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load YOLO models
person_model = YOLO('yolov8n.pt', verbose=False)
face_model = YOLO('best.pt', verbose=False)

# Constants
MOBILE_STREAM_URL = "http://192.168.1.64:8080/video"
FACE_IMAGES_FOLDER = "app/saved_faces"
FRAME_IMAGES_FOLDER="app/saved_frames"
os.makedirs(FRAME_IMAGES_FOLDER,exist_ok=True)
os.makedirs(FACE_IMAGES_FOLDER, exist_ok=True)

# Global variables for tracking
next_id = 91  # Unique global counter for track IDs
track_to_seq_id = {}  # Map tracker-assigned IDs to unique `track_id`
person_timestamps = defaultdict(lambda: {'entry': None, 'exit': None})
last_image_capture = defaultdict(lambda: None)
video_stream = None
use_webcam = False

def capture_frame() -> Optional[cv2.Mat]:
    """Captures a frame from the video stream."""
    global video_stream, use_webcam
    try:
        if use_webcam:
            video_stream = cv2.VideoCapture(0) if video_stream is None or not video_stream.isOpened() else video_stream
        else:
            video_stream = cv2.VideoCapture(MOBILE_STREAM_URL) if video_stream is None or not video_stream.isOpened() else video_stream
            if not video_stream.isOpened():
                logger.warning("IP camera failed. Switching to webcam.")
                use_webcam = True
                return capture_frame()

        ret, frame = video_stream.read()
        return frame if ret else None
    except Exception as e:
        logger.error(f"Frame capture error: {e}")
        return None

def detect_and_track_person(frame: cv2.Mat):
    """Detects and tracks persons in the given frame."""
    global next_id, track_to_seq_id, person_timestamps, last_image_capture
    try:
        results = person_model.track(source=frame, stream=True, tracker="bytetrack.yaml")
        current_frame_tracks = set()

        for result in results:
            for box in result.boxes:
                if box.id is None or box.cls[0].item() != 0:
                    continue

                tracker_id = int(box.id)

                # Check if this tracker_id has already been assigned a unique_track_id
                if tracker_id in track_to_seq_id:
                    unique_track_id = track_to_seq_id[tracker_id]
                else:
                    # Assign a new unique ID if not already assigned
                    unique_track_id = next_id
                    next_id += 1
                    track_to_seq_id[tracker_id] = unique_track_id

                current_frame_tracks.add(unique_track_id)

                # Set entry timestamp if not already set
                if unique_track_id not in person_timestamps:
                    person_timestamps[unique_track_id] = {'entry': datetime.now(), 'exit': None}

                # Draw bounding box for the person
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {unique_track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Extract the person region and process for face detection
                person_region = frame[y1:y2, x1:x2]
                if person_region.size > 0:
                    process_face_detection(frame, unique_track_id, person_region, (x1, y1))

        # Update exit timestamps for tracks no longer in the frame
        for track_id, unique_id in list(track_to_seq_id.items()):
            if unique_id not in current_frame_tracks and person_timestamps[unique_id]['exit'] is None:
                person_timestamps[unique_id]['exit'] = datetime.now()
                # Optional: Remove old tracker_id from mapping
                del track_to_seq_id[track_id]

    except Exception as e:
        logger.error(f"Person detection error: {e}")

def process_face_detection(frame: cv2.Mat, track_id: int, person_region: cv2.Mat, offset: tuple):
    """Processes face detection within a person's bounding box."""
    global last_image_capture
    try:
        face_results = face_model(person_region)
        for face_box in face_results[0].boxes:
            if face_box.conf[0] < 0.5:  # Confidence threshold for face detection
                continue

            # Extract face bounding box
            fx1, fy1, fx2, fy2 = map(int, face_box.xyxy[0])
            face_region = person_region[fy1:fy2, fx1:fx2]

            # Draw bounding box for the face
            global_x1, global_y1 = offset[0] + fx1, offset[1] + fy1
            global_x2, global_y2 = offset[0] + fx2, offset[1] + fy2
            cv2.rectangle(frame, (global_x1, global_y1), (global_x2, global_y2), (255, 0, 0), 2)  # Blue for face
            cv2.putText(frame, f"Face ID: {track_id}", (global_x1, global_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Save the face region image if necessary
            if face_region.size > 0:
                current_time = time.time()
                last_capture_time = last_image_capture[track_id]

                if last_capture_time is None or (current_time - last_capture_time) >= 5:
                    frame_filename = f"{FRAME_IMAGES_FOLDER}/track_{track_id}_{int(current_time)}.jpg"
                    face_filename = f"{FACE_IMAGES_FOLDER}/track_{track_id}_{int(current_time)}.jpg"
                    cv2.imwrite(face_filename, face_region)
                    cv2.imwrite(frame_filename, frame)
                    last_image_capture[track_id] = current_time

    except Exception as e:
        logger.error(f"Face detection error for track_id {track_id}: {e}")

async def continuous_tracking():
    """Continuously captures frames and performs detection and tracking."""
    while True:
        try:
            frame = await asyncio.to_thread(capture_frame)
            if frame is not None:
                await asyncio.to_thread(detect_and_track_person, frame)

                # Display the frame with bounding boxes
                cv2.imshow("Tracking", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
                    break
        except Exception as e:
            logger.error(f"Continuous tracking error: {e}")
        await asyncio.sleep(0.1)

async def insert_tracking_data_periodically():
    """Inserts tracking data periodically into the database."""
    entries_to_insert = set()
    while True:
        try:
            for track_id, timestamps in list(person_timestamps.items()):
                try:
                    # Entry insertion
                    if track_id not in entries_to_insert and timestamps['entry'] is not None:
                        # Get the face images associated with the current track_id
                        face_images = [
                            os.path.join(FACE_IMAGES_FOLDER, file)
                            for file in os.listdir(FACE_IMAGES_FOLDER)
                            if file.startswith(f"track_{track_id}_")
                        ]
                        frame_images = [
                            os.path.join(FRAME_IMAGES_FOLDER, file)
                            for file in os.listdir(FRAME_IMAGES_FOLDER)
                            if file.startswith(f"track_{track_id}_")
                        ]

                        # Prepare tracking data model with person_id defaulting to "unprocessed"
                        tracking_data = TrackingModel(
                            person_id="unprocessed",  # Default value
                            time_entered=timestamps['entry'],
                            time_exited=timestamps['exit'],
                            face_images=face_images,
                            frame_images=frame_images
                        )

                        # Serialize the data for database insertion
                        serialized_data = tracking_serializer(tracking_data.model_dump())

                        # Add track_id to the serialized data
                        serialized_data["track_id"] = track_id
                        
                        # Optionally, add an ObjectId explicitly (MongoDB will do it automatically if not provided)
                        serialized_data["_id"] = str(ObjectId())  # Generate a new ObjectId if needed

                        # Insert the serialized data into the database
                        result = await asyncio.to_thread(Tracking.insert_one, serialized_data)
                        
                        if result:
                            logger.info(f"Inserted tracking data for track_id {track_id}")
                            entries_to_insert.add(track_id)
                        else:
                            logger.warning(f"Failed to insert tracking data for track_id {track_id}")

                except Exception as e:
                    logger.error(f"Error processing track_id {track_id}: {e}")

        except Exception as e:
            logger.error(f"Tracking data insertion error: {e}")
        
        await asyncio.sleep(10)  

async def start_tracking():
    """Starts the tracking system."""
    tracking_tasks = [
        asyncio.create_task(insert_tracking_data_periodically()),
        asyncio.create_task(continuous_tracking())
    ]
    try:
        await asyncio.gather(*tracking_tasks)
    except Exception as e:
        logger.error(f"Tracking task error: {e}")
