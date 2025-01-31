from fastapi import APIRouter
from fastapi.responses import StreamingResponse
import cv2
import asyncio
import logging
from app.utilities.track import start_tracking

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cam", tags=["Camera"])

# Lock and global variable for safe frame access
frame_lock = asyncio.Lock()
latest_frame = None

async def update_latest_frame(frame):
    """Safely update the global latest frame."""
    global latest_frame
    async with frame_lock:
        latest_frame = frame

@router.get("/stream")
async def stream_video():
    """
    API endpoint to stream video frames processed by the tracking system.
    """
    async def frame_generator():
        """Generate frames from the tracking system."""
        while True:
            async with frame_lock:
                if latest_frame is not None:
                    # Encode the frame with annotations to JPEG
                    _, jpeg_frame = cv2.imencode('.jpg', latest_frame)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg_frame.tobytes() + b'\r\n')
            await asyncio.sleep(0.05)  # Delay for smooth streaming

    # Start the tracking system if not already running
    if latest_frame is None:
        logger.info("Starting tracking tasks...")
        asyncio.create_task(start_tracking())

    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

