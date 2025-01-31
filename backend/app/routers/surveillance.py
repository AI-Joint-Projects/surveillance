from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.staticfiles import StaticFiles
from app.database import Tracking
from datetime import datetime
from app.models import tracking_serializer
from pathlib import Path




# Create the router for the surveillance endpoints
router = APIRouter(
    prefix="/surveillance",
    tags=["Surveillance"]
)

@router.get("/search")
async def search(start_time: datetime, end_time: datetime):
    try:
        results = await Tracking.find({
            "time_entered": {
                "$gte": start_time,
                "$lte": end_time
            }
        }).to_list()

        if not results:
            raise HTTPException(status_code=404, detail="No records found for the given time range.")

        # Serialize the results before returning
        serialized_results = []
        for record in results:
            record_data = tracking_serializer(record)  # Use the updated serializer here
            # If needed, convert file paths to URLs here
            if 'face_images' in record:
                record_data['face_images_urls'] = [
                    f"http://localhost:8000/saved_faces/{Path(image).name}" for image in record['face_images']
                ]
            if 'frame_images' in record:
                record_data['frame_images_urls'] = [
                    f"http://localhost:8000/saved_frames/{Path(image).name}" for image in record['frame_images']
                ]
            serialized_results.append(record_data)

        return {"message": "Data retrieved successfully", "data": serialized_results}

    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error while fetching data: {str(e)}")
