import asyncio
from pathlib import Path
from fastapi import APIRouter,File, Request,Body,HTTPException,  Depends, Response, status,UploadFile, Form
from fastapi.responses import JSONResponse
from bson import ObjectId
from typing import List
from datetime import datetime
from .. import schema
from ..database import People, q_client     
from ..models import PeopleModel, person_serializer
from qdrant_client.models import PointStruct,FilterSelector,Filter,FieldCondition,MatchValue
import numpy as np
import os
from ..utilities import embedding
import uuid

router = APIRouter(
    prefix="/people",
    tags=["People"]
)

UPLOAD_DIR = "./uploads"

# Ensure the upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

async def insert_person_to_db(person_data: dict):
    result = await People.insert_one(person_data)
    return str(result.inserted_id)  # Return the inserted ID (if needed)

# Check if username is unique
async def is_username_unique(username: str):
    existing_user = await People.find_one({"person_id": username})
    return existing_user is None

# Save image to disk
async def save_image_to_disk(upload_file: UploadFile, save_dir: str) -> str:
    file_path = os.path.join(save_dir, upload_file.filename)
    with open(file_path, "wb") as image_file:
        image_file.write(await upload_file.read())
    return file_path

# Insert embedding into Qdrant
def save_embedding_to_qdrant(embedding: np.ndarray, metadata: dict, collection_name="faces"):
    point = PointStruct(
        id=str(uuid.uuid4()),  # Unique identifier
        vector=embedding,
        payload=metadata
    )
    q_client.upsert(collection_name=collection_name, points=[point])

@router.post("/register", status_code=status.HTTP_201_CREATED)
async def person_registration(
    first_name: str = Form(...),
    last_name: str = Form(...),
    phone_number: str = Form(...),
    birth_date: datetime = Form(...),
    role: str = Form(...),
    person_id: str = Form(...),
    images: List[UploadFile] = File(...)
):
    person_data = PeopleModel(
        first_name=first_name,
        last_name=last_name,
        phone_number=phone_number,
        birth_date=birth_date,
        role=role,
        person_id=person_id
    )
    # Ensure unique username
    if not await is_username_unique(person_id):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already exists")


    
    # Convert to dict for MongoDB
    person_dict = person_data.model_dump(by_alias=True)
    person_dict["_id"] = ObjectId()

    # Create a directory for this user's images
    user_dir = os.path.join(UPLOAD_DIR, person_id)
    os.makedirs(user_dir, exist_ok=True)

    # Process and save each image
    for image in images:
        image_path = await save_image_to_disk(image, user_dir)
        emb = embedding.get_image_embedding(image_path)

        metadata = {
            "image_path": image_path,
            "person_id": person_id,
            "identifier": str(uuid.uuid4())
        }

        save_embedding_to_qdrant(emb, metadata)

    # Insert data into MongoDB
    inserted_id = await insert_person_to_db(person_dict)

    return JSONResponse(content={
        "message": "Person registered successfully",
        "person_data": person_serializer(person_dict | {"_id": inserted_id}),
        "inserted_id": inserted_id
    })

@router.get("/list")
async def person_list():
    # Fetch all people from the MongoDB database
    people_cursor = People.find()  # Returns a cursor to iterate over documents
    
    # Convert the cursor to a list and serialize the data
    people_list = []
    async for person in people_cursor:
        # Use the serializer to format the person data
        serialized_person = person_serializer(person)
        people_list.append(serialized_person)
    
    return JSONResponse(content={
        "message": "People list retrieved successfully",
        "people": people_list
    })


@router.delete("/remove")
async def person_remove(person_id: str):
    # Delete from MongoDB
    result = await People.delete_one({"person_id": person_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Person not found")

    q_client.delete(
    collection_name="faces",
    points_selector=FilterSelector(
        filter=Filter(
            must=[
                FieldCondition(
                    key="person_id",
                    match=MatchValue(value=person_id),
                ),
            ],
        )
    ),
)

    return {"message": "Person removed successfully"}


async def get_images_for_person(person_id: str) -> List[str]:
    # Define the directory where the person's images are stored
    person_dir = Path(UPLOAD_DIR) / person_id
    
    # Check if the directory exists
    if not person_dir.exists() or not person_dir.is_dir():
        raise HTTPException(status_code=404, detail="Person's images not found")
    
    # Get all image files in the directory (assuming .jpg, .png, etc. extensions)
    image_files = list(person_dir.glob("*.{jpg,png}"))
    
    # Return the file paths (you can return URLs instead if serving images from a public URL)
    return [str(image_file) for image_file in image_files]


@router.get("/images/{person_id}", response_model=List[str])
async def get_images(person_id:str):
    # Path to the person's image folder
    person_folder = os.path.join(UPLOAD_DIR, person_id)

    # Check if the folder exists
    if not os.path.exists(person_folder):
        raise HTTPException(status_code=404, detail="Person folder not found")

    # List all files in the directory (you can filter by image file extensions if needed)
    image_files = [f for f in os.listdir(person_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]

    # If no images found, return an error
    if not image_files:
        raise HTTPException(status_code=404, detail="No images found for this person")

    # Return the list of image file names (or full URLs if needed)
    image_urls = [f"/uploads/{person_id}/{image_file}" for image_file in image_files]
    return image_urls

