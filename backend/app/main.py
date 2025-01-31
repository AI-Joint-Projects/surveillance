import asyncio
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
from .database import initialize_db, client
from .routers import person, live_cam, surveillance
from app.utilities.track import start_tracking
from app.utilities.recognise import start_recognition_task

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown tasks."""
    try:
        print("Initializing database...")
        await initialize_db()  # Initialize the database
        print("Database initialized successfully.")

        # Run tracking and recognition tasks asynchronously in the background
        print("Tracking tasks started.")
        asyncio.create_task(start_tracking())  # Start tracking tasks in the background

        print("Recognition tasks started.")
        asyncio.create_task(start_recognition_task())  # Start recognition tasks in the background
        
        
        yield  # Yield control back to FastAPI
        
    except Exception as e:
        print(f"Error during lifespan: {e}")
    finally:
        print("Shutting down application...")
        client.close()  # Close the database connection
        print("Database connection closed.")

app = FastAPI(lifespan=lifespan)

# Adding CORS middleware
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(person.router)
app.include_router(surveillance.router)
app.include_router(live_cam.router)
project_root = Path(__file__).resolve().parent  # Going one level up to the root folder

app.mount("/saved_faces", StaticFiles(directory=Path("app/saved_faces")), name="saved_faces")
app.mount("/saved_frames", StaticFiles(directory=Path("app/saved_frames")), name="saved_frames")


@app.get("/")
async def read_root():
    """Root endpoint."""
    return {"message": "Welcome to Surveillance App"}
