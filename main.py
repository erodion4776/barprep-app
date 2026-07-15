import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "https://barprepfront.netlify.app,https://djfydlfxefymvpxgemgy.supabase.co"
).split(",")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("BarPrep Processor starting...")
    logger.info(f"YouTube API: {bool(os.getenv('YOUTUBE_API_KEY'))}")
    logger.info(f"Groq API: {bool(os.getenv('GROQ_API_KEY'))}")
    yield
    logger.info("BarPrep Processor shutting down.")

app = FastAPI(
    title="BarPrep Video Processor",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

class VideoRequest(BaseModel):
    url: str
    topic: str = "Bar Exam"
    order_index: int = 0

@app.get("/")
def home():
    return {
        "status": "healthy",
        "service": "BarPrep Video Processor"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "youtube_api": bool(os.getenv("YOUTUBE_API_KEY")),
        "groq_api": bool(os.getenv("GROQ_API_KEY")),
        "supabase": bool(os.getenv("SUPABASE_URL"))
    }

@app.post("/process-video")
def process_video_endpoint(data: VideoRequest):
    """
    Process a YouTube video:
    1. Get metadata via YouTube Data API v3
    2. Try transcript (10s timeout)
    3. Fallback to description if blocked
    4. Generate AI course content
    5. Save to Supabase
    """
    try:
        from processor import process_video
        result = process_video(
            video_url=data.url,
            topic=data.topic,
            order_index=data.order_index
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Process video error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
