import os
import re
import time
import hashlib
import logging
import numpy as np
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

GROQ_API_KEY    = os.getenv("GROQ_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
SUPABASE_URL    = os.getenv("SUPABASE_URL")
SUPABASE_KEY    = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
HF_TOKEN        = os.getenv("HF_TOKEN")

HF_EMBED_URL  = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
MAX_CHUNKS    = 200
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200

# ---- Embedding ----

def generate_hash_embedding(text: str) -> list:
    embedding = np.zeros(EMBEDDING_DIM)
    for seed in range(EMBEDDING_DIM):
        hash_input = f"{seed}:{text}".encode("utf-8")
        hash_val = int(hashlib.md5(hash_input).hexdigest(), 16)
        embedding[seed] = (hash_val % 10000) / 10000.0 - 0.5
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding.tolist()

def get_embeddings_batch(texts: list) -> list:
    import requests
    clean_texts = [t.replace("\n", " ").strip() for t in texts]
    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    try:
        response = requests.post(
            HF_EMBED_URL,
            headers=headers,
            json={
                "inputs": clean_texts,
                "options": {"wait_for_model": True}
            },
            timeout=30
        )
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                logger.info("Embeddings via HuggingFace")
                return result
    except Exception as e:
        logger.warning(f"HuggingFace failed: {e}")

    logger.info("Using hash embeddings")
    return [generate_hash_embedding(text) for text in clean_texts]

def split_text(text: str) -> list:
    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    start = 0
    while start < len(text):
        chunks.append(text[start:start + CHUNK_SIZE])
        start += step
    return chunks

# ---- YouTube Data API v3 ----

def get_video_metadata(video_id: str) -> dict:
    """Get title, thumbnail, description via YouTube Data API v3"""
    fallback = {
        "title": f"YouTube Video {video_id}",
        "description": "",
        "thumbnail": f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg",
        "channel": ""
    }

    if not YOUTUBE_API_KEY:
        logger.warning("No YOUTUBE_API_KEY")
        return fallback

    try:
        from googleapiclient.discovery import build
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

        response = youtube.videos().list(
            part="snippet",
            id=video_id
        ).execute()

        if not response.get("items"):
            return fallback

        snippet = response["items"][0]["snippet"]
        thumbnails = snippet.get("thumbnails", {})
        thumbnail = (
            thumbnails.get("maxres", {}).get("url") or
            thumbnails.get("standard", {}).get("url") or
            thumbnails.get("high", {}).get("url") or
            fallback["thumbnail"]
        )

        logger.info(f"Got metadata: {snippet['title']}")
        return {
            "title": snippet.get("title", fallback["title"]),
            "description": snippet.get("description", "")[:1000],
            "thumbnail": thumbnail,
            "channel": snippet.get("channelTitle", "")
        }

    except Exception as e:
        logger.error(f"YouTube API metadata failed: {e}")
        return fallback

# ---- Transcript with Timeout + Fallback ----

def get_transcript_with_timeout(
    video_id: str,
    timeout_seconds: int = 10
) -> Optional[str]:
    """
    Tries youtube-transcript-api with a hard timeout.
    Returns None if blocked or timed out.
    Never hangs the pipeline.
    """
    import signal

    def timeout_handler(signum, frame):
        raise TimeoutError("Transcript fetch timed out")

    # Set timeout alarm
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)

    try:
        from youtube_transcript_api import (
            YouTubeTranscriptApi,
            TranscriptsDisabled,
            NoTranscriptFound
        )

        transcript_list = YouTubeTranscriptApi.get_transcript(
            video_id,
            languages=["en", "en-US", "en-GB"]
        )

        text = " ".join([entry["text"] for entry in transcript_list])
        logger.info(f"Transcript: {len(text)} chars")
        return text

    except TimeoutError:
        logger.warning(f"Transcript timed out after {timeout_seconds}s")
        return None
    except Exception as e:
        logger.warning(f"Transcript failed: {e}")
        return None
    finally:
        # Cancel the alarm
        signal.alarm(0)

# ---- Groq AI Content Generation ----

def generate_course_content(
    text: str,
    title: str,
    topic: str
) -> dict:
    """Generate AI summary and outline using Groq"""
    fallback = {
        "summary": f"This lecture covers essential {topic} concepts for the bar exam.",
        "outline": f"• Core {topic} principles\n• Key legal rules\n• Important cases\n• Bar exam applications\n• Practice tips"
    }

    if not GROQ_API_KEY:
        return fallback

    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)

        prompt = f"""Topic: {topic}
Video: {title}
Content: {text[:6000]}

Format EXACTLY:

SUMMARY:
[3-4 sentences about what bar exam students will learn]

OUTLINE:
• [Key concept 1]
• [Key concept 2]
• [Key concept 3]
• [Key concept 4]
• [Key concept 5]"""

        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a bar exam course creator. Create structured educational content."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.1-8b-instant",
            max_tokens=500,
            temperature=0.5
        )

        content = response.choices[0].message.content

        summary_match = re.search(
            r'SUMMARY:\n([\s\S]+?)(?=\n\nOUTLINE:)', content
        )
        outline_match = re.search(
            r'OUTLINE:\n([\s\S]+?)$', content
        )

        return {
            "summary": summary_match.group(1).strip() if summary_match else fallback["summary"],
            "outline": outline_match.group(1).strip() if outline_match else fallback["outline"]
        }

    except Exception as e:
        logger.error(f"Groq content generation failed: {e}")
        return fallback

# ---- Main Process Function ----

def process_video(
    video_url: str,
    topic: str = "Bar Exam",
    order_index: int = 0
) -> dict:
    """
    Full pipeline:
    1. Extract video ID
    2. Get metadata via YouTube Data API v3
    3. Try transcript (10s timeout)
    4. Fallback to description if transcript blocked
    5. Generate AI course content via Groq
    6. Save to Supabase
    7. Embed for RAG
    """
    from supabase import create_client

    # Step 1: Extract video ID
    match = re.search(r"(?:v=|\/)([a-zA-Z0-9_-]{11})", video_url)
    if not match:
        raise ValueError("Invalid YouTube URL")

    video_id = match.group(1)
    logger.info(f"=== Processing {video_id} | {topic} ===")

    # Step 2: Get metadata
    meta = get_video_metadata(video_id)
    logger.info(f"Title: {meta['title']}")

    # Step 3: Try transcript with 10 second timeout
    transcript = get_transcript_with_timeout(video_id, timeout_seconds=10)

    # Step 4: Fallback to description if transcript blocked
    if transcript and len(transcript) > 200:
        text_for_ai = transcript
        source_type = "transcript"
        logger.info("Using transcript for AI")
    elif meta["description"] and len(meta["description"]) > 100:
        text_for_ai = meta["description"]
        source_type = "description"
        logger.warning(
            "Transcript blocked/unavailable. "
            "Using video description as fallback."
        )
    else:
        text_for_ai = f"Bar exam lecture about {topic}: {meta['title']}"
        source_type = "title_only"
        logger.warning("Using title only as fallback")

    logger.info(f"Source: {source_type} | Length: {len(text_for_ai)}")

    # Step 5: Generate AI course content
    content = generate_course_content(text_for_ai, meta["title"], topic)
    logger.info("Course content generated")

    # Step 6: Save to Supabase
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    module_data = supabase.table("course_modules").insert({
        "title":         meta["title"],
        "description":   content["summary"],
        "topic":         topic,
        "video_id":      video_id,
        "video_url":     f"https://youtube.com/watch?v={video_id}",
        "thumbnail_url": meta["thumbnail"],
        "ai_summary":    content["summary"],
        "ai_outline":    content["outline"],
        "order_index":   order_index,
        "is_published":  True
    }).execute()

    module_id = module_data.data[0]["id"]

    # Save transcript or description
    supabase.table("youtube_videos").upsert({
        "video_id":   video_id,
        "title":      meta["title"],
        "transcript": text_for_ai
    }).execute()

    # Step 7: Embed for RAG
    chunks = split_text(text_for_ai)
    if len(chunks) > MAX_CHUNKS:
        chunks = chunks[:MAX_CHUNKS]

    embeddings = get_embeddings_batch(chunks)

    rows = [{
        "content": chunk,
        "metadata": {
            "source":      f"https://youtube.com/watch?v={video_id}",
            "video_id":    video_id,
            "type":        "course_module",
            "module_id":   module_id,
            "topic":       topic,
            "source_type": source_type,
            "chunk_index": i
        },
        "embedding": embeddings[i]
    } for i, chunk in enumerate(chunks)]

    supabase.table("documents").insert(rows).execute()

    logger.info(f"=== Success: {meta['title']} ===")

    return {
        "message": f"Course module created: {meta['title']}",
        "module_id": module_id,
        "title": meta["title"],
        "topic": topic,
        "source_type": source_type,
        "thumbnail": meta["thumbnail"],
        "chunks_embedded": len(chunks)
    }
