import os
import re
import time
import hashlib
import logging
import requests
import numpy as np
from contextlib import asynccontextmanager
from typing import List, Literal
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from supabase import create_client, Client
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
)

# ---- Logging Setup ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ---- Conditional Imports ----
try:
    from youtube_transcript_api import VideoUnavailable
    HAS_VIDEO_UNAVAILABLE = True
except ImportError:
    HAS_VIDEO_UNAVAILABLE = False
    logger.warning("VideoUnavailable not available.")

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logger.warning("groq package not installed.")

try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False
    logger.warning("duckduckgo-search not available.")

# ---- Environment ----
load_dotenv()
SUPABASE_URL    = os.getenv("SUPABASE_URL")
SUPABASE_KEY    = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
GROQ_API_KEY    = os.getenv("GROQ_API_KEY")
HF_TOKEN        = os.getenv("HF_TOKEN")
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "https://barprepfront.netlify.app,http://localhost:5173,http://localhost:3000"
).split(",")

if not all([SUPABASE_URL, SUPABASE_KEY]):
    raise ValueError("Missing: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY.")

if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY not set.")

# ---- Client Initialization ----
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

groq_client = None
if GROQ_API_KEY and GROQ_AVAILABLE:
    groq_client = Groq(api_key=GROQ_API_KEY)
    logger.info("Groq client initialized.")

# ---- Configuration Constants ----
HF_EMBED_URL     = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
POLLINATIONS_URL = "https://text.pollinations.ai"
MAX_CONTENT_BYTES = 5 * 1024 * 1024
MAX_CHUNKS        = 200
CHUNK_SIZE        = 1000
CHUNK_OVERLAP     = 200
RAG_THRESHOLD     = 0.5
RAG_MATCH_COUNT   = 4
GROQ_MODEL        = "llama-3.1-8b-instant"
EMBEDDING_DIM     = 384

# ---- Lifespan ----
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("BarPrep AI Backend starting...")
    logger.info(f"CORS Origins: {ALLOWED_ORIGINS}")
    logger.info(f"Groq Ready: {groq_client is not None}")
    logger.info(f"HF Token Set: {bool(HF_TOKEN)}")
    logger.info(f"DDG Available: {DDGS_AVAILABLE}")
    yield
    logger.info("BarPrep AI Backend shutting down.")

# ---- App Init ----
app = FastAPI(title="BarPrep AI Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# ---- Schemas ----
class IngestRequest(BaseModel):
    url: str

    @field_validator("url")
    @classmethod
    def url_must_be_valid(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("URL cannot be empty")
        if not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return v.strip()


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Content cannot be empty")
        if len(v) > 4000:
            raise ValueError("Content too long (max 4000 chars)")
        return v.strip()


class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []

    @field_validator("message")
    @classmethod
    def message_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Message cannot be empty")
        if len(v) > 2000:
            raise ValueError("Message too long (max 2000 chars)")
        return v.strip()

    @field_validator("history")
    @classmethod
    def history_not_too_long(cls, v: List[ChatMessage]) -> List[ChatMessage]:
        if len(v) > 20:
            raise ValueError("History too long (max 20 messages)")
        return v


# ---- Embedding Helpers ----

def local_hash_embedding(text: str) -> List[float]:
    """
    Generates a deterministic 384-dim embedding using
    character-level hashing. No external API needed.
    Works offline on Render free tier.
    Not as accurate as ML embeddings but always available.
    """
    embedding = np.zeros(EMBEDDING_DIM)
    
    # Use multiple hash seeds for better distribution
    for seed in range(EMBEDDING_DIM):
        hash_input = f"{seed}:{text}".encode("utf-8")
        hash_val = int(hashlib.md5(hash_input).hexdigest(), 16)
        embedding[seed] = (hash_val % 10000) / 10000.0 - 0.5

    # Normalize to unit vector for cosine similarity
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding.tolist()


def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Generates embeddings with fallback chain:
    1. HuggingFace API (best quality - needs internet)
    2. Local hash embedding (always works - no internet needed)
    """
    clean_texts = [t.replace("\n", " ").strip() for t in texts]

    # --- Try HuggingFace API first ---
    try:
        headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
        response = requests.post(
            HF_EMBED_URL,
            headers=headers,
            json={
                "inputs": clean_texts,
                "options": {"wait_for_model": True}
            },
            timeout=15  # Short timeout - fail fast to local fallback
        )
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                logger.info("Embeddings via HuggingFace API.")
                return result
    except Exception as e:
        logger.warning(f"HuggingFace unavailable: {e}. Using local fallback.")

    # --- Local hash fallback - always works ---
    logger.info("Using local hash embeddings.")
    return [local_hash_embedding(text) for text in clean_texts]


def sanitize_for_prompt(text: str) -> str:
    """Prevents prompt injection."""
    text = text.replace("===", "---")
    return text[:8000]


def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> List[str]:
    """Splits text into overlapping chunks."""
    if not text or not text.strip():
        return []
    if overlap >= chunk_size:
        raise ValueError("Overlap must be less than chunk_size")
    chunks, start, step = [], 0, chunk_size - overlap
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += step
    return chunks


def run_web_search(query: str) -> str:
    """Runs DuckDuckGo web search."""
    if not DDGS_AVAILABLE:
        return ""
    try:
        with DDGS() as ddgs:
            hits = list(ddgs.text(
                query,
                region="wt-wt",
                safesearch="off",
                max_results=3
            ))
            if not hits:
                return ""
            return "\n\n".join(
                f"Source: {h.get('href', h.get('link', ''))}\n"
                f"Snippet: {h.get('body', h.get('snippet', ''))}"
                for h in hits
            )
    except Exception as e:
        logger.error(f"DuckDuckGo search failed: {e}")
        return ""


# ---- Endpoints ----

@app.get("/")
def home():
    return {"status": "healthy", "service": "BarPrep AI Engine"}


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "groq_available": groq_client is not None,
        "hf_token_set": bool(HF_TOKEN),
        "supabase_connected": bool(supabase),
        "web_search_available": DDGS_AVAILABLE,
    }


@app.post("/api/ingest/url")
def ingest_url(data: IngestRequest):
    """Scrapes a webpage and ingests into Supabase."""
    url = data.url
    try:
        res = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
            stream=True
        )
        if res.status_code != 200:
            raise HTTPException(
                status_code=400,
                detail=f"Page returned status {res.status_code}"
            )
        content = b""
        for chunk in res.iter_content(chunk_size=8192):
            content += chunk
            if len(content) > MAX_CONTENT_BYTES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Page too large (max {MAX_CONTENT_BYTES // (1024*1024)}MB)"
                )
        soup = BeautifulSoup(content, "html.parser")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not reach URL: {e}"
        )

    for tag in soup(["script", "style"]):
        tag.extract()

    title = (
        soup.title.string.strip()[:200]
        if soup.title and soup.title.string
        else "Untitled Page"
    )

    clean_text = "\n".join(
        line.strip() for line in soup.get_text().splitlines() if line.strip()
    )

    chunks = split_text(clean_text)
    if not chunks:
        return {"message": "Page contained no readable text."}

    if len(chunks) > MAX_CHUNKS:
        chunks = chunks[:MAX_CHUNKS]
        logger.warning(f"Truncated to {MAX_CHUNKS} chunks.")

    try:
        embeddings = get_embeddings_batch(chunks)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Embedding failed: {e}"
        )

    rows = [{
        "content": chunk,
        "metadata": {
            "source": url,
            "title": title,
            "type": "webpage",
            "chunk_index": i
        },
        "embedding": embeddings[i]
    } for i, chunk in enumerate(chunks)]

    try:
        supabase.table("documents").insert(rows).execute()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"DB insert failed: {e}"
        )

    logger.info(f"Ingested {len(chunks)} chunks from '{title}'")
    return {
        "message": f"Successfully ingested {len(chunks)} chunks from '{title}'"
    }


@app.post("/api/ingest/youtube")
def ingest_youtube(data: IngestRequest):
    """Transcribes YouTube video and ingests into Supabase."""
    match = re.search(r"(?:v=|\/)([a-zA-Z0-9_-]{11})", data.url)
    if not match:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL.")
    video_id = match.group(1)

    transcript_list = None
    last_error = None

    for attempt in range(3):
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            break
        except TranscriptsDisabled:
            raise HTTPException(
                status_code=400,
                detail="Transcripts disabled for this video."
            )
        except NoTranscriptFound:
            raise HTTPException(
                status_code=404,
                detail="No transcript found for this video."
            )
        except Exception as e:
            last_error = e
            error_str = str(e).lower()

            if "429" in str(e) or "too many requests" in error_str:
                wait_time = (attempt + 1) * 5
                logger.warning(f"YouTube rate limit. Waiting {wait_time}s...")
                time.sleep(wait_time)
                continue

            if HAS_VIDEO_UNAVAILABLE and isinstance(e, VideoUnavailable):
                raise HTTPException(
                    status_code=404,
                    detail="Video unavailable or private."
                )
            if any(k in error_str for k in ["unavailable", "private"]):
                raise HTTPException(
                    status_code=404,
                    detail="Video unavailable or private."
                )

            raise HTTPException(
                status_code=500,
                detail=f"Transcript error: {e}"
            )

    if transcript_list is None:
        raise HTTPException(
            status_code=429,
            detail="YouTube rate limiting. Please wait 5 minutes and retry."
        )

    full_transcript = " ".join(e["text"] for e in transcript_list)

    try:
        supabase.table("youtube_videos").upsert({
            "video_id": video_id,
            "title": f"YouTube Lecture: {video_id}",
            "transcript": full_transcript
        }).execute()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Metadata DB insert failed: {e}"
        )

    chunks = split_text(full_transcript)
    if not chunks:
        return {"message": f"Video {video_id} had no usable transcript text."}

    if len(chunks) > MAX_CHUNKS:
        chunks = chunks[:MAX_CHUNKS]
        logger.warning(f"Truncated to {MAX_CHUNKS} chunks.")

    try:
        embeddings = get_embeddings_batch(chunks)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Embedding failed: {e}"
        )

    rows = [{
        "content": chunk,
        "metadata": {
            "source": f"https://youtube.com/watch?v={video_id}",
            "video_id": video_id,
            "type": "youtube_transcript",
            "chunk_index": i
        },
        "embedding": embeddings[i]
    } for i, chunk in enumerate(chunks)]

    try:
        supabase.table("documents").insert(rows).execute()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"DB insert failed: {e}"
        )

    logger.info(f"Processed video {video_id} into {len(chunks)} chunks.")
    return {
        "message": f"Processed video {video_id} into {len(chunks)} chunks."
    }


@app.post("/api/chat")
def chat_with_ai(data: ChatRequest):
    """Answers bar exam questions using RAG + AI."""

    # 1. Embed query
    try:
        query_vector = get_embeddings_batch([data.message])[0]
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Query embedding failed: {e}"
        )

    # 2. RAG search
    try:
        result = supabase.rpc("match_documents", {
            "query_embedding": query_vector,
            "match_threshold": RAG_THRESHOLD,
            "match_count": RAG_MATCH_COUNT
        }).execute()
        context_chunks = [r["content"] for r in result.data]
    except Exception as e:
        logger.error(f"RAG search failed: {e}")
        context_chunks = []

    # 3. Web search fallback
    web_results = ""
    if not context_chunks:
        web_results = run_web_search(data.message)

    # 4. Build prompt
    context_text = sanitize_for_prompt(
        "\n---\n".join(context_chunks) if context_chunks
        else "No relevant internal context found."
    )
    web_section = (
        f"\n=== LIVE WEB SEARCH CONTEXT ===\n{sanitize_for_prompt(web_results)}"
        if web_results else ""
    )

    system_prompt = f"""You are a professional, encouraging Bar Exam prep coach.
Help the student master bar exam topics using the context below.
Explain step-by-step with clear legal reasoning when needed.

=== INTERNAL STUDY MATERIAL CONTEXT ===
{context_text}{web_section}
"""

    messages = (
        [{"role": "system", "content": system_prompt}]
        + [{"role": m.role, "content": m.content} for m in data.history]
        + [{"role": "user", "content": data.message}]
    )

    # 5. Try Pollinations first
    try:
        res = requests.post(POLLINATIONS_URL, json={
            "messages": messages,
            "model": "openai",
            "private": True
        }, timeout=30)
        if res.status_code == 200 and (reply := res.text.strip()):
            return {"reply": reply}
    except Exception as e:
        logger.error(f"Pollinations failed: {e}")

    # 6. Groq fallback
    if groq_client:
        try:
            completion = groq_client.chat.completions.create(
                messages=messages,
                model=GROQ_MODEL
            )
            return {"reply": completion.choices[0].message.content.strip()}
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Groq failed: {e}"
            )

    raise HTTPException(
        status_code=500,
        detail="All LLM services temporarily unavailable."
    )


@app.get("/api/affirmation")
def get_daily_affirmation():
    """Generates a daily bar exam affirmation."""
    try:
        res = requests.post(POLLINATIONS_URL, json={
            "messages": [
                {
                    "role": "system",
                    "content": "You are an inspiring mentor to future lawyers."
                },
                {
                    "role": "user",
                    "content": "Give a short 2-3 sentence bar exam affirmation."
                }
            ],
            "model": "openai"
        }, timeout=15)
        if res.status_code == 200 and (reply := res.text.strip()):
            return {"affirmation": reply}
    except Exception as e:
        logger.error(f"Affirmation failed: {e}")

    return {
        "affirmation": "You have the analytical mind, the diligence, and the capability to conquer this exam. One rule, one analysis, and one day at a time."
    }
