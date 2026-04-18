import warnings
import os

# Suppress harmless warnings for cleaner output
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Lightning automatically upgraded.*")
warnings.filterwarnings("ignore", message=".*TF32.*")
warnings.filterwarnings("ignore", message=".*does not have many workers.*")

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import torch

from .database import init_db
from .api import router, get_engine
from .conversation_api import router as conversation_router
from .mcp_api import router as mcp_router
from .settings_api import router as settings_router
from .backup_api import router as profiles_router
from .streaming_websocket import router as streaming_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database and directories on startup"""
    print("Initializing database...")
    init_db()
    print("Database initialized!")

    # Create necessary directories (use relative paths for local dev, absolute for Docker)
    data_path = os.getenv("DATA_PATH", "./data")
    volumes_path = os.getenv("VOLUMES_PATH", "./volumes")

    os.makedirs(f"{data_path}/recordings", exist_ok=True)
    os.makedirs(f"{data_path}/temp", exist_ok=True)
    os.makedirs(volumes_path, exist_ok=True)

    # Preload AI models for faster first request
    print("\n=== Preloading AI models ===")
    engine = get_engine()

    # Load models (this loads into CPU/RAM but may not allocate VRAM yet)
    whisper = engine.whisper_model  # Loads Whisper
    diarization = engine.diarization_pipeline  # Loads diarization
    embedding = engine.embedding_model  # Loads embeddings
    emotion = engine.emotion_model  # Loads emotion2vec

    # Force VRAM allocation by running a warmup pass (only on GPU)
    if torch.cuda.is_available():
        print("Running GPU warmup to allocate VRAM...")
        import tempfile
        import wave
        import numpy as np

        # Create a longer test audio file (10 seconds with synthetic speech pattern)
        # This ensures all model buffers and VRAM are fully allocated
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            test_audio_path = f.name
            with wave.open(test_audio_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)

                # Generate 10 seconds of synthetic audio with speech-like patterns
                # This triggers VAD and allocates full model buffers
                duration = 10  # seconds
                samples = duration * 16000
                t = np.linspace(0, duration, samples)

                # Mix of frequencies to simulate speech (200-800 Hz range)
                audio = np.zeros(samples)
                audio += 0.3 * np.sin(2 * np.pi * 250 * t)  # Fundamental frequency
                audio += 0.2 * np.sin(2 * np.pi * 500 * t)  # First harmonic
                audio += 0.1 * np.sin(2 * np.pi * 750 * t)  # Second harmonic

                # Add amplitude modulation to simulate speech envelope
                envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)  # 3 Hz modulation
                audio *= envelope

                # Convert to int16
                audio = (audio * 32767 * 0.5).astype(np.int16)
                wf.writeframes(audio.tobytes())

        try:
            print(f"  - Created {duration}s test audio file")

            # Create dummy speaker to exercise speaker matching code path
            dummy_embedding = np.random.randn(512).astype(np.float32)
            dummy_speakers = [(1, "Warmup", dummy_embedding)]

            # Get threshold from settings
            from .config import get_config
            config = get_config()
            settings = config.get_settings()

            # Warmup: full pipeline with speaker matching
            print("  - Warming up models (diarization, embedding, whisper, matching)...")
            with torch.no_grad():
                _ = engine.transcribe_with_diarization(
                    test_audio_path,
                    known_speakers=dummy_speakers,
                    threshold=settings.speaker_threshold
                )

            print("GPU warmup complete")

            # Print VRAM usage (current = after cleanup, peak = during processing)
            if torch.cuda.is_available():
                current_allocated = torch.cuda.memory_allocated() / 1024**3
                peak_allocated = torch.cuda.max_memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"  - VRAM current: {current_allocated:.2f} GB (after cleanup)")
                print(f"  - VRAM peak: {peak_allocated:.2f} GB (during processing)")
                print(f"  - VRAM reserved: {reserved:.2f} GB")

        except Exception as e:
            print(f"Warmup error (non-critical): {e}")
        finally:
            # Clean up test file (os already imported at top)
            if os.path.exists(test_audio_path):
                os.unlink(test_audio_path)

    print("=== All models loaded and ready! ===\n")

    yield
    # Cleanup on shutdown (if needed)


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Speaker Diarization API",
    description="Speaker diarization and recognition service with GPU acceleration",
    version="1.0.0",
    lifespan=lifespan
)

# CORS: internal-network-only by default. Set CORS_ORIGINS to a comma-separated
# list of origins (e.g. "http://host:3000,http://host:8080") if a browser UI needs access.
# Browsers never send a trailing slash on Origin, so strip one if the user
# included it in CORS_ORIGINS — otherwise Starlette's exact-string match misses.
_cors_origins = [o.strip().rstrip("/") for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Include API routers
app.include_router(router, prefix="/api/v1", tags=["Speaker Diarization"])
app.include_router(conversation_router, prefix="/api/v1")
app.include_router(settings_router, prefix="/api/v1")
app.include_router(profiles_router, prefix="/api/v1")  # Voice Profiles
app.include_router(streaming_router, prefix="/api/v1")  # WebSocket streaming
app.include_router(mcp_router)  # MCP at /mcp


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Speaker Diarization API",
        "docs": "/docs",
        "api": "/api/v1",
        "mcp": "/mcp (AI agent interface - JSON-RPC)",
    }


if __name__ == "__main__":
    # Start FastAPI server
    print("Starting server...")

    # Check for SSL certificates (enables HTTPS/WSS)
    # Try local path first, then Docker path
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_cert = os.path.join(script_dir, "certs", "cert.pem")
    local_key = os.path.join(script_dir, "certs", "key.pem")

    if os.path.exists(local_cert) and os.path.exists(local_key):
        ssl_cert, ssl_key = local_cert, local_key
    else:
        ssl_cert = "/app/certs/cert.pem"
        ssl_key = "/app/certs/key.pem"

    ssl_opts = {}
    if os.path.exists(ssl_cert) and os.path.exists(ssl_key):
        ssl_opts = {"ssl_certfile": ssl_cert, "ssl_keyfile": ssl_key}
        print("🔒 SSL enabled - server will use HTTPS/WSS")
    else:
        print("⚠️  No SSL certs found - server will use HTTP/WS")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8418")),
        log_level="info",
        **ssl_opts
    )
