import os

# Fix for PyTorch 2.6+ weights_only=True breaking change
# Required for loading pyannote.audio model checkpoints which contain
# pickled objects (omegaconf, pytorch_lightning callbacks, etc.)
# This must be set BEFORE importing torch
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'

import torch

import logging
import numpy as np
from pyannote.audio import Pipeline, Model, Inference
from pyannote.core import Segment
from sklearn.metrics.pairwise import cosine_similarity
from typing import Any, List, Dict, Tuple, Optional
from faster_whisper import WhisperModel
from concurrent.futures import ThreadPoolExecutor
import time
import tempfile
import traceback
import uuid
import soundfile as sf
from pydub import AudioSegment
import threading
import queue
import gc

from .config import get_config
from .models import Speaker, SpeakerEmotionProfile

logger = logging.getLogger(__name__)

# Speaker matching constants
FALLBACK_THRESHOLD_REDUCTION = 0.05
MIN_FALLBACK_THRESHOLD = 0.20
UNKNOWN_NAME_MAX_ATTEMPTS = 5

# Emotion detection constants
MIN_VOICE_SAMPLES = 3

# Opt-in speed settings (all off by default; unset = the original behaviour).
#   WHISPER_BATCH_SIZE=16      faster-whisper's BatchedInferencePipeline: VAD chunks transcribed 16 at a time.
#                              ~8x faster transcription on a 3090 / CMP 170HX; returns ~30 s segments, so use it
#                              together with WORD_LEVEL_SPEAKERS.
#   WORD_LEVEL_SPEAKERS=true   give every word the diarization speaker at its midpoint and split segments where the
#                              speaker changes (or after a pause / a long run), instead of one speaker per segment.
#   PYANNOTE_ALLOW_TF32=true   keep TF32 matmuls on for pyannote (it turns them off for bit-reproducibility).
#   SPEAKER_ASSIGNMENT=cluster identify each diarization speaker once per file, from all of that speaker's segments,
#                              instead of each segment on its own (a short segment's embedding is too noisy: one
#                              person ends up as several Unknown_* speakers). Words get their speaker from pyannote's
#                              exclusive (non-overlapping) diarization. Implies word-level speakers.
WHISPER_BATCH_SIZE = int(os.getenv("WHISPER_BATCH_SIZE", "0") or 0)
WORD_LEVEL_SPEAKERS = os.getenv("WORD_LEVEL_SPEAKERS", "false").lower() == "true"
PYANNOTE_ALLOW_TF32 = os.getenv("PYANNOTE_ALLOW_TF32", "false").lower() == "true"
CLUSTER_SPEAKERS = os.getenv("SPEAKER_ASSIGNMENT", "segment").lower() == "cluster"
SPLIT_PAUSE_SEC = 1.0      # WORD_LEVEL_SPEAKERS: a pause this long between words starts a new segment
SPLIT_MAX_SEC = 30.0       # WORD_LEVEL_SPEAKERS: longest segment before a forced split at a word boundary
CENTROID_MAX_WEIGHT_SEC = 10.0  # SPEAKER_ASSIGNMENT=cluster: a segment counts by its duration, up to this
CENTROID_FALLBACK_TURNS = 3     # cluster with no embeddable segment: embed its longest diarization turns instead


def _speaker_at(t: float, diar_segments: List[Dict]) -> Optional[str]:
    """Diarization speaker whose turn contains time t, else the nearest turn's speaker."""
    for d in diar_segments:
        if d["start"] <= t <= d["end"]:
            return d["speaker"]
    best, label = float("inf"), None
    for d in diar_segments:
        dist = min(abs(d["start"] - t), abs(d["end"] - t))
        if dist < best:
            best, label = dist, d["speaker"]
    return label


def _split_by_speaker(transcription_segments: List[Dict], diar_segments: List[Dict]) -> List[Dict]:
    """Re-cut transcription segments at word boundaries so each piece has one diarization speaker."""
    out = []
    for seg in transcription_segments:
        words = seg.get("words") or []
        if not words or not diar_segments:
            out.append(seg)
            continue
        current, current_spk = [], None
        for w in words:
            spk = _speaker_at((w["start"] + w["end"]) / 2, diar_segments)
            if current and (spk != current_spk or w["start"] - current[-1]["end"] > SPLIT_PAUSE_SEC
                            or w["end"] - current[0]["start"] > SPLIT_MAX_SEC):
                out.append(_piece(seg, current, current_spk))
                current = []
            if not current:
                current_spk = spk
            current.append(w)
        if current:
            out.append(_piece(seg, current, current_spk))
    return out


def _piece(seg: Dict, words: List[Dict], speaker_label: Optional[str]) -> Dict:
    return {"start": words[0]["start"], "end": words[-1]["end"],
            "text": "".join(w["word"] for w in words).strip(), "words": words,
            "avg_logprob": seg.get("avg_logprob"), "speaker_label": speaker_label}


def _label_at_midpoint(seg: Dict, diar_segments: List[Dict]) -> Optional[str]:
    """Diarization speaker at the segment's midpoint, else the speaker of the turn whose midpoint is closest."""
    mid_time = (seg["start"] + seg["end"]) / 2
    for diar_seg in diar_segments:
        if diar_seg["start"] <= mid_time <= diar_seg["end"]:
            return diar_seg["speaker"]
    label, min_distance = None, float("inf")
    for diar_seg in diar_segments:
        distance = abs((diar_seg["start"] + diar_seg["end"]) / 2 - mid_time)
        if distance < min_distance:
            min_distance, label = distance, diar_seg["speaker"]
    return label


def _float_samples(segment: "AudioSegment") -> np.ndarray:
    """Mono float32 samples in [-1, 1]: what FunASR (torchaudio.load + channel mean) reads from the same WAV file."""
    samples = np.array(segment.get_array_of_samples(), dtype=np.float32)
    if segment.channels > 1:
        samples = samples.reshape(-1, segment.channels).mean(axis=1)
    return samples / float(1 << (8 * segment.sample_width - 1))


def _centroid(embeddings: List[np.ndarray], weights: List[float]) -> Optional[np.ndarray]:
    """Weighted mean of L2-normalised embeddings (a cosine-similarity centroid)."""
    if not embeddings:
        return None
    unit = [e / (np.linalg.norm(e) or 1.0) for e in embeddings]
    return np.average(np.stack(unit), axis=0, weights=weights).astype(np.float32)


def _keep_tf32_on():
    """pyannote calls fix_reproducibility() on every inference, which switches TF32 off; make it a no-op."""
    import sys as _sys
    import pyannote.audio.utils.reproducibility as rep
    original = rep.fix_reproducibility
    noop = lambda device: None  # noqa: E731
    for mod in list(_sys.modules.values()):
        if getattr(mod, "fix_reproducibility", None) is original:
            mod.fix_reproducibility = noop
    rep.fix_reproducibility = noop
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
DUAL_DETECTOR_AGREE_D1 = 0.70
DUAL_DETECTOR_AGREE_D2 = 0.80
VOICE_STRONG_THRESHOLD = 0.85
MAX_EMOTION_DURATION_SEC = 30.0
MIN_SEGMENT_DURATION_SEC = 0.5

# Hallucination filters — cheap text-only checks, applied before embedding / emotion extraction
_HALLUCINATION_FILLERS = {"...", "ah", "um", "uh", "oh", "eh", "hmm", "mm"}
_HALLUCINATION_PATTERNS = (
    "thank you",
    "thanks for watching",
    "please subscribe",
    "like and subscribe",
    "don't forget to subscribe",
    "see you next time",
    "thanks for listening",
)
_HALLUCINATION_MIN_LEN = 3
_HALLUCINATION_PATTERN_MAX_DURATION = 0.4
_HALLUCINATION_PATTERN_MAX_LOGPROB = -0.6


def _is_hallucination(text: str, duration: float, avg_logprob: Optional[float]) -> bool:
    """Cheap text+duration heuristics applied before any GPU work."""
    text = text.strip().lower()
    if len(text) < _HALLUCINATION_MIN_LEN:
        return True
    if len(text) == 1 or text in _HALLUCINATION_FILLERS:
        return True
    if any(pattern in text for pattern in _HALLUCINATION_PATTERNS):
        short = duration < _HALLUCINATION_PATTERN_MAX_DURATION
        low_conf = avg_logprob is not None and avg_logprob < _HALLUCINATION_PATTERN_MAX_LOGPROB
        if short or low_conf:
            return True
    return False


def _build_profile_list(speakers: list) -> List[Dict[str, Any]]:
    """Flatten Speaker ORM rows into the cache's profile-entry format.

    Used by both load_speaker_cache and the DB-fallback in match_speaker_with_all_profiles
    so the two paths can never drift apart.
    """
    profiles: List[Dict[str, Any]] = []
    for speaker in speakers:
        general_emb = speaker.get_embedding()
        if general_emb is not None and not np.isnan(general_emb).any():
            profiles.append({
                "speaker_id": speaker.id,
                "speaker_name": speaker.name,
                "embedding": general_emb,
                "emotion": None,
                "profile_type": "general",
            })
        for emotion_profile in speaker.emotion_profiles:
            if emotion_profile.voice_sample_count < MIN_VOICE_SAMPLES:
                continue
            voice_emb = emotion_profile.get_voice_embedding()
            if voice_emb is None or np.isnan(voice_emb).any():
                continue
            profiles.append({
                "speaker_id": speaker.id,
                "speaker_name": speaker.name,
                "embedding": voice_emb,
                "emotion": emotion_profile.emotion_category,
                "profile_type": "emotion_voice",
                "sample_count": emotion_profile.voice_sample_count,
            })
    return profiles


def auto_enroll_unknown_speaker(embedding: np.ndarray, db_session, threshold: float = 0.30):
    """
    Auto-enroll unknown speaker with timestamp-based naming and clustering.

    CRITICAL: This is a FALLBACK for segments that didn't match in transcribe_with_diarization().
    Checks ALL speakers (not just Unknown_*) with a slightly lower threshold as a second chance.
    This prevents creating duplicate Unknown speakers for people who are already enrolled.

    Args:
        embedding: Numpy array of speaker embedding
        db_session: SQLAlchemy database session
        threshold: Similarity threshold from settings (will use threshold-0.05 as fallback)

    Returns:
        Tuple of (speaker_id, speaker_name) for the enrolled/matched speaker
    """
    fallback_threshold = max(threshold - FALLBACK_THRESHOLD_REDUCTION, MIN_FALLBACK_THRESHOLD)

    best_match: Optional[Speaker] = None
    best_similarity = 0.0

    if embedding is not None:
        all_speakers = db_session.query(Speaker).all()
        if all_speakers:
            logger.info(f"  🔍 Auto-enroll fallback: Checking {len(all_speakers)} speakers (fallback threshold: {fallback_threshold:.2%})")
            for speaker in all_speakers:
                existing_embedding = speaker.get_embedding()
                if existing_embedding is None or np.isnan(existing_embedding).any():
                    continue
                similarity = cosine_similarity(
                    embedding.reshape(1, -1),
                    existing_embedding.reshape(1, -1),
                )[0][0]
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = speaker

    if best_match is not None and best_similarity > fallback_threshold:
        logger.info(f"✓ Fallback match: Identified as '{best_match.name}' (similarity: {best_similarity:.2%}, fallback threshold: {fallback_threshold:.2%})")
        return (best_match.id, best_match.name)
    if best_match is not None:
        logger.info(f"  ℹ️ Best match was '{best_match.name}' with {best_similarity:.2%} (below fallback threshold {fallback_threshold:.2%})")

    return _create_unknown_speaker(embedding, db_session)


def _create_unknown_speaker(embedding: np.ndarray, db_session):
    """New Unknown_<timestamp> speaker with this embedding (flushed, not committed). Returns (speaker_id, name)."""
    timestamp_name = None
    for _ in range(UNKNOWN_NAME_MAX_ATTEMPTS):
        timestamp_name = f"Unknown_{int(time.time() * 1000000)}"
        existing = db_session.query(Speaker).filter(Speaker.name == timestamp_name).first()
        if not existing:
            break
        time.sleep(0.001)
    else:
        timestamp_name = f"Unknown_{uuid.uuid4().hex[:12]}"

    # Create new speaker with embedding
    new_speaker = Speaker(name=timestamp_name)
    new_speaker.set_embedding(embedding)
    db_session.add(new_speaker)
    db_session.flush()  # Get the ID without committing

    logger.info(f"✓ Auto-enrolled new speaker: {timestamp_name} (ID: {new_speaker.id})")

    return (new_speaker.id, new_speaker.name)


class SpeakerRecognitionEngine:
    """
    Speaker diarization and recognition engine using pyannote.audio
    """

    def __init__(self, hf_token: Optional[str] = None):
        """
        Initialize the speaker recognition engine

        Args:
            hf_token: HuggingFace token for accessing pyannote models
        """
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ConfigManager already handles persisted values plus non-empty env
        # overrides. Reading the raw Compose env here made CONTEXT_PADDING=""
        # crash startup and ignored later Settings API updates.
        self.context_padding = get_config().get_settings().context_padding

        # Initialize models (lazy loading). `_EMOTION_MODEL_FAILED` is a sentinel
        # so a one-time load failure (e.g. transient HF outage) doesn't cause every
        # later segment to re-attempt the full download path.
        self._diarization_pipeline = None
        self._embedding_model = None
        self._whisper_model = None
        self._emotion_model = None
        self._emotion_model_failed = False

        # Serializes lazy-load on the four models. The startup warmup already
        # populates them, so on hot paths this is uncontended — but if warmup
        # is removed, two parallel requests must not both construct a model.
        self._model_lock = threading.Lock()

        # Speaker profile cache (reduces DB queries during streaming)
        self._speaker_cache = None
        self._cache_lock = threading.Lock()

        # Background cleanup thread (non-blocking)
        self._cleanup_running = True  # Set BEFORE starting thread
        self._cleanup_queue = queue.Queue(maxsize=10)  # Limit queue size
        self._cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self._cleanup_thread.start()

        # Periodic safety-net cleanup (every 30s). It exists because this box
        # shares the GPU with other processes; on-demand cleanup alone has
        # missed allocations in long-lived processes. Quiet unless it actually
        # freed something worth logging.
        self._periodic_cleanup_thread = threading.Thread(target=self._periodic_cleanup_worker, daemon=True)
        self._periodic_cleanup_thread.start()

        logger.info(f"Speaker Recognition Engine initialized on device: {self.device}")
        logger.info(f"Background GPU cleanup thread started")

    def _cleanup_worker(self):
        """Background worker that processes GPU cleanup requests"""
        while self._cleanup_running:
            try:
                # Wait for cleanup request (blocks until available)
                cleanup_type = self._cleanup_queue.get(timeout=1.0)

                if cleanup_type and torch.cuda.is_available():
                    # Perform cleanup
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                        torch.cuda.reset_peak_memory_stats()

                self._cleanup_queue.task_done()

            except queue.Empty:
                continue  # No cleanup requests, keep waiting
            except Exception as e:
                logger.info(f"Background cleanup error: {e}")

    def _periodic_cleanup_worker(self):
        """Every 30s, free VRAM. Silent unless the cleanup actually freed something."""
        while self._cleanup_running:
            try:
                time.sleep(30)
                if not torch.cuda.is_available():
                    continue

                vram_before_gb = torch.cuda.memory_reserved() / (1024**3)
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                vram_after_gb = torch.cuda.memory_reserved() / (1024**3)

                freed_gb = vram_before_gb - vram_after_gb
                if freed_gb > 0.1:
                    logger.info(f"⏰ Periodic cleanup freed {freed_gb:.1f}GB VRAM (now {vram_after_gb:.1f}GB)")
            except Exception as e:
                logger.info(f"Periodic cleanup error: {e}")

    def clear_gpu_cache_async(self, cleanup_type="standard"):
        """Queue GPU cleanup to run in background (non-blocking, conditional on VRAM usage)"""
        if not torch.cuda.is_available():
            return

        # Get VRAM threshold from env (default: 12GB)
        threshold_gb = float(os.getenv("CLEANUP_VRAM_THRESHOLD_GB") or "12")

        # Check current VRAM usage (reserved = what nvidia-smi shows)
        vram_used_gb = torch.cuda.memory_reserved() / (1024**3)

        # Only cleanup if VRAM exceeds threshold
        if vram_used_gb < threshold_gb:
            return  # Skip cleanup - VRAM usage is fine

        try:
            # Non-blocking put - if queue is full, just skip (cleanup will happen eventually)
            self._cleanup_queue.put_nowait(cleanup_type)
        except queue.Full:
            pass  # Queue full, cleanup will happen from pending requests

    def clear_gpu_cache(self):
        """Clear GPU memory cache immediately (blocking - use sparingly)"""
        if torch.cuda.is_available():
            # Force garbage collection first
            gc.collect()

            # Clear PyTorch CUDA cache
            torch.cuda.empty_cache()

            # Synchronize to ensure all operations are complete
            torch.cuda.synchronize()

            # Additional cleanup: clear any lingering computation graphs
            if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                torch.cuda.reset_peak_memory_stats()

    def load_speaker_cache(self, db_session):
        """Load all speaker profiles into memory cache (one-time DB query)."""
        with self._cache_lock:
            speakers = db_session.query(Speaker).all()
            profiles = _build_profile_list(speakers)
            self._speaker_cache = profiles
            logger.info(f"📦 Speaker cache loaded: {len(speakers)} speakers → {len(profiles)} profiles")
            return len(profiles)

    def add_speaker_to_cache(self, speaker_id, speaker_name, embedding, emotion=None, profile_type='general'):
        """Add newly enrolled speaker to cache (avoids cache invalidation)"""
        if self._speaker_cache is None:
            return  # Cache not initialized

        with self._cache_lock:
            self._speaker_cache.append({
                'speaker_id': speaker_id,
                'speaker_name': speaker_name,
                'embedding': embedding,
                'emotion': emotion,
                'profile_type': profile_type
            })
            logger.info(f"➕ Added to speaker cache: {speaker_name} ({profile_type})")

    def clear_speaker_cache(self):
        """Clear speaker cache (forces reload on next use)"""
        with self._cache_lock:
            self._speaker_cache = None
            logger.info(f"🗑️ Speaker cache cleared")

    @property
    def diarization_pipeline(self):
        """Lazy load diarization pipeline (thread-safe)."""
        if self._diarization_pipeline is not None:
            return self._diarization_pipeline
        with self._model_lock:
            if self._diarization_pipeline is None:
                logger.info("Loading pyannote diarization pipeline...")
                if self.hf_token:
                    os.environ["HF_TOKEN"] = self.hf_token
                pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1")
                pipeline.to(self.device)
                if PYANNOTE_ALLOW_TF32:
                    _keep_tf32_on()
                    logger.info("pyannote: TF32 kept on (PYANNOTE_ALLOW_TF32=true)")
                self._diarization_pipeline = pipeline
        return self._diarization_pipeline

    @property
    def embedding_model(self):
        """Lazy load pyannote embedding model (thread-safe)."""
        if self._embedding_model is not None:
            return self._embedding_model
        with self._model_lock:
            if self._embedding_model is None:
                logger.info("Loading pyannote embedding model...")
                if self.hf_token:
                    os.environ["HF_TOKEN"] = self.hf_token
                model = Model.from_pretrained("pyannote/embedding")
                inference = Inference(model, window="whole")
                inference.to(self.device)
                self._embedding_model = inference
        return self._embedding_model

    @property
    def whisper_model(self):
        """Lazy load faster-whisper model (thread-safe)."""
        if self._whisper_model is not None:
            return self._whisper_model
        with self._model_lock:
            if self._whisper_model is None:
                model_name = os.getenv("WHISPER_MODEL", "large-v3")
                device_name = "cuda" if torch.cuda.is_available() else "cpu"
                compute_type = "float16" if torch.cuda.is_available() else "int8"
                logger.info(f"Loading faster-whisper model ({model_name}) on {device_name} / {compute_type}...")
                self._whisper_model = WhisperModel(model_name, device=device_name, compute_type=compute_type)
                logger.info(f"faster-whisper model '{model_name}' loaded")
        return self._whisper_model

    @property
    def batched_whisper(self):
        """faster-whisper BatchedInferencePipeline over the same model (only used when WHISPER_BATCH_SIZE > 0)."""
        if getattr(self, "_batched_whisper", None) is None:
            model = self.whisper_model
            with self._model_lock:
                if getattr(self, "_batched_whisper", None) is None:
                    from faster_whisper import BatchedInferencePipeline
                    self._batched_whisper = BatchedInferencePipeline(model=model)
                    logger.info(f"faster-whisper batched pipeline on (batch size {WHISPER_BATCH_SIZE})")
        return self._batched_whisper

    @property
    def emotion_model(self):
        """Lazy load emotion2vec via FunASR. Sentinel-caches load failures so one
        transient error doesn't cause every segment to retry the full download."""
        if self._emotion_model is not None:
            return self._emotion_model
        if self._emotion_model_failed:
            return None
        with self._model_lock:
            if self._emotion_model is not None:
                return self._emotion_model
            if self._emotion_model_failed:
                return None
            try:
                from funasr import AutoModel
                model_name = os.getenv("EMOTION_MODEL", "iic/emotion2vec_plus_large")
                use_offline = os.getenv("OFFLINE_MODE", "false").lower() == "true"
                logger.info(f"Loading emotion2vec model via FunASR ({model_name})...")
                self._emotion_model = AutoModel(
                    model=model_name,
                    hub="hf",
                    disable_update=True,
                    local_files_only=use_offline,
                )
                logger.info(f"emotion2vec model loaded successfully ({model_name})")
            except ImportError:
                logger.warning("FunASR not installed. Emotion detection disabled. Install with: pip install funasr")
                self._emotion_model_failed = True
                return None
            except Exception as e:
                logger.warning(f"Failed to load emotion2vec model: {e}")
                self._emotion_model_failed = True
                return None
        return self._emotion_model

    def transcribe(self, audio_file: str) -> List[Dict]:
        """
        Transcribe audio file with timestamps

        Args:
            audio_file: Path to audio file

        Returns:
            List of transcription segments with timestamps
        """
        logger.info(f"Transcribing {audio_file}...")
        # Get language from environment variable (default: "en")
        # Use "auto" for auto-detection, or specify language code (e.g., "es", "fr", "de")
        language = os.getenv("WHISPER_LANGUAGE", "en")
        # faster-whisper transcription with word-level timestamps and confidence
        if WHISPER_BATCH_SIZE > 0:
            # Batched pipeline (VAD-chunked, always filters non-speech); see WHISPER_BATCH_SIZE above.
            segments_generator, info = self.batched_whisper.transcribe(
                audio_file,
                language=None if language == "auto" else language,
                task="transcribe",
                beam_size=5,
                batch_size=WHISPER_BATCH_SIZE,
                word_timestamps=True
            )
        else:
            segments_generator, info = self.whisper_model.transcribe(
                audio_file,
                language=None if language == "auto" else language,  # None = auto-detect
                task="transcribe",
                beam_size=5,
                vad_filter=True,  # Use VAD to filter out non-speech
                word_timestamps=True  # Enable word-level timestamps and probabilities
            )

        logger.info(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")

        transcription_segments = []
        # Convert generator to list - transcription happens during iteration
        for segment in segments_generator:
            # Extract word-level data if available
            words_data = []
            if hasattr(segment, 'words') and segment.words:
                for word in segment.words:
                    words_data.append({
                        "word": word.word,
                        "start": word.start,
                        "end": word.end,
                        "probability": word.probability  # Word-level confidence
                    })

            transcription_segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
                "words": words_data,  # Include word-level data with confidence
                "avg_logprob": segment.avg_logprob  # Segment-level confidence indicator
            })

        # Explicit cleanup: delete generator and info objects
        del segments_generator, info
        # Queue async cleanup (non-blocking)
        self.clear_gpu_cache_async("transcription")

        return transcription_segments

    def extract_embedding(self, audio_file: str) -> np.ndarray:
        """
        Extract speaker embedding from audio file

        Args:
            audio_file: Path to audio file

        Returns:
            Speaker embedding as numpy array
        """
        with torch.no_grad():
            embedding = self.embedding_model(audio_file)
        return np.array(embedding)

    def diarize(self, audio_file: str) -> Dict:
        """
        Perform speaker diarization on audio file

        Args:
            audio_file: Path to audio file (WAV format)

        Returns:
            Diarization result with speaker segments
        """
        logger.info(f"Running diarization on {audio_file}...")
        with torch.no_grad():
            output = self.diarization_pipeline(audio_file)

        # Convert to dictionary format. In pyannote.audio 4.0+, speaker_diarization holds the turns (overlapping where
        # two people talk at once); exclusive_speaker_diarization has one speaker at a time, for aligning words.
        def turns(annotation):
            return [{"start": turn.start, "end": turn.end, "speaker": speaker, "duration": turn.end - turn.start}
                    for turn, speaker in annotation]

        segments = turns(output.speaker_diarization)
        result = {
            "segments": segments,
            "exclusive_segments": turns(output.exclusive_speaker_diarization),
            "num_speakers": len(set(s["speaker"] for s in segments))
        }

        # Explicit cleanup: delete output object and clear references
        del output
        # Queue async cleanup (non-blocking)
        self.clear_gpu_cache_async("diarization")

        return result

    def match_speaker(
        self,
        segment_embedding: np.ndarray,
        known_speakers: List[Tuple[int, str, np.ndarray]],
        threshold: float = 0.7
    ) -> Optional[Tuple[int, str, float]]:
        """
        Match a segment embedding to known speakers

        Args:
            segment_embedding: Embedding to match
            known_speakers: List of (id, name, embedding) tuples
            threshold: Minimum similarity threshold for a match

        Returns:
            (speaker_id, speaker_name, confidence) or None if no match
        """
        if not known_speakers:
            return None

        # Validate segment embedding - check for NaN values
        if np.isnan(segment_embedding).any():
            logger.info(f"  ⚠️ Segment embedding contains NaN values - skipping matching")
            return None

        # Calculate similarities
        best_match = None
        best_similarity = threshold

        for speaker_id, speaker_name, speaker_embedding in known_speakers:
            # Validate known speaker embedding
            if np.isnan(speaker_embedding).any():
                logger.info(f"  ⚠️ Speaker '{speaker_name}' embedding contains NaN - skipping")
                continue

            similarity = cosine_similarity(
                segment_embedding.reshape(1, -1),
                speaker_embedding.reshape(1, -1)
            )[0][0]

            # Debug: Always print similarity score
            logger.info(f"  Similarity with {speaker_name}: {similarity:.4f} (threshold: {threshold})")

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = (speaker_id, speaker_name, float(similarity))

        return best_match

    def match_emotion_to_profile(
        self,
        emotion_embedding: np.ndarray,
        speaker_emotion_profiles: List[Tuple[str, np.ndarray, Optional[float]]],
        global_threshold: float = 0.6,
        speaker_threshold: Optional[float] = None
    ) -> Optional[Tuple[str, float]]:
        """
        Match emotion embedding to speaker's learned emotion profiles

        Args:
            emotion_embedding: Emotion embedding to match (1024-D from emotion2vec)
            speaker_emotion_profiles: List of (emotion_category, profile_embedding, custom_threshold) tuples
            global_threshold: Fallback threshold if no speaker/profile threshold set
            speaker_threshold: Per-speaker emotion threshold (overrides global, overridden by per-emotion)

        Returns:
            (emotion_category, confidence) or None if no match above threshold
        """
        if not speaker_emotion_profiles:
            return None

        # Validate emotion embedding
        if np.isnan(emotion_embedding).any():
            logger.info(f"  ⚠️  Emotion embedding contains NaN - skipping personalized matching")
            return None

        best_match = None
        best_similarity = 0.0

        # Threshold hierarchy: per-emotion > per-speaker > global
        default_threshold = speaker_threshold if speaker_threshold is not None else global_threshold

        for emotion_cat, profile_emb, custom_threshold in speaker_emotion_profiles:
            # Validate profile embedding
            if np.isnan(profile_emb).any():
                logger.info(f"  ⚠️  Emotion profile '{emotion_cat}' contains NaN - skipping")
                continue

            # Use custom threshold if set, otherwise use speaker/global threshold
            threshold = custom_threshold if custom_threshold is not None else default_threshold

            similarity = cosine_similarity(
                emotion_embedding.reshape(1, -1),
                profile_emb.reshape(1, -1)
            )[0][0]

            logger.info(f"  Emotion similarity with '{emotion_cat}' profile: {similarity:.4f} (threshold: {threshold:.4f})")

            if similarity > threshold and similarity > best_similarity:
                best_similarity = similarity
                best_match = (emotion_cat, float(similarity))

        return best_match

    def extract_segment_embedding(
        self,
        audio_file: str,
        start_time: float,
        end_time: float,
        context_padding: float = None
    ) -> np.ndarray:
        """
        Extract embedding from a specific segment of audio with context padding

        Args:
            audio_file: Path to audio file
            start_time: Start time in seconds
            end_time: End time in seconds
            context_padding: Seconds to include before/after for more reliable embeddings (default: 0.15s)
                           Optimal for movie audio with background music/effects
                           Based on comprehensive ground truth testing: 67.4% matching + only 3 misidentifications
                           Lower padding reduces background music corruption

        Returns:
            Segment embedding as numpy array
        """
        if context_padding is None:
            # Read at use time so a Settings API update applies without
            # reconstructing the heavyweight engine.
            context_padding = get_config().get_settings().context_padding
            self.context_padding = context_padding

        # Get actual audio duration to prevent out-of-bounds
        try:
            info = sf.info(audio_file)
            duration = info.duration

            # Add context padding for more reliable embeddings
            padded_start = start_time - context_padding
            padded_end = end_time + context_padding

            # Clamp times to valid range with small safety margin
            start_time = max(0, min(padded_start, duration - 0.5))
            end_time = min(padded_end, duration - 0.01)

            # If start is beyond end after clamping, adjust start
            if start_time >= end_time:
                start_time = max(0, end_time - 0.5)

            # Ensure segment is at least 0.1s
            if end_time - start_time < 0.1:
                end_time = min(start_time + 0.1, duration - 0.01)
                if end_time - start_time < 0.1:
                    start_time = max(0, end_time - 0.1)

        except Exception as e:
            logger.warning(f"Could not get audio duration, using original times: {e}")

        segment = Segment(start_time, end_time)
        with torch.no_grad():
            embedding = self.embedding_model.crop(audio_file, segment)
        return np.array(embedding)

    def extract_segment_embeddings_batch(
        self,
        segments: list,
        context_padding: float = None
    ) -> list:
        """
        Extract embeddings for multiple segments efficiently.

        Args:
            segments: List of dicts with keys: 'audio_file', 'start_time', 'end_time'
            context_padding: Optional padding override (uses instance default if None)

        Returns:
            List of embeddings (numpy arrays) in same order as input segments
            Skips segments that fail extraction (returns None in that position)
        """
        embeddings = []
        for seg in segments:
            try:
                emb = self.extract_segment_embedding(
                    seg['audio_file'], seg['start_time'], seg['end_time'], context_padding
                )
                embeddings.append(emb)
            except Exception as e:
                logger.info(f"Could not extract embedding from {os.path.basename(seg['audio_file'])}: {e}")
                embeddings.append(None)
        return embeddings

    def _preload_emotion_audio(self, audio_file: str) -> Optional["AudioSegment"]:
        """Decode/resample once only when emotion inference is available."""
        if self.emotion_model is None:
            return None
        full_audio = AudioSegment.from_file(audio_file)
        if full_audio.frame_rate != 16000:
            full_audio = full_audio.set_frame_rate(16000)
        return full_audio

    def preload_emotion_audio(self, audio_file: str) -> Optional["AudioSegment"]:
        """Public decode-once contract for recalculation/profile callers."""
        return self._preload_emotion_audio(audio_file)

    def extract_emotion(
        self,
        audio_file: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        extract_embedding: bool = False,
        preloaded_audio: Optional["AudioSegment"] = None,
    ) -> Optional[Dict]:
        """
        Extract emotion from audio file or segment using emotion2vec via FunASR

        Args:
            audio_file: Path to audio file
            start_time: Optional start time for segment (seconds)
            end_time: Optional end time for segment (seconds)
            extract_embedding: If True, also extract emotion embedding (default: False)

        Returns:
            Dictionary with emotion data or None if extraction fails:
            {
                'emotion_category': str,  # Primary emotion label
                'emotion_confidence': float,  # 0-1
                'embedding': np.ndarray  # 1024-D embedding (only if extract_embedding=True)
            }
        """
        if self.emotion_model is None:
            return None

        try:
            # FunASR emotion2vec can process audio file directly or with timestamps
            # Format: model.generate(input, granularity="utterance")

            if start_time is not None and end_time is not None:
                # For segments, we need to extract the audio first.
                # Cap segment duration to avoid OOM in emotion2vec attention (O(n^2) memory)
                if (end_time - start_time) > MAX_EMOTION_DURATION_SEC:
                    end_time = start_time + MAX_EMOTION_DURATION_SEC

                # Extract segment from preloaded audio when available.
                # Callers processing many segments of one file (the main
                # diarization loop) used to re-decode the ENTIRE file per
                # segment (measured ~0.5 s/segment on a 30 min file); decode
                # once and slice instead.
                if preloaded_audio is not None:
                    audio = preloaded_audio
                else:
                    audio = AudioSegment.from_file(audio_file)
                start_ms = int(start_time * 1000)
                end_ms = int(end_time * 1000)
                segment = audio[start_ms:end_ms]

                # Resample to 16kHz if needed (emotion2vec requirement).
                # Preloaded audio is already 16kHz by contract.
                if preloaded_audio is None and segment.frame_rate != 16000:
                    segment = segment.set_frame_rate(16000)

                # Hand the samples over in memory: a temp WAV per segment stalls on a busy disk.
                result = self.emotion_model.generate(
                    _float_samples(segment),
                    granularity="utterance",
                    extract_embedding=extract_embedding,
                )
            else:
                # Process a bounded, normalized copy. Passing the original
                # filename when it was already 16 kHz bypassed the duration
                # slice above and sent the entire recording to emotion2vec.
                audio = AudioSegment.from_file(audio_file)

                # Cap duration to avoid OOM in emotion2vec attention (O(n^2) memory)
                max_emotion_duration_ms = int(MAX_EMOTION_DURATION_SEC * 1000)
                needs_temp_input = (
                    len(audio) > max_emotion_duration_ms
                    or audio.frame_rate != 16000
                )
                if len(audio) > max_emotion_duration_ms:
                    audio = audio[:max_emotion_duration_ms]

                # Resample to 16kHz if needed
                if audio.frame_rate != 16000:
                    audio = audio.set_frame_rate(16000)

                if needs_temp_input:
                    # Long already-16 kHz sources must still use the processed
                    # object or the in-memory duration cap is silently bypassed.
                    with tempfile.NamedTemporaryFile(
                        suffix=".wav", delete=False
                    ) as temp_file:
                        audio.export(temp_file.name, format="wav")
                        temp_path = temp_file.name

                    try:
                        result = self.emotion_model.generate(
                            temp_path,
                            granularity="utterance",
                            extract_embedding=extract_embedding,
                        )
                    finally:
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
                else:
                    result = self.emotion_model.generate(
                        audio_file,
                        granularity="utterance",
                        extract_embedding=extract_embedding,
                    )

            # FunASR emotion2vec returns list of results
            # Format: [[{'labels': ['angry'], 'scores': [0.9]}]]
            if result and len(result) > 0:
                # Unwrap nested structure
                res = result[0] if isinstance(result, list) else result
                if isinstance(res, list) and len(res) > 0:
                    res = res[0]

                # Extract emotion label and scores
                if isinstance(res, dict):
                    labels = res.get('labels', ['unknown'])
                    scores = res.get('scores', [0.0])

                    # Find the index with highest score
                    if isinstance(scores, list) and len(scores) > 0:
                        max_idx = scores.index(max(scores))
                        max_score = scores[max_idx]

                        # Get the label at that index
                        if isinstance(labels, list) and len(labels) > max_idx:
                            raw_label = labels[max_idx]
                        else:
                            raw_label = labels[0] if isinstance(labels, list) else str(labels)
                    else:
                        raw_label = labels[0] if isinstance(labels, list) else str(labels)
                        max_score = 0.0

                    # Extract English label (format: "中文/english")
                    if '/' in raw_label:
                        emotion_label = raw_label.split('/')[-1].strip()
                    else:
                        emotion_label = raw_label
                else:
                    emotion_label = 'unknown'
                    max_score = 0.0

                # Return emotion category and confidence
                # Note: emotion2vec does NOT return arousal/valence - it only does categorical classification
                emotion_data = {
                    'emotion_category': emotion_label,
                    'emotion_confidence': float(max_score)
                }

                # Extract embedding if requested
                if extract_embedding and isinstance(res, dict):
                    feats = res.get('feats')
                    if feats is not None:
                
                        # Convert to numpy array (1024-D from emotion2vec)
                        embedding = np.array(feats, dtype=np.float32)
                        emotion_data['embedding'] = embedding

                # Explicit cleanup: delete result object and clear references
                del result, res
                # Queue async cleanup (non-blocking)
                self.clear_gpu_cache_async("emotion")

                return emotion_data

            # Cleanup even if no result
            del result
            self.clear_gpu_cache_async("emotion")
            return None

        except Exception as e:
            logger.warning(f"Failed to extract emotion: {e}")
            traceback.print_exc()
            self.clear_gpu_cache_async("emotion_error")
            return None

    def transcribe_with_diarization(
        self,
        audio_file: str,
        known_speakers: List[Tuple[int, str, np.ndarray]] = None,
        threshold: float = 0.7,
        db_session = None
    ) -> Dict:
        """
        Full pipeline: transcription + diarization + speaker recognition + personalized emotions

        Args:
            audio_file: Path to audio file
            known_speakers: Optional list of (id, name, embedding) tuples
            threshold: Similarity threshold for speaker matching
            db_session: Optional database session for personalized emotion matching

        Returns:
            Dictionary with transcribed segments and speaker labels
        """
        known_speakers = known_speakers or []
        logger.info(f"Known speakers: {len(known_speakers)}")
        for speaker_id, speaker_name, _ in known_speakers:
            logger.info(f"  - ID: {speaker_id}, Name: {speaker_name}")

        # Run transcription and diarization IN PARALLEL
        logger.info("Running transcription and diarization in parallel...")
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both tasks to run concurrently
            transcription_future = executor.submit(self.transcribe, audio_file)
            diarization_future = executor.submit(self.diarize, audio_file)

            # Wait for both to complete
            transcription_segments = transcription_future.result()
            diarization_result = diarization_future.result()

        elapsed = time.time() - start_time
        logger.info(f"Parallel processing completed in {elapsed:.2f}s")

        if CLUSTER_SPEAKERS or WORD_LEVEL_SPEAKERS:
            turns = diarization_result["exclusive_segments" if CLUSTER_SPEAKERS else "segments"]
            before = len(transcription_segments)
            transcription_segments = _split_by_speaker(transcription_segments, turns)
            logger.info(f"Word-level speakers: {before} transcription segments -> {len(transcription_segments)}")

        # Explicit cleanup after parallel processing to free VRAM before emotion extraction
        del transcription_future, diarization_future
        gc.collect()
        self.clear_gpu_cache_async("parallel")

        # Hoist settings lookups out of the per-segment loops.
        settings = get_config().get_settings()
        filter_hallucinations = settings.filter_hallucinations
        global_emotion_threshold = settings.emotion_threshold
        enable_personalized_emotions = os.getenv("ENABLE_PERSONALIZED_EMOTIONS", "true").lower() == "true"

        # Pass 1: drop hallucinations, give each segment its diarization speaker and its own voice embedding.
        # The cheap text-only filter runs BEFORE any embedding or emotion extraction so Whisper's
        # "thank you for watching" ghosts don't cost us a GPU pass each.
        segments = []
        for trans_seg in transcription_segments:
            if filter_hallucinations and _is_hallucination(
                    trans_seg["text"], trans_seg["end"] - trans_seg["start"], trans_seg.get("avg_logprob")):
                logger.info(f"⏭️ Skipping hallucination: '{trans_seg['text'][:60]}'")
                continue
            label = trans_seg.get("speaker_label") or _label_at_midpoint(trans_seg, diarization_result["segments"])
            segments.append((trans_seg, label, self._segment_embedding_or_none(audio_file, trans_seg)))

        # Pass 2: who is speaking in each segment.
        if CLUSTER_SPEAKERS:
            identities = self._identify_clusters(audio_file, segments, diarization_result, known_speakers,
                                                 threshold, db_session)
            speakers = [identities[label] for _, label, _ in segments]
        else:
            speakers = self._identify_segments(segments, known_speakers, threshold, db_session)

        # Pass 3: emotion, then the output segment.
        # Decode/resample the conversation audio ONCE for the emotion pass
        # (lazy: only if any segment actually needs emotion extraction).
        preloaded_emotion_audio = None
        emotion_preload_failed = False
        transcribed_with_speakers = []

        for (trans_seg, speaker_label, embedding), who in zip(segments, speakers):
            if preloaded_emotion_audio is None and not emotion_preload_failed:
                try:
                    preloaded_emotion_audio = self.preload_emotion_audio(audio_file)
                    if preloaded_emotion_audio is None:
                        emotion_preload_failed = True
                except Exception as e:  # noqa: BLE001 - decoder/model failures use per-segment fallback
                    logger.info(f"Emotion preload failed ({e}); falling back to per-segment decode")
                    emotion_preload_failed = True

            emotion_data = self.extract_emotion(
                audio_file,
                trans_seg["start"],
                trans_seg["end"],
                extract_embedding=enable_personalized_emotions,
                preloaded_audio=preloaded_emotion_audio,
            )
            if emotion_data:
                if who["matched_emotion"] is not None:
                    self._apply_voice_profile_emotion(emotion_data, who)
                elif enable_personalized_emotions and who["is_known"] and db_session:
                    self._apply_personalized_emotion(emotion_data, who["speaker"], embedding, db_session,
                                                     global_emotion_threshold)

            segment_data = {
                "start": trans_seg["start"],
                "end": trans_seg["end"],
                "text": trans_seg["text"],
                "speaker": who["speaker"],
                "speaker_label": speaker_label,
                "is_known": who["is_known"],
                "confidence": float(who["confidence"] or 0.0),  # Convert to Python float
                "embedding": embedding,  # Include embedding for unknown speakers
                "words": trans_seg.get("words", []),  # Include word-level data
                "avg_logprob": trans_seg.get("avg_logprob")  # Include segment confidence
            }
            if who["speaker_id"] is not None:   # already resolved (SPEAKER_ASSIGNMENT=cluster)
                segment_data["speaker_id"] = who["speaker_id"]

            # Add emotion data if available (include embedding for caching)
            if emotion_data:
                emotion_conf = emotion_data.get("emotion_confidence")
                segment_data.update({
                    "emotion_category": emotion_data.get("emotion_category"),
                    "emotion_confidence": float(emotion_conf) if emotion_conf is not None else None,
                    "emotion_embedding": emotion_data.get("embedding"),
                    "detector_breakdown": emotion_data.get("detector_breakdown"),
                })

            transcribed_with_speakers.append(segment_data)

        return {
            "segments": transcribed_with_speakers,
            "num_speakers": diarization_result["num_speakers"]
        }

    def _segment_embedding_or_none(self, audio_file: str, seg: Dict) -> Optional[np.ndarray]:
        """Voice embedding of one segment, or None if it is too short, at a file boundary or NaN."""
        segment_duration = seg["end"] - seg["start"]
        if segment_duration < MIN_SEGMENT_DURATION_SEC:
            logger.info(f"⏭️ Skipping embedding extraction (segment too short: {segment_duration:.2f}s)")
            return None
        try:
            embedding = self.extract_segment_embedding(audio_file, seg["start"], seg["end"])
        except RuntimeError as e:
            error_str = str(e)
            if "Kernel size" in error_str or "input size" in error_str:
                logger.info(f"⏭️ Skipping embedding extraction (segment too short for embedding model)")
            elif "beyond file duration" in error_str:
                logger.info(f"⏭️ Skipping embedding extraction (segment at file boundary)")
            else:
                raise
            return None
        except Exception as e:
            # Catch pyannote errors like sample mismatches at file boundaries
            if "samples instead of the expected" in str(e) or "requested chunk" in str(e):
                logger.info(f"⏭️ Skipping embedding extraction (segment beyond file duration: {seg['start']:.2f}s-{seg['end']:.2f}s)")
                return None
            raise
        if np.isnan(embedding).any():
            logger.info(f"⏭️ Segment embedding has NaN - skipping")
            return None
        return embedding

    @staticmethod
    def _speaker(name, speaker_id=None, is_known=False, confidence=0.0, matched_emotion=None) -> Dict:
        return {"speaker": name, "speaker_id": speaker_id, "is_known": is_known, "confidence": confidence,
                "matched_emotion": matched_emotion}

    def _identify_segments(self, segments, known_speakers, threshold, db_session) -> List[Dict]:
        """Default (SPEAKER_ASSIGNMENT=segment): match every segment's own embedding against the stored profiles.

        Unmatched segments are named Unknown_NN per diarization label here, then auto-enrolled one segment at a
        time when they are saved (create_segment_from_result).
        """
        unknown_speaker_map: Dict[str, str] = {}
        unknown_counter = 1
        out = []
        for _, speaker_label, embedding in segments:
            who = self._speaker(speaker_label)
            if embedding is not None:
                if db_session:
                    # Enhanced matching against ALL voice profiles (general + emotion-specific)
                    match = self.match_speaker_with_all_profiles(embedding, db_session, threshold=threshold)
                    if match:
                        who = self._speaker(match["speaker_name"], is_known=True, confidence=match["confidence"],
                                            matched_emotion=match.get("matched_emotion"))
                elif known_speakers:
                    # Fallback to old method if no db_session
                    match = self.match_speaker(embedding, known_speakers, threshold)
                    if match:
                        _, name, confidence = match
                        who = self._speaker(name, is_known=True, confidence=confidence)
                        logger.info(f"Matched segment to {name} with confidence {confidence:.2%}")
            if not who["is_known"]:
                # Consistent mapping from diarization label to Unknown_XX (no label: a generic unknown)
                if speaker_label and speaker_label in unknown_speaker_map:
                    who["speaker"] = unknown_speaker_map[speaker_label]
                else:
                    who["speaker"] = f"Unknown_{unknown_counter:02d}"
                    unknown_counter += 1
                    if speaker_label:
                        unknown_speaker_map[speaker_label] = who["speaker"]
            out.append(who)
        return out

    def _identify_clusters(self, audio_file, segments, diarization_result, known_speakers, threshold,
                           db_session) -> Dict[Optional[str], Dict]:
        """SPEAKER_ASSIGNMENT=cluster: one identity per diarization speaker, decided once from all of its speech.

        A speaker's voice is the duration-weighted centroid of its segments' embeddings. Speakers are matched to the
        stored people one-to-one (pyannote has already decided they are different people), above the configured
        threshold and no lower. A speaker that matches nobody becomes ONE new Unknown_* (auto-enrolled with the
        centroid), so an unsure match stays unknown instead of being given someone else's name.
        """
        groups: Dict[Optional[str], Tuple[List[np.ndarray], List[float]]] = {}
        for seg, label, embedding in segments:
            embeddings, weights = groups.setdefault(label, ([], []))
            if embedding is not None:
                embeddings.append(embedding)
                weights.append(min(seg["end"] - seg["start"], CENTROID_MAX_WEIGHT_SEC))
        labels = list(groups)   # order of first appearance
        centroids = {label: _centroid(*groups[label]) for label in labels}
        for label in labels:
            if centroids[label] is None and label is not None:
                centroids[label] = self._turns_centroid(audio_file, label, diarization_result["exclusive_segments"])

        # Candidate people: speaker_id -> (name, all their voice profiles: general + emotion-specific)
        people: Dict[Any, Tuple[str, List[np.ndarray]]] = {}
        if db_session:
            for p in self._all_profiles(db_session):
                people.setdefault(p["speaker_id"], (p["speaker_name"], []))[1].append(p["embedding"])
        else:
            for speaker_id, name, embedding in known_speakers:
                if not np.isnan(embedding).any():
                    people.setdefault(speaker_id, (name, []))[1].append(embedding)

        ids = list(people)
        matchable = [label for label in labels if centroids[label] is not None]
        identities: Dict[Optional[str], Dict] = {}
        if ids and matchable:
            sims = np.array([[max(float(cosine_similarity(centroids[label].reshape(1, -1), e.reshape(1, -1))[0][0])
                                  for e in people[i][1]) for i in ids] for label in matchable])
            for label, row in zip(matchable, sims):
                top = sorted(((s, people[i][0]) for s, i in zip(row, ids)), reverse=True)[:3]
                logger.info(f"  Speaker {label}: best matches " + ", ".join(f"{n} {s:.3f}" for s, n in top))
            from scipy.optimize import linear_sum_assignment
            rows, cols = linear_sum_assignment(np.where(sims > threshold, -sims, 0.0))
            for r, c in zip(rows, cols):
                if sims[r, c] > threshold:
                    label, speaker_id = matchable[r], ids[c]
                    identities[label] = self._speaker(people[speaker_id][0], speaker_id, True, sims[r, c])
                    logger.info(f"  ✅ Speaker {label} → {people[speaker_id][0]} ({sims[r, c]:.2%})")

        unknown_counter = 1
        for label in labels:
            if label in identities:
                continue
            centroid = centroids[label]
            if db_session and centroid is not None:
                speaker_id, name = _create_unknown_speaker(centroid, db_session)
                self.add_speaker_to_cache(speaker_id, name, centroid)
                identities[label] = self._speaker(name, speaker_id, confidence=1.0)
            else:
                identities[label] = self._speaker(f"Unknown_{unknown_counter:02d}")
                unknown_counter += 1
            logger.info(f"  ❔ Speaker {label} → {identities[label]['speaker']} (no stored voice above {threshold:.2%})")
        return identities

    def _turns_centroid(self, audio_file: str, label: str, turns: List[Dict]) -> Optional[np.ndarray]:
        """Centroid of a diarization speaker's longest turns (for a speaker none of whose segments could be embedded)."""
        longest = sorted((t for t in turns if t["speaker"] == label), key=lambda t: t["duration"], reverse=True)
        embeddings, weights = [], []
        for turn in longest[:CENTROID_FALLBACK_TURNS]:
            embedding = self._segment_embedding_or_none(audio_file, turn)
            if embedding is not None:
                embeddings.append(embedding)
                weights.append(min(turn["duration"], CENTROID_MAX_WEIGHT_SEC))
        return _centroid(embeddings, weights)

    def _apply_voice_profile_emotion(self, emotion_data: Dict, who: Dict):
        """The segment matched one of the speaker's emotion voice profiles: use that as the primary emotion signal."""
        matched_emotion, confidence, speaker_name = who["matched_emotion"], who["confidence"], who["speaker"]
        logger.info(f"  🎯 Voice profile match indicates emotion: {matched_emotion}")

        # Create detector breakdown showing voice match as primary
        emotion_data['detector_breakdown'] = {
            'emotion2vec_detector': {
                'emotion': emotion_data['emotion_category'],
                'confidence': float(emotion_data.get('emotion_confidence', 0.0))
            },
            'voice_profile_detector': {
                'emotion': matched_emotion,
                'confidence': float(confidence)  # Use speaker match confidence
            },
            'final_decision': {
                'emotion': matched_emotion,
                'confidence': float(confidence),
                'reason': f'Matched {speaker_name}_{matched_emotion} voice profile',
                'voice_profile_available': True
            }
        }

        # Override emotion with voice profile match
        emotion_data['emotion_category'] = matched_emotion
        emotion_data['emotion_confidence'] = float(confidence)  # Convert to Python float

        logger.info(f"  🔬 Enhanced Match Results:")
        logger.info(f"     emotion2vec: {emotion_data['detector_breakdown']['emotion2vec_detector']['emotion']} ({emotion_data['detector_breakdown']['emotion2vec_detector']['confidence']:.2%})")
        logger.info(f"     Voice profile: {matched_emotion} ({confidence:.2%})")
        logger.info(f"     Final: {matched_emotion} - Matched emotion voice profile")

    def _apply_personalized_emotion(self, emotion_data: Dict, speaker_name: str, voice_embedding, db_session,
                                    global_emotion_threshold: float):
        """Known speaker with learned emotion profiles: let the dual detector make the final emotion decision."""
        emotion_embedding = emotion_data.get('embedding')
        if emotion_embedding is None:
            return
        try:
            speaker = db_session.query(Speaker).filter(Speaker.name == speaker_name).first()
            if not (speaker and speaker.emotion_profiles):
                return
            dual_result = self.match_emotion_dual_detector(
                emotion_embedding=emotion_embedding,
                voice_embedding=voice_embedding,
                speaker_emotion_profiles=speaker.emotion_profiles,
                global_threshold=global_emotion_threshold,
                speaker_threshold=speaker.emotion_threshold,
                generic_emotion=emotion_data['emotion_category'],
                generic_confidence=emotion_data['emotion_confidence'],
            )

            # Use final decision from dual-detector
            final = dual_result['final_decision']
            emotion_data['emotion_category'] = final['emotion']
            emotion_data['emotion_confidence'] = final['confidence']

            # Store detector breakdown for client display
            emotion_data['detector_breakdown'] = dual_result

            logger.info(f"  🔬 Dual-Detector Results:")
            logger.info(f"     Detector 1 (emotion2vec): {dual_result['emotion2vec_detector']['emotion']} ({dual_result['emotion2vec_detector']['confidence']:.2%})")
            if dual_result['voice_profile_detector']:
                logger.info(f"     Detector 2 (voice profile): {dual_result['voice_profile_detector']['emotion']} ({dual_result['voice_profile_detector']['confidence']:.2%})")
            else:
                logger.info(f"     Detector 2 (voice profile): Not available")
            logger.info(f"     Final: {final['emotion']} ({final['confidence']:.2%}) - {final['reason']}")

        except Exception as e:
            logger.info(f"  Warning: Personalized emotion matching failed: {e}")
            # Fall back to generic emotion detection

    def _all_profiles(self, db_session) -> List[Dict[str, Any]]:
        """Every voice profile (general + emotion-specific): the warm cache, else one DB scan (memoized)."""
        with self._cache_lock:
            if self._speaker_cache is not None:
                logger.info(f"  ⚡ Using cached profiles: {len(self._speaker_cache)} profiles")
                return self._speaker_cache
            logger.info(f"  ⚠️ Cache not loaded, querying DB...")
            self._speaker_cache = _build_profile_list(db_session.query(Speaker).all())
            logger.info(f"  🔍 Checking against {len(self._speaker_cache)} voice profiles (general + emotion-specific)")
            return self._speaker_cache

    def match_speaker_with_all_profiles(self, segment_embedding, db_session, threshold=0.35):
        """
        Enhanced speaker matching that checks ALL voice profiles (general + emotion-specific).

        This matches against:
        1. Speaker's general voice profile
        2. ALL speaker emotion voice profiles (Andy_angry, Andy_happy, etc.)

        Returns best match with speaker AND emotion info.
        """
        if db_session is None:
            return None

        best_match = None
        best_similarity = threshold

        for profile in self._all_profiles(db_session):
            similarity = cosine_similarity(
                segment_embedding.reshape(1, -1),
                profile['embedding'].reshape(1, -1)
            )[0][0]

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = {
                    'speaker_id': profile['speaker_id'],
                    'speaker_name': profile['speaker_name'],
                    'confidence': float(similarity),  # Convert numpy type to Python float
                    'matched_emotion': profile['emotion'],  # None if general, emotion name if emotion profile
                    'profile_type': profile['profile_type']
                }

        if best_match:
            if best_match['matched_emotion']:
                logger.info(f"  ✅ BEST MATCH: {best_match['speaker_name']}_{best_match['matched_emotion']} ({best_match['confidence']:.2%}) - Emotion voice profile")
            else:
                logger.info(f"  ✅ BEST MATCH: {best_match['speaker_name']} ({best_match['confidence']:.2%}) - General profile")
        else:
            logger.info(f"  ❌ No match found above threshold {threshold:.2%}")

        return best_match

    def match_emotion_dual_detector(
        self,
        emotion_embedding,
        voice_embedding,
        speaker_emotion_profiles,
        global_threshold=0.6,
        speaker_threshold=None,
        generic_emotion="neutral",
        generic_confidence=0.0
    ):
        """Dual-detector emotion matching using both emotion2vec and voice profiles"""


        # Validate inputs
        if np.isnan(emotion_embedding).any() or np.isnan(voice_embedding).any():
            return {
                "emotion2vec_detector": {"emotion": generic_emotion, "confidence": float(generic_confidence)},
                "voice_profile_detector": None,
                "final_decision": {"emotion": generic_emotion, "confidence": float(generic_confidence), "reason": "Invalid embeddings", "voice_profile_available": False}
            }

        if not speaker_emotion_profiles:
            return {
                "emotion2vec_detector": {"emotion": generic_emotion, "confidence": float(generic_confidence)},
                "voice_profile_detector": None,
                "final_decision": {"emotion": generic_emotion, "confidence": float(generic_confidence), "reason": "No profiles", "voice_profile_available": False}
            }

        # Get SPEAKER_THRESHOLD for voice profile matching (512-D embeddings)
        default_speaker_threshold = get_config().get_settings().speaker_threshold

        # DETECTOR 1: emotion2vec (uses EMOTION_THRESHOLD for 1024-D embeddings)
        detector1_best = None
        detector1_confidence = 0.0
        default_emotion_threshold = speaker_threshold if speaker_threshold is not None else global_threshold
        
        for profile in speaker_emotion_profiles:
            profile_emb = profile.get_embedding()
            if profile_emb is None or np.isnan(profile_emb).any():
                continue

            threshold = profile.confidence_threshold if profile.confidence_threshold is not None else default_emotion_threshold
            similarity = cosine_similarity(emotion_embedding.reshape(1, -1), profile_emb.reshape(1, -1))[0][0]

            if similarity > threshold and similarity > detector1_confidence:
                detector1_confidence = similarity
                detector1_best = profile.emotion_category
        
        detector1_result = {
            "emotion": detector1_best or generic_emotion,
            "confidence": float(detector1_confidence) if detector1_best else float(generic_confidence)
        }

        # DETECTOR 2: Voice profile (uses SPEAKER_THRESHOLD for 512-D embeddings)
        voice_matches = []
        voice_best = None
        voice_best_confidence = 0.0
        # MIN_VOICE_SAMPLES defined at module level

        for profile in speaker_emotion_profiles:
            voice_emb = profile.get_voice_embedding()
            if voice_emb is None or np.isnan(voice_emb).any() or profile.voice_sample_count < MIN_VOICE_SAMPLES:
                continue

            voice_threshold = profile.voice_threshold if profile.voice_threshold is not None else default_speaker_threshold
            voice_sim = cosine_similarity(voice_embedding.reshape(1, -1), voice_emb.reshape(1, -1))[0][0]

            voice_matches.append({
                "emotion": profile.emotion_category,
                "similarity": float(voice_sim),
                "threshold": float(voice_threshold),
                "samples": profile.voice_sample_count
            })

            if voice_sim > voice_threshold and voice_sim > voice_best_confidence:
                voice_best_confidence = voice_sim
                voice_best = profile.emotion_category

        detector2_result = {
            "emotion": voice_best,
            "confidence": float(voice_best_confidence),
            "matches": voice_matches
        }
        detector2_has_diagnostics = bool(voice_matches)
        # detector2_available is True only when at least one voice profile had
        # enough samples AND produced an above-threshold opinion. A missing
        # detector-2 result must be treated as "no second opinion", not as a
        # literal "neutral" prediction — otherwise the combiner neutralises
        # every confident detector-1 label as soon as the speaker has fewer
        # than MIN_VOICE_SAMPLES corrections.
        detector2_available = voice_best is not None

        # COMBINE
        d1_emotion = detector1_result["emotion"]
        d1_conf = detector1_result["confidence"]
        d2_emotion = detector2_result["emotion"]
        d2_conf = detector2_result["confidence"]

        if d1_emotion == "neutral" or d1_emotion == "<unk>":
            final = {"emotion": "neutral", "confidence": float(d1_conf), "reason": "emotion2vec neutral", "voice_profile_available": detector2_available}
        elif not detector2_available:
            final = {"emotion": d1_emotion, "confidence": float(d1_conf), "reason": "emotion2vec only (no voice profile match)", "voice_profile_available": False}
        elif d1_emotion == d2_emotion and d1_conf > DUAL_DETECTOR_AGREE_D1 and d2_conf > DUAL_DETECTOR_AGREE_D2:
            final = {"emotion": d1_emotion, "confidence": float((d1_conf + d2_conf) / 2), "reason": "Both agree", "voice_profile_available": True}
        elif d2_conf > VOICE_STRONG_THRESHOLD:
            final = {"emotion": d2_emotion, "confidence": float(d2_conf), "reason": f"Voice strong: {d2_emotion}", "voice_profile_available": True}
        elif d1_emotion != d2_emotion:
            final = {"emotion": "neutral", "confidence": float(max(d1_conf, d2_conf)), "reason": f"Disagree: {d1_emotion} vs {d2_emotion}", "voice_profile_available": True}
        else:
            final = {"emotion": d1_emotion, "confidence": float(d1_conf), "reason": "Agree low conf", "voice_profile_available": True}

        return {
            "emotion2vec_detector": detector1_result,
            # Keep below-threshold similarities visible for diagnostics while
            # still treating them as no second opinion in decision logic.
            "voice_profile_detector": (
                detector2_result if detector2_has_diagnostics else None
            ),
            "final_decision": final
        }

