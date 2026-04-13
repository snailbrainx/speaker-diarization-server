import os

# Fix for PyTorch 2.6+ weights_only=True breaking change
# Required for loading pyannote.audio model checkpoints which contain
# pickled objects (omegaconf, pytorch_lightning callbacks, etc.)
# This must be set BEFORE importing torch
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'

import torch

import numpy as np
from pyannote.audio import Pipeline, Model, Inference
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional
from faster_whisper import WhisperModel
from concurrent.futures import ThreadPoolExecutor
import time
from pydub import AudioSegment
import threading
import queue
import gc

# Speaker matching constants
FALLBACK_THRESHOLD_REDUCTION = 0.05
MIN_FALLBACK_THRESHOLD = 0.20
UNKNOWN_NAME_MAX_ATTEMPTS = 5

# Emotion detection constants
MIN_VOICE_SAMPLES = 3
DUAL_DETECTOR_AGREE_D1 = 0.70
DUAL_DETECTOR_AGREE_D2 = 0.80
VOICE_STRONG_THRESHOLD = 0.85
MAX_EMOTION_DURATION_SEC = 30.0
MIN_SEGMENT_DURATION_SEC = 0.5


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
    from .models import Speaker

    # FIRST: Check ALL speakers (including enrolled ones like "Bob", "Alice") as safety fallback
    # Use slightly LOWER threshold (5% less) to catch borderline cases that transcribe_with_diarization missed
    # This prevents creating duplicate Unknowns for people already in the database
    fallback_threshold = max(threshold - FALLBACK_THRESHOLD_REDUCTION, MIN_FALLBACK_THRESHOLD)

    all_speakers = db_session.query(Speaker).all()
    best_match = None
    best_similarity = 0.0
    best_above_threshold = None

    if all_speakers and embedding is not None:
        print(f"  🔍 Auto-enroll fallback: Checking {len(all_speakers)} speakers (fallback threshold: {fallback_threshold:.2%})")

        for speaker in all_speakers:
            existing_embedding = speaker.get_embedding()

            # Calculate cosine similarity
            similarity = cosine_similarity(
                embedding.reshape(1, -1),
                existing_embedding.reshape(1, -1)
            )[0][0]

            # Track best overall (for logging)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = speaker

            # Track best above threshold (for matching)
            if similarity > fallback_threshold and (best_above_threshold is None or similarity > best_similarity):
                best_above_threshold = speaker
                print(f"    → Match: '{speaker.name}' (similarity: {similarity:.2%})")

    if best_above_threshold:
        print(f"✓ Fallback match: Identified as '{best_above_threshold.name}' (similarity: {best_similarity:.2%}, fallback threshold: {fallback_threshold:.2%})")
        return (best_above_threshold.id, best_above_threshold.name)
    elif best_match:
        print(f"  ℹ️ Best match was '{best_match.name}' with {best_similarity:.2%} (below fallback threshold {fallback_threshold:.2%})")

    # No match found - create new Unknown speaker with timestamp
    # Use microsecond precision for better uniqueness
    max_attempts = UNKNOWN_NAME_MAX_ATTEMPTS
    timestamp_name = None

    for attempt in range(max_attempts):
        timestamp_name = f"Unknown_{int(time.time() * 1000000)}"

        # Check if name already exists in DB
        existing = db_session.query(Speaker).filter(Speaker.name == timestamp_name).first()
        if not existing:
            break
        time.sleep(0.001)  # Wait 1ms and try again
    else:
        # Fallback to UUID if all attempts fail
        import uuid
        timestamp_name = f"Unknown_{uuid.uuid4().hex[:12]}"

    # Create new speaker with embedding
    new_speaker = Speaker(name=timestamp_name)
    new_speaker.set_embedding(embedding)
    db_session.add(new_speaker)
    db_session.flush()  # Get the ID without committing

    print(f"✓ Auto-enrolled new speaker: {timestamp_name} (ID: {new_speaker.id})")

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

        # Load configuration from environment
        self.context_padding = float(os.getenv("CONTEXT_PADDING", "0.15"))

        # Initialize models (lazy loading)
        self._diarization_pipeline = None
        self._embedding_model = None
        self._whisper_model = None
        self._emotion_model = None

        # Speaker profile cache (reduces DB queries during streaming)
        self._speaker_cache = None
        self._cache_lock = threading.Lock()

        # Background cleanup thread (non-blocking)
        self._cleanup_running = True  # Set BEFORE starting thread
        self._cleanup_queue = queue.Queue(maxsize=10)  # Limit queue size
        self._cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self._cleanup_thread.start()

        # Periodic cleanup timer (every 2 minutes)
        self._periodic_cleanup_thread = threading.Thread(target=self._periodic_cleanup_worker, daemon=True)
        self._periodic_cleanup_thread.start()

        print(f"Speaker Recognition Engine initialized on device: {self.device}")
        print(f"Background GPU cleanup thread started")
        print(f"Periodic cleanup timer started (every 30 seconds)")

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
                print(f"Background cleanup error: {e}")

    def _periodic_cleanup_worker(self):
        """Periodic cleanup timer - runs every 30 seconds to free VRAM"""
        import time
        while self._cleanup_running:
            try:
                # Wait 30 seconds (more frequent than 2 minutes)
                time.sleep(30)

                if torch.cuda.is_available():
                    vram_used_gb = torch.cuda.memory_reserved() / (1024**3)

                    # ALWAYS cleanup every 30 seconds (unconditional for stability with other models)
                    print(f"⏰ Periodic cleanup (VRAM: {vram_used_gb:.1f}GB)")

                    # Force cleanup
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                    vram_after_gb = torch.cuda.memory_reserved() / (1024**3)
                    freed_gb = vram_used_gb - vram_after_gb
                    if freed_gb > 0.1:
                        print(f"  ✅ Freed {freed_gb:.1f}GB VRAM (now {vram_after_gb:.1f}GB)")

            except Exception as e:
                print(f"Periodic cleanup error: {e}")

    def clear_gpu_cache_async(self, cleanup_type="standard"):
        """Queue GPU cleanup to run in background (non-blocking, conditional on VRAM usage)"""
        if not torch.cuda.is_available():
            return

        # Get VRAM threshold from env (default: 12GB)
        threshold_gb = float(os.getenv("CLEANUP_VRAM_THRESHOLD_GB", "12"))

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
        """
        Load all speaker profiles into memory cache (one-time DB query).
        Dramatically reduces DB overhead during streaming (153 queries → 1 query).
        """

        from .models import Speaker, SpeakerEmotionProfile

        with self._cache_lock:
            all_profiles = []
            speakers = db_session.query(Speaker).all()

            print(f"📦 Loading speaker cache: {len(speakers)} speakers...")

            for speaker in speakers:
                # 1. Add general speaker profile
                general_emb = speaker.get_embedding()
                if general_emb is not None and not np.isnan(general_emb).any():
                    all_profiles.append({
                        'speaker_id': speaker.id,
                        'speaker_name': speaker.name,
                        'embedding': general_emb,
                        'emotion': None,
                        'profile_type': 'general'
                    })

                # 2. Add ALL emotion voice profiles for this speaker
                for emotion_profile in speaker.emotion_profiles:
                    voice_emb = emotion_profile.get_voice_embedding()
                    if voice_emb is not None and not np.isnan(voice_emb).any():
                        # Only include if has enough samples (min 3)
                        if emotion_profile.voice_sample_count >= 3:
                            all_profiles.append({
                                'speaker_id': speaker.id,
                                'speaker_name': speaker.name,
                                'embedding': voice_emb,
                                'emotion': emotion_profile.emotion_category,
                                'profile_type': 'emotion_voice'
                            })

            self._speaker_cache = all_profiles
            print(f"✅ Speaker cache loaded: {len(all_profiles)} total profiles")
            return len(all_profiles)

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
            print(f"➕ Added to speaker cache: {speaker_name} ({profile_type})")

    def clear_speaker_cache(self):
        """Clear speaker cache (forces reload on next use)"""
        with self._cache_lock:
            self._speaker_cache = None
            print(f"🗑️ Speaker cache cleared")

    @property
    def diarization_pipeline(self):
        """Lazy load diarization pipeline"""
        if self._diarization_pipeline is None:
            print("Loading pyannote diarization pipeline...")
            # Set HF_TOKEN environment variable for authentication
            if self.hf_token:
                os.environ["HF_TOKEN"] = self.hf_token
            self._diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-community-1"
            )
            self._diarization_pipeline.to(self.device)
        return self._diarization_pipeline

    @property
    def embedding_model(self):
        """Lazy load embedding model"""
        if self._embedding_model is None:
            print("Loading pyannote embedding model...")
            # Set HF_TOKEN environment variable for authentication
            if self.hf_token:
                os.environ["HF_TOKEN"] = self.hf_token
            # In pyannote.audio 4.0, load Model first then create Inference
            model = Model.from_pretrained("pyannote/embedding")
            self._embedding_model = Inference(model, window="whole")
            self._embedding_model.to(self.device)
        return self._embedding_model

    @property
    def whisper_model(self):
        """Lazy load Whisper model"""
        if self._whisper_model is None:
            # Get model from environment variable (default: large-v3)
            model_name = os.getenv("WHISPER_MODEL", "large-v3")
            print(f"Loading faster-whisper model ({model_name})...")
            # Load faster-whisper model with FP16 for GPU acceleration
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if torch.cuda.is_available() else "int8"
            self._whisper_model = WhisperModel(model_name, device=device_name, compute_type=compute_type)
            print(f"faster-whisper model '{model_name}' loaded on {device_name} with {compute_type}")
        return self._whisper_model

    @property
    def emotion_model(self):
        """Lazy load emotion2vec model via FunASR"""
        if self._emotion_model is None:
            print("Loading emotion2vec model via FunASR...")
            try:
                from funasr import AutoModel
                # Load emotion2vec+ model from ModelScope/HuggingFace
                # Using 'plus_large' for better accuracy (can switch to 'plus_base' for speed)
                model_name = os.getenv("EMOTION_MODEL", "iic/emotion2vec_plus_large")

                # Try loading from local cache only (offline mode)
                # Falls back to downloading if not cached
                use_offline = os.getenv("OFFLINE_MODE", "false").lower() == "true"

                self._emotion_model = AutoModel(
                    model=model_name,
                    hub="hf",  # Use HuggingFace hub (fix for overseas users)
                    disable_update=True,  # Don't auto-update models
                    local_files_only=use_offline  # True = never download, use cache only
                )
                print(f"emotion2vec model loaded successfully ({model_name})")
            except ImportError:
                print("Warning: FunASR not installed. Emotion detection will be disabled.")
                print("Install with: pip install funasr")
                return None
            except Exception as e:
                print(f"Warning: Failed to load emotion2vec model: {e}")
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
        print(f"Transcribing {audio_file}...")
        # Get language from environment variable (default: "en")
        # Use "auto" for auto-detection, or specify language code (e.g., "es", "fr", "de")
        language = os.getenv("WHISPER_LANGUAGE", "en")
        # faster-whisper transcription with word-level timestamps and confidence
        segments_generator, info = self.whisper_model.transcribe(
            audio_file,
            language=None if language == "auto" else language,  # None = auto-detect
            task="transcribe",
            beam_size=5,
            vad_filter=True,  # Use VAD to filter out non-speech
            word_timestamps=True  # Enable word-level timestamps and probabilities
        )

        print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")

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
        print(f"Running diarization on {audio_file}...")
        with torch.no_grad():
            output = self.diarization_pipeline(audio_file)

        # Convert to dictionary format
        segments = []
        # In pyannote.audio 4.0+, use the speaker_diarization attribute
        for turn, speaker in output.speaker_diarization:
            segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker,
                "duration": turn.end - turn.start
            })

        result = {
            "segments": segments,
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
            print(f"  ⚠️ Segment embedding contains NaN values - skipping matching")
            return None

        # Calculate similarities
        best_match = None
        best_similarity = threshold

        for speaker_id, speaker_name, speaker_embedding in known_speakers:
            # Validate known speaker embedding
            if np.isnan(speaker_embedding).any():
                print(f"  ⚠️ Speaker '{speaker_name}' embedding contains NaN - skipping")
                continue

            similarity = cosine_similarity(
                segment_embedding.reshape(1, -1),
                speaker_embedding.reshape(1, -1)
            )[0][0]

            # Debug: Always print similarity score
            print(f"  Similarity with {speaker_name}: {similarity:.4f} (threshold: {threshold})")

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
            print(f"  ⚠️  Emotion embedding contains NaN - skipping personalized matching")
            return None

        best_match = None
        best_similarity = 0.0

        # Threshold hierarchy: per-emotion > per-speaker > global
        default_threshold = speaker_threshold if speaker_threshold is not None else global_threshold

        for emotion_cat, profile_emb, custom_threshold in speaker_emotion_profiles:
            # Validate profile embedding
            if np.isnan(profile_emb).any():
                print(f"  ⚠️  Emotion profile '{emotion_cat}' contains NaN - skipping")
                continue

            # Use custom threshold if set, otherwise use speaker/global threshold
            threshold = custom_threshold if custom_threshold is not None else default_threshold

            similarity = cosine_similarity(
                emotion_embedding.reshape(1, -1),
                profile_emb.reshape(1, -1)
            )[0][0]

            print(f"  Emotion similarity with '{emotion_cat}' profile: {similarity:.4f} (threshold: {threshold:.4f})")

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
        from pyannote.core import Segment
        import soundfile as sf

        # Use instance padding if not specified
        if context_padding is None:
            context_padding = self.context_padding

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
            print(f"Warning: Could not get audio duration, using original times: {e}")

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
                print(f"Could not extract embedding from {os.path.basename(seg['audio_file'])}: {e}")
                embeddings.append(None)
        return embeddings

    def extract_emotion(
        self,
        audio_file: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        extract_embedding: bool = False
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
                # For segments, we need to extract the audio first
                import torchaudio
                from pydub import AudioSegment
                import tempfile

                # Cap segment duration to avoid OOM in emotion2vec attention (O(n^2) memory)
                if (end_time - start_time) > MAX_EMOTION_DURATION_SEC:
                    end_time = start_time + MAX_EMOTION_DURATION_SEC

                # Extract segment to temporary file
                audio = AudioSegment.from_file(audio_file)
                start_ms = int(start_time * 1000)
                end_ms = int(end_time * 1000)
                segment = audio[start_ms:end_ms]

                # Resample to 16kHz if needed (emotion2vec requirement)
                # This won't affect faster-whisper or pyannote which handle their own resampling
                if segment.frame_rate != 16000:
                    segment = segment.set_frame_rate(16000)

                # Save to temp file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    segment.export(temp_file.name, format='wav')
                    temp_path = temp_file.name

                # Process with emotion2vec
                result = self.emotion_model.generate(
                    temp_path,
                    granularity="utterance",
                    extract_embedding=extract_embedding
                )

                # Clean up temp file
                import os
                os.unlink(temp_path)
            else:
                # Process entire file - need to check/resample first
                from pydub import AudioSegment
                import tempfile

                audio = AudioSegment.from_file(audio_file)

                # Cap duration to avoid OOM in emotion2vec attention (O(n^2) memory)
                max_emotion_duration_ms = int(MAX_EMOTION_DURATION_SEC * 1000)
                if len(audio) > max_emotion_duration_ms:
                    audio = audio[:max_emotion_duration_ms]

                # Resample to 16kHz if needed
                if audio.frame_rate != 16000:
                    audio = audio.set_frame_rate(16000)
                    # Create temp file with resampled audio
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                        audio.export(temp_file.name, format='wav')
                        temp_path = temp_file.name

                    result = self.emotion_model.generate(
                        temp_path,
                        granularity="utterance",
                        extract_embedding=extract_embedding
                    )

                    # Clean up
                    import os
                    os.unlink(temp_path)
                else:
                    # Already 16kHz, use directly
                    result = self.emotion_model.generate(
                        audio_file,
                        granularity="utterance",
                        extract_embedding=extract_embedding
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
            print(f"Warning: Failed to extract emotion: {e}")
            import traceback
            traceback.print_exc()
            # Cleanup on error
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
        print(f"Known speakers: {len(known_speakers)}")
        for speaker_id, speaker_name, _ in known_speakers:
            print(f"  - ID: {speaker_id}, Name: {speaker_name}")

        # Run transcription and diarization IN PARALLEL
        print("Running transcription and diarization in parallel...")
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both tasks to run concurrently
            transcription_future = executor.submit(self.transcribe, audio_file)
            diarization_future = executor.submit(self.diarize, audio_file)

            # Wait for both to complete
            transcription_segments = transcription_future.result()
            diarization_result = diarization_future.result()

        elapsed = time.time() - start_time
        print(f"Parallel processing completed in {elapsed:.2f}s")

        # Explicit cleanup after parallel processing to free VRAM before emotion extraction
        del transcription_future, diarization_future
        gc.collect()
        # Queue async cleanup (non-blocking) - this was blocking processing before
        self.clear_gpu_cache_async("parallel")
        print(f"🧹 GPU cache cleanup queued (async)")

        # Step 3: Match transcription segments to speakers
        # Map diarization labels (SPEAKER_00, SPEAKER_01) to Unknown_XX
        unknown_speaker_map = {}  # Maps SPEAKER_XX -> Unknown_YY
        unknown_counter = 1

        transcribed_with_speakers = []

        for trans_seg in transcription_segments:
            # Find which speaker was talking during this transcription segment
            # Use the middle of the transcription segment for matching
            mid_time = (trans_seg["start"] + trans_seg["end"]) / 2

            # Find the diarization segment that contains this time
            speaker_label = None
            for diar_seg in diarization_result["segments"]:
                if diar_seg["start"] <= mid_time <= diar_seg["end"]:
                    speaker_label = diar_seg["speaker"]
                    break

            if speaker_label is None:
                # If no exact match, find the closest diarization segment
                min_distance = float('inf')
                for diar_seg in diarization_result["segments"]:
                    diar_mid = (diar_seg["start"] + diar_seg["end"]) / 2
                    distance = abs(diar_mid - mid_time)
                    if distance < min_distance:
                        min_distance = distance
                        speaker_label = diar_seg["speaker"]

            # Try to match to known speaker if provided
            speaker_name = speaker_label  # Default to diarization label (will be converted to Unknown_XX)
            is_known = False
            confidence = 0.0
            embedding = None
            matched_emotion_from_voice = None  # Will be set if matched via emotion voice profile

            # ALWAYS extract embeddings (needed for embedding verification and speaker matching)
            # Check if segment is long enough for embedding extraction
            segment_duration = trans_seg["end"] - trans_seg["start"]
            if segment_duration < MIN_SEGMENT_DURATION_SEC:
                print(f"⏭️ Skipping embedding extraction (segment too short: {segment_duration:.2f}s)")
            else:
                try:
                    # Extract embedding for this segment
                    segment_embedding = self.extract_segment_embedding(
                        audio_file,
                        trans_seg["start"],
                        trans_seg["end"]
                    )
                    # Validate embedding
                    if np.isnan(segment_embedding).any():
                        print(f"⏭️ Segment embedding has NaN - skipping")
                    else:
                        # Store valid embedding for later use (auto-enrollment or verification)
                        embedding = segment_embedding

                        # NEW: Enhanced matching against ALL voice profiles (general + emotion-specific)
                        if db_session:
                            enhanced_match = self.match_speaker_with_all_profiles(
                                segment_embedding,
                                db_session,
                                threshold=threshold
                            )

                            if enhanced_match:
                                speaker_name = enhanced_match['speaker_name']
                                confidence = enhanced_match['confidence']
                                is_known = True
                                matched_emotion_from_voice = enhanced_match.get('matched_emotion')  # May be None
                                # Keep embedding for caching
                        elif known_speakers:
                            # Fallback to old method if no db_session
                            match = self.match_speaker(segment_embedding, known_speakers, threshold)
                            if match:
                                _, speaker_name, confidence = match
                                is_known = True
                                print(f"Matched segment to {speaker_name} with confidence {confidence:.2%}")
                                # Keep embedding for caching
                except RuntimeError as e:
                    error_str = str(e)
                    if "Kernel size" in error_str or "input size" in error_str:
                        print(f"⏭️ Skipping embedding extraction (segment too short for embedding model)")
                    elif "beyond file duration" in error_str:
                        print(f"⏭️ Skipping embedding extraction (segment at file boundary)")
                    else:
                        raise
                except Exception as e:
                    # Catch pyannote errors like sample mismatches at file boundaries
                    if "samples instead of the expected" in str(e) or "requested chunk" in str(e):
                        print(f"⏭️ Skipping embedding extraction (segment beyond file duration: {trans_seg['start']:.2f}s-{trans_seg['end']:.2f}s)")
                    else:
                        raise

            # If still not matched to known speaker (or embedding failed), map to Unknown_XX
            if not is_known:
                if speaker_label:
                    # Create consistent mapping from diarization label to Unknown_XX
                    if speaker_label not in unknown_speaker_map:
                        unknown_speaker_map[speaker_label] = f"Unknown_{unknown_counter:02d}"
                        unknown_counter += 1
                    speaker_name = unknown_speaker_map[speaker_label]
                else:
                    # No speaker label at all - create generic unknown
                    speaker_name = f"Unknown_{unknown_counter:02d}"
                    unknown_counter += 1

            # Extract emotion for this segment (with embeddings for personalized matching)
            # Feature flag for personalized emotions
            ENABLE_PERSONALIZED_EMOTIONS = os.getenv("ENABLE_PERSONALIZED_EMOTIONS", "true").lower() == "true"

            emotion_data = self.extract_emotion(
                audio_file,
                trans_seg["start"],
                trans_seg["end"],
                extract_embedding=ENABLE_PERSONALIZED_EMOTIONS  # Extract embeddings if personalization enabled
            )

            # NEW: If we matched via emotion voice profile, use that as primary signal
            if matched_emotion_from_voice is not None and emotion_data:
                print(f"  🎯 Voice profile match indicates emotion: {matched_emotion_from_voice}")

                # Create detector breakdown showing voice match as primary
                emotion_data['detector_breakdown'] = {
                    'emotion2vec_detector': {
                        'emotion': emotion_data['emotion_category'],
                        'confidence': float(emotion_data.get('emotion_confidence', 0.0))
                    },
                    'voice_profile_detector': {
                        'emotion': matched_emotion_from_voice,
                        'confidence': float(confidence)  # Use speaker match confidence
                    },
                    'final_decision': {
                        'emotion': matched_emotion_from_voice,
                        'confidence': float(confidence),
                        'reason': f'Matched {speaker_name}_{matched_emotion_from_voice} voice profile',
                        'voice_profile_available': True
                    }
                }

                # Override emotion with voice profile match
                emotion_data['emotion_category'] = matched_emotion_from_voice
                emotion_data['emotion_confidence'] = float(confidence)  # Convert to Python float

                print(f"  🔬 Enhanced Match Results:")
                print(f"     emotion2vec: {emotion_data['detector_breakdown']['emotion2vec_detector']['emotion']} ({emotion_data['detector_breakdown']['emotion2vec_detector']['confidence']:.2%})")
                print(f"     Voice profile: {matched_emotion_from_voice} ({confidence:.2%})")
                print(f"     Final: {matched_emotion_from_voice} - Matched emotion voice profile")

            # Personalized emotion matching (if enabled and speaker has profiles and NOT already matched via voice)
            elif emotion_data and ENABLE_PERSONALIZED_EMOTIONS and is_known and db_session:
                emotion_embedding = emotion_data.get('embedding')

                if emotion_embedding is not None:
                    try:
                        # Import models here to avoid circular dependency
                        from .models import Speaker, SpeakerEmotionProfile

                        # Get speaker's emotion profiles
                        speaker = db_session.query(Speaker).filter(Speaker.name == speaker_name).first()

                        if speaker and speaker.emotion_profiles:
                            # Get global emotion threshold
                            from .config import get_config
                            global_threshold = get_config().get_settings().emotion_threshold

                            # NEW: DUAL-DETECTOR MATCHING (emotion2vec + voice profile)
                            # Use both embeddings for better accuracy
                            voice_emb = embedding  # Voice embedding from pyannote (already extracted)

                            dual_result = self.match_emotion_dual_detector(
                                emotion_embedding=emotion_embedding,
                                voice_embedding=voice_emb,
                                speaker_emotion_profiles=speaker.emotion_profiles,  # Pass full objects
                                global_threshold=global_threshold,
                                speaker_threshold=speaker.emotion_threshold,
                                generic_emotion=emotion_data['emotion_category'],
                                generic_confidence=emotion_data['emotion_confidence']
                            )

                            # Use final decision from dual-detector
                            final = dual_result['final_decision']
                            emotion_data['emotion_category'] = final['emotion']
                            emotion_data['emotion_confidence'] = final['confidence']

                            # Store detector breakdown for frontend display
                            emotion_data['detector_breakdown'] = dual_result

                            print(f"  🔬 Dual-Detector Results:")
                            print(f"     Detector 1 (emotion2vec): {dual_result['emotion2vec_detector']['emotion']} ({dual_result['emotion2vec_detector']['confidence']:.2%})")
                            if dual_result['voice_profile_detector']:
                                print(f"     Detector 2 (voice profile): {dual_result['voice_profile_detector']['emotion']} ({dual_result['voice_profile_detector']['confidence']:.2%})")
                            else:
                                print(f"     Detector 2 (voice profile): Not available")
                            print(f"     Final: {final['emotion']} ({final['confidence']:.2%}) - {final['reason']}")

                    except Exception as e:
                        print(f"  Warning: Personalized emotion matching failed: {e}")
                        # Fall back to generic emotion detection

            segment_data = {
                "start": trans_seg["start"],
                "end": trans_seg["end"],
                "text": trans_seg["text"],
                "speaker": speaker_name,
                "speaker_label": speaker_label,
                "is_known": is_known,
                "confidence": float(confidence) if confidence is not None else 0.0,  # Convert to Python float
                "embedding": embedding,  # Include embedding for unknown speakers
                "words": trans_seg.get("words", []),  # Include word-level data
                "avg_logprob": trans_seg.get("avg_logprob")  # Include segment confidence
            }

            # Add emotion data if available (include embedding for caching)
            if emotion_data:
                # Ensure all numeric values are Python native types for JSON serialization
                emotion_conf = emotion_data.get("emotion_confidence")
                segment_data.update({
                    "emotion_category": emotion_data.get("emotion_category"),
                    "emotion_confidence": float(emotion_conf) if emotion_conf is not None else None,
                    "emotion_embedding": emotion_data.get("embedding"),  # Store for fast recalculation
                    "detector_breakdown": emotion_data.get("detector_breakdown")  # NEW: Dual-detector results
                })

            # Hallucination filtering (if enabled)
            from .config import get_config
            config = get_config()
            settings = config.get_settings()

            if settings.filter_hallucinations:
                text = trans_seg["text"].strip().lower()
                duration = trans_seg["end"] - trans_seg["start"]
                avg_logprob = trans_seg.get("avg_logprob", 0)

                # Filter 1: Minimum text length (3 characters)
                if len(text) < 3:
                    print(f"⏭️ Skipping hallucination (text too short): '{trans_seg['text']}'")
                    continue

                # Filter 2: Single character or filler sounds
                if len(text) == 1 or text in ["...", "ah", "um", "uh", "oh", "eh", "hmm", "mm"]:
                    print(f"⏭️ Skipping hallucination (filler): '{trans_seg['text']}'")
                    continue

                # Filter 3: Low confidence hallucinations (Whisper hallucinating on silence/music)
                # avg_logprob < -1.0 = very low confidence, likely hallucination
                hallucination_patterns = [
                    "thank you",
                    "thanks for watching",
                    "please subscribe",
                    "like and subscribe",
                    "don't forget to subscribe",
                    "see you next time",
                    "thanks for listening",
                ]

                # Check if text matches hallucination patterns
                if any(pattern in text for pattern in hallucination_patterns):
                    # Filter if either: very short (<0.4s) OR low confidence (<-0.6)
                    if duration < 0.4 or (avg_logprob is not None and avg_logprob < -0.6):
                        print(f"⏭️ Skipping hallucination (dur={duration:.2f}s, conf={avg_logprob:.2f}): '{trans_seg['text']}'")
                        continue

            transcribed_with_speakers.append(segment_data)

        return {
            "segments": transcribed_with_speakers,
            "num_speakers": diarization_result["num_speakers"]
        }

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


        from sklearn.metrics.pairwise import cosine_similarity
        from .models import Speaker, SpeakerEmotionProfile

        best_match = None
        best_similarity = threshold

        # Use cached profiles if available (fast), otherwise load from DB (slow)
        with self._cache_lock:
            if self._speaker_cache is not None:
                all_profiles = self._speaker_cache
                print(f"  ⚡ Using cached profiles: {len(all_profiles)} profiles")
            else:
                # Cache not loaded - fall back to DB query (will be slow)
                print(f"  ⚠️ Cache not loaded, querying DB...")
                speakers = db_session.query(Speaker).all()

                all_profiles = []

                for speaker in speakers:
                    # 1. Add general speaker profile
                    general_emb = speaker.get_embedding()
                    if general_emb is not None and not np.isnan(general_emb).any():
                        all_profiles.append({
                            'speaker_id': speaker.id,
                            'speaker_name': speaker.name,
                            'embedding': general_emb,
                            'emotion': None,
                            'profile_type': 'general'
                        })

                    # 2. Add ALL emotion voice profiles for this speaker
                    for emotion_profile in speaker.emotion_profiles:
                        voice_emb = emotion_profile.get_voice_embedding()
                        if (voice_emb is not None and
                            not np.isnan(voice_emb).any() and
                            emotion_profile.voice_sample_count >= 3):
                            all_profiles.append({
                                'speaker_id': speaker.id,
                                'speaker_name': speaker.name,
                                'embedding': voice_emb,
                                'emotion': emotion_profile.emotion_category,
                                'profile_type': 'emotion_voice',
                                'sample_count': emotion_profile.voice_sample_count
                            })

                print(f"  🔍 Checking against {len(all_profiles)} voice profiles (general + emotion-specific)")

        # Match against ALL profiles (silent, only print best match)
        for profile in all_profiles:
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
                print(f"  ✅ BEST MATCH: {best_match['speaker_name']}_{best_match['matched_emotion']} ({best_match['confidence']:.2%}) - Emotion voice profile")
            else:
                print(f"  ✅ BEST MATCH: {best_match['speaker_name']} ({best_match['confidence']:.2%}) - General profile")
        else:
            print(f"  ❌ No match found above threshold {threshold:.2%}")

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

        from sklearn.metrics.pairwise import cosine_similarity

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
        from .config import get_config
        config = get_config()
        settings = config.get_settings()
        default_speaker_threshold = settings.speaker_threshold

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
            "emotion": voice_best or "neutral",
            "confidence": float(voice_best_confidence),
            "matches": voice_matches
        }

        # COMBINE
        d1_emotion = detector1_result["emotion"]
        d1_conf = detector1_result["confidence"]
        d2_emotion = detector2_result["emotion"]
        d2_conf = detector2_result["confidence"]

        if d1_emotion == d2_emotion and d1_conf > DUAL_DETECTOR_AGREE_D1 and d2_conf > DUAL_DETECTOR_AGREE_D2:
            final = {"emotion": d1_emotion, "confidence": float((d1_conf + d2_conf) / 2), "reason": "Both agree", "voice_profile_available": True}
        elif d1_emotion == "neutral" or d1_emotion == "<unk>":
            final = {"emotion": "neutral", "confidence": float(d1_conf), "reason": "emotion2vec neutral", "voice_profile_available": len(voice_matches) > 0}
        elif d2_conf > VOICE_STRONG_THRESHOLD:
            final = {"emotion": d2_emotion, "confidence": float(d2_conf), "reason": f"Voice strong: {d2_emotion}", "voice_profile_available": True}
        elif d1_emotion != d2_emotion:
            final = {"emotion": "neutral", "confidence": float(max(d1_conf, d2_conf)), "reason": f"Disagree: {d1_emotion} vs {d2_emotion}", "voice_profile_available": len(voice_matches) > 0}
        else:
            final = {"emotion": d1_emotion, "confidence": float(d1_conf), "reason": "Agree low conf", "voice_profile_available": len(voice_matches) > 0}

        return {
            "emotion2vec_detector": detector1_result,
            "voice_profile_detector": detector2_result if len(voice_matches) > 0 else None,
            "final_decision": final
        }

