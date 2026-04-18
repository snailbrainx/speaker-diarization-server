from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, LargeBinary, Text, Boolean, UniqueConstraint
from sqlalchemy.orm import relationship
from .database import Base, utc_now
import json
import numpy as np

class Speaker(Base):
    __tablename__ = "speakers"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)
    embedding = Column(LargeBinary, nullable=False)  # Stored as numpy array bytes
    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)

    # Per-speaker emotion matching threshold (NULL = use global default)
    emotion_threshold = Column(Float, nullable=True)

    # Relationships
    emotion_profiles = relationship("SpeakerEmotionProfile", back_populates="speaker", cascade="all, delete-orphan")

    def get_embedding(self):
        """Convert binary embedding back to numpy array"""

        return np.frombuffer(self.embedding, dtype=np.float32)

    def set_embedding(self, embedding_array):
        """Convert numpy array to binary for storage"""

        self.embedding = embedding_array.astype(np.float32).tobytes()


class SpeakerEmotionProfile(Base):
    """
    Emotion embeddings for a specific speaker and emotion category.
    Stores learned emotional signatures that improve with corrections.
    """
    __tablename__ = "speaker_emotion_profiles"

    id = Column(Integer, primary_key=True, index=True)
    speaker_id = Column(Integer, ForeignKey("speakers.id", ondelete="CASCADE"), nullable=False, index=True)
    emotion_category = Column(String, nullable=False)  # 'angry', 'happy', 'sad', etc.

    # Emotion embedding (1024-D from emotion2vec)
    embedding = Column(LargeBinary, nullable=False)  # Numpy array -> bytes

    # Metadata
    sample_count = Column(Integer, default=1)  # How many corrections went into this
    confidence_threshold = Column(Float, nullable=True)  # Per-speaker-per-emotion threshold (NULL = use speaker/global)
    
    # Voice embedding for this specific emotion (512-D from pyannote) - NEW for dual-detector system
    voice_embedding = Column(LargeBinary, nullable=True)  # Voice signature when expressing THIS emotion
    voice_sample_count = Column(Integer, default=0)  # How many voice samples in this emotion profile
    voice_threshold = Column(Float, nullable=True)  # Custom threshold for voice matching (NULL = use speaker/global)

    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)

    # Unique constraint: one profile per speaker per emotion
    __table_args__ = (
        UniqueConstraint('speaker_id', 'emotion_category', name='_speaker_emotion_uc'),
    )

    # Relationships
    speaker = relationship("Speaker", back_populates="emotion_profiles")

    def get_embedding(self):
        """Convert binary emotion embedding to numpy array"""

        return np.frombuffer(self.embedding, dtype=np.float32)

    def set_embedding(self, embedding_array):
        """Convert numpy array to binary (emotion embedding)"""

        self.embedding = embedding_array.astype(np.float32).tobytes()
    
    def get_voice_embedding(self):
        """Convert binary voice embedding to numpy array"""
        if self.voice_embedding is None:
            return None

        return np.frombuffer(self.voice_embedding, dtype=np.float32)
    
    def set_voice_embedding(self, embedding_array):
        """Convert numpy array to binary (voice embedding)"""
        if embedding_array is None:
            self.voice_embedding = None
            return

        self.voice_embedding = embedding_array.astype(np.float32).tobytes()


class Conversation(Base):
    """
    Represents a continuous recording session (e.g., a meeting, interview, etc.)
    """
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=True)  # Auto-generated or user-set
    start_time = Column(DateTime, nullable=False, default=utc_now)
    end_time = Column(DateTime, nullable=True)  # Null while recording
    duration = Column(Float, nullable=True)  # Duration in seconds
    status = Column(String, default="recording")  # recording, processing, completed, failed
    audio_path = Column(String, nullable=True)  # Path to WAV or MP3 file
    audio_format = Column(String, default="wav")  # wav or mp3
    num_segments = Column(Integer, default=0)
    num_speakers = Column(Integer, default=0)
    created_at = Column(DateTime, default=utc_now)

    # Relationships
    transcript_segments = relationship("ConversationSegment", back_populates="conversation", cascade="all, delete-orphan")


class ConversationSegment(Base):
    """
    Individual speech segment within a conversation with transcription
    """
    __tablename__ = "conversation_segments"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False, index=True)
    speaker_id = Column(Integer, ForeignKey("speakers.id", ondelete="SET NULL"), nullable=True, index=True)  # Null for unknown - auto-set to NULL when speaker deleted
    speaker_name = Column(String, nullable=True)  # Denormalized for quick access
    text = Column(Text, nullable=True)  # Transcription text

    # Absolute timestamps (for AI context)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)

    # Relative timestamps (seconds from conversation start, for audio playback)
    start_offset = Column(Float, nullable=False)
    end_offset = Column(Float, nullable=False)

    # Individual segment audio file (for streaming playback before concatenation)
    segment_audio_path = Column(String, nullable=True)

    confidence = Column(Float, nullable=True)  # Speaker identification confidence

    # Emotion detection (from emotion2vec)
    emotion_category = Column(String, nullable=True)  # Primary emotion label (happy, sad, angry, etc.)
    emotion_confidence = Column(Float, nullable=True)  # Confidence score for emotion prediction

    # Emotion correction tracking (for personalized learning)
    emotion_corrected = Column(Boolean, default=False, nullable=False)  # True if user manually corrected
    emotion_corrected_at = Column(DateTime, nullable=True)  # When correction was made
    emotion_misidentified = Column(Boolean, default=False, nullable=False)  # True if emotion correction was wrong (exclude from profile)

    # Dual-detector breakdown (JSON string with both detector results)
    detector_breakdown = Column(Text, nullable=True)  # JSON string with detector breakdown for transparency

    # Word-level transcription data with confidence scores (JSON)
    words_data = Column(Text, nullable=True)  # Stores JSON array of {word, start, end, probability}
    avg_logprob = Column(Float, nullable=True)  # Segment-level average log probability

    processed_at = Column(DateTime, default=utc_now)

    # Misidentification tracking
    is_misidentified = Column(Boolean, default=False, nullable=False)  # True if this segment was wrongly assigned to current speaker

    # Cached embeddings (for fast recalculation without audio re-extraction)
    # Stored as binary (numpy array -> bytes) to avoid reprocessing audio files
    speaker_embedding = Column(LargeBinary, nullable=True)  # 512-D pyannote embedding (~2KB)
    emotion_embedding = Column(LargeBinary, nullable=True)  # 1024-D emotion2vec embedding (~4KB)

    # Relationships
    conversation = relationship("Conversation", back_populates="transcript_segments")
    speaker = relationship("Speaker")

    @property
    def words(self):
        """Parse words_data JSON and return as list."""
        if not self.words_data:
            return None
        try:
            return json.loads(self.words_data)
        except (json.JSONDecodeError, TypeError):
            return None

    def get_speaker_embedding(self):
        """Convert binary speaker embedding back to numpy array"""
        if self.speaker_embedding is None:
            return None

        return np.frombuffer(self.speaker_embedding, dtype=np.float32)

    def set_speaker_embedding(self, embedding_array):
        """Convert numpy array to binary for storage"""
        if embedding_array is None:
            self.speaker_embedding = None
            return

        self.speaker_embedding = embedding_array.astype(np.float32).tobytes()

    def get_emotion_embedding(self):
        """Convert binary emotion embedding back to numpy array"""
        if self.emotion_embedding is None:
            return None

        return np.frombuffer(self.emotion_embedding, dtype=np.float32)

    def set_emotion_embedding(self, embedding_array):
        """Convert numpy array to binary for storage"""
        if embedding_array is None:
            self.emotion_embedding = None
            return

        self.emotion_embedding = embedding_array.astype(np.float32).tobytes()


