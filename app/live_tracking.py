"""Live speaker tracking: who is speaking across the chunks of one live session.

A live chunk holds a few seconds of speech, and one 2-4 s segment's voice embedding scores about as high against the
wrong person as against the right one, so naming every segment on its own gives wrong names and a new Unknown for
every few seconds of a stranger. Instead, as SPEAKER_ASSIGNMENT=cluster does for a whole file, a session keeps one
track per voice:
- each chunk's diarization speakers are matched one-to-one to the session's tracks by voice, or start new tracks;
- a track's voice is the duration-weighted mean of everything it has said so far;
- a track is named from that voice: the stored speaker (a person, or an Unknown from an earlier session) it matches
  with at least NAME_SIMILARITY and NAME_MARGIN above the next one, once it has MIN_NAME_SECONDS of speech; failing
  that, another track of this session it matches the same way (two tracks of one voice share one Unknown); failing
  that, once it has PROVISIONAL_SECONDS of speech, an Unknown of its own (before that its segments are a plain
  "Unknown" with no stored speaker, so a few words nobody can place do not each become a stranger to name). When a
  track gets a speaker, its earlier segments follow; when an Unknown of this session gets a name, every segment that
  carries it follows (SessionTracks.renames, applied by the caller).
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# Tuned on live replays of the AMI test meetings with the default voice model (WeSpeaker ResNet34): people named in a
# first meeting are named in the next with 93.8% of their speech right and 2.9% wrong in its first 8 minutes (most of
# the wrong share is diarization mixing two voices in a chunk); speakers nobody named are never named, one Unknown each.
JOIN_SIMILARITY = 0.50   # a chunk's speaker joins a track whose voice is at least this similar (else a new track)
NAME_SIMILARITY = 0.70   # a track takes a stored speaker's name at this similarity
NAME_MARGIN = 0.05       # ... and only this far above the next stored speaker
MIN_NAME_SECONDS = 2.0   # ... and only once it has this much speech
PROVISIONAL_SECONDS = 3.0  # a voice nobody matches gets its own Unknown only after this much speech
WEIGHT_CAP_SEC = 10.0    # one chunk counts for at most this long in a track's voice (as cluster mode's centroids)


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).ravel()
    return v / (np.linalg.norm(v) or 1.0)


@dataclass
class Track:
    id: int
    voice_sum: np.ndarray            # sum of unit voice vectors weighted by seconds
    seconds: float
    speaker_id: Optional[int] = None  # the stored speaker its segments carry
    own_unknown: bool = False         # speaker_id is the Unknown created for this track (else a matched speaker)
    segment_ids: List[int] = field(default_factory=list)   # its saved segments, in order

    @property
    def voice(self) -> np.ndarray:
        return _unit(self.voice_sum)

    def add(self, voice: np.ndarray, seconds: float):
        w = min(seconds, WEIGHT_CAP_SEC)
        self.voice_sum = self.voice_sum + w * _unit(voice)
        self.seconds += seconds


def best_speaker(voice: np.ndarray, seconds: float, people: Dict[int, List[np.ndarray]],
                 taken: Tuple[int, ...] = ()) -> Optional[Tuple[int, float]]:
    """The speaker a voice is: (speaker id, similarity) if it clears NAME_SIMILARITY, NAME_MARGIN over the next one and
    MIN_NAME_SECONDS, else None. people: {speaker id: their voice profiles}; taken: speakers already given to another
    speaker of the same chunk (two voices in one chunk are two people)."""
    if seconds < MIN_NAME_SECONDS or not people:
        return None
    v = _unit(voice)
    sims = sorted(((max(float(v @ _unit(e)) for e in embs), sid) for sid, embs in people.items() if embs),
                  reverse=True)
    if not sims:
        return None
    best, sid = sims[0]
    runner_up = sims[1][0] if len(sims) > 1 else -1.0
    if sid in taken or best < NAME_SIMILARITY or best - runner_up < NAME_MARGIN:
        return None
    return sid, best


class SessionTracks:
    """The tracks of one live session (chunks are processed one at a time, in order)."""

    def __init__(self):
        self.tracks: List[Track] = []
        self.renames: List[Tuple[int, int]] = []   # (Unknown of this session, speaker it became), for the caller

    def assign(self, voices: Dict[str, Tuple[np.ndarray, float]]) -> Dict[str, Track]:
        """{chunk speaker label: (voice, seconds of speech)} -> {label: its track}, one-to-one: the most similar
        pairs first, each above JOIN_SIMILARITY; a label left over starts a new track. Updates the tracks' voices."""
        pairs = sorted(((float(_unit(v) @ t.voice), label, t) for label, (v, _) in voices.items() for t in self.tracks),
                       key=lambda x: -x[0])
        out: Dict[str, Track] = {}
        for sim, label, track in pairs:
            if sim < JOIN_SIMILARITY:
                break
            if label not in out and all(t is not track for t in out.values()):
                out[label] = track
        for label, (voice, seconds) in voices.items():
            if label not in out:
                out[label] = Track(id=len(self.tracks) + 1, voice_sum=np.zeros(len(_unit(voice))), seconds=0.0)
                self.tracks.append(out[label])
            out[label].add(voice, seconds)
        return out

    def session_unknowns(self) -> set:
        return {t.speaker_id for t in self.tracks if t.own_unknown}

    def identify(self, track: Track, stored: Dict[int, List[np.ndarray]],
                 taken: Tuple[int, ...] = ()) -> Optional[Tuple[int, float]]:
        """Who an unnamed track (no speaker yet, or its own Unknown) is: a stored speaker from before this session,
        else another track's Unknown of this session (the same voice tracked twice), else None."""
        mine = self.session_unknowns()
        before = {sid: embs for sid, embs in stored.items() if sid not in mine}
        match = best_speaker(track.voice, track.seconds, before, taken)
        if match is None:
            others = {t.speaker_id: [t.voice] for t in self.tracks
                      if t.own_unknown and t is not track and t.speaker_id != track.speaker_id}
            match = best_speaker(track.voice, track.seconds, others, taken)
        return match

    def adopt(self, track: Track, speaker_id: int):
        """The track (and, if it had its own Unknown, every track sharing that Unknown) becomes speaker_id."""
        old = track.speaker_id if track.own_unknown else None
        if old is None:
            track.speaker_id, track.own_unknown = speaker_id, False
            return
        for t in self.tracks:
            if t.speaker_id == old:
                t.speaker_id, t.own_unknown = speaker_id, False
        self.renames.append((old, speaker_id))
