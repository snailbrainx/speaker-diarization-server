"""Overlapping speech: the words of the second voice.

One speech-to-text pass over mixed audio mostly keeps the louder voice where two people talk at once. On AMI meetings
a quarter of the words are spoken over someone else and half of those were missing from the transcript. With
OVERLAP_SEPARATION on, every stretch where the diarization hears two people at once, plus CONTEXT_SEC either side,
is separated into two voices (MossFormer2 from ClearerVoice-Studio, Apache-2.0), both voices are transcribed, and each
voice's embedding is compared with every speaker's voice:
- a voice that sounds like another speaker than the stretch's main one (by VOICE_MARGIN or more) adds that speaker's
  words as new segments;
- the other voice is the main speaker's: its words the transcript lacks join the main speaker's segment there.
Words the transcript already has (the same word within SAME_WORD_SEC) or below MIN_WORD_PROB are not added.
"""
import logging
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

SEPARATION_MODEL = "alibabasglab/MossFormer2_SS_16K"
SR = 16000
CONTEXT_SEC = 1.0     # audio either side of an overlap that goes into the separation (0.5 and 2 s were no better)
MIN_SEC = 0.2         # shorter overlaps are skipped
MERGE_GAP_SEC = 0.5   # overlaps closer than this are separated together
EDGE_SEC = 0.2        # a word counts if its middle is this close to the overlap
MIN_WORD_PROB = 0.8   # below this, about half of the added words were never said
VOICE_MARGIN = 0.1    # a voice goes to another speaker only if it is this much closer to them than to the main one
SAME_WORD_SEC = 0.3   # a word the transcript has within this distance is not added again

Segment = Tuple[Dict, Optional[str], Optional[np.ndarray]]  # (segment, diarization label, voice embedding)


class Separator:
    """MossFormer2 two-voice separation at 16 kHz."""

    def __init__(self, device: torch.device):
        from types import SimpleNamespace
        from huggingface_hub import hf_hub_download
        net = _mossformer2()(SimpleNamespace(encoder_embedding_dim=512, mossformer_sequence_dim=512,
                                                 num_mossformer_layer=24, encoder_kernel_size=16, num_spks=2)).model
        checkpoint = torch.load(hf_hub_download(SEPARATION_MODEL, "last_best_checkpoint.pt"), map_location="cpu",
                                weights_only=True)
        net.load_state_dict(checkpoint["model"])
        self.model, self.device = net.to(device).eval(), device

    @torch.no_grad()
    def __call__(self, wav: np.ndarray) -> np.ndarray:
        """(samples,) -> (2, samples), each voice at the mixture's level."""
        voices = self.model(torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32))[None].to(self.device))
        out = torch.stack([v[0] for v in voices])[:, :len(wav)].float().cpu().numpy()
        level = float(np.sqrt((wav ** 2).mean())) + 1e-8
        return out / (np.sqrt((out ** 2).mean(axis=1, keepdims=True)) + 1e-8) * level


def _mossformer2():
    """The MossFormer2_SS_16K class from the clearvoice package, loaded on its own: importing clearvoice itself pulls
    in its other models' dependencies and switches Python warnings off for the whole process."""
    import importlib
    import importlib.util
    import os
    import sys
    root = os.path.join(os.path.dirname(importlib.util.find_spec("clearvoice").origin), "models", "mossformer2_ss")
    if "mossformer2_ss" not in sys.modules:
        spec = importlib.util.spec_from_file_location("mossformer2_ss", os.path.join(root, "__init__.py"),
                                                      submodule_search_locations=[root])
        sys.modules["mossformer2_ss"] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sys.modules["mossformer2_ss"])
    return importlib.import_module("mossformer2_ss.mossformer2").MossFormer2_SS_16K


def overlap_regions(turns: Sequence[Dict]) -> List[Tuple[float, float]]:
    """(start, end) of the stretches where two or more diarization turns overlap, joined across gaps shorter than
    MERGE_GAP_SEC, at least MIN_SEC long."""
    events = sorted([(t["start"], 1) for t in turns] + [(t["end"], -1) for t in turns])  # ends first at a tie
    spans: List[List[float]] = []
    active, begin = 0, 0.0
    for t, step in events:
        if active < 2 <= active + step:
            begin = t
        elif active >= 2 > active + step:
            if spans and begin - spans[-1][1] < MERGE_GAP_SEC:
                spans[-1][1] = t
            else:
                spans.append([begin, t])
        active += step
    return [(s, e) for s, e in spans if e - s >= MIN_SEC]


def _mid(w: Dict) -> float:
    return (w["start"] + w["end"]) / 2


def _norm(word: str) -> str:
    return "".join(c for c in word.lower() if c.isalnum() or c == "'")


def _missing(words: List[Dict], have: List[Dict], start: float, end: float) -> List[Dict]:
    """Words of a voice within the overlap that are sure enough and not in the transcript already."""
    def known(w):
        return any(_norm(h["word"]) == _norm(w["word"]) and abs(_mid(h) - _mid(w)) <= SAME_WORD_SEC for h in have)
    return [w for w in words if start - EDGE_SEC <= _mid(w) <= end + EDGE_SEC
            and (w.get("probability") or 0.0) >= MIN_WORD_PROB and not known(w)]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.ravel(a), np.ravel(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def recover(segments: List[Segment], turns: Sequence[Dict], audio: np.ndarray, voices: Dict[str, np.ndarray],
            separate: Callable[[np.ndarray], np.ndarray], transcribe: Callable[[List[np.ndarray]], List[List[Dict]]],
            embed: Callable[[np.ndarray], Optional[np.ndarray]], pause_sec: float) -> List[Segment]:
    """The segments (in time order) with the overlapped words added: new segments for a second speaker, missing
    words of the main speaker merged into their segment (or new segments where they have none there). New segments
    carry no voice embedding: their audio span holds two voices, so it must not shape anyone's profile.

    turns: the diarization turns (overlapping); voices: each diarization label's voice embedding; separate: one
    stretch -> two voices; transcribe: clips -> their words (times from the clip's start); embed: a voice -> its
    embedding; pause_sec: a pause this long starts a new segment."""
    duration = len(audio) / SR
    windows = [(s, e, max(0.0, s - CONTEXT_SEC), min(duration, e + CONTEXT_SEC)) for s, e in overlap_regions(turns)]
    windows = [w for w in windows if w[3] - w[2] >= 0.5]
    if not windows or not voices:
        return segments
    separated = [separate(audio[int(lo * SR):int(hi * SR)]) for _, _, lo, hi in windows]
    words = transcribe([v for pair in separated for v in pair])
    all_words = [w for seg, _, _ in segments for w in seg.get("words") or []]
    extra: Dict[int, List[Dict]] = {}   # index of a segment -> the main speaker's words to merge into it
    new: Dict[str, List[List[Dict]]] = {}   # speaker -> word groups for new segments
    for k, (start, end, lo, hi) in enumerate(windows):
        cover: Dict[Optional[str], float] = {}
        for seg, label, _ in segments:
            ov = min(end, seg["end"]) - max(start, seg["start"])
            if ov > 0:
                cover[label] = cover.get(label, 0.0) + ov
        main = max(cover, key=cover.get) if cover else None
        if main is not None and main not in voices:  # nothing to tell the main voice from the other by
            continue
        have = [w for w in all_words if lo - 1 <= w["start"] <= hi + 1]
        voice_words = [[dict(w, start=w["start"] + lo, end=w["end"] + lo) for w in words[2 * k + j]] for j in (0, 1)]
        embs = [embed(v) for v in separated[k]]
        main_sims = [_cosine(e, voices[main]) if e is not None and main in voices else -1.0 for e in embs]
        other, other_label, gain = None, None, VOICE_MARGIN
        for j, e in enumerate(embs):
            if e is None:
                continue
            label, sim = max(((l, _cosine(e, v)) for l, v in voices.items()), key=lambda x: x[1])
            if label != main and sim - main_sims[j] >= gain:
                other, other_label, gain = j, label, sim - main_sims[j]
        if other is not None:
            new.setdefault(other_label, []).append(_missing(voice_words[other], have, start, end))
        if main is None:
            continue
        mine = 1 - other if other is not None else int(main_sims[1] > main_sims[0])
        rest = []
        for w in _missing(voice_words[mine], have, start, end):
            host = next((i for i, (seg, label, _) in enumerate(segments)
                         if label == main and seg["start"] <= _mid(w) <= seg["end"]), None)
            (extra.setdefault(host, []) if host is not None else rest).append(w)
        new.setdefault(main, []).append(rest)
    out = []
    for i, (seg, label, emb) in enumerate(segments):
        if i in extra:
            merged = sorted((seg.get("words") or []) + extra[i], key=lambda w: w["start"])
            seg = dict(seg, words=merged, text="".join(w["word"] for w in merged).strip())
        out.append((seg, label, emb))
    added = 0
    for label, groups in new.items():
        for ws in groups:
            for piece in _pieces(ws, pause_sec):
                out.append(({"start": piece[0]["start"], "end": piece[-1]["end"], "words": piece,
                             "text": "".join(w["word"] for w in piece).strip(), "avg_logprob": _avg_logprob(piece),
                             "speaker_label": label, "overlap": True}, label, None))
                added += len(piece)
    logger.info(f"Overlapping speech: {len(windows)} stretches, {added} words in new segments, "
                f"{sum(map(len, extra.values()))} merged into existing ones")
    return sorted(out, key=lambda x: x[0]["start"])


def _pieces(words: List[Dict], pause_sec: float) -> List[List[Dict]]:
    groups: List[List[Dict]] = []
    for w in words:
        if groups and w["start"] - groups[-1][-1]["end"] < pause_sec:
            groups[-1].append(w)
        else:
            groups.append([w])
    return groups


def _avg_logprob(words: List[Dict]) -> Optional[float]:
    probs = [w["probability"] for w in words if w.get("probability") is not None]
    return float(np.mean(np.log(np.clip(probs, 1e-6, 1.0)))) if probs else None
