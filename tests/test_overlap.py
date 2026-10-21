"""OVERLAP_SEPARATION: where two people talk at once, the second voice's words are added for the speaker it sounds
like, and the main speaker's missed words join their segment (app/overlap.py)."""
import numpy as np

from app import overlap

A, B = np.array([1.0, 0.0]), np.array([0.0, 1.0])   # the two speakers' voices


def turn(s, e, spk):
    return {"start": s, "end": e, "speaker": spk}


def word(text, s, e, p=0.99):
    return {"word": " " + text, "start": s, "end": e, "probability": p}


def seg(words, label):
    return {"start": words[0]["start"], "end": words[-1]["end"], "words": words,
            "text": "".join(w["word"] for w in words).strip(), "speaker_label": label}


def test_overlap_regions_join_close_overlaps_and_drop_short_ones():
    turns = [turn(0, 10, "A"), turn(2, 3, "B"), turn(3.3, 4, "C"), turn(8, 8.1, "B")]
    assert overlap.overlap_regions(turns) == [(2, 4)]   # 2-3 and 3.3-4 joined; 8-8.1 too short


def run(segments, turns, voices_out, words_out, voices=None):
    """overlap.recover with fake models: the separator gives two voices, transcribe gives each voice's words (times
    from the clip start, which is 1 s before the overlap), embed tells the voices apart by their first sample."""
    audio = np.zeros(16000 * 20, dtype=np.float32)
    separate = lambda wav: np.stack([np.full_like(wav, 1.0), np.full_like(wav, 2.0)])  # noqa: E731
    transcribe = lambda clips: [words_out[i % 2] for i in range(len(clips))]           # noqa: E731
    embed = lambda wav: voices_out[int(wav[0]) - 1]                                     # noqa: E731
    return overlap.recover(segments, turns, audio, voices or {"A": A, "B": B}, separate, transcribe, embed, 1.0)


def test_second_voice_becomes_the_other_speakers_segment_and_main_words_join_theirs():
    main = seg([word("so", 4.0, 4.3), word("we", 4.4, 4.6), word("go", 6.0, 6.3)], "A")
    segments = [(main, "A", A)]
    turns = [turn(4.0, 6.3, "A"), turn(5.0, 5.6, "B")]
    # clip starts at 4.0 (1 s before the 5.0 overlap): voice 1 = A's words, voice 2 = B's "yeah right"
    voice_a = [word("we", 0.4, 0.6), word("then", 1.1, 1.3)]           # "we" is already there, "then" was missed
    voice_b = [word("yeah", 1.0, 1.2), word("right", 1.3, 1.5)]
    out = run(segments, turns, [A, B], [voice_a, voice_b])
    assert [(s["text"], label) for s, label, _ in out] == [("so we then go", "A"), ("yeah right", "B")]
    added = out[1][0]
    assert (added["start"], added["end"], added["speaker_label"], added["overlap"]) == (5.0, 5.5, "B", True)
    assert out[1][2] is None   # two voices in its span: no embedding for anyone's profile


def test_unsure_words_and_a_voice_like_the_main_speaker_add_nothing():
    main = seg([word("so", 4.0, 4.3), word("go", 6.0, 6.3)], "A")
    turns = [turn(4.0, 6.3, "A"), turn(5.0, 5.6, "B")]
    # both voices sound like A (the separation failed): nothing goes to B; A's unsure word stays out
    out = run([(main, "A", A)], turns, [A, A * 0.9 + B * 0.1], [[word("um", 1.1, 1.3, p=0.5)], [word("x", 1.2, 1.3)]])
    assert [(s["text"], label) for s, label, _ in out] == [("so go", "A")]


def test_no_overlap_leaves_the_segments_as_they_are():
    segments = [(seg([word("hi", 0.0, 0.5)], "A"), "A", A)]
    assert run(segments, [turn(0, 1, "A"), turn(2, 3, "B")], [A, B], [[], []]) is segments
