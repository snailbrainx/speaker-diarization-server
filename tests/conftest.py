"""Test bootstrap: make the application importable without the GPU/ML stack.

The app imports torch/pyannote/faster-whisper at module level, but the defects
under regression test (restore atomicity, streaming finalisation, config races,
the dual-detector combiner, SQLite pragmas) live in plain Python + SQLAlchemy.
When the heavy stack is absent we install minimal import-time stubs; when it is
present the real modules are used unchanged.

DATABASE_URL/DATA_PATH are redirected to per-session temp locations BEFORE any
app import so no test can touch a real database or data directory.
"""
import os
import sys
import tempfile
import types
from pathlib import Path

# Isolate storage before any app module import.
_TMP = tempfile.mkdtemp(prefix="sds-tests-")
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP}/test.db"
os.environ["DATA_PATH"] = os.path.join(_TMP, "data")
os.environ["BACKUPS_DIR_OVERRIDE"] = ""  # tests monkeypatch backup_api._BACKUPS_DIR
os.environ.setdefault("ENABLE_PERSONALIZED_EMOTIONS", "true")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _try_import(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:  # noqa: BLE001 — stub fallback must survive any import failure
        return False


class _NoGrad:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


if not _try_import("torch"):
    torch_stub = types.ModuleType("torch")
    torch_stub.no_grad = lambda: _NoGrad()
    cuda_stub = types.ModuleType("torch.cuda")
    cuda_stub.is_available = lambda: False
    torch_stub.cuda = cuda_stub
    sys.modules["torch"] = torch_stub
    sys.modules["torch.cuda"] = cuda_stub

if not _try_import("pyannote.audio"):
    pa = types.ModuleType("pyannote.audio")
    pa.Pipeline = object
    pa.Model = object
    pa.Inference = object
    sys.modules["pyannote.audio"] = pa

if not _try_import("pyannote.core"):
    pc = types.ModuleType("pyannote.core")

    class _Segment(tuple):
        def __new__(cls, start, end):
            return super().__new__(cls, (start, end))

    pc.Segment = _Segment
    sys.modules["pyannote.core"] = pc

if not _try_import("pyannote"):
    pn = types.ModuleType("pyannote")
    sys.modules["pyannote"] = pn

if not _try_import("sklearn"):
    import numpy as _np

    sk = types.ModuleType("sklearn")
    skm = types.ModuleType("sklearn.metrics")
    skp = types.ModuleType("sklearn.metrics.pairwise")

    def _cosine_similarity(a, b):
        a = _np.asarray(a, dtype=float)
        b = _np.asarray(b, dtype=float)
        denom = _np.linalg.norm(a) * _np.linalg.norm(b)
        if denom == 0:
            return _np.zeros((a.shape[0], b.shape[0]))
        return (a @ b.T) / denom

    skp.cosine_similarity = _cosine_similarity
    sk.metrics = skm
    skm.pairwise = skp
    sys.modules["sklearn"] = sk
    sys.modules["sklearn.metrics"] = skm
    sys.modules["sklearn.metrics.pairwise"] = skp

if not _try_import("faster_whisper"):
    fw = types.ModuleType("faster_whisper")
    fw.WhisperModel = object
    sys.modules["faster_whisper"] = fw

if not _try_import("soundfile"):
    sf_stub = types.ModuleType("soundfile")

    class _Info:
        def __init__(self, duration=0.0, samplerate=16000):
            self.duration = duration
            self.samplerate = samplerate

    sf_stub.info = lambda path: _Info()
    sf_stub.read = lambda path, **kw: (__import__("numpy").zeros(16000, dtype="float32"), 16000)
    sys.modules["soundfile"] = sf_stub

if not _try_import("funasr"):
    funasr_stub = types.ModuleType("funasr")
    funasr_stub.AutoModel = object
    sys.modules["funasr"] = funasr_stub
