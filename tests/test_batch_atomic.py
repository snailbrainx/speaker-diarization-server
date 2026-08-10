"""Atomic batch-output publication tests."""
import inspect

import pytest


def test_atomic_write_replaces_only_after_complete_success(tmp_path):
    from batch_process import _atomic_write

    target = tmp_path / "result.txt"
    target.write_text("old")

    def fail_after_partial(stream):
        stream.write("partial")
        raise RuntimeError("injected failure")

    with pytest.raises(RuntimeError, match="injected"):
        _atomic_write(str(target), fail_after_partial, encoding="utf-8")

    assert target.read_text() == "old"
    assert list(tmp_path.glob("result.txt.*.tmp")) == []


def test_atomic_write_publishes_complete_content(tmp_path):
    from batch_process import _atomic_write

    target = tmp_path / "result.txt"
    _atomic_write(str(target), lambda stream: stream.write("complete"), encoding="utf-8")
    assert target.read_text() == "complete"


def test_batch_json_completion_marker_is_published_after_transcript():
    """Resume checks JSON; source order must keep JSON as the last marker."""
    import batch_process

    source = inspect.getsource(batch_process.worker_main)
    txt_publish = source.index("_atomic_write(txt_path")
    json_publish = source.index("_atomic_write(\n                json_path")
    assert txt_publish < json_publish
