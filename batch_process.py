#!/usr/bin/env python3
"""
Multi-GPU batch processing for speaker diarization.
Processes MP3 files across multiple GPUs in parallel.

Usage:
    ./run_batch.sh /mnt/data/otherdatas/audio/ --gpus 0,1,2
    ./run_batch.sh /mnt/data/otherdatas/audio/ --gpus 0 --limit 5  # test with 5 files
    ./run_batch.sh /mnt/data/otherdatas/audio/ --resume             # skip already-done
    ./run_batch.sh /mnt/data/otherdatas/audio/ --dry-run             # list files only
"""

import os
import sys
import json
import time
import glob
import signal
import argparse
import sqlite3
import multiprocessing as mp
from multiprocessing import Process, Queue, Manager
from pathlib import Path
from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# Worker function - runs in a spawned child process
# ---------------------------------------------------------------------------

def worker_main(gpu_id, file_queue, result_queue, known_speakers_raw,
                input_dir, output_dir, threshold, script_dir):
    """
    Worker process. Processes files from the queue on a single GPU.

    CRITICAL: CUDA_VISIBLE_DEVICES must be set BEFORE importing torch.
    Since we use multiprocessing spawn, this function is the entry point
    and no torch has been imported yet in this process.
    """
    # Pin to a single GPU before any CUDA import
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Suppress verbose output from models
    import logging
    logging.disable(logging.WARNING)
    os.environ["FUNASR_DISABLE_PROGRESS"] = "1"

    # Redirect stdout/stderr to suppress engine spam (similarity prints, progress bars)
    import io
    class QuietStream(io.TextIOBase):
        """Swallows all output except lines we explicitly print."""
        def __init__(self, gpu_id, result_queue):
            self.gpu_id = gpu_id
            self.result_queue = result_queue
        def write(self, s):
            return len(s)
        def flush(self):
            pass

    # Save real stdout for our own output
    real_stdout = sys.stdout
    real_stderr = sys.stderr

    # Set up model cache paths
    os.environ["HF_HOME"] = os.path.join(script_dir, "volumes", "huggingface_cache")
    os.environ["WHISPER_CACHE_DIR"] = os.path.join(script_dir, "volumes", "whisper_cache")

    # Load .env for HF_TOKEN and other settings
    env_path = os.path.join(script_dir, ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key.strip(), val.strip())

    # NOW import torch and heavy dependencies
    sys.path.insert(0, script_dir)

    import gc
    import torch
    import numpy as np
    import traceback
    from pydub import AudioSegment as PydubSegment
    import tempfile

    gpu_name = "CPU"
    total_vram_gb = 0
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[GPU {gpu_id}] {gpu_name} | VRAM: {total_vram_gb:.1f} GB")
    else:
        print(f"[GPU {gpu_id}] WARNING: CUDA not available, using CPU")

    # Import and create engine
    # Set high threshold so engine's own periodic cleanup doesn't interfere
    os.environ["CLEANUP_VRAM_THRESHOLD_GB"] = "99"
    from app.diarization import SpeakerRecognitionEngine
    engine = SpeakerRecognitionEngine()

    def force_gpu_cleanup():
        """Aggressive GPU memory cleanup between files."""
        if not torch.cuda.is_available():
            return
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if hasattr(torch.cuda, 'reset_peak_memory_stats'):
            torch.cuda.reset_peak_memory_stats()

    def check_vram(label=""):
        """Log VRAM usage and warn if high."""
        if not torch.cuda.is_available():
            return 0
        used_gb = torch.cuda.memory_reserved() / 1024**3
        if used_gb > total_vram_gb * 0.7:
            print(f"[GPU {gpu_id}] WARNING: VRAM high ({used_gb:.1f}/{total_vram_gb:.1f} GB) {label}")
            force_gpu_cleanup()
            used_gb = torch.cuda.memory_reserved() / 1024**3
            print(f"[GPU {gpu_id}] After cleanup: {used_gb:.1f} GB")
        return used_gb

    # Force-load all models upfront
    print(f"[GPU {gpu_id}] Loading models...")
    load_start = time.time()
    _ = engine.diarization_pipeline
    _ = engine.embedding_model
    _ = engine.whisper_model
    _ = engine.emotion_model
    load_time = time.time() - load_start
    print(f"[GPU {gpu_id}] All models loaded in {load_time:.1f}s")

    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_reserved() / 1024**3
        print(f"[GPU {gpu_id}] VRAM after model load: {vram_used:.1f} GB")

    # Deserialize known speakers
    known_speakers = []
    for speaker_id, name, emb_bytes in known_speakers_raw:
        if emb_bytes:
            embedding = np.frombuffer(emb_bytes, dtype=np.float32).copy()
            known_speakers.append((speaker_id, name, embedding))

    print(f"[GPU {gpu_id}] {len(known_speakers)} known speakers loaded")

    # NOW suppress all engine output (similarity prints, progress bars, warnings)
    sys.stdout = QuietStream(gpu_id, result_queue)
    sys.stderr = QuietStream(gpu_id, result_queue)

    # Signal ready
    result_queue.put(("ready", gpu_id, None))

    # Process loop
    files_processed = 0
    files_failed = 0

    while True:
        file_path = file_queue.get()
        if file_path is None:
            break  # Sentinel - shutdown

        relative_path = os.path.relpath(file_path, input_dir)
        output_base = os.path.join(output_dir, "results", relative_path)
        json_path = output_base.rsplit(".", 1)[0] + ".json"
        txt_path = output_base.rsplit(".", 1)[0] + ".txt"

        # Skip if already processed
        if os.path.exists(json_path):
            files_processed += 1
            result_queue.put(("skip", gpu_id, file_path))
            continue

        os.makedirs(os.path.dirname(json_path), exist_ok=True)

        # Report start
        result_queue.put(("start", gpu_id, os.path.basename(file_path)))

        wav_path = None
        try:
            file_start = time.time()

            # Convert MP3 to WAV
            audio = PydubSegment.from_file(file_path)
            duration_sec = len(audio) / 1000.0

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_temp:
                wav_path = wav_temp.name
            audio.export(wav_path, format="wav")

            # Run diarization pipeline (no db_session - file-only mode)
            result = engine.transcribe_with_diarization(
                wav_path,
                known_speakers=known_speakers,
                threshold=threshold,
                db_session=None
            )

            processing_time = time.time() - file_start
            num_speakers = result["num_speakers"]
            num_segments = len(result["segments"])

            # Build JSON result (exclude numpy arrays)
            json_result = {
                "source_file": file_path,
                "relative_path": relative_path,
                "duration_seconds": round(duration_sec, 2),
                "processing_time_seconds": round(processing_time, 2),
                "gpu_id": gpu_id,
                "num_speakers": num_speakers,
                "num_segments": num_segments,
                "threshold": threshold,
                "processed_at": datetime.now().isoformat(),
                "segments": []
            }

            for seg in result["segments"]:
                seg_data = {
                    "start": round(seg["start"], 3),
                    "end": round(seg["end"], 3),
                    "speaker": seg.get("speaker", "Unknown"),
                    "speaker_label": seg.get("speaker_label", ""),
                    "text": seg.get("text", ""),
                    "confidence": round(seg.get("confidence", 0.0), 4),
                    "is_known": seg.get("is_known", False),
                    "emotion_category": seg.get("emotion_category"),
                    "emotion_confidence": round(seg.get("emotion_confidence", 0.0), 4) if seg.get("emotion_confidence") else None,
                    "avg_logprob": round(seg.get("avg_logprob", 0.0), 4) if seg.get("avg_logprob") else None,
                }
                # Word-level data
                if seg.get("words"):
                    seg_data["words"] = [
                        {"word": w.get("word", w) if isinstance(w, dict) else str(w),
                         "start": round(w["start"], 3) if isinstance(w, dict) and "start" in w else None,
                         "end": round(w["end"], 3) if isinstance(w, dict) and "end" in w else None,
                         "probability": round(w["probability"], 4) if isinstance(w, dict) and "probability" in w else None}
                        for w in seg["words"]
                    ]
                json_result["segments"].append(seg_data)

            with open(json_path, "w") as f:
                json.dump(json_result, f, indent=2, default=str)

            # Write human-readable transcript
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"# Transcript: {os.path.basename(file_path)}\n")
                f.write(f"# Duration: {duration_sec:.1f}s | Speakers: {result['num_speakers']} | Segments: {len(result['segments'])}\n")
                f.write(f"# Processed: {datetime.now().isoformat()} | GPU: {gpu_id}\n")
                f.write(f"# Source: {file_path}\n\n")

                for seg in result["segments"]:
                    start = seg["start"]
                    end = seg["end"]
                    speaker = seg.get("speaker", "Unknown")
                    text = seg.get("text", "").strip()
                    emotion = seg.get("emotion_category", "")

                    ts = f"[{int(start//60):02d}:{start%60:05.2f} -> {int(end//60):02d}:{end%60:05.2f}]"
                    emotion_tag = f" ({emotion})" if emotion else ""
                    f.write(f"{ts} {speaker}{emotion_tag}: {text}\n")

            files_processed += 1

            # Aggressive cleanup after every file
            del result
            force_gpu_cleanup()
            vram = check_vram(f"after {os.path.basename(file_path)}")

            result_queue.put(("done", gpu_id, {
                "file": os.path.basename(file_path),
                "duration": duration_sec,
                "time": processing_time,
                "segments": num_segments,
                "speakers": num_speakers,
                "vram_gb": round(vram, 1),
            }))

        except Exception as e:
            files_failed += 1
            error_msg = f"{type(e).__name__}: {str(e)}"

            result_queue.put(("error", gpu_id, {
                "file": os.path.basename(file_path),
                "error": error_msg,
            }))

            # Append to error log
            error_log = os.path.join(output_dir, "errors.log")
            with open(error_log, "a") as f:
                f.write(f"[{datetime.now().isoformat()}] GPU {gpu_id} | {file_path}\n")
                f.write(f"  {traceback.format_exc()}\n\n")

            # Extra aggressive cleanup after errors
            try:
                force_gpu_cleanup()
            except Exception:
                pass

        finally:
            # Clean up temp WAV
            if wav_path and os.path.exists(wav_path):
                try:
                    os.unlink(wav_path)
                except Exception:
                    pass

    # Final report
    result_queue.put(("finished", gpu_id, {
        "processed": files_processed,
        "failed": files_failed,
    }))


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def discover_mp3_files(input_dir):
    """Recursively find all MP3 files."""
    files = []
    for root, dirs, filenames in os.walk(input_dir):
        for f in filenames:
            if f.lower().endswith('.mp3'):
                files.append(os.path.join(root, f))
    files.sort()
    return files


def find_already_processed(output_dir, input_dir):
    """Find input files that already have output JSON."""
    processed = set()
    results_dir = os.path.join(output_dir, "results")
    if not os.path.exists(results_dir):
        return processed
    for json_file in glob.glob(os.path.join(results_dir, "**/*.json"), recursive=True):
        # Read the source_file from the JSON to get exact match
        try:
            with open(json_file) as f:
                data = json.load(f)
            if "source_file" in data:
                processed.add(data["source_file"])
        except Exception:
            pass
    return processed


def load_known_speakers(db_path):
    """Load speakers from SQLite (read-only). Returns list of (id, name, embedding_bytes)."""
    if not os.path.exists(db_path):
        print(f"Warning: Database not found at {db_path}")
        return []
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cursor = conn.execute("SELECT id, name, embedding FROM speakers")
    speakers = [(row[0], row[1], row[2]) for row in cursor]
    conn.close()
    return speakers


def format_time(seconds):
    """Format seconds as HH:MM:SS."""
    return str(timedelta(seconds=int(seconds)))


def main():
    parser = argparse.ArgumentParser(
        description="Multi-GPU batch speaker diarization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /mnt/data/audio/ --gpus 0,1,2          # All 3 GPUs
  %(prog)s /mnt/data/audio/ --gpus 0 --limit 5    # Test 5 files on GPU 0
  %(prog)s /mnt/data/audio/ --resume               # Skip already-processed
  %(prog)s /mnt/data/audio/ --dry-run              # List files only
        """
    )
    parser.add_argument("input_dir", help="Directory with MP3 files")
    parser.add_argument("--output", "-o", default=None,
                        help="Output directory (default: <input_dir>_transcripts)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-processed files")
    parser.add_argument("--threshold", type=float, default=0.30,
                        help="Speaker matching threshold (default: 0.30)")
    parser.add_argument("--gpus", type=str, default="0,1,2",
                        help="Comma-separated GPU IDs (default: 0,1,2)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of files to process (0=all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List files without processing")
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = args.output or (input_dir.rstrip("/") + "_transcripts")
    gpu_ids = [int(g) for g in args.gpus.split(",")]
    script_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"{'='*60}")
    print(f"Multi-GPU Batch Speaker Diarization")
    print(f"{'='*60}")
    print(f"Input:     {input_dir}")
    print(f"Output:    {output_dir}")
    print(f"GPUs:      {gpu_ids}")
    print(f"Threshold: {args.threshold}")
    print()

    # Discover files
    all_files = discover_mp3_files(input_dir)
    print(f"Found {len(all_files)} MP3 files")

    if args.resume:
        processed = find_already_processed(output_dir, input_dir)
        all_files = [f for f in all_files if f not in processed]
        print(f"Resume mode: {len(processed)} already done, {len(all_files)} remaining")

    if args.limit > 0:
        all_files = all_files[:args.limit]
        print(f"Limited to {len(all_files)} files")

    if not all_files:
        print("No files to process!")
        return

    if args.dry_run:
        print(f"\nFiles to process ({len(all_files)}):")
        for f in all_files[:30]:
            size_mb = os.path.getsize(f) / (1024 * 1024)
            print(f"  {size_mb:6.1f} MB  {os.path.relpath(f, input_dir)}")
        if len(all_files) > 30:
            print(f"  ... and {len(all_files) - 30} more")
        return

    # Load known speakers
    db_path = os.path.join(script_dir, "volumes", "speakers.db")
    known_speakers = load_known_speakers(db_path)
    print(f"Loaded {len(known_speakers)} known speakers from DB")

    # Create output directory
    os.makedirs(os.path.join(output_dir, "results"), exist_ok=True)

    # Set up multiprocessing with spawn (required for CUDA)
    ctx = mp.get_context("spawn")
    file_queue = ctx.Queue()
    result_queue = ctx.Queue()

    # Fill the queue
    for f in all_files:
        file_queue.put(f)
    # Add sentinels (one per worker)
    for _ in gpu_ids:
        file_queue.put(None)

    total_files = len(all_files)
    start_time = time.time()

    print(f"\nSpawning {len(gpu_ids)} workers...")

    # Spawn workers
    workers = []
    for gpu_id in gpu_ids:
        p = ctx.Process(
            target=worker_main,
            args=(gpu_id, file_queue, result_queue, known_speakers,
                  input_dir, output_dir, args.threshold, script_dir),
            name=f"GPU-{gpu_id}"
        )
        p.start()
        workers.append((gpu_id, p))
        print(f"  Worker GPU {gpu_id} started (PID: {p.pid})")

    # Wait for all workers to signal ready
    ready_count = 0
    print(f"\nWaiting for models to load on {len(gpu_ids)} GPUs...")
    while ready_count < len(gpu_ids):
        msg_type, gpu_id, data = result_queue.get()
        if msg_type == "ready":
            ready_count += 1
            print(f"  GPU {gpu_id} ready ({ready_count}/{len(gpu_ids)})")

    print(f"\nAll workers ready! Processing {total_files} files...\n")

    # Monitor progress
    completed = 0
    failed = 0
    skipped = 0
    finished_workers = 0
    gpu_current = {}  # what each GPU is working on
    gpu_vram = {}     # last known VRAM per GPU
    total_audio_duration = 0
    last_files = []   # rolling log of last few completed files

    def draw_progress():
        """Draw a clean progress display."""
        done = completed + failed + skipped
        pct = (done / total_files * 100) if total_files > 0 else 0
        elapsed = time.time() - start_time
        rate = completed / elapsed if elapsed > 0 and completed > 0 else 0
        remaining = total_files - done
        eta = remaining / rate if rate > 0 else 0

        # Progress bar
        bar_width = 40
        filled = int(bar_width * done / total_files) if total_files > 0 else 0
        bar = "█" * filled + "░" * (bar_width - filled)

        # Build status lines
        lines = []
        lines.append(f"  [{bar}] {pct:.1f}%")
        lines.append(f"  {done}/{total_files} done | {completed} ok | {failed} err | {skipped} skip | {format_time(elapsed)} elapsed | ETA: {format_time(eta)} | {rate:.2f} files/s")

        # GPU status line
        gpu_parts = []
        for gid in sorted(gpu_current.keys()):
            fname = gpu_current.get(gid, "idle")
            if isinstance(fname, str) and len(fname) > 25:
                fname = "..." + fname[-22:]
            vram = gpu_vram.get(gid, "?")
            gpu_parts.append(f"GPU{gid}:{vram}GB {fname}")
        if gpu_parts:
            lines.append(f"  {' | '.join(gpu_parts)}")

        # Last completed file
        if last_files:
            last = last_files[-1]
            lines.append(f"  Last: GPU{last[0]} {last[1]} ({last[2]:.0f}s, {last[3]}spk)")

        # Move cursor up and overwrite
        output = "\r" + "\033[K" + ("\033[1A\033[K" * (len(lines) - 1))
        output += "\n".join(lines)
        print(output, end="", flush=True)

    # Print initial empty lines for the progress display
    print("\n\n\n", end="")

    try:
        while finished_workers < len(gpu_ids):
            msg_type, gpu_id, data = result_queue.get()

            if msg_type == "start":
                gpu_current[gpu_id] = data
                draw_progress()
            elif msg_type == "done":
                completed += 1
                total_audio_duration += data.get("duration", 0)
                gpu_vram[gpu_id] = data.get("vram_gb", "?")
                gpu_current[gpu_id] = "idle"
                last_files.append((gpu_id, data['file'], data['time'], data['speakers']))
                if len(last_files) > 5:
                    last_files.pop(0)
                draw_progress()
            elif msg_type == "error":
                failed += 1
                gpu_current[gpu_id] = "idle"
                # Print error on its own line above progress
                print(f"\n  ERROR GPU{gpu_id}: {data['file']}: {data['error']}", flush=True)
                print("\n\n\n", end="")
                draw_progress()
            elif msg_type == "skip":
                skipped += 1
            elif msg_type == "finished":
                finished_workers += 1

        print()  # final newline

    except KeyboardInterrupt:
        print(f"\n\nInterrupted! Waiting for workers to finish current file...")
        # Workers will finish their current file and exit on next queue.get()

    # Wait for workers
    for gpu_id, p in workers:
        p.join(timeout=120)
        if p.is_alive():
            print(f"  Force-killing GPU {gpu_id} worker")
            p.terminate()
            p.join(timeout=10)

    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total files:    {total_files}")
    print(f"Completed:      {completed}")
    print(f"Failed:         {failed}")
    print(f"Skipped:        {skipped}")
    print(f"Total time:     {format_time(elapsed)}")
    print(f"Audio duration: {format_time(total_audio_duration)}")
    if completed > 0:
        print(f"Avg time/file:  {elapsed / completed:.1f}s")
        print(f"Throughput:     {completed / elapsed:.2f} files/s")
        print(f"Speedup:        {total_audio_duration / elapsed:.1f}x realtime")
    print(f"Output:         {output_dir}")
    if failed > 0:
        print(f"Error log:      {os.path.join(output_dir, 'errors.log')}")
    print(f"{'='*60}")

    # Save summary JSON
    summary = {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "total_files": total_files,
        "completed": completed,
        "failed": failed,
        "skipped": skipped,
        "elapsed_seconds": round(elapsed, 2),
        "total_audio_seconds": round(total_audio_duration, 2),
        "gpus": gpu_ids,
        "threshold": args.threshold,
        "started_at": datetime.fromtimestamp(start_time).isoformat(),
        "finished_at": datetime.now().isoformat(),
    }
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
