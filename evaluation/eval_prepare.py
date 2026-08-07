#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare PDF-derived artifacts for citation-quality evaluation.

Default input:
  eval_inputs/example_method/*.pdf

Default output:
  eval_cache/example_method/md/001.md
  eval_cache/example_method/png/001/001.png
  eval_cache/example_method/logs/prepare_summary.json

Only md/, png/, and logs/ are kept under eval_cache/{dataset}. Runtime scratch
files live under .eval_prepare_tmp/{dataset} and are cleaned up
by default.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import queue
import re
import shutil
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

try:
    from loguru import logger
except Exception:
    class _FallbackLogger:
        def __init__(self) -> None:
            self._logger = logging.getLogger("eval_prepare")
            self._logger.setLevel(logging.INFO)
            if not self._logger.handlers:
                handler = logging.StreamHandler(sys.stderr)
                handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
                self._logger.addHandler(handler)

        def remove(self):
            for handler in list(self._logger.handlers):
                self._logger.removeHandler(handler)

        def configure(self, *args, **kwargs):
            return None

        def add(self, path, **kwargs):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(path, encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
            self._logger.addHandler(handler)

        def info(self, msg):
            self._logger.info(msg)

        def warning(self, msg):
            self._logger.warning(msg)

        def error(self, msg):
            self._logger.error(msg)

    logger = _FallbackLogger()

try:
    from tqdm import tqdm
except Exception:
    class tqdm:
        def __init__(self, total=None, unit=None, desc=None):
            self.total = total or 0
            self.count = 0
            self.desc = desc or "progress"

        def __enter__(self):
            print(f"{self.desc}: 0/{self.total}")
            return self

        def __exit__(self, exc_type, exc, tb):
            print(f"{self.desc}: {self.count}/{self.total}")

        def update(self, n=1):
            self.count += n


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_pdf_extract_kit_root = os.environ.get("PDF_EXTRACT_KIT_ROOT", "").strip()
PDF_EXTRACT_KIT_ROOT = Path(_pdf_extract_kit_root).expanduser() if _pdf_extract_kit_root else None
PDF_EXTRACT_KIT_HASH = "1d9a3cd772329d0f83d84638a789296863f940f9"
HF_PDF_EXTRACT_KIT_CACHE = Path.home() / ".cache/huggingface/hub/models--opendatalab--PDF-Extract-Kit-1.0"


THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENCV_NUM_THREADS": "1",
    "OPENCV_OPENCL_RUNTIME": "disabled",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "DISABLE_UPDATE_CHECK": "true",
    "YOLO_OFFLINE": "True",
    "YOLO_UPDATE_CHECK": "False",
    "YOLO_VERBOSE": "False",
    "TORCH_COMPILE_DISABLE": "1",
}


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("single PDF parse timed out")


@dataclass
class Task:
    pdf_path: str
    stem: str
    md_path: str
    png_dir: str
    size: int


def apply_runtime_env(tmp_dir: Path) -> None:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for key, value in THREAD_ENV.items():
        os.environ[key] = value
    os.environ["TMPDIR"] = str(tmp_dir)
    os.environ["TEMP"] = str(tmp_dir)
    os.environ["TMP"] = str(tmp_dir)


def write_magic_pdf_config(work_dir: Path) -> Path:
    if PDF_EXTRACT_KIT_ROOT is None:
        raise RuntimeError(
            "PDF_EXTRACT_KIT_ROOT is required for MinerU preprocessing. "
            "Set it to the local PDF-Extract-Kit directory."
        )
    work_dir.mkdir(parents=True, exist_ok=True)
    config_path = work_dir / "magic-pdf.json"
    config = {
        "bucket_info": {"bucket-name-1": ["ak", "sk", "endpoint"]},
        "models-dir": str(PDF_EXTRACT_KIT_ROOT / "models"),
        "device-mode": "cuda",
        "table-config": {"model": "TableMaster", "enable": False, "max_time": 400},
    }
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return config_path


def forge_offline_environment(work_dir: Path) -> None:
    """Expose local PDF-Extract-Kit through the HF cache layout MinerU expects."""
    work_config = write_magic_pdf_config(work_dir)
    home_config = Path.home() / "magic-pdf.json"
    try:
        home_config.write_text(work_config.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception as exc:
        logger.warning(f"failed to write {home_config}: {exc}")

    refs_dir = HF_PDF_EXTRACT_KIT_CACHE / "refs"
    snapshots_dir = HF_PDF_EXTRACT_KIT_CACHE / "snapshots"
    target_snapshot = snapshots_dir / PDF_EXTRACT_KIT_HASH

    try:
        refs_dir.mkdir(parents=True, exist_ok=True)
        snapshots_dir.mkdir(parents=True, exist_ok=True)
        (refs_dir / "main").write_text(PDF_EXTRACT_KIT_HASH, encoding="utf-8")

        if target_snapshot.is_symlink() or target_snapshot.exists():
            if target_snapshot.resolve(strict=False) == PDF_EXTRACT_KIT_ROOT.resolve(strict=False):
                logger.info(f"HF cache already mapped: {target_snapshot} -> {PDF_EXTRACT_KIT_ROOT}")
                return
            logger.warning(f"replacing stale HF cache mapping: {target_snapshot} -> {target_snapshot.resolve(strict=False)}")
            if target_snapshot.is_dir() and not target_snapshot.is_symlink():
                shutil.rmtree(target_snapshot)
            else:
                target_snapshot.unlink()

        target_snapshot.symlink_to(PDF_EXTRACT_KIT_ROOT, target_is_directory=True)
        logger.info(f"HF cache mapped: {target_snapshot} -> {PDF_EXTRACT_KIT_ROOT}")
    except Exception as exc:
        logger.warning(f"failed to map HF cache for PDF-Extract-Kit: {exc}")


def natural_pdf_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)", path.stem)
    if match:
        return (int(match.group(1)), path.name)
    return (10**9, path.name)


def discover_pdf_tasks(input_root: Path, md_dir: Path, png_root: Path, overwrite: bool) -> List[Task]:
    pdfs = sorted(input_root.glob("*.pdf"), key=natural_pdf_key)
    tasks: List[Task] = []
    for pdf_path in pdfs:
        stem = pdf_path.stem
        md_path = md_dir / f"{stem}.md"
        png_dir = png_root / stem
        if md_path.exists():
            continue
        tasks.append(Task(str(pdf_path), stem, str(md_path), str(png_dir), pdf_path.stat().st_size))
    return tasks


def setup_logger(log_file: Path, gpu_id: int | str) -> None:
    def patch(record):
        record["extra"].setdefault("gpu", gpu_id)

    logger.remove()
    try:
        logger.configure(patcher=patch)
        logger.add(
            log_file,
            enqueue=True,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | GPU-{extra[gpu]} | {message}",
            level="INFO",
        )
    except TypeError:
        logger.add(log_file)


def choose_markdown(generated_root: Path, stem: str) -> Path:
    candidates = sorted(generated_root.rglob("*.md"))
    if not candidates:
        raise FileNotFoundError(f"no markdown output found under {generated_root}")
    exact = [path for path in candidates if path.stem == stem]
    if exact:
        return exact[0]
    return sorted(candidates, key=lambda path: path.stat().st_size, reverse=True)[0]


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def render_pdf_pages_to_png(pdf_path: Path, png_dir: Path, dpi: int) -> int:
    try:
        import fitz  # PyMuPDF
    except Exception as exc:
        raise RuntimeError("PyMuPDF/fitz is required to render full-page PDF PNGs") from exc

    reset_dir(png_dir)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    count = 0
    with fitz.open(pdf_path) as doc:
        for page_index, page in enumerate(doc, start=1):
            pix = page.get_pixmap(matrix=mat, alpha=False)
            pix.save(str(png_dir / f"{page_index:03d}.png"))
            count += 1
    return count


def collect_markdown(generated_root: Path, task: Task) -> Path:
    md_src = choose_markdown(generated_root, task.stem)
    md_dst = Path(task.md_path)
    md_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(md_src, md_dst)
    return md_dst


def worker_process(
    gpu_id: int,
    task_queue: mp.Queue,
    progress_queue: mp.Queue,
    log_file: str,
    tmp_dir: str,
    mineru_tmp_dir: str,
    work_dir: str,
    timeout_seconds: int,
    render_dpi: int,
) -> None:
    apply_runtime_env(Path(tmp_dir))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["MINERU_DEVICE_MODE"] = "cuda"
    os.environ["MINERU_VIRTUAL_VRAM_SIZE"] = "10"
    os.chdir(work_dir)
    setup_logger(Path(log_file), gpu_id)
    signal.signal(signal.SIGALRM, timeout_handler)

    try:
        import cv2
        cv2.setNumThreads(0)
    except Exception:
        pass

    try:
        import torch  # noqa: F401
        from mineru.cli.common import do_parse, read_fn
    except Exception as exc:
        progress_queue.put({"status": "fatal", "error": f"MinerU import failed: {exc}"})
        return

    while True:
        item = task_queue.get()
        if item is None:
            break

        task = Task(**item)
        temp_out = Path(mineru_tmp_dir) / f"worker_{os.getpid()}_{task.stem}_{int(time.time())}"
        temp_out.mkdir(parents=True, exist_ok=True)

        try:
            signal.alarm(timeout_seconds)
            pdf_bytes = [read_fn(Path(task.pdf_path))]
            do_parse(
                output_dir=str(temp_out),
                pdf_file_names=[task.stem],
                pdf_bytes_list=pdf_bytes,
                p_lang_list=["en"],
                backend="pipeline",
                parse_method="auto",
                formula_enable=False,
                table_enable=True,
                f_dump_orig_pdf=False,
                f_dump_middle_json=False,
                f_dump_model_output=False,
                f_dump_content_list=False,
                f_dump_md=True,
                f_draw_span_bbox=False,
                f_draw_layout_bbox=False,
            )
            signal.alarm(0)

            generated_root = temp_out / task.stem
            if not generated_root.exists():
                generated_root = temp_out
            md_path = collect_markdown(generated_root, task)
            page_count = render_pdf_pages_to_png(Path(task.pdf_path), Path(task.png_dir), dpi=render_dpi)

            progress_queue.put(
                {
                    "status": "ok",
                    "stem": task.stem,
                    "md": str(md_path),
                    "png_dir": task.png_dir,
                    "page_png_count": page_count,
                }
            )
        except Exception as exc:
            signal.alarm(0)
            logger.error(f"failed {task.pdf_path}: {exc}\n{traceback.format_exc()}")
            progress_queue.put({"status": "error", "stem": task.stem, "error": str(exc)})
        finally:
            signal.alarm(0)
            shutil.rmtree(temp_out, ignore_errors=True)
            try:
                import torch as raw_torch
                raw_torch.cuda.empty_cache()
                raw_torch.cuda.ipc_collect()
            except Exception:
                pass


def run_pool(args: argparse.Namespace, tasks: List[Task], paths: dict[str, Path], log_file: Path) -> List[dict]:
    mp.set_start_method("spawn", force=True)
    task_queue: mp.Queue = mp.Queue()
    progress_queue: mp.Queue = mp.Queue()

    for task in tasks:
        task_queue.put(task.__dict__)

    worker_count = max(1, len(args.gpus) * args.workers_per_gpu)
    for _ in range(worker_count):
        task_queue.put(None)

    processes: List[mp.Process] = []
    for gpu in args.gpus:
        for _ in range(args.workers_per_gpu):
            proc = mp.Process(
                target=worker_process,
                args=(
                    gpu,
                    task_queue,
                    progress_queue,
                    str(log_file),
                    str(paths["tmp_dir"]),
                    str(paths["mineru_tmp_dir"]),
                    str(paths["work_dir"]),
                    args.timeout_seconds,
                    args.render_dpi,
                ),
            )
            proc.start()
            processes.append(proc)
            time.sleep(args.worker_start_gap)

    results: List[dict] = []
    with tqdm(total=len(tasks), unit="pdf", desc=f"prepare:{args.dataset_name}") as pbar:
        completed = 0
        idle_seconds = 0
        while completed < len(tasks):
            try:
                result = progress_queue.get(timeout=60)
                results.append(result)
                completed += 1
                pbar.update(1)
                idle_seconds = 0
            except queue.Empty:
                alive = [proc for proc in processes if proc.is_alive()]
                if not alive:
                    results.append({"status": "fatal", "error": "all workers exited before completion"})
                    break
                idle_seconds += 60
                if idle_seconds >= args.global_idle_timeout:
                    for proc in processes:
                        proc.terminate()
                    results.append({"status": "fatal", "error": "global idle timeout"})
                    break

    for proc in processes:
        proc.join(timeout=5)
    return results


def parse_gpus(raw) -> List[int]:
    if isinstance(raw, list):
        return raw
    gpus = [int(part.strip()) for part in str(raw).split(",") if part.strip()]
    if not gpus:
        raise argparse.ArgumentTypeError("--gpus must contain at least one GPU id")
    return gpus


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare MinerU Markdown and full-page PNG artifacts for eval PDFs.")
    parser.add_argument("--method", default=None, help="Submission method name. If set, input-root becomes eval_inputs/{method}.")
    parser.add_argument("--topic-id", default=None, help="Single topic id such as 001. If omitted, process all PDFs for the method/input-root.")
    parser.add_argument("--input-root", type=Path, default=None)
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--cache-root", type=Path, default=PROJECT_ROOT / "eval_cache")
    parser.add_argument("--temp-root", type=Path, default=PROJECT_ROOT / ".eval_prepare_tmp")
    parser.add_argument("--gpus", type=parse_gpus, default=[0], help="Comma-separated GPU ids. Default: 0")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--global-idle-timeout", type=int, default=600)
    parser.add_argument("--worker-start-gap", type=float, default=3.0)
    parser.add_argument("--api-mode", choices=("on", "off", "config"), default=None, help="Accepted for interface compatibility and ignored by prepare.")
    parser.add_argument("--model-name", default=None, help="Accepted for interface compatibility and ignored by prepare.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Optional number of PDFs to process for smoke tests.")
    parser.add_argument("--dry-run", action="store_true", help="List pending tasks and exit without parsing PDFs.")
    parser.add_argument("--render-dpi", type=int, default=160)
    parser.add_argument("--keep-temp", action="store_true", help="Keep runtime scratch files under --temp-root.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.method:
        args.dataset_name = args.method
        if args.input_root is None:
            args.input_root = PROJECT_ROOT / "eval_inputs" / args.method
    if args.dataset_name is None:
        args.dataset_name = "submission"
    if args.input_root is None:
        args.input_root = PROJECT_ROOT / "eval_inputs" / args.dataset_name

    dataset_root = args.cache_root / args.dataset_name
    dataset_tmp_root = args.temp_root / args.dataset_name
    paths = {
        "dataset_root": dataset_root,
        "md_dir": dataset_root / "md",
        "png_root": dataset_root / "png",
        "log_dir": dataset_root / "logs",
        "tmp_dir": dataset_tmp_root / "tmp",
        "mineru_tmp_dir": dataset_tmp_root / "mineru_tmp",
        "work_dir": dataset_tmp_root / "workdir",
    }

    for key in ["md_dir", "png_root", "log_dir", "tmp_dir", "mineru_tmp_dir", "work_dir"]:
        paths[key].mkdir(parents=True, exist_ok=True)

    tasks = discover_pdf_tasks(args.input_root, paths["md_dir"], paths["png_root"], overwrite=args.overwrite)
    if args.topic_id:
        topic_id = str(args.topic_id).zfill(3)
        tasks = [task for task in tasks if task.stem == topic_id]
        pdf_path = args.input_root / f"{topic_id}.pdf"
        if not tasks and not pdf_path.exists():
            print(f"[SKIP] missing PDF: method={args.dataset_name} topic_id={topic_id} path={pdf_path}")
    if args.limit > 0:
        tasks = tasks[: args.limit]

    print(f"input_root={args.input_root}")
    print(f"dataset_name={args.dataset_name}")
    print(f"cache_root={args.cache_root}")
    print(f"dataset_cache={dataset_root}")
    print(f"md_dir={paths['md_dir']}")
    print(f"png_root={paths['png_root']}")
    print(f"log_dir={paths['log_dir']}")
    print(f"temp_root={dataset_tmp_root}")
    print(f"gpus={args.gpus}; workers_per_gpu={args.workers_per_gpu}")
    print(f"tasks={len(tasks)}")

    if args.dry_run:
        for task in tasks[:20]:
            print("DRY_RUN_TASK %s: %s -> %s, %s" % (task.stem, task.pdf_path, task.md_path, task.png_dir))
        if len(tasks) > 20:
            print("DRY_RUN_TASK ... %d more" % (len(tasks) - 20))
        return 0

    summary_path = paths["log_dir"] / "prepare_summary.json"
    if not tasks:
        summary_path.write_text(json.dumps({"status": "nothing_to_do", "results": []}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"No pending PDFs. Summary: {summary_path}")
        return 0

    apply_runtime_env(paths["tmp_dir"])
    forge_offline_environment(paths["work_dir"])

    log_file = paths["log_dir"] / f"prepare_{time.strftime('%Y%m%d_%H%M%S')}.log"
    setup_logger(log_file, "main")

    try:
        results = run_pool(args, tasks, paths, log_file)
    finally:
        if not args.keep_temp:
            shutil.rmtree(dataset_tmp_root, ignore_errors=True)

    ok = [result for result in results if result.get("status") == "ok"]
    errors = [result for result in results if result.get("status") != "ok"]
    summary = {
        "input_root": str(args.input_root),
        "dataset_name": args.dataset_name,
        "cache_root": str(args.cache_root),
        "dataset_cache": str(dataset_root),
        "md_dir": str(paths["md_dir"]),
        "png_root": str(paths["png_root"]),
        "log_file": str(log_file),
        "num_tasks": len(tasks),
        "num_ok": len(ok),
        "num_errors": len(errors),
        "results": results,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"ok={len(ok)} errors={len(errors)} summary={summary_path}")
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
