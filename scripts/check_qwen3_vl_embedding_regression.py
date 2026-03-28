#!/usr/bin/env python3

from __future__ import annotations

import argparse
import contextlib
import gc
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import numpy as np
import torch


@dataclass
class RunTimings:
    load_time_s: float | None = None
    run_time_s: float | None = None
    total_time_s: float | None = None
    wall_time_s: float | None = None
    prompt_eval_time_s: float | None = None
    prompt_eval_tokens: int | None = None
    prompt_eval_tokens_per_s: float | None = None
    device: str | None = None
    vision_backend: str | None = None


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Compare Python Qwen3-VL-Embedding outputs with llama-vl-embedding."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=repo_root,
        help="Repository root that contains llama.cpp, models, and Qwen3-VL-Embedding.",
    )
    parser.add_argument(
        "--inputs-file",
        type=Path,
        default=repo_root / "scripts" / "data" / "qwen3_vl_embedding_regression_inputs.json",
        help="JSON array input fixture.",
    )
    parser.add_argument(
        "--python-model-root",
        type=Path,
        default=repo_root / "Qwen3-VL-Embedding",
        help="Root directory of the Python reference implementation.",
    )
    parser.add_argument(
        "--hf-model-dir",
        type=Path,
        default=repo_root / "models" / "Qwen3-VL-Embedding-2B",
        help="Hugging Face model directory used by the Python reference.",
    )
    parser.add_argument(
        "--gguf-model",
        type=Path,
        default=repo_root / "Qwen3-VL-Embedding-2B-f16.gguf",
        help="GGUF model used by llama-vl-embedding.",
    )
    parser.add_argument(
        "--mmproj",
        type=Path,
        default=repo_root / "mmproj-Qwen3-VL-Embedding-2B-f16.gguf",
        help="MMProj file used by llama-vl-embedding.",
    )
    parser.add_argument(
        "--llama-bin",
        type=Path,
        default=repo_root / "llama.cpp" / "build" / "bin" / "llama-vl-embedding",
        help="llama-vl-embedding executable path.",
    )
    parser.add_argument(
        "--ctx-size",
        type=int,
        default=4096,
        help="Context size passed to llama-vl-embedding.",
    )
    parser.add_argument(
        "--python-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device selection for the Python reference implementation.",
    )
    parser.add_argument(
        "--python-torch-dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
        help="Optional torch_dtype override for the Python reference model.",
    )
    parser.add_argument(
        "--regression-mode",
        choices=("strict", "retrieval"),
        default="strict",
        help="strict checks numerical similarity; retrieval checks ranking consistency.",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        help="Override CUDA_VISIBLE_DEVICES before loading the Python reference or launching llama-vl-embedding.",
    )
    parser.add_argument(
        "--llama-ngl",
        default="0",
        help="Value passed to llama-vl-embedding as -ngl. Use 'auto' for GPU offload.",
    )
    parser.add_argument(
        "--llama-no-mmproj-offload",
        action="store_true",
        help="Pass --no-mmproj-offload to llama-vl-embedding.",
    )
    parser.add_argument(
        "--min-cosine",
        type=float,
        default=0.9995,
        help="Minimum per-sample cosine similarity.",
    )
    parser.add_argument(
        "--max-mean-abs-diff",
        type=float,
        default=5e-4,
        help="Maximum per-sample mean absolute difference.",
    )
    parser.add_argument(
        "--max-abs-diff",
        type=float,
        default=2e-3,
        help="Maximum per-sample elementwise absolute difference.",
    )
    parser.add_argument(
        "--max-pairwise-diff",
        type=float,
        default=2e-3,
        help="Maximum absolute difference between pairwise similarity matrices.",
    )
    parser.add_argument(
        "--min-pairwise-pearson",
        type=float,
        default=0.999,
        help="Minimum pairwise similarity Pearson correlation in retrieval mode.",
    )
    parser.add_argument(
        "--min-spearman-mean",
        type=float,
        default=0.999,
        help="Minimum mean Spearman rank correlation in retrieval mode.",
    )
    parser.add_argument(
        "--min-recall-at-1",
        type=float,
        default=0.99,
        help="Minimum Recall@1 in retrieval mode.",
    )
    parser.add_argument(
        "--min-recall-at-k",
        type=float,
        default=0.99,
        help="Minimum Recall@K in retrieval mode.",
    )
    parser.add_argument(
        "--min-mrr",
        type=float,
        default=0.99,
        help="Minimum MRR in retrieval mode.",
    )
    return parser.parse_args()


def assert_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def resolve_media_value(value: Any, repo_root: Path) -> Any:
    if isinstance(value, str):
        if value.startswith(("http://", "https://", "file://", "oss://")):
            return value
        path = Path(value)
        if not path.is_absolute():
            path = (repo_root / path).resolve()
        return str(path)

    if isinstance(value, list):
        return [resolve_media_value(v, repo_root) for v in value]

    return value


def resolve_input_paths(item: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    resolved = dict(item)
    for key in ("image", "video"):
        if key in resolved and resolved[key] is not None:
            resolved[key] = resolve_media_value(resolved[key], repo_root)
    return resolved


def load_inputs(inputs_file: Path, repo_root: Path) -> tuple[list[str], list[dict[str, Any]]]:
    with inputs_file.open("r", encoding="utf-8") as f:
        raw_inputs = json.load(f)

    if not isinstance(raw_inputs, list):
        raise ValueError(f"inputs file must contain a JSON array: {inputs_file}")

    labels: list[str] = []
    resolved_inputs: list[dict[str, Any]] = []

    for idx, item in enumerate(raw_inputs):
        if not isinstance(item, dict):
            raise ValueError(f"inputs[{idx}] must be an object")
        labels.append(str(item.get("name", f"input_{idx}")))
        resolved_inputs.append(resolve_input_paths(item, repo_root))

    return labels, resolved_inputs


def configure_cuda_env(args: argparse.Namespace) -> None:
    if args.python_device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return

    if args.python_device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices or os.environ.get("CUDA_VISIBLE_DEVICES") or "0"
        return

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices


def maybe_torch_cuda_synchronize() -> None:
    try:
        import torch
    except Exception:
        return

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def release_torch_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()


def configure_python_reference_logging() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r"Skipping import of cpp extensions due to incompatible torch version .*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"`torch_dtype` is deprecated! Use `dtype` instead!",
    )

    try:
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_error()
        if hasattr(transformers_logging, "disable_progress_bar"):
            transformers_logging.disable_progress_bar()
    except Exception:
        pass


def run_python_reference(
    python_model_root: Path,
    hf_model_dir: Path,
    inputs: list[dict[str, Any]],
    python_torch_dtype: str,
) -> tuple[np.ndarray, RunTimings]:
    configure_python_reference_logging()
    quiet = io.StringIO()
    with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
        sys.path.insert(0, str(python_model_root))
        try:
            from src.models.qwen3_vl_embedding import Qwen3VLEmbedder
        finally:
            sys.path.pop(0)

        t0 = time.perf_counter()
        embedder_kwargs: dict[str, Any] = {}
        if python_torch_dtype != "auto":
            dtype_map = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }
            embedder_kwargs["torch_dtype"] = dtype_map[python_torch_dtype]

        embedder = Qwen3VLEmbedder(model_name_or_path=str(hf_model_dir), **embedder_kwargs)
        maybe_torch_cuda_synchronize()
        load_time_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        embeddings = embedder.process(inputs)
        maybe_torch_cuda_synchronize()
        run_time_s = time.perf_counter() - t1
        device = str(embedder.model.device)

    timings = RunTimings(
        load_time_s=load_time_s,
        run_time_s=run_time_s,
        total_time_s=load_time_s + run_time_s,
        device=device,
    )
    # Some GPU-capable environments return BF16 embeddings from the HF reference
    # path, but torch->numpy BF16 conversion is not universally supported.
    return embeddings.detach().to(dtype=torch.float32, device="cpu").numpy(), timings


def parse_llama_timings(output: str) -> RunTimings:
    timings = RunTimings()

    if m := re.search(r"load time =\s*([0-9.]+)\s*ms", output):
        timings.load_time_s = float(m.group(1)) / 1000.0
    if m := re.search(
        r"prompt eval time =\s*([0-9.]+)\s*ms\s*/\s*([0-9]+)\s*tokens.*?([0-9.]+)\s*tokens per second",
        output,
    ):
        timings.prompt_eval_time_s = float(m.group(1)) / 1000.0
        timings.prompt_eval_tokens = int(m.group(2))
        timings.prompt_eval_tokens_per_s = float(m.group(3))
        timings.run_time_s = timings.prompt_eval_time_s
    if m := re.search(r"total time =\s*([0-9.]+)\s*ms", output):
        timings.total_time_s = float(m.group(1)) / 1000.0

    if "clip_ctx: CLIP using CUDA0 backend" in output:
        timings.vision_backend = "cuda"
    elif "clip_ctx: CLIP using CPU backend" in output:
        timings.vision_backend = "cpu"

    return timings


def run_llama_embedding(
    llama_bin: Path,
    gguf_model: Path,
    mmproj: Path,
    inputs: list[dict[str, Any]],
    ctx_size: int,
    llama_ngl: str,
    llama_no_mmproj_offload: bool,
    env: dict[str, str] | None = None,
) -> tuple[np.ndarray, RunTimings]:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tmp:
        json.dump(inputs, tmp, ensure_ascii=False)
        tmp.flush()
        temp_path = Path(tmp.name)

    cmd = [
        str(llama_bin),
        "-m",
        str(gguf_model),
        "--mmproj",
        str(mmproj),
        "--inputs-file",
        str(temp_path),
        "--pooling",
        "last",
        "--embd-normalize",
        "2",
        "--embd-output-format",
        "array",
        "-c",
        str(ctx_size),
        "-ngl",
        str(llama_ngl),
        "--no-warmup",
    ]
    if llama_no_mmproj_offload:
        cmd.append("--no-mmproj-offload")

    try:
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
        wall_time_s = time.perf_counter() - t0
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "llama-vl-embedding failed\n"
            f"command: {' '.join(cmd)}\n"
            f"returncode: {exc.returncode}\n"
            f"stdout:\n{exc.stdout}\n"
            f"stderr:\n{exc.stderr}"
        ) from exc
    finally:
        temp_path.unlink(missing_ok=True)

    match = re.search(r"(\[\[.*\]\])", proc.stdout, re.S)
    if not match:
        raise RuntimeError(
            "failed to parse llama-vl-embedding output\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )

    timings = parse_llama_timings(proc.stdout + "\n" + proc.stderr)
    timings.wall_time_s = wall_time_s
    return np.array(json.loads(match.group(1)), dtype=np.float32), timings


def fmt_time(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}s"


def descending_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(values))
    return ranks


def print_timing_summary(python_timings: RunTimings, llama_timings: RunTimings, args: argparse.Namespace) -> None:
    print(
        "python_reference: "
        f"device={python_timings.device or '-'} "
        f"load={fmt_time(python_timings.load_time_s)} "
        f"run={fmt_time(python_timings.run_time_s)} "
        f"total={fmt_time(python_timings.total_time_s)}"
    )
    prompt_tokens = "-" if llama_timings.prompt_eval_tokens is None else str(llama_timings.prompt_eval_tokens)
    prompt_tps = "-" if llama_timings.prompt_eval_tokens_per_s is None else f"{llama_timings.prompt_eval_tokens_per_s:.2f}"
    print(
        "llama_vl_embedding: "
        f"ngl={args.llama_ngl} "
        f"mmproj_offload={'off' if args.llama_no_mmproj_offload else 'on'} "
        f"vision_backend={llama_timings.vision_backend or '-'} "
        f"wall={fmt_time(llama_timings.wall_time_s)} "
        f"load={fmt_time(llama_timings.load_time_s)} "
        f"prompt_eval={fmt_time(llama_timings.prompt_eval_time_s)} "
        f"prompt_tokens={prompt_tokens} "
        f"prompt_tps={prompt_tps} "
        f"total={fmt_time(llama_timings.total_time_s)}"
    )


def main() -> int:
    args = parse_args()
    configure_cuda_env(args)

    assert_exists(args.inputs_file, "inputs file")
    assert_exists(args.python_model_root, "python model root")
    assert_exists(args.hf_model_dir, "HF model dir")
    assert_exists(args.gguf_model, "GGUF model")
    assert_exists(args.mmproj, "mmproj")
    assert_exists(args.llama_bin, "llama-vl-embedding binary")

    labels, inputs = load_inputs(args.inputs_file, args.repo_root)

    ref, python_timings = run_python_reference(
        args.python_model_root,
        args.hf_model_dir,
        inputs,
        args.python_torch_dtype,
    )
    release_torch_cuda_memory()
    llama_env = os.environ.copy()
    got, llama_timings = run_llama_embedding(
        args.llama_bin,
        args.gguf_model,
        args.mmproj,
        inputs,
        args.ctx_size,
        args.llama_ngl,
        args.llama_no_mmproj_offload,
        env=llama_env,
    )

    if ref.shape != got.shape:
        raise RuntimeError(f"shape mismatch: python={ref.shape}, llama={got.shape}")

    print(f"inputs_file: {args.inputs_file}")
    print(f"samples: {len(labels)}")
    print(f"regression_mode: {args.regression_mode}")
    print(
        f"cuda_visible_devices: {os.environ.get('CUDA_VISIBLE_DEVICES', '')!r} "
        f"python_device_request={args.python_device} "
        f"python_torch_dtype={args.python_torch_dtype}"
    )
    print_timing_summary(python_timings, llama_timings, args)
    print()

    failed = False
    for idx, label in enumerate(labels):
        ref_i = ref[idx]
        got_i = got[idx]
        cos = float(np.dot(ref_i, got_i) / (np.linalg.norm(ref_i) * np.linalg.norm(got_i)))
        abs_diff = np.abs(ref_i - got_i)
        mean_abs_diff = float(abs_diff.mean())
        max_abs_diff = float(abs_diff.max())

        print(
            f"{idx:02d} {label}: "
            f"cos={cos:.9f} "
            f"mean_abs_diff={mean_abs_diff:.9f} "
            f"max_abs_diff={max_abs_diff:.9f}"
        )

        if args.regression_mode == "strict":
            if cos < args.min_cosine:
                failed = True
            if mean_abs_diff > args.max_mean_abs_diff:
                failed = True
            if max_abs_diff > args.max_abs_diff:
                failed = True

    ref_sim = ref @ ref.T
    got_sim = got @ got.T
    sim_diff = np.abs(ref_sim - got_sim)
    pairwise_max = float(sim_diff.max())
    pairwise_mean = float(sim_diff.mean())

    if len(labels) >= 2:
        ref_pairs = ref_sim[np.triu_indices(len(labels), k=1)]
        got_pairs = got_sim[np.triu_indices(len(labels), k=1)]

        ref_centered = ref_pairs - ref_pairs.mean()
        got_centered = got_pairs - got_pairs.mean()
        denom = np.linalg.norm(ref_centered) * np.linalg.norm(got_centered)
        pairwise_pearson = float(np.dot(ref_centered, got_centered) / denom) if denom > 0 else 1.0

        ref_rank = np.argsort(-np.where(np.eye(len(labels), dtype=bool), -np.inf, ref_sim), axis=1)
        got_rank = np.argsort(-np.where(np.eye(len(labels), dtype=bool), -np.inf, got_sim), axis=1)
        nn_top1_acc = float((ref_rank[:, 0] == got_rank[:, 0]).mean())
        topk = min(3, len(labels) - 1)
        topk_overlap = float(
            np.mean(
                [
                    len(set(ref_rank[i, :topk]).intersection(set(got_rank[i, :topk]))) / topk
                    for i in range(len(labels))
                ]
            )
        )

        recall_at_1 = nn_top1_acc
        recall_at_k_hits = []
        reciprocal_ranks = []
        spearman_values = []
        for i in range(len(labels)):
            mask = np.ones(len(labels), dtype=bool)
            mask[i] = False

            ref_scores = ref_sim[i, mask]
            got_scores = got_sim[i, mask]
            ref_candidates = np.arange(len(labels))[mask]

            ref_order = ref_candidates[np.argsort(-ref_scores, kind="stable")]
            got_order = ref_candidates[np.argsort(-got_scores, kind="stable")]

            target = ref_order[0]
            target_rank = int(np.where(got_order == target)[0][0])
            recall_at_k_hits.append(float(target_rank < topk))
            reciprocal_ranks.append(1.0 / float(target_rank + 1))

            ref_ranks = descending_ranks(ref_scores)
            got_ranks = descending_ranks(got_scores)
            if len(ref_ranks) > 1:
                ref_centered = ref_ranks.astype(np.float64) - ref_ranks.mean()
                got_centered = got_ranks.astype(np.float64) - got_ranks.mean()
                denom = np.linalg.norm(ref_centered) * np.linalg.norm(got_centered)
                spearman = float(np.dot(ref_centered, got_centered) / denom) if denom > 0 else 1.0
            else:
                spearman = 1.0
            spearman_values.append(spearman)

        recall_at_k = float(np.mean(recall_at_k_hits))
        mrr = float(np.mean(reciprocal_ranks))
        spearman_mean = float(np.mean(spearman_values))
        spearman_min = float(np.min(spearman_values))
    else:
        pairwise_pearson = 1.0
        nn_top1_acc = 1.0
        topk_overlap = 1.0
        recall_at_1 = 1.0
        recall_at_k = 1.0
        mrr = 1.0
        spearman_mean = 1.0
        spearman_min = 1.0

    print()
    print(
        "pairwise_similarity: "
        f"max_abs_diff={pairwise_max:.9f} "
        f"mean_abs_diff={pairwise_mean:.9f}"
    )
    topk_label = min(3, max(1, len(labels) - 1))

    print(
        "retrieval_consistency: "
        f"pairwise_pearson={pairwise_pearson:.9f} "
        f"nn_top1_acc={nn_top1_acc:.9f} "
        f"top{topk_label}_overlap={topk_overlap:.9f}"
    )
    print(
        "retrieval_metrics: "
        f"spearman_mean={spearman_mean:.9f} "
        f"spearman_min={spearman_min:.9f} "
        f"recall_at_1={recall_at_1:.9f} "
        f"recall_at_{topk_label}={recall_at_k:.9f} "
        f"mrr={mrr:.9f}"
    )

    if args.regression_mode == "strict":
        if pairwise_max > args.max_pairwise_diff:
            failed = True
    else:
        if pairwise_pearson < args.min_pairwise_pearson:
            failed = True
        if spearman_mean < args.min_spearman_mean:
            failed = True
        if recall_at_1 < args.min_recall_at_1:
            failed = True
        if recall_at_k < args.min_recall_at_k:
            failed = True
        if mrr < args.min_mrr:
            failed = True

    if failed:
        print()
        print("regression_check: FAILED")
        return 1

    print()
    print("regression_check: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
