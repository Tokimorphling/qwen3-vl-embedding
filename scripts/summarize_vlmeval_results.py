#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize VLMEvalKit CSV outputs into a compact markdown report."
    )
    parser.add_argument("--work-dir", required=True, help="VLMEvalKit work dir")
    parser.add_argument("--output", required=True, help="Markdown output path")
    return parser.parse_args()


def safe_read_csv(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def markdown_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    rows = [[str(v) for v in row] for row in df.fillna("").values.tolist()]
    widths = [len(c) for c in cols]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(row: list[str]) -> str:
        return "| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) + " |"

    header = fmt_row(cols)
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(cols))) + " |"
    body = [fmt_row(row) for row in rows]
    return "\n".join([header, sep] + body)


def dataset_key(model_name: str, csv_path: Path) -> str:
    stem = csv_path.stem
    prefix = f"{model_name}_"
    if stem.startswith(prefix):
        stem = stem[len(prefix):]
    return stem


def main() -> int:
    args = parse_args()
    work_dir = Path(args.work_dir).resolve()
    output = Path(args.output).resolve()

    model_dirs = sorted(
        p for p in work_dir.iterdir() if p.is_dir() and p.name not in {"logs"}
    )

    grouped: dict[str, list[tuple[str, Path, pd.DataFrame]]] = {}
    for model_dir in model_dirs:
        for csv_path in sorted(model_dir.glob("*.csv")):
            df = safe_read_csv(csv_path)
            if df is None:
                continue
            key = dataset_key(model_dir.name, csv_path)
            grouped.setdefault(key, []).append((model_dir.name, csv_path, df))

    lines: list[str] = []
    lines.append("# VLMEval Summary")
    lines.append("")
    lines.append(f"- work_dir: `{work_dir}`")
    lines.append(f"- models: `{', '.join(p.name for p in model_dirs)}`")
    lines.append("")

    manifest: dict[str, list[dict[str, str]]] = {}

    for key in sorted(grouped):
        lines.append(f"## {key}")
        lines.append("")
        manifest[key] = []
        for model_name, csv_path, df in grouped[key]:
            manifest[key].append(
                {
                    "model": model_name,
                    "csv": str(csv_path),
                }
            )
            lines.append(f"### {model_name}")
            lines.append("")
            lines.append(f"- csv: `{csv_path}`")
            lines.append(f"- shape: `{df.shape[0]} x {df.shape[1]}`")
            lines.append("")
            if df.shape[0] <= 20 and df.shape[1] <= 16:
                lines.append(markdown_table(df))
                lines.append("")
            else:
                lines.append("_table omitted due to size_")
                lines.append("")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    output.with_suffix(".json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
