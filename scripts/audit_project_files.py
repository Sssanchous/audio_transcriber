from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_DATASETS = {
    "training_dataset.jsonl",
    "train.jsonl",
    "val.jsonl",
    "test.jsonl",
    "dataset_stats.json",
    "split_summary.json",
}


def move(path: Path, target_root: Path) -> str:
    target = target_root / path.relative_to(ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target = target.with_name(f"{target.stem}_moved{target.suffix}")
    shutil.move(str(path), str(target))
    return str(target)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit project files without deleting data.")
    parser.add_argument("--move-to-legacy", action="store_true")
    args = parser.parse_args()

    extra_datasets = [path for path in (ROOT / "datasets").glob("*") if path.is_file() and path.name not in ACTIVE_DATASETS]
    extra_results = [
        path
        for path in (ROOT / "results").glob("*.json")
        if any(token in path.name.lower() for token in ["check", "demo", "kpi", "fixed", "final"])
    ]
    ignored_dirs = {".git", ".venv", "venv", "node_modules", "dist"}
    empty_files = [
        path
        for path in ROOT.rglob("*")
        if path.is_file() and path.stat().st_size == 0 and not (set(path.parts) & ignored_dirs)
    ]
    report = {
        "extra_dataset_files": [str(path.relative_to(ROOT)) for path in extra_datasets],
        "old_check_demo_results": [str(path.relative_to(ROOT)) for path in extra_results],
        "empty_files": [str(path.relative_to(ROOT)) for path in empty_files],
        "moved": [],
    }
    if args.move_to_legacy:
        legacy_root = ROOT / "legacy"
        for path in extra_datasets + extra_results:
            if path.exists():
                report["moved"].append(move(path, legacy_root))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
