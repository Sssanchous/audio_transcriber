from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Promote a checked RuBERT candidate model to active model.")
    parser.add_argument("--candidate", default="models/rubert_classifier_candidate")
    parser.add_argument("--target", default="models/rubert_classifier")
    args = parser.parse_args()

    candidate = Path(args.candidate)
    target = Path(args.target)
    if not candidate.exists():
        raise SystemExit(f"Candidate model not found: {candidate}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if target.exists():
        backup = target.with_name(f"{target.name}_backup_{timestamp}")
        shutil.copytree(target, backup)
    else:
        backup = None
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(candidate, target)
    version = {"promoted_at": timestamp, "candidate": str(candidate), "backup": str(backup) if backup else None}
    (target / "model_version.json").write_text(json.dumps(version, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "promoted", **version}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
