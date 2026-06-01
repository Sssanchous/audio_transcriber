from __future__ import annotations

import json
from pathlib import Path

from pm_insights.settings import RESULTS_DIR

INDEX_FILE = RESULTS_DIR / "meetings_index.json"


def save_meeting_result(result: dict, output_dir: str | Path = RESULTS_DIR) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{result['meeting_id']}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    upsert_index(result, root)
    return path


def upsert_index(result: dict, output_dir: str | Path = RESULTS_DIR) -> None:
    index_path = Path(output_dir) / "meetings_index.json"
    meetings = load_meetings(output_dir)
    meetings = [item for item in meetings if item.get("meeting_id") != result.get("meeting_id")]
    meetings.append(result)
    index_path.write_text(json.dumps(meetings, ensure_ascii=False, indent=2), encoding="utf-8")


def load_meetings(output_dir: str | Path = RESULTS_DIR) -> list[dict]:
    index_path = Path(output_dir) / "meetings_index.json"
    if not index_path.exists():
        return []
    try:
        return json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def load_meeting(meeting_id: str, output_dir: str | Path = RESULTS_DIR) -> dict | None:
    path = Path(output_dir) / f"{meeting_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def delete_meeting(meeting_id: str, output_dir: str | Path = RESULTS_DIR) -> bool:
    root = Path(output_dir)
    path = root / f"{meeting_id}.json"
    if path.exists():
        path.unlink()
    meetings = [item for item in load_meetings(root) if item.get("meeting_id") != meeting_id]
    (root / "meetings_index.json").write_text(json.dumps(meetings, ensure_ascii=False, indent=2), encoding="utf-8")
    return True
