from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


def build_upload_metadata(original_filename: str, stored_path: str | Path, status: str = "uploaded") -> dict:
    path = Path(stored_path)
    return {
        "original_filename": original_filename,
        "stored_filename": path.name,
        "extension": path.suffix.lower(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "upload_date": datetime.now(timezone.utc).isoformat(),
        "processing_status": status,
    }
