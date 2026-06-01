from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass
class MeetingAnalysisResult:
    meeting_id: str
    project_name: str
    meeting_date: str | None
    source_audio: str
    transcript: list[dict[str, Any]] = field(default_factory=list)
    tasks: list[dict[str, Any]] = field(default_factory=list)
    questions_answers: list[dict[str, Any]] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    deadlines: list[dict[str, Any]] = field(default_factory=list)
    responsibles: list[dict[str, Any]] = field(default_factory=list)
    sentiment: list[dict[str, Any]] = field(default_factory=list)
    aspects: list[dict[str, Any]] = field(default_factory=list)
    topics: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
