from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ProtocolRecord:
    source_file: str
    meeting_date: str | None = None
    meeting_time: str | None = None
    format: str | None = None
    participants: list[str] = field(default_factory=list)
    project_title: str | None = None
    agenda: list[dict] = field(default_factory=list)
    discussion_items: list[dict] = field(default_factory=list)
    tasks: list[dict] = field(default_factory=list)
    decisions: list[dict] = field(default_factory=list)
    assignments: list[dict] = field(default_factory=list)
    deadlines: list[dict] = field(default_factory=list)
    key_points: list[dict] = field(default_factory=list)
    summary: str | None = None
    raw_sections: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


SECTION_PATTERNS = [
    ("tasks", re.compile(r"^(до\s+следующей\s+встречи\s+подготовить|поставленные\s+задачи|поручения|задачи|что\s+нужно\s+сделать|к\s+следующей\s+встрече)\b", re.IGNORECASE)),
    ("decisions", re.compile(r"^(принятые\s+решения|решили|решения|было\s+согласовано|согласовано|зафиксировано)\b", re.IGNORECASE)),
    ("discussion_items", re.compile(r"^(на\s+встрече\s+обсуждались|обсудили|повестка|ход\s+встречи|ключевые\s+моменты|тема\s+\d+)\b", re.IGNORECASE)),
    ("summary", re.compile(r"^(итог\s+встречи|вывод|результат\s+встречи)\b", re.IGNORECASE)),
]
META_RE = re.compile(r"^(Дата|Время|Формат|Присутствовали|Тема встречи|Тема|Проект)\s*[:：]\s*(.+)$", re.IGNORECASE)
LIST_PREFIX_RE = re.compile(r"^\s*(?:[-•*]|\d+[\).]|[а-я]\))\s*", re.IGNORECASE)


def _read_docx(path: Path) -> str:
    try:
        from docx import Document  # type: ignore
    except Exception as exc:
        raise RuntimeError("python-docx is required to parse .docx protocols.") from exc
    document = Document(str(path))
    return "\n".join(paragraph.text for paragraph in document.paragraphs if paragraph.text.strip())


def read_protocol_text(path: Path) -> str:
    if path.suffix.lower() == ".docx":
        try:
            return _read_docx(path)
        except Exception:
            return path.read_text(encoding="utf-8", errors="ignore")
    return path.read_text(encoding="utf-8", errors="ignore")


def _clean_item(line: str) -> str:
    return LIST_PREFIX_RE.sub("", line).strip(" \t:;.")


def _split_participants(value: str) -> list[str]:
    return [item.strip(" .") for item in re.split(r"[,;\n]+", value or "") if item.strip(" .")]


def _section_for_line(line: str) -> str | None:
    normalized = _clean_item(line).strip(":")
    for section, pattern in SECTION_PATTERNS:
        if pattern.search(normalized):
            return section
    return None


def _append_item(record: ProtocolRecord, section: str, text: str) -> None:
    item = {"text": text, "label": section[:-1] if section.endswith("s") else section, "source_section": section}
    if section == "tasks":
        item["label"] = "task"
        if "следующ" in " ".join(record.raw_sections.get(section, [])).lower():
            item["deadline"] = "до следующей встречи"
        record.tasks.append(item)
    elif section == "decisions":
        item["label"] = "decision"
        record.decisions.append(item)
    elif section == "discussion_items":
        item["label"] = "discussion_item"
        record.discussion_items.append(item)
    elif section == "summary":
        record.summary = (record.summary + " " if record.summary else "") + text
    else:
        record.key_points.append(item)


def parse_protocol_text(text: str, source_file: str) -> ProtocolRecord:
    record = ProtocolRecord(source_file=source_file)
    current_section: str | None = None

    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        meta = META_RE.match(line)
        if meta:
            key = meta.group(1).lower()
            value = meta.group(2).strip()
            if key == "дата":
                record.meeting_date = value
            elif key == "время":
                record.meeting_time = value
            elif key == "формат":
                record.format = value
            elif key == "присутствовали":
                record.participants = _split_participants(value)
            elif key in {"тема встречи", "тема", "проект"}:
                record.project_title = value
            continue

        if current_section and LIST_PREFIX_RE.match(line):
            item = _clean_item(line)
            if item:
                record.raw_sections.setdefault(current_section, []).append(line)
                _append_item(record, current_section, item)
            continue

        section = _section_for_line(line)
        if section:
            current_section = section
            record.raw_sections.setdefault(section, []).append(line)
            suffix = line.split(":", 1)[1].strip() if ":" in line else ""
            if suffix and len(suffix.split()) > 2:
                _append_item(record, section, _clean_item(suffix))
            continue

        if current_section:
            item = _clean_item(line)
            if item:
                record.raw_sections.setdefault(current_section, []).append(line)
                _append_item(record, current_section, item)
        else:
            record.agenda.append({"text": _clean_item(line), "label": "discussion_item", "source_section": "agenda"})

    return record


def parse_protocol_file(path: Path) -> ProtocolRecord:
    return parse_protocol_text(read_protocol_text(path), path.name)
