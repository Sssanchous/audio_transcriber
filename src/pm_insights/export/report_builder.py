from __future__ import annotations

import copy
import io
import json
import re
import tempfile
from pathlib import Path
from typing import Any

from pm_insights.nlp.postprocessing import normalize_analysis_result


MEETING_TYPE_LABELS = {
    "project_meeting": "проектная встреча",
    "technical_research": "техническая / исследовательская встреча",
    "education_consultation": "учебная консультация",
    "commercial_meeting": "коммерческая встреча",
    "commercial_oil_gas": "нефтегазовая коммерческая встреча",
    "oil_gas_commercial": "нефтегазовая коммерческая встреча",
    "mixed": "смешанная встреча",
    "general_discussion": "общее обсуждение",
    "unknown": "тип встречи не определён",
}

SECTION_TITLES_RU = {
    "tasks": "Задачи",
    "research_actions": "Исследовательские действия",
    "recommendations": "Рекомендации",
    "research_notes": "Исследовательские заметки / технический контекст",
    "questions_answers": "Вопросы и ответы",
    "qa": "Вопросы и ответы",
    "deadlines": "Дедлайны / следующая встреча",
    "topics": "Ключевые темы",
    "aspects": "Аспекты",
    "aspects_topics": "Аспекты и темы",
    "sentiment": "Тональность",
    "review_items": "Требует проверки",
    "review": "Требует проверки",
    "transcript": "Транскрипт",
    "commercial_terms": "Коммерческие условия",
    "agreements": "Договорённости",
    "commitments": "Обещания сторон",
    "responsibles": "Ответственные",
    "responsible_sides": "Ответственные стороны",
    "decisions": "Решения",
}


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _looks_corrupted_heading(text: str) -> bool:
    if not text:
        return True
    question_marks = text.count("?")
    if question_marks >= 3:
        return True
    if question_marks and question_marks / max(len(text), 1) > 0.25:
        return True
    return False


def _compact(value: Any, limit: int = 900) -> str:
    text = re.sub(r"\s+", " ", _as_text(value)).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _meeting_type(result: dict) -> str:
    return (
        (result.get("meeting_type") or {}).get("label")
        or (result.get("analysis_summary") or {}).get("meeting_type")
        or "unknown"
    )


def _meeting_type_label(result: dict) -> str:
    meeting_type = _meeting_type(result)
    return (result.get("meeting_type") or {}).get("display_name") or MEETING_TYPE_LABELS.get(meeting_type, meeting_type)


def _metadata(result: dict) -> dict:
    metadata = result.get("metadata") or {}
    info = metadata.get("meeting_info") or result.get("meeting_info") or {}
    processing = metadata.get("processing_time") or (result.get("metrics") or {}).get("processing_time") or {}
    return {
        "system": "PM Insights",
        "meeting_id": result.get("meeting_id") or result.get("id") or "",
        "meeting_title": info.get("meeting_title") or result.get("meeting_title") or result.get("source_audio") or "",
        "project_name": info.get("project_name") or result.get("project_name") or "Без проекта",
        "meeting_date": info.get("meeting_date") or result.get("meeting_date") or result.get("created_at") or "",
        "meeting_type": _meeting_type(result),
        "meeting_type_label": _meeting_type_label(result),
        "duration_seconds": metadata.get("duration_seconds") or processing.get("audio_duration_seconds"),
        "analysis_date": result.get("created_at") or metadata.get("analysis_date") or "",
        "asr_status": metadata.get("asr_status") or "completed",
        "asr_model": metadata.get("asr_model") or metadata.get("model") or "",
        "processing_time_seconds": processing.get("total_processing_seconds"),
    }


def _section_items(result: dict, section_id: str, clean_field: str, raw_field: str | None = None) -> list[dict]:
    for section in result.get("report_sections") or []:
        if section.get("id") == section_id and isinstance(section.get("items"), list):
            return section["items"]
    if isinstance(result.get(clean_field), list):
        return result[clean_field]
    if raw_field and isinstance(result.get(raw_field), list):
        return result[raw_field]
    return []


def _section_items_any(
    result: dict,
    section_ids: tuple[str, ...],
    clean_fields: tuple[str, ...],
    raw_fields: tuple[str, ...] = (),
) -> list[dict]:
    for section in result.get("report_sections") or []:
        if section.get("id") in section_ids and isinstance(section.get("items"), list) and section["items"]:
            return section["items"]
    for field in clean_fields:
        if isinstance(result.get(field), list):
            return result[field]
    for field in raw_fields:
        if isinstance(result.get(field), list):
            return result[field]
    return []


def _topic_aspect_items(result: dict) -> list[dict]:
    items: list[dict] = []
    for section in result.get("report_sections") or []:
        if section.get("id") in {"topics", "aspects", "aspects_topics"} and isinstance(section.get("items"), list):
            items.extend(section["items"])
    for topic in result.get("topics") or []:
        if isinstance(topic, dict):
            items.append(topic)
    aspect_frequencies = (result.get("metrics") or {}).get("aspect_frequencies") or {}
    for aspect, count in aspect_frequencies.items():
        items.append({"topic_name": aspect, "count": count})
    return items


def _simplified_sections(result: dict) -> list[dict]:
    meeting_type = _meeting_type(result)
    sections: list[dict] = []

    def add(section_id: str, title: str, items: list[dict]) -> None:
        if items:
            sections.append({"id": section_id, "title": title, "items": items})

    if meeting_type in {"technical_research", "education_consultation"}:
        add(
            "tasks",
            "Задачи",
            _section_items_any(
                result,
                ("research_actions", "tasks"),
                ("clean_research_actions", "clean_tasks"),
                ("tasks",),
            ),
        )
    else:
        add("tasks", "Задачи", _section_items_any(result, ("tasks",), ("clean_tasks",), ("tasks",)))

    add("qa", "Вопросы и ответы", _section_items_any(result, ("questions_answers", "qa"), ("clean_questions_answers",), ("questions_answers",)))
    add("responsibles", "Ответственные", _section_items_any(result, ("responsibles",), ("clean_responsibles",), ("responsibles",)))
    add("deadlines", "Дедлайны", _section_items_any(result, ("deadlines",), ("clean_deadlines",), ("deadlines",)))
    add("decisions", "Решения", _section_items_any(result, ("decisions",), ("clean_decisions",), ("decisions",)))
    add("topics", "Аспекты и темы", _topic_aspect_items(result))
    add("sentiment", "Тональность", result.get("sentiment") or [])
    return sections


def build_report_payload(result: dict) -> dict:
    source = copy.deepcopy(result)
    normalized = normalize_analysis_result(copy.deepcopy(result))
    if source.get("report_sections"):
        normalized["report_sections"] = (normalized.get("report_sections") or []) + source["report_sections"]
    for field in (
        "clean_tasks",
        "clean_research_actions",
        "clean_questions_answers",
        "clean_deadlines",
        "clean_responsibles",
        "clean_decisions",
        "sentiment",
        "topics",
    ):
        if not normalized.get(field) and source.get(field):
            normalized[field] = source[field]
    sections = _simplified_sections(normalized)
    summary = normalized.get("analysis_summary") or {}
    transcript = normalized.get("transcript") or []
    return {
        "metadata": _metadata(normalized),
        "summary": summary,
        "sections": sections,
        "transcript": transcript,
        "technical_metrics": {
            "processing_time": (normalized.get("metadata") or {}).get("processing_time") or {},
        },
        "result": normalized,
    }


def export_json(report: dict) -> bytes:
    return json.dumps(report["result"], ensure_ascii=False, indent=2).encode("utf-8")


def _section_items_list(section: dict) -> list[dict]:
    items = section.get("items") or []
    return items if isinstance(items, list) else []


def _section_title(section: dict) -> str:
    section_id = section.get("id") or ""
    title = _as_text(section.get("title")).strip()
    if _looks_corrupted_heading(title):
        return SECTION_TITLES_RU.get(section_id, section_id or "Раздел")
    return title


def assert_no_question_marks_in_headings(report: dict) -> None:
    headings = [
        "PM Insights — отчёт по встрече",
        "Краткая сводка",
        "Основные результаты",
        "Транскрипт",
        *[_section_title(section) for section in report.get("sections") or []],
    ]
    damaged = [heading for heading in headings if _looks_corrupted_heading(heading)]
    if damaged:
        raise ValueError(f"Повреждены заголовки отчёта: {damaged}")


def _item_title(item: dict) -> str:
    return (
        item.get("title")
        or item.get("question_title")
        or item.get("topic_name")
        or item.get("deadline")
        or item.get("clean_title")
        or item.get("text")
        or "Элемент"
    )


def _item_summary(item: dict) -> str:
    return item.get("summary") or item.get("answer_summary") or item.get("context") or item.get("reason") or ""


def _append_sheet(workbook, title: str, rows: list[dict], columns: list[str]) -> None:
    sheet = workbook.create_sheet(title[:31])
    sheet.append(columns)
    for row in rows:
        sheet.append([_compact(row.get(column, "")) for column in columns])
    for cell in sheet[1]:
        cell.font = cell.font.copy(bold=True)
    sheet.freeze_panes = "A2"
    for column_cells in sheet.columns:
        max_len = max(len(_as_text(cell.value)) for cell in column_cells)
        sheet.column_dimensions[column_cells[0].column_letter].width = min(max(max_len + 2, 12), 48)
    for row_cells in sheet.iter_rows():
        for cell in row_cells:
            cell.alignment = cell.alignment.copy(wrap_text=True, vertical="top")


def export_xlsx(report: dict) -> bytes:
    from openpyxl import Workbook
    from openpyxl.styles import Border, Side

    workbook = Workbook()
    workbook.remove(workbook.active)
    metadata = report["metadata"]
    summary = report.get("summary") or {}
    _append_sheet(
        workbook,
        "Summary",
        [
            {"field": "Система", "value": metadata.get("system")},
            {"field": "Встреча", "value": metadata.get("meeting_title")},
            {"field": "Проект", "value": metadata.get("project_name")},
            {"field": "Дата", "value": metadata.get("meeting_date")},
            {"field": "Тип", "value": metadata.get("meeting_type_label")},
            {"field": "ASR модель", "value": metadata.get("asr_model")},
            {"field": "Время обработки", "value": metadata.get("processing_time_seconds")},
            {"field": "Основные темы", "value": ", ".join(summary.get("main_topics") or [])},
        ],
        ["field", "value"],
    )
    columns = {
        "tasks": ("Tasks", ["title", "summary", "responsible", "responsible_side", "deadline", "status", "source_fragment"]),
        "qa": ("QA", ["question_title", "answer_summary", "status", "source_fragments"]),
        "questions_answers": ("QA", ["question_title", "answer_summary", "status", "source_fragments"]),
        "responsibles": ("Responsibles", ["name", "responsible", "source_fragment"]),
        "deadlines": ("Deadlines", ["deadline", "context", "kind", "source_fragment"]),
        "decisions": ("Decisions", ["title", "summary", "source_fragment"]),
        "topics": ("Topics", ["topic_name", "title", "count", "keywords"]),
    }
    added = set()
    for section in report.get("sections") or []:
        section_id = section.get("id")
        if section_id in columns:
            sheet_name, sheet_columns = columns[section_id]
            if sheet_name not in added:
                _append_sheet(workbook, sheet_name, _section_items_list(section), sheet_columns)
                added.add(sheet_name)
    _append_sheet(workbook, "Sentiment", report["result"].get("sentiment") or [], ["text", "sentiment", "score", "source_fragment"])
    _append_sheet(workbook, "Transcript", report.get("transcript") or [], ["start", "end", "text", "sentiment"])
    thin = Side(style="thin", color="D9D9D9")
    for sheet in workbook.worksheets:
        for row in sheet.iter_rows():
            for cell in row:
                cell.border = Border(left=thin, right=thin, top=thin, bottom=thin)
    output = io.BytesIO()
    workbook.save(output)
    return output.getvalue()


def _set_docx_font(document) -> None:
    from docx.oxml.ns import qn

    for style_name in ("Normal", "Heading 1", "Heading 2", "Heading 3"):
        style = document.styles[style_name]
        style.font.name = "Calibri"
        if style._element.rPr is not None:
            style._element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
            style._element.rPr.rFonts.set(qn("w:cs"), "Calibri")


def export_docx(report: dict) -> bytes:
    from docx import Document

    assert_no_question_marks_in_headings(report)
    document = Document()
    _set_docx_font(document)
    document.add_heading("PM Insights — отчёт по встрече", level=1)
    metadata = report["metadata"]
    table = document.add_table(rows=0, cols=2)
    for key, value in [
        ("Встреча", metadata.get("meeting_title")),
        ("Проект", metadata.get("project_name")),
        ("Дата", metadata.get("meeting_date")),
        ("Тип", metadata.get("meeting_type_label")),
        ("Длительность", metadata.get("duration_seconds")),
        ("ASR модель", metadata.get("asr_model")),
        ("Время обработки", metadata.get("processing_time_seconds")),
    ]:
        row = table.add_row().cells
        row[0].text = key
        row[1].text = _as_text(value)
    document.add_heading("Краткая сводка", level=2)
    summary = report.get("summary") or {}
    if summary.get("main_topics"):
        document.add_paragraph("Основные темы: " + ", ".join(summary["main_topics"]))
    document.add_paragraph("Отчёт содержит транскрипт, задачи, вопросы и ответы, ответственных, сроки, аспекты, темы и тональность встречи.")
    document.add_heading("Основные результаты", level=2)
    for section in report.get("sections") or []:
        document.add_heading(_section_title(section), level=3)
        items = _section_items_list(section)
        if not items:
            document.add_paragraph("Нет данных")
            continue
        for item in items[:30]:
            document.add_paragraph(_compact(_item_title(item), 240), style="List Bullet")
            summary_text = _item_summary(item)
            if summary_text:
                document.add_paragraph(_compact(summary_text, 600))
    document.add_heading("Транскрипт", level=2)
    transcript_text = " ".join(segment.get("text", "") for segment in report.get("transcript") or [])
    document.add_paragraph(_compact(transcript_text, 20000) or "Нет данных")
    output = io.BytesIO()
    document.save(output)
    return output.getvalue()


def register_cyrillic_pdf_font() -> str:
    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
        Path("C:/Windows/Fonts/times.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
        Path("/usr/share/fonts/truetype/freefont/FreeSans.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise RuntimeError("Не найден системный Unicode-шрифт для PDF")


def export_pdf(report: dict) -> bytes:
    from fpdf import FPDF

    assert_no_question_marks_in_headings(report)
    metadata = report["metadata"]
    lines = [
        "PM Insights — отчёт по встрече",
        f"Встреча: {metadata.get('meeting_title')}",
        f"Проект: {metadata.get('project_name')}",
        f"Дата: {metadata.get('meeting_date')}",
        f"Тип: {metadata.get('meeting_type_label')}",
        "",
        "Краткая сводка:",
    ]
    summary = report.get("summary") or {}
    if summary.get("main_topics"):
        lines.append("Основные темы: " + ", ".join(summary["main_topics"][:8]))
    lines.append("Отчёт содержит транскрипт, задачи, вопросы и ответы, ответственных, сроки, аспекты, темы и тональность встречи.")
    lines.append("")
    lines.append("Основные результаты:")
    for section in report.get("sections") or []:
        lines.append(f"- {_section_title(section)}: {len(_section_items_list(section))}")
        for item in _section_items_list(section)[:5]:
            lines.append(f"  • {_compact(_item_title(item), 160)}")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.add_font("PMInsightsUnicode", "", register_cyrillic_pdf_font())
    pdf.set_font("PMInsightsUnicode", "", 11)
    pdf.multi_cell(0, 6, "\n".join(lines[:100]), wrapmode="CHAR")
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        pdf.output(tmp_path)
        return tmp_path.read_bytes()
    finally:
        tmp_path.unlink(missing_ok=True)
