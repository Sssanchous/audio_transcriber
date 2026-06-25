from __future__ import annotations

import time
from pathlib import Path
from uuid import uuid4

from pm_insights import db, settings
from pm_insights.analytics.metrics import calculate_metrics
from pm_insights.asr.transcriber import build_transcriber
from pm_insights.audio.metadata import build_upload_metadata
from pm_insights.audio.upload_validator import validate_audio_file
from pm_insights.dataset.cleaner import clean_fragment, normalize_text
from pm_insights.dataset.segmenter import split_long_text
from pm_insights.nlp.aspects import extract_aspects
from pm_insights.nlp.deadline_extractor import extract_deadlines
from pm_insights.nlp.decision_extractor import extract_agreements, extract_decisions
from pm_insights.nlp.fragment_classifier import score_fragment_confidence
from pm_insights.nlp.meeting_type import detect_meeting_type
from pm_insights.nlp.qa_extractor import extract_qa_pairs
from pm_insights.nlp.postprocessing import normalize_analysis_result
from pm_insights.nlp.responsible_extractor import extract_responsibles
from pm_insights.nlp.semantic_blocks import build_semantic_blocks
from pm_insights.nlp.sentiment import analyze_sentiment
from pm_insights.nlp.task_extractor import extract_tasks
from pm_insights.nlp.text_normalization import normalize_text_for_nlp
from pm_insights.nlp.topic_modeling import extract_topics
from pm_insights.schemas import MeetingAnalysisResult

from .participants import parse_participants, participant_names
from .storage import save_meeting_result


def _segments_from_transcription(transcription: dict) -> list[dict]:
    fragments = []
    fragment_index = 0
    source_segments = transcription.get("segments") or [{"start": 0.0, "end": 0.0, "text": transcription.get("text", "")}]

    for segment in source_segments:
        start = float(segment.get("start", 0.0) or 0.0)
        end = float(segment.get("end", 0.0) or 0.0)
        for part in split_long_text(segment.get("text", ""), max_words=settings.MAX_SEGMENT_WORDS):
            cleaned, reason = clean_fragment(part, min_length=settings.MIN_SEGMENT_CHARS)
            if reason or not cleaned:
                continue
            fragment_index += 1
            fragments.append(
                {
                    "fragment_index": fragment_index,
                    "text": cleaned,
                    "normalized_text": normalize_text_for_nlp(cleaned),
                    "start": start,
                    "end": end,
                }
            )
    return fragments


def _enrich_tasks_with_confidence(tasks: list[dict], qa_pairs: list[dict]) -> None:
    for task in tasks:
        task.update(score_fragment_confidence(task.get("text", ""), "task"))

    for pair in qa_pairs:
        classifier_result = score_fragment_confidence(pair.get("question", ""), "question")
        pair["classifier_label"] = classifier_result["classifier_label"]
        pair["classifier_confidence"] = classifier_result["classifier_confidence"]
        pair["needs_review"] = classifier_result["needs_review"]

        answer_text = pair.get("answer")
        if answer_text:
            answer_classifier_result = score_fragment_confidence(answer_text, "answer")
            pair["answer_classifier_label"] = answer_classifier_result["classifier_label"]
            pair["answer_classifier_confidence"] = answer_classifier_result["classifier_confidence"]
            pair["needs_review"] = pair["needs_review"] or answer_classifier_result["needs_review"]


def _merge_items_by_text(*collections: list[dict]) -> list[dict]:
    merged: list[dict] = []
    seen: set[str] = set()
    for collection in collections:
        for item in collection:
            key = str(item.get("text") or item).strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(item)
    return merged


def analyze_meeting(
    audio_path: str | Path,
    project_name: str = "PM Insights",
    meeting_title: str = "",
    meeting_date: str | None = None,
    participants: str | list[dict] = "",
    output_dir: str | Path | None = None,
    transcript_text: str | None = None,
    meeting_id: str | None = None,
    persist_db: bool = False,
) -> dict:
    total_start = time.perf_counter()
    validation_start = time.perf_counter()
    path = Path(audio_path)
    validation = validate_audio_file(path, path.name)
    if not validation.valid:
        raise ValueError(f"Invalid audio file: {validation.reason}")
    upload_validation_seconds = time.perf_counter() - validation_start

    meeting_id = meeting_id or f"meeting_{uuid4().hex[:12]}"
    db_save_seconds = 0.0
    try:
        if persist_db:
            db_start = time.perf_counter()
            db.update_meeting_status(meeting_id, "processing")
            db.log_processing(meeting_id, "pipeline", "processing", "Analysis started")
            db_save_seconds += time.perf_counter() - db_start

        transcriber = build_transcriber(test_mode=bool(transcript_text), fallback_text=transcript_text)
        asr_start = time.perf_counter()
        transcription = transcriber.transcribe(path)
        asr_seconds = time.perf_counter() - asr_start
        transcription["text"] = normalize_text(transcript_text or transcription.get("text", ""))
        fragments = _segments_from_transcription(transcription)
        participant_items = parse_participants(participants)
        participant_name_items = participant_names(participant_items)

        nlp_start = time.perf_counter()
        meeting_type = detect_meeting_type(fragments)
        semantic_blocks = build_semantic_blocks(fragments, meeting_type.get("label"))
        analysis_units = semantic_blocks or fragments
        tasks = _merge_items_by_text(
            extract_tasks(fragments, participants=participant_name_items),
            extract_tasks(analysis_units, participants=participant_name_items),
        )
        qa_pairs = extract_qa_pairs(analysis_units)
        decisions = extract_decisions(analysis_units)
        agreements = extract_agreements(analysis_units)
        deadlines = _merge_items_by_text(extract_deadlines(fragments), extract_deadlines(analysis_units))
        responsibles = extract_responsibles(fragments, participants=participant_name_items)
        aspects = extract_aspects(analysis_units)
        _enrich_tasks_with_confidence(tasks, qa_pairs)
        nlp_seconds = time.perf_counter() - nlp_start

        topic_start = time.perf_counter()
        topic_result = extract_topics(
            analysis_units,
            meeting_type=meeting_type.get("label"),
            engine=settings.TOPIC_MODEL_ENGINE,
            max_topics=settings.TOPIC_MAX_TOPICS,
        )
        topics = topic_result["topics"]
        topic_modeling_seconds = time.perf_counter() - topic_start

        sentiment_start = time.perf_counter()
        sentiment = analyze_sentiment(analysis_units)
        sentiment_seconds = time.perf_counter() - sentiment_start

        audio_duration_seconds = float(transcription.get("duration", 0.0) or 0.0)
        total_processing_seconds = time.perf_counter() - total_start
        estimated_1h_processing_minutes = (
            total_processing_seconds / audio_duration_seconds * 3600 / 60 if audio_duration_seconds else None
        )
        processing_time = {
            "upload_validation_seconds": round(upload_validation_seconds, 3),
            "asr_seconds": round(asr_seconds, 3),
            "nlp_seconds": round(nlp_seconds, 3),
            "topic_modeling_seconds": round(topic_modeling_seconds, 3),
            "sentiment_seconds": round(sentiment_seconds, 3),
            "db_save_seconds": round(db_save_seconds, 3),
            "total_processing_seconds": round(total_processing_seconds, 3),
            "audio_duration_seconds": audio_duration_seconds,
            "estimated_1h_processing_minutes": round(estimated_1h_processing_minutes, 3)
            if estimated_1h_processing_minutes is not None
            else None,
        }

        result = MeetingAnalysisResult(
            meeting_id=meeting_id,
            project_name=project_name,
            meeting_date=meeting_date,
            source_audio=path.name,
            transcript=fragments,
            tasks=tasks,
            questions_answers=qa_pairs,
            decisions=decisions,
            deadlines=deadlines,
            responsibles=responsibles,
            sentiment=sentiment,
            aspects=aspects,
            topics=topics,
            metadata={
                "participants": participant_items,
                "meeting_info": {
                    "meeting_title": meeting_title,
                    "meeting_date": meeting_date,
                    "project_name": project_name,
                    "participants": participant_items,
                    "meeting_key": db.normalize_meeting_key(project_name, meeting_title),
                },
                "upload": build_upload_metadata(path.name, path, "completed"),
                "asr_status": transcription.get("status", "completed"),
                "asr_model": transcription.get("model", ""),
                "language": transcription.get("language", "ru"),
                "duration_seconds": transcription.get("duration", 0.0),
                "processing_time": processing_time,
            },
        ).to_dict()
        result["semantic_blocks"] = semantic_blocks
        result["meeting_type"] = meeting_type
        result["agreements"] = agreements
        result["commercial_terms"] = [item for item in agreements if item.get("type") == "commercial_term"]
        result["topic_source"] = topic_result.get("source")
        result["topic_warnings"] = topic_result.get("warnings", [])
        result["aspect_frequencies"] = topic_result.get("aspect_frequencies", {})
        result["learning_status"] = {
            "feedback_required": True,
            "feedback_saved": False,
            "ready_for_training": False,
            "message": "Проверьте результаты. Исправления будут использованы для будущего дообучения модели.",
        }
        result = normalize_analysis_result(result)
        # Mark so request-time reads (dashboard, /result) can skip re-normalizing.
        result["is_normalized"] = True
        result["metrics"] = calculate_metrics(result)
        result["metrics"]["processing_time"] = processing_time

        save_meeting_result(result, output_dir or settings.RESULTS_DIR)

        if persist_db:
            db_start = time.perf_counter()
            db.save_transcript(meeting_id, transcription)
            db.save_analysis_result(meeting_id, result)
            db.update_meeting_status(meeting_id, "completed")
            db.log_processing(meeting_id, "pipeline", "completed", "Analysis completed")
            db_save_seconds += time.perf_counter() - db_start
            processing_time["db_save_seconds"] = round(db_save_seconds, 3)
        return result
    except Exception as exc:
        if persist_db:
            db.update_meeting_status(meeting_id, "failed")
            db.log_processing(meeting_id, "pipeline", "failed", str(exc))
        raise
