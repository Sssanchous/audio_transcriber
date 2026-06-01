from __future__ import annotations

import re

from .domain_profiles import get_domain_profile


ACTION_RE = re.compile(
    r"\b(подготовь|подготовить|сделай|сделать|проверь|проверить|исправь|исправить|собери|собрать|"
    r"отправь|отправить|обнови|обновить|согласуй|согласовать|добавь|добавить|оформи|оформить|"
    r"пришли|прислать|сделать\s+развертку|разделить\s+по\s+параметрам|сопоставить\s+кейсы|"
    r"скинуть\s+промежуточные\s+результаты)\b",
    re.IGNORECASE,
)
DISCUSSION_RE = re.compile(
    r"\b(как\s+правило|в\s+теории|в\s+принципе|как\s+гипотеза|получается|я\s+правильно\s+понимаю|"
    r"задача\s+эксперта|задача\s+интерпретации|обратная\s+задача|математическая\s+задача)\b",
    re.IGNORECASE,
)
PERSON_RE = re.compile(r"^\s*[А-ЯЁ][а-яё]{2,}\s*[,.:]")


def score_task_candidate(
    text: str,
    meeting_type_label: str | None,
    responsible: str | None = None,
    deadline: str | None = None,
    classifier_prediction: dict | None = None,
) -> dict:
    score = 0.0
    reasons: list[str] = []
    clean = text or ""

    if ACTION_RE.search(clean):
        score += 0.42
        reasons.append("action_marker")
    if PERSON_RE.search(clean):
        score += 0.16
        reasons.append("person_prefix")
    if responsible:
        score += 0.14
        reasons.append("responsible")
    if deadline:
        score += 0.12
        reasons.append("deadline")
    if "из ближайших задач" in clean.lower() or "к следующей встрече" in clean.lower():
        score += 0.18
        reasons.append("explicit_action_item")
    if classifier_prediction and classifier_prediction.get("label") == "task":
        score += min(float(classifier_prediction.get("confidence", 0.0) or 0.0), 1.0) * 0.12
        reasons.append("classifier_task")

    if DISCUSSION_RE.search(clean):
        score -= 0.35
        reasons.append("technical_discussion_penalty")
    if meeting_type_label in {"technical_research", "education_consultation", "mixed"} and not (
        responsible or deadline or "из ближайших задач" in clean.lower() or "к следующей встрече" in clean.lower()
    ):
        score -= 0.12
        reasons.append("meeting_type_penalty")

    profile = get_domain_profile(meeting_type_label)
    score = max(0.0, min(1.0, score))
    if score >= profile["task_accept_threshold"]:
        decision = "accept"
    elif score >= profile["task_review_threshold"]:
        decision = "review"
    else:
        decision = "reject"
    return {
        "candidate_type": "task",
        "score": round(score, 3),
        "decision": decision,
        "review_required": decision != "accept",
        "reasons": reasons,
    }


def build_review_items(result: dict) -> list[dict]:
    items: list[dict] = []
    for task in result.get("clean_tasks", []):
        if task.get("review_required") or float(task.get("confidence", 0.0) or 0.0) < 0.75:
            items.append(
                {
                    "type": "task_candidate",
                    "text": task.get("source_text") or task.get("title"),
                    "clean_title": task.get("title"),
                    "reason": "low_confidence_or_missing_owner",
                    "confidence": task.get("confidence"),
                    "source_fragments": [task.get("source_fragment")] if task.get("source_fragment") else [],
                }
            )
    for pair in result.get("clean_questions_answers", []):
        has_partial_answer = pair.get("status") == "partial" and len(pair.get("answer") or "") >= 40
        if pair.get("status") != "answered" and not has_partial_answer:
            items.append(
                {
                    "type": "qa_candidate",
                    "text": pair.get("question"),
                    "reason": f"qa_status_{pair.get('status')}",
                    "confidence": 0.55 if pair.get("status") == "partial" else 0.35,
                    "source_fragments": pair.get("source_fragments", []),
                }
            )
    for deadline in result.get("clean_deadlines", []):
        if deadline.get("review_required"):
            items.append(
                {
                    "type": "deadline_candidate",
                    "text": deadline.get("text"),
                    "reason": "low_confidence_deadline",
                    "confidence": deadline.get("confidence"),
                    "source_fragments": [deadline.get("source_fragment")] if deadline.get("source_fragment") else [],
                }
            )
    return items
