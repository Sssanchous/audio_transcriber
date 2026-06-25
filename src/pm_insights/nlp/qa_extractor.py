from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from .fragment_classifier import score_fragment_confidence


KNOWN_SPEAKERS = "Алексей|Анна|Иван|Мария|Илья|Ольга|Дмитрий|Сергей|Екатерина|Павел|Николай"
SPEAKER_PREFIX_RE = re.compile(rf"^\s*(?:{KNOWN_SPEAKERS})\s*[,.:]\s*", re.IGNORECASE)
QUESTION_START_RE = re.compile(
    r"^\s*(кто|что|когда|где|почему|зачем|сколько|какой|какая|какие|какое)\b|"
    r"^\s*(можно|нужно|есть|готово|правильно|успеваем)\s+ли\b|"
    r"^\s*(я\s+правильно\s+понимаю|хотел(?:а)?\s+уточнить|подскажите|скажите|можно\s+уточнить)\b",
    re.IGNORECASE,
)
QUESTION_STOP_RE = re.compile(
    r"\b(как\s+правило|как\s+вариант|как\s+раз|как\s+раз-таки|как\s+конструкция|"
    r"как\s+первичный\s+подход|как\s+для\s+инженера|как\s+в\s+сетках|"
    r"где\s+меньше\s+ошибку\s+получим|почему\s+я\s+спрашиваю|в\s+чем\s+смысл\s+есть|"
    r"так,\s*где-то\s+было|где-то\s+было|какой-то|что\s+мы\s+тогда|"
    r"когда\s+мы\s+обратную\s+задачу\s+решаем|что\s+оно\s+только)\b",
    re.IGNORECASE,
)
ANSWER_RE = re.compile(
    r"^\s*(да|нет|готово|сделано|пока\s+нет|хорошо|понял|поняла|окей)\b|"
    r"\b(я\s+отправлю|отправлю|уточню|лежит|отвечаю\s+я|открыты|закрыты|планируем|"
    r"сделаю|исправлю|проверю|подготовлю|соберу|пришлю|обновлю|проверил|проверила|"
    r"исправил|исправила|уже|в\s+процессе|я\s+сделал|мы\s+сделали|возьму\s+в\s+работу|"
    r"стабильно|ошибок\s+нет|готова|готов|для\s+этого|в\s+таком\s+случае|это\s+означает)\b",
    re.IGNORECASE,
)


SIMILARITY_PREFERENCE_THRESHOLD = 0.3


@lru_cache(maxsize=1)
def _load_qa_embedding_model() -> Any | None:
    try:
        from sentence_transformers import SentenceTransformer

        from pm_insights import settings

        return SentenceTransformer(settings.TOPIC_EMBEDDING_MODEL)
    except Exception:
        return None


def _cosine_similarity(vec_a: Any, vec_b: Any) -> float:
    try:
        numerator = float(sum(float(a) * float(b) for a, b in zip(vec_a, vec_b, strict=False)))
        norm_a = sum(float(a) * float(a) for a in vec_a) ** 0.5
        norm_b = sum(float(b) * float(b) for b in vec_b) ** 0.5
        if not norm_a or not norm_b:
            return 0.0
        return numerator / (norm_a * norm_b)
    except Exception:
        return 0.0


def _question_answer_similarity(question: str, candidate: str) -> float:
    if not question or not candidate:
        return 0.0
    model = _load_qa_embedding_model()
    if model is None:
        return 0.0
    try:
        embeddings = model.encode([question, candidate], show_progress_bar=False)
        return _cosine_similarity(embeddings[0], embeddings[1])
    except Exception:
        return 0.0


def strip_speaker_prefix(text: str) -> str:
    return SPEAKER_PREFIX_RE.sub("", text or "").strip()


def _words_count(text: str) -> int:
    return len(re.findall(r"[А-Яа-яЁёA-Za-z0-9]+", text or ""))


def is_question(text: str) -> bool:
    clean = strip_speaker_prefix(text)
    if not clean:
        return False
    lower = clean.lower()
    if QUESTION_STOP_RE.search(lower):
        return False
    if "?" in clean:
        if _words_count(clean) < 3 and not QUESTION_START_RE.search(clean):
            return False
        if re.search(r"^[А-Яа-яЁёA-Za-z-]+\?\s*(ну\s+да|да|нет)\b", clean, re.IGNORECASE):
            return False
        return True
    return bool(QUESTION_START_RE.search(clean))


def is_answer(text: str) -> bool:
    clean = strip_speaker_prefix(text)
    return bool(ANSWER_RE.search(clean))


def split_question_spans(text: str) -> list[str]:
    clean = strip_speaker_prefix(text)
    if not is_question(clean):
        return []
    if clean.count("?") <= 1:
        return [clean]

    spans: list[str] = []
    start = 0
    for match in re.finditer(r"\?", clean):
        part = clean[start : match.end()].strip(" ,.;")
        start = match.end()
        if is_question(part):
            spans.append(part)
    tail = clean[start:].strip(" ,.;")
    if tail and is_question(tail):
        spans.append(tail)
    return spans or [clean]


def _is_plausible_answer(candidate_text: str) -> bool:
    clean = strip_speaker_prefix(candidate_text)
    if not clean or is_question(clean):
        return False
    if is_answer(clean):
        return True
    return _words_count(clean) >= 5 and not QUESTION_STOP_RE.search(clean)


MIN_QUESTION_WORDS = 5
LOW_CONFIDENCE_QUESTION_THRESHOLD = 0.55
SHORT_FRAGMENT_WORDS = 8


def _shares_leading_word(text_a: str, text_b: str) -> bool:
    words_a = text_a.split()
    words_b = text_b.split()
    return bool(words_a and words_b and words_a[0].lower() == words_b[0].lower())


def _likely_same_speaker_continuation(question_text: str, next_text: str) -> bool:
    if not next_text:
        return False
    if SPEAKER_PREFIX_RE.search(next_text) or is_answer(next_text):
        return False
    q_clean = strip_speaker_prefix(question_text).strip()
    n_clean = strip_speaker_prefix(next_text).strip()
    if not q_clean or not n_clean:
        return False
    if _shares_leading_word(q_clean, n_clean):
        return True
    return _words_count(q_clean) < SHORT_FRAGMENT_WORDS and _words_count(n_clean) < SHORT_FRAGMENT_WORDS


def _trim_answer(answer: str | None, limit: int = 1000) -> tuple[str | None, bool]:
    if not answer or len(answer) <= limit:
        return answer, False
    clipped = answer[:limit]
    sentence_end = max(clipped.rfind("."), clipped.rfind("?"), clipped.rfind("!"))
    if sentence_end > 150:
        return clipped[: sentence_end + 1].strip(), True
    return clipped.rstrip(" ,.;") + "...", True


def extract_qa_pairs(fragments: list[dict], max_gap: int = 5) -> list[dict]:
    pairs = []
    for index, fragment in enumerate(fragments):
        text = fragment.get("text", "")
        questions = split_question_spans(text)
        if not questions:
            continue

        next_text = fragments[index + 1].get("text", "") if index + 1 < len(fragments) else ""
        same_speaker_continuation = _likely_same_speaker_continuation(text, next_text)

        window: list[tuple[int, dict]] = []
        for offset in range(1, max_gap + 1):
            if index + offset >= len(fragments):
                break
            candidate = fragments[index + offset]
            if is_question(candidate.get("text", "")):
                break
            window.append((offset, candidate))

        plausible = [
            (offset, candidate) for offset, candidate in window if _is_plausible_answer(candidate.get("text", ""))
        ]

        start_at = None
        if plausible:
            start_at = plausible[0][0]
            best_score = _question_answer_similarity(text, plausible[0][1].get("text", ""))
            for offset, candidate in plausible[1:]:
                score = _question_answer_similarity(text, candidate.get("text", ""))
                if score > SIMILARITY_PREFERENCE_THRESHOLD and score > best_score:
                    start_at, best_score = offset, score

        answer_parts: list[str] = []
        answer_fragments: list[int] = []
        status = "not_answered"
        if start_at is not None:
            for offset, candidate in window:
                if offset < start_at:
                    continue
                candidate_text = candidate.get("text", "")
                if _is_plausible_answer(candidate_text):
                    answer_parts.append(candidate_text)
                    answer_fragments.append(candidate.get("fragment_index") or candidate.get("block_index"))
                    status = "answered" if is_answer(candidate_text) else "partial"
                    if is_answer(candidate_text):
                        continue
                    break
                elif answer_parts:
                    break

        answer_text = " ".join(part for part in answer_parts if part).strip() or None
        answer_text, was_trimmed = _trim_answer(answer_text)
        if was_trimmed and status == "answered":
            status = "partial"

        for question in questions:
            if _words_count(question) < MIN_QUESTION_WORDS:
                continue
            if same_speaker_continuation:
                continue
            confidence_signal = score_fragment_confidence(question, "question")
            if (confidence_signal.get("classifier_confidence") or 0.0) < LOW_CONFIDENCE_QUESTION_THRESHOLD and "?" not in question:
                continue
            pairs.append(
                {
                    "question": question,
                    "answer": answer_text,
                    "status": status,
                    "question_fragment": fragment.get("fragment_index") or fragment.get("block_index"),
                    "answer_fragment": answer_fragments[0] if answer_fragments else None,
                    "answer_fragments": [item for item in answer_fragments if item is not None],
                }
            )
    return pairs
