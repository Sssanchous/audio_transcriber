from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pm_insights import settings

from .aspects import ASPECT_KEYWORDS, TECHNICAL_ASPECT_KEYWORDS, is_technical_meeting


STOPWORDS = {
    "это", "как", "что", "для", "или", "если", "так", "там", "тут", "вот", "уже",
    "надо", "нужно", "можно", "будет", "были", "было", "быть", "есть", "нет",
    "они", "она", "оно", "мы", "вы", "нам", "вам", "при", "про", "без",
    "the", "and", "for", "with", "that", "this",
}

OIL_GAS_TOPIC_KEYWORDS = {
    "объемы поставки": ["объем", "тонн", "партия", "поставка"],
    "ценовая формула": ["brent", "брент", "dated", "формула", "премия", "дифференциал"],
    "логистика": ["терминал", "труба", "отгрузка", "окно", "маршрут"],
    "фрахт и демередж": ["фрахт", "демередж", "чартер", "сталийное время"],
    "качество сырья": ["качество", "сера", "плотность", "лаборатор", "сертификат"],
    "платежные условия": ["предоплата", "оплата", "отсрочка", "банковская гарантия", "аккредитив", "платеж"],
    "хеджирование": ["хедж", "фьючерс", "своп", "форвард", "кривая", "basis risk"],
    "комплаенс": ["судно", "страховка", "санкционные ограничения", "флаг", "порт"],
    "инспекция": ["инспектор", "sgs", "проба", "коносамент", "shore tank", "инспекция"],
}

DOMAIN_TOPIC_LABELS = {
    "ценовая формула": {"брент", "brent", "премия", "дифференциал", "формула"},
    "релиз": {"релиз", "сборка", "сервер", "деплой", "frontend", "backend"},
    "параметры пласта": {"скин", "пласт", "проницаемость", "дебит", "скважина"},
    "платежные условия": {"оплата", "отсрочка", "гарантия", "аккредитив", "предоплата"},
}


def _fragment_id(fragment: dict, fallback: int) -> int:
    return int(fragment.get("fragment_index") or fragment.get("source_fragment") or fallback)


def _valid_fragments(fragments: list[dict]) -> list[dict]:
    valid = []
    for index, fragment in enumerate(fragments, start=1):
        text = re.sub(r"\s+", " ", str(fragment.get("text") or "")).strip()
        if len(text) < 20 or len(text.split()) < 3:
            continue
        item = dict(fragment)
        item["text"] = text
        item["_topic_fragment_id"] = _fragment_id(fragment, index)
        valid.append(item)
    return valid


def _keywords(texts: list[str], limit: int = 5) -> list[str]:
    counter: Counter[str] = Counter()
    for text in texts:
        for word in re.findall(r"[A-Za-zА-Яа-яЁё0-9][A-Za-zА-Яа-яЁё0-9-]{2,}", text.lower()):
            if word in STOPWORDS or word.isdigit():
                continue
            counter[word] += 1
    return [word for word, _ in counter.most_common(limit)]


def _topic_name_from_keywords(keywords: list[str]) -> str:
    keyword_set = set(keywords)
    for label, markers in DOMAIN_TOPIC_LABELS.items():
        if len(keyword_set & markers) >= 2:
            return label
    return " ".join(keywords[:3]) if keywords else "обсуждаемая тема"


def _aspect_frequencies(topics: list[dict]) -> dict[str, int]:
    return {
        str(topic.get("topic_name")): int(topic.get("count") or len(topic.get("fragment_ids") or topic.get("fragments") or []) or 1)
        for topic in topics
        if topic.get("topic_name")
    }


def _normalize_topic(topic: dict, source: str, topic_id: int) -> dict:
    fragment_ids = topic.get("fragment_ids") or topic.get("fragments") or []
    count = int(topic.get("count") or len(fragment_ids) or 1)
    return {
        "topic_id": topic.get("topic_id", topic_id),
        "topic_name": topic.get("topic_name") or topic.get("name") or f"topic_{topic_id}",
        "keywords": list(topic.get("keywords") or [])[:8],
        "count": count,
        "fragment_ids": fragment_ids,
        "fragments": fragment_ids,
        "confidence": float(topic.get("confidence", 0.55)),
        "source": source,
    }


def is_oil_gas_meeting(fragments: list[dict]) -> bool:
    joined = " ".join(fragment.get("text", "") for fragment in fragments).lower()
    hits = 0
    for keywords in OIL_GAS_TOPIC_KEYWORDS.values():
        hits += sum(1 for keyword in keywords if keyword in joined)
    return hits >= 5


def build_rule_based_topics(fragments: list[dict]) -> list[dict]:
    topic_id = 0
    topics = []
    if is_oil_gas_meeting(fragments):
        keywords_map = OIL_GAS_TOPIC_KEYWORDS
    elif is_technical_meeting(fragments):
        keywords_map = TECHNICAL_ASPECT_KEYWORDS
    else:
        keywords_map = ASPECT_KEYWORDS

    for topic_name, keywords in keywords_map.items():
        matched_fragments = []
        matched_texts = []
        for index, fragment in enumerate(fragments, start=1):
            lower = fragment.get("text", "").lower()
            if any(keyword.lower() in lower for keyword in keywords):
                matched_fragments.append(_fragment_id(fragment, index))
                matched_texts.append(fragment.get("text", ""))
        if matched_fragments:
            topics.append(
                {
                    "topic_id": topic_id,
                    "topic_name": topic_name,
                    "keywords": _keywords(matched_texts, 5) or keywords[:5],
                    "count": len(matched_fragments),
                    "fragment_ids": matched_fragments,
                    "fragments": matched_fragments,
                    "confidence": 0.62,
                    "source": "rule_based_fallback",
                }
            )
            topic_id += 1

    if not topics:
        texts = [fragment.get("text", "") for fragment in _valid_fragments(fragments)]
        keywords = _keywords(texts, 8)
        if keywords:
            topics.append(
                {
                    "topic_id": 0,
                    "topic_name": _topic_name_from_keywords(keywords),
                    "keywords": keywords[:5],
                    "count": len(texts),
                    "fragment_ids": [_fragment_id(fragment, index) for index, fragment in enumerate(fragments, start=1) if fragment.get("text")],
                    "fragments": [_fragment_id(fragment, index) for index, fragment in enumerate(fragments, start=1) if fragment.get("text")],
                    "confidence": 0.4,
                    "source": "rule_based_fallback",
                }
            )
    return topics


def _fit_bertopic_topics(valid: list[dict], max_topics: int, allow_fit: bool = False) -> list[dict]:
    if not settings.TOPIC_MODEL_PATH.exists() and not allow_fit:
        raise RuntimeError(f"BERTopic model is not found at {settings.TOPIC_MODEL_PATH}.")
    try:
        from bertopic import BERTopic  # type: ignore
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as exc:
        raise RuntimeError("BERTopic optional dependencies are not installed.") from exc

    texts = [fragment["text"] for fragment in valid]
    embedding_model = SentenceTransformer(settings.TOPIC_EMBEDDING_MODEL)
    if settings.TOPIC_MODEL_PATH.exists():
        model = BERTopic.load(str(settings.TOPIC_MODEL_PATH), embedding_model=embedding_model)
        topic_ids, _ = model.transform(texts)
    elif allow_fit:
        model = BERTopic(
            language=getattr(settings, "BERTOPIC_LANGUAGE", "multilingual"),
            min_topic_size=max(2, min(settings.TOPIC_MIN_FRAGMENTS, max(2, len(texts) // 3))),
            embedding_model=embedding_model,
        )
        topic_ids, _ = model.fit_transform(texts)

    grouped: dict[int, list[int]] = defaultdict(list)
    text_by_topic: dict[int, list[str]] = defaultdict(list)
    for index, topic_id in enumerate(topic_ids):
        if topic_id == -1:
            continue
        grouped[int(topic_id)].append(valid[index]["_topic_fragment_id"])
        text_by_topic[int(topic_id)].append(valid[index]["text"])

    topics = []
    for index, (topic_id, fragment_ids) in enumerate(sorted(grouped.items(), key=lambda pair: len(pair[1]), reverse=True)[:max_topics]):
        words = model.get_topic(topic_id) or []
        keywords = [word for word, _ in words[:6]] or _keywords(text_by_topic[topic_id], 6)
        topics.append(
            _normalize_topic(
                {
                    "topic_id": topic_id,
                    "topic_name": _topic_name_from_keywords(keywords),
                    "keywords": keywords,
                    "count": len(fragment_ids),
                    "fragment_ids": fragment_ids,
                    "confidence": 0.78,
                },
                "bertopic",
                index,
            )
        )
    return topics


def _embedding_cluster_topics(valid: list[dict], max_topics: int) -> list[dict]:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        from sklearn.cluster import KMeans  # type: ignore
    except Exception as exc:
        raise RuntimeError("Embedding clustering optional dependencies are not installed.") from exc

    texts = [fragment["text"] for fragment in valid]
    model = SentenceTransformer(settings.TOPIC_EMBEDDING_MODEL)
    embeddings = model.encode(texts, show_progress_bar=False)
    clusters_count = max(1, min(max_topics, max(2, len(texts) // 4)))
    if clusters_count >= len(texts):
        clusters_count = max(1, len(texts) // 2)
    labels = KMeans(n_clusters=clusters_count, random_state=42, n_init="auto").fit_predict(embeddings)

    grouped: dict[int, list[dict]] = defaultdict(list)
    for label, fragment in zip(labels, valid, strict=False):
        grouped[int(label)].append(fragment)

    topics = []
    for topic_id, rows in sorted(grouped.items(), key=lambda pair: len(pair[1]), reverse=True)[:max_topics]:
        texts = [row["text"] for row in rows]
        keywords = _keywords(texts, 6)
        fragment_ids = [row["_topic_fragment_id"] for row in rows]
        topics.append(
            _normalize_topic(
                {
                    "topic_id": topic_id,
                    "topic_name": _topic_name_from_keywords(keywords),
                    "keywords": keywords,
                    "count": len(fragment_ids),
                    "fragment_ids": fragment_ids,
                    "confidence": 0.68,
                },
                "embedding_clustering",
                len(topics),
            )
        )
    return topics


def extract_topics(
    fragments: list[dict],
    meeting_type: str | None = None,
    engine: str = "auto",
    max_topics: int = 12,
    allow_fit: bool = False,
) -> dict[str, Any]:
    selected = (engine or settings.TOPIC_MODEL_ENGINE or "auto").lower()
    if selected == "fallback":
        selected = "rule_based"
    warnings: list[str] = []
    valid = _valid_fragments(fragments)
    min_fragments = max(1, settings.TOPIC_MIN_FRAGMENTS)

    def fallback(reason: str | None = None) -> dict[str, Any]:
        if reason:
            warnings.append(reason)
        topics = [_normalize_topic(topic, "rule_based_fallback", index) for index, topic in enumerate(build_rule_based_topics(fragments)[:max_topics])]
        return {
            "source": "rule_based_fallback",
            "topics": topics,
            "aspect_frequencies": _aspect_frequencies(topics),
            "warnings": warnings,
            "fragments_total": len(fragments),
            "fragments_used": len(valid),
        }

    if selected == "rule_based":
        return fallback()
    if len(valid) < min_fragments:
        return fallback(f"Not enough fragments for embedding topic modeling: {len(valid)} < {min_fragments}.")

    if selected in {"auto", "bertopic"}:
        try:
            topics = _fit_bertopic_topics(valid, max_topics=max_topics, allow_fit=allow_fit)
            if topics:
                return {
                    "source": "bertopic",
                    "topics": topics,
                    "aspect_frequencies": _aspect_frequencies(topics),
                    "warnings": warnings,
                    "fragments_total": len(fragments),
                    "fragments_used": len(valid),
                }
            warnings.append("BERTopic returned no topics.")
        except Exception as exc:
            warnings.append(f"BERTopic unavailable, using fallback: {exc}")

    if selected in {"auto", "embedding", "bertopic"}:
        try:
            topics = _embedding_cluster_topics(valid, max_topics=max_topics)
            if topics:
                return {
                    "source": "embedding_clustering",
                    "topics": topics,
                    "aspect_frequencies": _aspect_frequencies(topics),
                    "warnings": warnings,
                    "fragments_total": len(fragments),
                    "fragments_used": len(valid),
                }
            warnings.append("Embedding clustering returned no topics.")
        except Exception as exc:
            warnings.append(f"Embedding clustering unavailable, using rule_based_fallback: {exc}")

    return fallback()


def build_topics(fragments: list[dict], engine: str | None = None, allow_fit: bool = False) -> list[dict]:
    return extract_topics(
        fragments,
        engine=engine or settings.TOPIC_MODEL_ENGINE,
        max_topics=settings.TOPIC_MAX_TOPICS,
        allow_fit=allow_fit,
    )["topics"]
