from __future__ import annotations

import re

from .responsible_side import find_responsible_side


DECISION_RE = re.compile(
    r"\b(решили|договорились|принимаем\s+решение|фиксируем\s+решение|"
    r"выбрали\s+вариант|оставляем\s+вариант|переходим\s+к\s+варианту|согласовали|утвердили)\b",
    re.IGNORECASE,
)
TECHNICAL_DECISION_STOP_RE = re.compile(
    r"\b(математическое\s+решение|решение\s+задачи|решение\s+которое\s+получается|"
    r"решение\s+реализовано|решение\s+которое\s+у\s+вас\s+получается|классические\s+решения)\b",
    re.IGNORECASE,
)

AGREEMENT_RE = re.compile(
    r"\b(согласовано|договорились|фиксируем|зафиксируем|давайте\s+запишем|это\s+приемлемо|"
    r"это\s+можно\s+записать|как\s+рабочий\s+вариант|предварительно|подтверждаем|оставляем|"
    r"дели(?:м|ть)\s+50\s+на\s+50|цена\s+по\s+формуле|согласованный\s+обеими\s+сторонами)\b",
    re.IGNORECASE,
)
COMMERCIAL_TERM_RE = re.compile(
    r"\b(преми[яи]|дифференциал|опцион|котировочн\w+\s+дн|коносамент|демередж|"
    r"фрахт|чартер|инспекц|хедж|хеджирован|терминал|труба|партии?|тонн|баррель|"
    r"цено(?:вое|вой)\s+окно|банковская\s+гарантия|аккредитив|предоплата|"
    r"до\s+10\s+июля|25\s*[–-]\s*27\s+июля|25\s*[–-]\s*27\s+числ)\b",
    re.IGNORECASE,
)


def extract_decisions(fragments: list[dict]) -> list[dict]:
    result = []
    for fragment in fragments:
        text = fragment.get("text", "")
        if TECHNICAL_DECISION_STOP_RE.search(text):
            continue
        matches = [match.group(0) for match in DECISION_RE.finditer(text)]
        if matches:
            result.append(
                {
                    "text": text,
                    "source_fragment": fragment.get("fragment_index") or fragment.get("block_index"),
                    "matched_rules": sorted(set(matches)),
                    "confidence": 0.9,
                }
            )
    return result


def is_agreement(text: str) -> bool:
    text = text or ""
    return bool(AGREEMENT_RE.search(text) or COMMERCIAL_TERM_RE.search(text))


def extract_agreements(fragments: list[dict]) -> list[dict]:
    result = []
    seen = set()
    for fragment in fragments:
        text = fragment.get("text", "")
        if not text or TECHNICAL_DECISION_STOP_RE.search(text):
            continue
        marker_matches = [match.group(0) for match in AGREEMENT_RE.finditer(text)]
        term_matches = [match.group(0) for match in COMMERCIAL_TERM_RE.finditer(text)]
        if not marker_matches and not term_matches:
            continue
        key = re.sub(r"\s+", " ", text.lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        item_type = "commercial_term" if term_matches else "agreement"
        confidence = 0.82 if marker_matches else 0.72
        result.append(
            {
                "text": text,
                "type": item_type,
                "source_fragment": fragment.get("fragment_index") or fragment.get("block_index"),
                "matched_rules": sorted(set(marker_matches + term_matches)),
                "confidence": confidence,
                "responsible_side": find_responsible_side(text),
            }
        )
    return result
