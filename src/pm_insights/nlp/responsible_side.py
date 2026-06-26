from __future__ import annotations

import re


SIDE_PATTERNS: dict[str, tuple[str, ...]] = {
    "поставщик": (
        "поставщик",
        "со стороны поставщика",
        "продавец",
        "продавца",
        "поставщик направляет",
        "продавец обязан",
    ),
    "покупатель": (
        "покупатель",
        "со стороны покупателя",
        "завод",
        "нпз",
        "покупатель должен",
        "покупатель направляет",
    ),
    "обе стороны": (
        "обе стороны",
        "стороны делят",
        "делим 50 на 50",
        "50 на 50",
        "согласованный обеими сторонами",
        "инспекцию делим",
    ),
    "каждая сторона": (
        "каждая сторона",
        "самостоятельно хеджирует",
        "каждая сторона самостоятельно",
    ),
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def find_responsible_side(text: str) -> str | None:
    clean = _normalize(text)
    if not clean:
        return None

    for side in ("каждая сторона", "обе стороны", "покупатель", "поставщик"):
        if any(pattern in clean for pattern in SIDE_PATTERNS[side]):
            return side
    return None
