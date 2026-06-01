from __future__ import annotations

import re
from functools import lru_cache


NAME_RE = re.compile(r"\b[А-ЯЁ][а-яё]{2,}\b")
ORG_RE = re.compile(r"\b(?:ООО|АО|ИП|компания|отдел|команда)\s+[A-Za-zА-ЯЁа-яё0-9\"«» -]{2,40}", re.IGNORECASE)
DATE_RE = re.compile(r"\b\d{1,2}[.]\d{1,2}[.]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b")
PERSON_STOPWORDS = {"берёт", "берет", "нужно", "надо", "задача", "ответственный", "исполнитель"}


@lru_cache(maxsize=1)
def _natasha_objects():
    try:
        from natasha import DatesExtractor, Doc, MorphVocab, NamesExtractor, NewsEmbedding, NewsNERTagger, Segmenter
    except Exception:
        return None
    try:
        segmenter = Segmenter()
        morph = MorphVocab()
        emb = NewsEmbedding()
        ner = NewsNERTagger(emb)
        dates = DatesExtractor(morph)
        names = NamesExtractor(morph)
        return {
            "Doc": Doc,
            "segmenter": segmenter,
            "morph": morph,
            "ner": ner,
            "dates": dates,
            "names": names,
        }
    except Exception:
        return None


def extract_entities(text: str) -> dict:
    text = text or ""
    people = set(NAME_RE.findall(text))
    dates = set(DATE_RE.findall(text))
    organizations = set(match.group(0).strip() for match in ORG_RE.finditer(text))

    n = _natasha_objects()
    if n:
        try:
            for match in n["names"](text):
                fact = match.fact
                name = " ".join(part for part in [fact.first, fact.last] if part)
                if name and name.lower() not in PERSON_STOPWORDS:
                    people.add(name)
            for match in n["dates"](text):
                dates.add(match.text)
            doc = n["Doc"](text)
            doc.segment(n["segmenter"])
            doc.tag_ner(n["ner"])
            for span in doc.spans:
                if span.type == "PER":
                    if span.text.lower() not in PERSON_STOPWORDS:
                        people.add(span.text)
                elif span.type == "ORG":
                    organizations.add(span.text)
        except Exception:
            pass

    return {
        "people": sorted(person for person in people if person.lower() not in PERSON_STOPWORDS),
        "dates": sorted(dates),
        "organizations": sorted(organizations),
        "natasha_available": bool(n),
    }
