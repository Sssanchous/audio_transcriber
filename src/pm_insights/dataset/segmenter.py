from __future__ import annotations

import re

from .cleaner import normalize_text
from .reader import RawBlock


SENTENCE_RE = re.compile(r"(?<=[.!?…])\s+|\s*[;]\s+")


def split_long_text(text: str, max_words: int = 38) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []

    words = text.split()
    if len(words) <= max_words:
        return [text]

    parts = [normalize_text(part) for part in SENTENCE_RE.split(text) if normalize_text(part)]
    if len(parts) > 1:
        result: list[str] = []
        for part in parts:
            result.extend(split_long_text(part, max_words=max_words))
        return result

    return [" ".join(words[start : start + max_words]) for start in range(0, len(words), max_words)]


def segment_blocks(blocks: list[RawBlock], max_words: int = 38) -> list[dict]:
    fragments: list[dict] = []
    fragment_index = 0
    for block in blocks:
        for text in split_long_text(block.text, max_words=max_words):
            fragment_index += 1
            fragments.append(
                {
                    "source_file": block.source_file,
                    "paragraph_index": block.paragraph_index,
                    "fragment_index": fragment_index,
                    "text": text,
                    "block_type": block.block_type,
                }
            )
    return fragments
