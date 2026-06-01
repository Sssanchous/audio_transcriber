from pm_insights.dataset.reader import RawBlock
from pm_insights.dataset.segmenter import segment_blocks, split_long_text


def test_segmenter_splits_long_russian_text():
    text = "Нужно подготовить отчёт до пятницы. Когда будет готов макет? Решили оставить архитектуру."

    parts = split_long_text(text, max_words=4)

    assert len(parts) >= 3
    assert all(parts)


def test_segmenter_preserves_source_metadata():
    fragments = segment_blocks([RawBlock("a.docx", 3, "Нужно подготовить отчёт.")])

    assert fragments[0]["source_file"] == "a.docx"
    assert fragments[0]["paragraph_index"] == 3
