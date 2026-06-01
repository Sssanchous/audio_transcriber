from pm_insights.nlp.semantic_blocks import build_semantic_blocks


def frag(index, text, start, end):
    return {"fragment_index": index, "text": text, "start": start, "end": end}


def test_short_fragments_are_merged_into_semantic_block():
    blocks = build_semantic_blocks(
        [
            frag(1, "То есть мы сначала проверяем параметры", 0.0, 2.0),
            frag(2, "и тогда смотрим аппроксимацию модели", 2.4, 4.0),
        ],
        "technical_research",
    )
    assert len(blocks) == 1
    assert blocks[0]["source_fragments"] == [1, 2]


def test_pause_breaks_semantic_block():
    blocks = build_semantic_blocks(
        [
            frag(1, "Обсудим параметры модели.", 0.0, 2.0),
            frag(2, "Нужно скинуть промежуточные результаты", 10.0, 12.0),
        ],
        "technical_research",
    )
    assert len(blocks) == 2


def test_question_block_gets_answer_context():
    blocks = build_semantic_blocks(
        [
            frag(1, "Илья, можно вязкость тоже сделать?", 0.0, 2.0),
            frag(2, "Да, для нефтяного кейса это отдельное значение.", 2.3, 5.0),
            frag(3, "Если рассматриваем газовый случай, данные идут через PVT.", 5.1, 8.0),
        ],
        "technical_research",
    )
    assert blocks[0]["has_question"] is True
    assert blocks[0]["answer_context_blocks"]


def test_long_blocks_do_not_exceed_limit():
    fragments = [frag(i, "слово " * 20, i * 0.5, i * 0.5 + 0.2) for i in range(1, 12)]
    blocks = build_semantic_blocks(fragments, "project_meeting")
    assert all(block["word_count"] <= 120 for block in blocks)
