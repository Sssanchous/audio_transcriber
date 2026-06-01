from pm_insights.nlp.qa_extractor import extract_qa_pairs


def test_qa_extractor_pairs_question_and_answer():
    fragments = [
        {"fragment_index": 1, "text": "Когда будет готов макет?"},
        {"fragment_index": 2, "text": "Да, я уже проверил первую версию."},
    ]

    pairs = extract_qa_pairs(fragments)

    assert pairs[0]["status"] == "answered"
    assert pairs[0]["answer_fragment"] == 2
