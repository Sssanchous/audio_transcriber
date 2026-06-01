from pm_insights.dataset.cleaner import clean_fragment, has_mojibake, normalize_text


def test_cleaner_normalizes_spaces_and_keeps_russian_text():
    text, reason = clean_fragment("  Нужно\t подготовить   отчёт.  ", min_length=5)

    assert reason is None
    assert text == "Нужно подготовить отчёт."


def test_cleaner_drops_service_and_mojibake():
    assert clean_fragment("Протокол встречи", min_length=5)[1] == "service"
    assert has_mojibake("Íó ëàäíî.")
    assert normalize_text(" а  б ") == "а б"
