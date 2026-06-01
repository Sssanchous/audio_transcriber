from pm_insights.nlp.meeting_type import detect_meeting_type


def test_pm_example_detected_as_project_meeting():
    result = detect_meeting_type(
        "Анна подготовь отчет до пятницы. Кто отвечает за релиз? Есть открытые задачи по клиенту."
    )
    assert result["label"] in {"project_meeting", "mixed"}


def test_technical_terms_detected_as_technical_research():
    result = detect_meeting_type(
        "Гидродинамика, дебит, скважина, МГРП, скин-фактор, параметры пласта, интерпретация и аппроксимация модели."
    )
    assert result["label"] in {"technical_research", "mixed"}


def test_vkr_terms_detected_as_education_or_mixed():
    result = detect_meeting_type(
        "ВКР, диплом, научный руководитель, глава, страница, обоснование и черновик исследования."
    )
    assert result["label"] in {"education_consultation", "technical_research", "mixed"}
