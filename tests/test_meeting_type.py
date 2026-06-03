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


def test_vkr_terms_detected_as_education_consultation():
    result = detect_meeting_type(
        "ВКР, диплом, научный руководитель, глава, страница, обоснование и черновик исследования."
    )
    assert result["label"] in {"education_consultation", "technical_research"}


def test_vkr_consultation_markers_do_not_fall_back_to_mixed():
    result = detect_meeting_type(
        "На консультации по ВКР обсуждали черновик, страницы, семестр, пары и следующую встречу."
    )
    assert result["label"] in {"education_consultation", "technical_research"}


def test_commercial_oil_gas_detected_as_commercial():
    result = detect_meeting_type(
        "Покупатель и поставщик обсуждают Brent, премию, дифференциал, партию 60 тысяч тонн, отгрузку, коносамент, фрахт, демередж и судно."
    )
    assert result["label"] in {"commercial_oil_gas", "commercial_meeting"}


def test_non_meeting_speech_is_not_project_meeting():
    result = detect_meeting_type(
        "Дорогие коллеги, поздравляю вас с праздником. Сегодня хочу обратиться к команде и поблагодарить всех за работу."
    )
    assert result["label"] == "non_meeting_speech"
