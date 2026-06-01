from pm_insights.nlp.postprocessing import normalize_analysis_result


def frag(index, text):
    return {"fragment_index": index, "text": text, "start": float(index), "end": float(index + 1)}


def test_clean_task_normalizes_need_to_do_phrase():
    result = {
        "transcript": [frag(1, "надо в этом плане именно развертку сделать.")],
        "tasks": [{"text": "надо в этом плане именно развертку сделать.", "source_fragment": 1, "confidence": 0.8}],
        "questions_answers": [],
        "topics": [{"topic_name": "гидродинамика"}],
        "aspects": [{"aspects": ["гидродинамика"]}],
    }
    normalized = normalize_analysis_result(result)
    assert normalized["clean_tasks"][0]["title"].startswith("Сделать развертку")
    assert normalized["clean_tasks"][0]["review_required"] is True
    assert normalized["tasks"]


def test_clean_task_keeps_parameter_split_action_item():
    text = "Тогда из ближайших задач получается попробовать описанный вариант разделения на S, N, A, L параметры"
    normalized = normalize_analysis_result(
        {
            "transcript": [frag(1, text)],
            "tasks": [{"text": text, "source_fragment": 1, "confidence": 0.7}],
            "questions_answers": [],
            "topics": [{"topic_name": "параметры пласта"}],
            "aspects": [{"aspects": ["параметры пласта"]}],
        }
    )
    titles = [item["title"] for item in normalized["clean_tasks"]]
    assert "Проверить разделение исследования по группам параметров S, N, A, L" in titles


def test_clean_qa_finds_answer_in_following_window():
    result = {
        "transcript": [
            frag(1, "Илья, можно вязкость тоже сделать?"),
            frag(2, "Да, для нефтяного кейса это получается значение отдельное."),
            frag(3, "Если рассматриваем такой вариант, оно идет как параметр флюида."),
        ],
        "tasks": [],
        "questions_answers": [
            {
                "question": "Илья, можно вязкость тоже сделать?",
                "answer": None,
                "status": "not_answered",
                "question_fragment": 1,
            }
        ],
        "topics": [{"topic_name": "вязкость"}],
        "aspects": [{"aspects": ["вязкость"]}],
    }
    normalized = normalize_analysis_result(result)
    pair = normalized["clean_questions_answers"][0]
    assert pair["status"] == "answered"
    assert "нефтяного кейса" in pair["answer"]
    assert pair["source_fragments"] == [1, 2, 3]


def test_technical_summary_and_raw_fields_are_preserved():
    normalized = normalize_analysis_result(
        {
            "transcript": [frag(1, "гидродинамика, дебит, скважина, МГРП, параметры")],
            "tasks": [{"text": "надо в этом плане именно развертку сделать.", "source_fragment": 1}],
            "questions_answers": [],
            "topics": [{"topic_name": "гидродинамика"}, {"topic_name": "ВКР"}],
            "aspects": [{"aspects": ["гидродинамика", "ВКР"]}],
        }
    )
    assert normalized["analysis_summary"]["meeting_type"] == "technical_research"
    assert normalized["analysis_summary"]["requires_manual_review"] is True
    assert normalized["tasks"]
    assert normalized["clean_tasks"]


def test_clean_schema_adds_review_and_quality_fields():
    normalized = normalize_analysis_result(
        {
            "transcript": [frag(1, "Нужно скинуть промежуточные результаты к следующей встрече")],
            "semantic_blocks": [frag(1, "Нужно скинуть промежуточные результаты к следующей встрече")],
            "tasks": [{"text": "Нужно скинуть промежуточные результаты к следующей встрече", "source_fragment": 1}],
            "questions_answers": [],
            "deadlines": [
                {
                    "text": "Нужно скинуть промежуточные результаты к следующей встрече",
                    "deadlines": ["к следующей встрече"],
                    "kind": "task_deadline",
                    "source_fragment": 1,
                }
            ],
            "responsibles": [],
            "decisions": [],
            "topics": [],
            "aspects": [],
        }
    )
    assert "clean_deadlines" in normalized
    assert "clean_responsibles" in normalized
    assert "review_items" in normalized
    assert normalized["quality_metrics"]["clean_tasks_count"] == len(normalized["clean_tasks"])


def test_commercial_oil_gas_postprocessing_compacts_and_filters_tasks():
    result = {
        "transcript": [
            frag(1, "На июль подтверждаем 180 тысяч тонн. Премия по первой партии 1,4 доллара за баррель."),
            frag(2, "Мы можем прислать последние лабораторные данные отдельно."),
            frag(3, "Тогда по платежам нужно прописать точные даты."),
            frag(4, "Документы можем отправить сегодня-завтра."),
            frag(5, "А период ценообразования какой вы хотите?"),
            frag(6, "Мы предпочитаем среднее за 5 котировочных дней вокруг коносамента. Это приемлемо."),
        ],
        "tasks": [
            {"text": "Мы можем прислать последние лабораторные данные отдельно.", "source_fragment": 2},
            {"text": "Тогда по платежам нужно прописать точные даты.", "source_fragment": 3},
            {"text": "Документы можем отправить сегодня-завтра.", "source_fragment": 4},
        ],
        "questions_answers": [
            {
                "question": "А период ценообразования какой вы хотите? Следующая фраза уже не вопрос.",
                "answer": "Мы предпочитаем среднее за 5 котировочных дней вокруг коносамента. Это приемлемо. Тогда хедж тоже проще поставить.",
                "status": "answered",
                "question_fragment": 5,
                "answer_fragment": 6,
            }
        ],
        "deadlines": [
            {
                "text": "Документы можем отправить сегодня-завтра.",
                "deadlines": ["сегодня-завтра"],
                "kind": "mention",
                "source_fragment": 4,
            }
        ],
        "agreements": [
            {
                "text": "Премия по первой партии 1,4 доллара за баррель.",
                "type": "commercial_term",
                "source_fragment": 1,
                "confidence": 0.72,
            }
        ],
        "topics": [],
        "aspects": [],
    }
    normalized = normalize_analysis_result(result)
    assert normalized["meeting_type"]["label"] == "commercial_oil_gas"
    titles = [item["title"] for item in normalized["clean_tasks"]]
    assert titles == ["Прописать точные даты платежей"]
    assert len(normalized["commitments"]) == 2
    assert normalized["clean_questions_answers"][0]["question_title"].endswith("?")
    assert len(normalized["clean_questions_answers"][0]["answer_summary"]) <= 330
    assert normalized["clean_deadlines"][0]["context"]
    assert normalized["clean_commercial_terms"][0]["category"] == "премия"


def test_universal_report_layer_adds_sections_and_display_config():
    result = {
        "transcript": [
            frag(1, "По контракту нужно прописать точные даты платежей."),
            frag(2, "Покупатель подтверждает оплату, поставщик направляет term sheet."),
        ],
        "tasks": [{"text": "По контракту нужно прописать точные даты платежей.", "source_fragment": 1}],
        "questions_answers": [
            {
                "question": "Когда покупатель подтверждает оплату?",
                "answer": "Покупатель подтверждает оплату после согласования term sheet.",
                "status": "answered",
                "question_fragment": 1,
                "answer_fragment": 2,
            }
        ],
        "deadlines": [
            {
                "text": "Покупатель подтверждает оплату в течение 24 часов.",
                "deadlines": ["в течение 24 часов"],
                "kind": "answer_deadline",
                "source_fragment": 2,
            }
        ],
        "agreements": [
            {
                "text": "Поставщик направляет term sheet, покупатель подтверждает оплату.",
                "type": "commercial_term",
                "source_fragment": 2,
                "confidence": 0.76,
            }
        ],
        "topics": [],
        "aspects": [],
    }
    normalized = normalize_analysis_result(result)
    assert normalized["meeting_type"]["label"] == "commercial_meeting"
    assert "report_sections" in normalized
    assert "display_config" in normalized
    assert normalized["clean_tasks"][0]["summary"]
    assert normalized["clean_questions_answers"][0]["question_title"].endswith("?")
    assert normalized["clean_deadlines"][0]["deadline"] == "в течение 24 часов"
    section_ids = [section["id"] for section in normalized["report_sections"]]
    assert section_ids.index("commercial_terms") < section_ids.index("tasks")
    assert normalized["display_config"]["transcript_preview_chars"] == 1600


def test_clean_output_filters_intro_weak_qa_and_frequency_deadlines():
    result = {
        "transcript": [
            frag(1, "Доброе утро, коллеги. У нас сегодня основная тема — поставка."),
            frag(2, "На июль фиксируем 180 тысяч тонн, три партии по 60 тысяч тонн."),
            frag(3, "Давайте запишем: инспекцию делим 50 на 50."),
            frag(4, "Насколько небольшой?"),
            frag(5, "Обновления по логистике два раза в неделю, за пять дней до отгрузки — ежедневный статус."),
        ],
        "tasks": [
            {"text": "Доброе утро, коллеги. У нас сегодня основная тема — поставка.", "source_fragment": 1},
        ],
        "questions_answers": [
            {
                "question": "Насколько небольшой?",
                "answer": "Небольшой, это обсуждается отдельно.",
                "status": "answered",
                "question_fragment": 4,
                "answer_fragment": 5,
            }
        ],
        "deadlines": [
            {
                "text": "Обновления по логистике два раза в неделю, за пять дней до отгрузки — ежедневный статус.",
                "deadlines": ["два раза в неделю", "за пять дней до отгрузки"],
                "kind": "mention",
                "source_fragment": 5,
            }
        ],
        "agreements": [
            {
                "text": "Доброе утро, коллеги. У нас сегодня основная тема — поставка.",
                "type": "agreement",
                "source_fragment": 1,
                "confidence": 0.8,
            },
            {
                "text": "На июль фиксируем 180 тысяч тонн, три партии по 60 тысяч тонн.",
                "type": "commercial_term",
                "source_fragment": 2,
                "confidence": 0.82,
            },
            {
                "text": "Давайте запишем: инспекцию делим 50 на 50.",
                "type": "agreement",
                "source_fragment": 3,
                "confidence": 0.82,
            },
        ],
        "topics": [],
        "aspects": [],
    }
    normalized = normalize_analysis_result(result)
    assert not normalized["clean_tasks"]
    assert all("Доброе утро" not in item.get("title", "") for item in normalized["clean_commercial_terms"])
    assert [item["title"] for item in normalized["clean_commercial_terms"]] == ["График поставки: три партии по 60 тыс. тонн"]
    assert any(item["title"] == "Инспекция: 50/50 по базовой инспекции" for item in normalized["clean_agreements"])
    assert normalized["clean_questions_answers"] == []
    deadline_values = [item["deadline"] for item in normalized["clean_deadlines"]]
    assert "два раза в неделю" not in deadline_values
    assert "за пять дней до отгрузки" in deadline_values
    assert any(item.get("frequency") == "два раза в неделю" for item in normalized["clean_commitments"])


def test_oil_gas_commercial_term_titles_are_specific():
    result = {
        "transcript": [
            frag(1, "Мы бы предлагали три партии по 60 тысяч тонн."),
            frag(2, "Тогда можем подтвердить минимальный августовский объем 150 тысяч тонн, а еще 70 тысяч тонн оставить опционам до 10 июля."),
            frag(3, "По премии корректировки примерно на 1,8 доллара за баррель."),
            frag(4, "По первой партии премия 1,4, вторая и третья плюс 1,2. Если считать по всем трем партиям средняя премия получится около 1,27."),
        ],
        "tasks": [],
        "questions_answers": [],
        "deadlines": [],
        "agreements": [
            {"text": "Мы бы предлагали три партии по 60 тысяч тонн.", "type": "commercial_term", "source_fragment": 1},
            {
                "text": "Тогда можем подтвердить минимальный августовский объем 150 тысяч тонн, а еще 70 тысяч тонн оставить опционам до 10 июля.",
                "type": "commercial_term",
                "source_fragment": 2,
            },
            {"text": "По премии корректировки примерно на 1,8 доллара за баррель.", "type": "commercial_term", "source_fragment": 3},
            {
                "text": "По первой партии премия 1,4, вторая и третья плюс 1,2. Если считать по всем трем партиям средняя премия получится около 1,27.",
                "type": "commercial_term",
                "source_fragment": 4,
            },
        ],
        "topics": [],
        "aspects": [],
    }
    normalized = normalize_analysis_result(result)
    titles = [item["title"] for item in normalized["clean_commercial_terms"]]
    assert "График поставки: три партии по 60 тыс. тонн" in titles
    assert "Августовский твердый объем: 150 тыс. тонн" in titles
    assert "Августовский опцион: 70 тыс. тонн до 10 июля" in titles
    assert "Исходное предложение по премии: +1,8 доллара/баррель" in titles
    assert "Премия первой партии: +1,4 доллара/баррель" in titles
    assert "Премия второй и третьей партии: +1,2 доллара/баррель" in titles
    assert "Средняя премия по трем партиям: около +1,27 доллара/баррель" in titles
    assert not normalized["clean_agreements"]


def test_oil_gas_source_mapping_uses_specific_august_fragment():
    result = {
        "transcript": [
            frag(1, "Доброе утро, коллеги. У нас сегодня основная тема – объемы на июль и август."),
            frag(2, "На июль можем подтвердить 180 тысяч тонн сырой нефти."),
            frag(3, "На август – ориентировочно 220 тысяч тонн, но есть зависимость от транспортного окна."),
        ],
        "tasks": [],
        "questions_answers": [],
        "deadlines": [],
        "agreements": [
            {
                "text": "Доброе утро, коллеги. У нас сегодня основная тема – объемы на июль и август. На июль можем подтвердить 180 тысяч тонн сырой нефти. На август – ориентировочно 220 тысяч тонн, но есть зависимость от транспортного окна.",
                "type": "commercial_term",
                "source_fragment": 1,
            }
        ],
        "topics": [],
        "aspects": [],
    }
    normalized = normalize_analysis_result(result)
    august = next(item for item in normalized["clean_commercial_terms"] if item["title"] == "Августовский ориентировочный объем: 220 тыс. тонн")
    assert "На август" in august["source_text"]
    assert "220 тысяч тонн" in august["source_text"]
    assert "На июль" not in august["source_text"]
    assert august["source_fragment"] == 3


def test_oil_gas_premium_terms_use_direct_split_source():
    result = {
        "transcript": [
            frag(
                7,
                "Премию тогда можно разделить. По первой партии премия 1,4, потому что она срочная. По второй и третьей – 1,2. Если считать по всем трем партиям средняя премия получится около 1,27.",
            )
        ],
        "tasks": [],
        "questions_answers": [],
        "deadlines": [],
        "agreements": [
            {
                "text": "Премию тогда можно разделить. По первой партии премия 1,4, потому что она срочная. По второй и третьей – 1,2. Если считать по всем трем партиям средняя премия получится около 1,27.",
                "type": "commercial_term",
                "source_fragment": 7,
            }
        ],
        "topics": [],
        "aspects": [],
    }
    normalized = normalize_analysis_result(result)
    terms = {item["title"]: item for item in normalized["clean_commercial_terms"]}
    first = terms["Премия первой партии: +1,4 доллара/баррель"]
    second = terms["Премия второй и третьей партии: +1,2 доллара/баррель"]
    for item in (first, second):
        assert "По первой партии премия 1,4" in item["source_text"]
        assert "По второй и третьей" in item["source_text"]
        assert item["source_fragment"] == 7


def test_oil_gas_premium_breakdown_qa_deduplicated_and_summarized():
    result = {
        "transcript": [],
        "tasks": [],
        "questions_answers": [
            {
                "question": "Можете разложить премию 1,8 доллара по фрахту, страховке и риску?",
                "answer": "Около 90 центов относится к фрахту, 30–40 центов — к страховке и портовым расходам, остальное — к риску задержек и доступности логистики.",
                "status": "answered",
                "question_fragment": 1,
                "answer_fragment": 2,
            },
            {
                "question": "Сколько там фрахт, сколько страховка и сколько риск по премии?",
                "answer": "Около 90 центов относится к фрахту, 30–40 центов — к страховке и портовым расходам, остальное — к риску задержек и доступности логистики.",
                "status": "answered",
                "question_fragment": 3,
                "answer_fragment": 4,
            },
        ],
        "deadlines": [],
        "agreements": [],
        "topics": [],
        "aspects": [],
    }
    normalized = normalize_analysis_result(result)
    premium_pairs = [item for item in normalized["clean_questions_answers"] if "премия 1,8" in item["question_title"]]
    assert len(premium_pairs) == 1
    assert premium_pairs[0]["question_title"] == "Как раскладывается премия 1,8 доллара по фрахту, страховке и риску?"
    assert premium_pairs[0]["answer_summary"] == "Около 90 центов относится к фрахту, 30–40 центов — к страховке и портовым расходам, остальное — к риску задержек и доступности логистики."


def test_oil_gas_deadline_keeps_single_specific_five_working_days_value():
    result = {
        "transcript": [],
        "tasks": [],
        "questions_answers": [],
        "deadlines": [
            {
                "text": "Номинация судна направляется минимум за 5 рабочих дней до отгрузки.",
                "deadlines": ["за 5 рабочих дней до", "минимум за 5 рабочих дней"],
                "kind": "task_deadline",
                "source_fragment": 9,
            }
        ],
        "agreements": [],
        "topics": [],
        "aspects": [],
    }
    normalized = normalize_analysis_result(result)
    values = [item["deadline"] for item in normalized["clean_deadlines"]]
    assert values == ["минимум за 5 рабочих дней до отгрузки"]


def test_technical_research_splits_actions_recommendations_and_notes():
    result = {
        "meeting_type": {"label": "technical_research", "confidence": 0.9},
        "transcript": [
            frag(1, "Метрокубический разделить на сутки – это дебит."),
            frag(2, "Это необходимо сделать."),
            frag(3, "Проверить устойчивость модели и точность."),
            frag(4, "Лучше проверить более частные версии модели."),
            frag(5, "Сделать развертку по параметрам/формуле безразмеривания."),
            frag(6, "Я попробую поискать кейсы."),
        ],
        "tasks": [
            {"text": "Метрокубический разделить на сутки – это дебит.", "source_fragment": 1, "confidence": 0.75},
            {"text": "Это необходимо сделать.", "source_fragment": 2, "confidence": 0.75},
            {"text": "Проверить устойчивость модели и точность.", "source_fragment": 3, "confidence": 0.82},
            {"text": "Лучше проверить более частные версии модели.", "source_fragment": 4, "confidence": 0.74},
            {"text": "Сделать развертку по параметрам/формуле безразмеривания.", "source_fragment": 5, "confidence": 0.82},
            {"text": "Я попробую поискать кейсы.", "source_fragment": 6, "confidence": 0.72},
        ],
        "questions_answers": [],
        "deadlines": [],
        "topics": [{"topic_name": "модель"}],
        "aspects": [{"aspects": ["модель"]}],
    }
    normalized = normalize_analysis_result(result)
    action_titles = [item["title"] for item in normalized["clean_research_actions"]]
    recommendation_titles = [item["title"] for item in normalized["clean_recommendations"]]
    note_titles = [item["title"] for item in normalized["clean_research_notes"]]

    assert normalized["clean_tasks"] == normalized["clean_research_actions"]
    assert "Проверить устойчивость модели и точность" in action_titles
    assert "Сделать развертку по параметрам/формуле безразмеривания" in action_titles
    assert not any("Метрокубический" in title for title in action_titles)
    assert not any(title == "Это необходимо сделать" for title in action_titles)
    assert not any("попробую поискать" in title.lower() for title in action_titles)
    assert "Проверить более частные версии модели" in recommendation_titles
    assert any("Метрокубический" in title for title in note_titles)
    assert any(item["reason"] == "too_generic_for_research_action" for item in normalized["review_items"])
    assert any(item["reason"] == "speaker_intent_without_clear_deliverable" for item in normalized["review_items"])


def test_technical_report_sections_prioritize_research_protocol_blocks():
    result = {
        "meeting_type": {"label": "education_consultation", "confidence": 0.9},
        "transcript": [frag(1, "Проверить устойчивость модели и точность.")],
        "tasks": [{"text": "Проверить устойчивость модели и точность.", "source_fragment": 1, "confidence": 0.82}],
        "questions_answers": [],
        "deadlines": [],
        "topics": [{"topic_name": "ВКР"}],
        "aspects": [{"aspects": ["ВКР"]}],
    }
    normalized = normalize_analysis_result(result)
    section_ids = [section["id"] for section in normalized["report_sections"]]
    assert "research_actions" in section_ids
    assert "recommendations" in section_ids
    assert "research_notes" in section_ids
    assert section_ids.index("research_actions") < section_ids.index("qa")
    assert section_ids.index("recommendations") < section_ids.index("qa")
