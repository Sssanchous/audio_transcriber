from pm_insights.nlp.aspects import extract_aspects
from pm_insights.nlp.deadline_extractor import extract_deadlines, find_deadlines
from pm_insights.nlp.decision_extractor import extract_agreements, extract_decisions, is_agreement
from pm_insights.nlp.qa_extractor import extract_qa_pairs, is_answer, is_question, split_question_spans
from pm_insights.nlp.responsible_extractor import find_responsibles
from pm_insights.nlp.responsible_side import find_responsible_side
from pm_insights.nlp.sentiment import analyze_text_sentiment
from pm_insights.nlp.task_extractor import extract_tasks, is_task_fragment
from pm_insights.nlp.topic_modeling import build_rule_based_topics


def frag(text, index=1):
    return {"fragment_index": index, "text": text, "start": 0.0, "end": 4.0}


def test_task_responsible_and_deadline_rules():
    text = "Алексей, Анна, подготовь отчет до пятницы"
    tasks = extract_tasks([frag(text)])
    assert tasks
    assert tasks[0]["responsible"] == "Анна"
    assert "до пятницы" in tasks[0]["deadline"]


def test_questions_are_not_tasks():
    text = "Алексей. Нужно ли нам переносить релиз на следующую неделю?"
    assert is_question(text)
    assert not is_task_fragment(text)
    assert not extract_tasks([frag(text)])


def test_negative_answer_is_not_task():
    text = "Иван. Нет, релиз переносить не нужно"
    assert is_answer(text)
    assert not is_task_fragment(text)


def test_responsible_and_answer_from_self_reference():
    text = "Мария, за релизную сборку отвечаю я"
    assert find_responsibles(text) == ["Мария"]
    assert is_answer(text)


def test_budget_question_not_task():
    text = "Алексей, можно ли согласовать бюджет сегодня?"
    assert is_question(text)
    assert not is_task_fragment(text)


def test_problem_task_deadline_aspect_and_sentiment():
    text = "Алексей, Иван, исправь ошибку авторизации до завтра к 12 часам"
    tasks = extract_tasks([frag(text)])
    assert tasks
    assert tasks[0]["responsible"] == "Иван"
    assert tasks[0]["deadline"] == "до завтра к 12 часам"
    deadline = extract_deadlines([frag(text)])[0]
    assert deadline["kind"] == "task_deadline"
    aspects = extract_aspects([frag(text)])[0]["aspects"]
    assert "проблемы" in aspects
    assert "авторизация" in aspects
    assert analyze_text_sentiment(text)[0] == "negative"


def test_qa_answered_statuses():
    fragments = [
        frag("Когда мы отправим презентацию клиенту?", 1),
        frag("Анна, я отправлю презентацию сегодня после обеда", 2),
        frag("Кто отвечает за релизную сборку?", 3),
        frag("Мария, за релизную сборку отвечаю я", 4),
    ]
    pairs = extract_qa_pairs(fragments)
    assert [pair["status"] for pair in pairs] == ["answered", "answered"]


def test_release_is_not_deadline_aspect_by_itself():
    aspects = extract_aspects([frag("Кто отвечает за релизную сборку?")])[0]["aspects"]
    assert "релиз" in aspects
    assert "сроки" not in aspects


def test_mixed_sentiment_is_not_strong_negative():
    assert analyze_text_sentiment("Критических ошибок нет.")[0] in {"neutral", "positive"}
    assert analyze_text_sentiment("Был небольшой сбой, но сейчас все нормально.")[0] in {"neutral", "positive"}
    assert find_deadlines("Нужно сдать отчёт до пятницы.")


def test_technical_phrases_are_not_false_tasks():
    non_tasks = [
        "И, собственно, задача эксперта гидродинамика",
        "Для корректной интерпретации необходимо большое время",
        "Как правило, его задают последовательно по времени",
        "Это как гипотеза именно того, чтобы проверить",
    ]
    for text in non_tasks:
        assert not is_task_fragment(text, technical_mode=True)


def test_technical_action_items_are_tasks():
    tasks = [
        "Тогда из ближайших задач получается попробовать разделение на S, N, A, L параметры",
        "Надо в этом плане сделать развертку",
        "Давайте следующую встречу поставим на четверг в 7.30",
        "Нужно скинуть промежуточные результаты",
    ]
    for text in tasks:
        assert is_task_fragment(text, technical_mode=True)


def test_technical_false_questions_are_rejected():
    non_questions = [
        "Как правило, его задают последовательно по времени",
        "как в сетках",
        "Где меньше ошибку получим",
        "почему я спрашиваю",
    ]
    for text in non_questions:
        assert not is_question(text)


def test_technical_real_questions_are_detected():
    questions = [
        "Илья, можно вязкость тоже сделать?",
        "Хорошо, какие будние дни у вас свободные?",
        "Так, после 7, да, вы сказали?",
        "Я правильно понимаю, что нужно выделить группы параметров?",
    ]
    for text in questions:
        assert is_question(text)


def test_responsible_stopwords_are_never_people():
    for text in ["и", "а", "в", "по", "мы", "можем", "так", "уже"]:
        assert find_responsibles(text) == []
    assert find_responsibles("Илья, проверь расчеты") == ["Илья"]
    assert find_responsibles("Анна, подготовь отчет") == ["Анна"]
    assert find_responsibles("за релиз отвечает Мария") == ["Мария"]
    assert find_responsibles("Сидоров, подготовь расчеты", participants=["Сидоров Владислав Евгеньевич"]) == [
        "Сидоров Владислав Евгеньевич"
    ]
    assert find_responsibles("Я сделаю расчеты", participants=["Сидоров Владислав Евгеньевич"]) == []


def test_technical_aspects_and_topics_do_not_emit_it_false_positives():
    fragments = [
        frag("скин-фактор, МГРП, гидродинамика, дебит и скважина", 1),
        frag("Пользователь входит в программу", 2),
        frag("его нету во входных параметрах", 3),
    ]
    aspects = extract_aspects(fragments)
    flat = {aspect for item in aspects for aspect in item["aspects"]}
    assert {"скин-фактор", "МГРП", "гидродинамика", "дебит", "скважина"} & flat
    assert "frontend" not in flat
    assert "авторизация" not in flat
    topics = {topic["topic_name"] for topic in build_rule_based_topics(fragments)}
    assert "frontend" not in topics
    assert "авторизация" not in topics


def test_technical_solution_word_is_not_decision():
    assert not extract_decisions([frag("можно будет посмотреть решение которое у вас получается")])
    assert extract_decisions([frag("Договорились, оставляем этот вариант")])


def test_organizational_deadlines_not_physical_hours():
    assert find_deadlines("Давайте следующую встречу поставим на четверг в 7.30")
    assert find_deadlines("После 7 можно созвониться")
    assert not find_deadlines("модель считается 50 и более тысяч часов")


def test_oil_gas_modal_discussion_is_not_task():
    false_tasks = [
        "Можно сделать пять дней",
        "Нужно понимать график платежей",
        "Нам нужно чтобы риск был справедливо распределен",
        "Если на стороне независимого инспектора, нужно смотреть причину",
        "Ценовое окно — 5 котировочных дней вокруг коносамента",
    ]
    for text in false_tasks:
        assert not is_task_fragment(text)


def test_oil_gas_contract_actions_are_tasks():
    assert is_task_fragment("Тогда по платежам нужно прописать точные даты")
    assert is_task_fragment("Документы можем отправить сегодня-завтра")
    assert is_task_fragment("Покупатель должен подтвердить приемлемость судна в течение 24 часов")

    tasks = extract_tasks(
        [
            frag("Документы можем отправить сегодня-завтра", 1),
            frag("Покупатель должен подтвердить приемлемость судна в течение 24 часов", 2),
        ]
    )
    assert tasks[0]["deadline"] == "сегодня-завтра"
    assert tasks[1]["responsible_side"] == "покупатель"
    assert tasks[1]["deadline"] == "в течение 24 часов"


def test_oil_gas_agreements_and_sides():
    text = "Давайте делить 50 на 50 по базовой инспекции"
    assert not is_task_fragment(text)
    assert is_agreement(text)
    assert find_responsible_side(text) == "обе стороны"

    agreements = extract_agreements([frag(text), frag("Ценовое окно — 5 котировочных дней вокруг коносамента", 2)])
    assert len(agreements) == 2
    assert agreements[0]["responsible_side"] == "обе стороны"
    assert agreements[1]["type"] == "commercial_term"


def test_oil_gas_deadline_patterns():
    cases = [
        "До 10 июля нужно подтвердить августовский опцион",
        "Документы предоставляются за 5 рабочих дней до отгрузки",
        "Покупатель отвечает в течение 24 часов",
        "Оплата идет через 15 календарных дней",
        "Завтра до конца дня направим term sheet",
    ]
    for text in cases:
        assert find_deadlines(text), text


def test_qa_splits_multiple_questions_and_keeps_answer_short():
    questions = split_question_spans("Это подтвержденный объем или может измениться? Одной партией или несколькими?")
    assert questions == ["Это подтвержденный объем или может измениться?", "Одной партией или несколькими?"]

    pairs = extract_qa_pairs(
        [
            frag("Это подтвержденный объем или может измениться? Одной партией или несколькими?", 1),
            frag("Да, июльский объем подтвержден, разбивка идет тремя партиями по 60 тысяч тонн.", 2),
            frag("Какой потолок по демереджу?", 3),
        ]
    )
    assert len(pairs) == 3
    assert [pair["status"] for pair in pairs[:2]] == ["answered", "answered"]
    assert pairs[2]["status"] == "not_answered"
    assert all("Это подтвержденный объем" not in (pair["answer"] or "") for pair in pairs[:2])
