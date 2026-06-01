from __future__ import annotations

import re
from collections import defaultdict


TECHNICAL_TERMS_RE = re.compile(
    r"\b(гидродинамик|дебит|скважин|трещин|мгрп|скин[-\s]?фактор|проницаем|"
    r"интерпретац|безразмерн|аппроксимац|калькулированн|эталонн|параметр|"
    r"вязкост|pvt|вкр|пласт|флюид|модель|генерализац)\b",
    re.IGNORECASE,
)

PM_ASPECT_KEYWORDS = {
    "сроки": ["срок", "дедлайн", "до пятницы", "до завтра", "к понедельнику", "до конца недели", "сегодня до", "к 12", "на следующей неделе"],
    "релиз": ["релиз", "релизная сборка", "сборка", "деплой", "выпуск"],
    "ресурсы": ["ресурсы", "нагрузка", "занятость", "команда", "люди"],
    "бюджет": ["бюджет", "оплата", "стоимость", "деньги"],
    "дизайн": ["дизайн", "макет", "ui", "ux", "интерфейс"],
    "сервер": ["сервер", "backend", "бэкенд", "api"],
    "frontend": ["frontend", "фронтенд"],
    "база данных": ["база данных", "postgres", "sql"],
    "документация": ["документация", "документ", "описание", "readme"],
    "тестирование": ["тестирование", "тесты", "проверка"],
    "интеграция": ["интеграция", "интегрировать"],
    "риски": ["риск", "риски", "блокер"],
    "проблемы": ["ошибк", "сбо", "не работает", "проблем"],
    "авторизация": ["авторизац", "логин"],
    "качество": ["качество", "корректно", "некорректно"],
    "коммуникация": ["коммуникация", "согласование", "обсудить"],
}

TECHNICAL_ASPECT_KEYWORDS = {
    "гидродинамика": ["гидродинамика", "гидродинамик"],
    "дебит": ["дебит"],
    "скважина": ["скважина", "скважины"],
    "трещины": ["трещина", "трещины"],
    "МГРП": ["мгрп"],
    "скин-фактор": ["скин-фактор", "скин фактор"],
    "параметры пласта": ["параметр", "пласт", "проницаемость"],
    "вязкость": ["вязкость"],
    "PVT": ["pvt"],
    "безразмерные кривые": ["безразмерные кривые", "безразмерн"],
    "интерпретация": ["интерпретация", "интерпретац"],
    "аппроксимация": ["аппроксимация", "аппроксимац"],
    "модель": ["модель", "модели", "генерализация"],
    "калькуляция": ["калькулированные", "калькуляция"],
    "эталонные данные": ["эталонные данные", "эталонн"],
    "промысловые данные": ["промысловые данные", "промыслов"],
    "ВКР": ["вкр"],
    "документация": ["документ", "страница", "обоснование"],
    "график встреч": ["встреча", "четверг", "будние дни", "после 7", "7.30"],
}

OIL_GAS_ASPECT_KEYWORDS = {
    "объемы поставки": ["объем", "тонн", "партия", "поставка"],
    "ценовая формула": ["brent", "брент", "dated", "формула", "премия", "дифференциал"],
    "логистика": ["терминал", "труба", "отгрузка", "окно", "маршрут"],
    "фрахт и демередж": ["фрахт", "демередж", "чартер", "сталийное время"],
    "качество сырья": ["качество", "сера", "плотность", "лаборатор", "сертификат"],
    "платежные условия": ["предоплата", "оплата", "отсрочка", "банковская гарантия", "аккредитив", "платеж"],
    "хеджирование": ["хедж", "фьючерс", "своп", "форвард", "кривая", "basis risk"],
    "комплаенс": ["судно", "страховка", "санкционные ограничения", "флаг", "порт"],
    "инспекция": ["инспектор", "sgs", "проба", "коносамент", "shore tank", "инспекция"],
}

ASPECT_KEYWORDS = {**PM_ASPECT_KEYWORDS, **TECHNICAL_ASPECT_KEYWORDS}


def is_technical_meeting(fragments: list[dict]) -> bool:
    joined = " ".join(fragment.get("text", "") for fragment in fragments[:120])
    return len(TECHNICAL_TERMS_RE.findall(joined)) >= 3


def is_oil_gas_meeting(fragments: list[dict]) -> bool:
    joined = " ".join(fragment.get("text", "") for fragment in fragments).lower()
    hits = 0
    for keywords in OIL_GAS_ASPECT_KEYWORDS.values():
        hits += sum(1 for keyword in keywords if keyword in joined)
    return hits >= 5


def _has_keyword(text: str, keyword: str) -> bool:
    keyword = keyword.lower()
    if " " in keyword or keyword in {"ui", "ux", "api", "sql", "pvt"}:
        return keyword in text
    return bool(re.search(rf"\b{re.escape(keyword)}\w*\b", text, flags=re.IGNORECASE))


def extract_aspects(fragments: list[dict]) -> list[dict]:
    result = []
    technical = is_technical_meeting(fragments)
    if is_oil_gas_meeting(fragments):
        keywords_map = OIL_GAS_ASPECT_KEYWORDS
    elif technical:
        keywords_map = TECHNICAL_ASPECT_KEYWORDS
    else:
        keywords_map = PM_ASPECT_KEYWORDS

    for fragment in fragments:
        text = fragment.get("text", "")
        lower = text.lower()
        found = [aspect for aspect, keywords in keywords_map.items() if any(_has_keyword(lower, keyword) for keyword in keywords)]
        if found:
            result.append({"text": text, "aspects": found, "source_fragment": fragment.get("fragment_index")})
    return result


def aspect_frequencies(aspect_items: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for item in aspect_items:
        for aspect in item.get("aspects", []):
            counts[aspect] += 1
    return dict(counts)
