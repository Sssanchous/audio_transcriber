from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pm_insights import settings

from .aspects import ASPECT_KEYWORDS, TECHNICAL_ASPECT_KEYWORDS, is_technical_meeting


STOPWORDS = {
    "это", "как", "что", "для", "или", "если", "так", "там", "тут", "вот", "уже",
    "надо", "нужно", "можно", "будет", "были", "было", "быть", "есть", "нет",
    "они", "она", "оно", "мы", "вы", "нам", "вам", "при", "про", "без",
    "именно", "собственно", "получается", "вообще", "потому", "почему", "вас",
    "меня", "нас", "ага", "раз", "пока", "его", "руках", "эти", "этих",
    "которые", "того", "тогда", "все", "соответственно", "случае", "можем",
    "возможно", "понимаю", "сейчас", "давайте", "более", "менее", "например",
    "этот", "эта", "который", "которая", "которое", "куда", "когда", "чтобы",
    "тоже", "даже", "очень", "просто", "хорошо", "вроде", "сразу", "будем",
    "будут", "нужна", "нужны", "должен", "должна", "должны", "типа", "наверное",
    "здесь", "кто", "где", "чем", "либо", "после", "перед", "дня", "дней",
    "день", "часов", "часа", "часть", "части", "какой", "какая", "какие",
    "том", "был", "была", "могут", "может", "могу", "плане", "другое",
    "остальные", "далее", "основных", "также", "еще", "одна", "правда",
    "использовать", "сделать", "проверить", "подумать", "хотелось", "всей",
    "что-нибудь", "какие-то", "что-то", "правило", "сказали", "спрашиваю",
    "принципе", "илья", "отталкиваться", "время",
    "ну", "да", "ага", "как-бы", "значит", "короче", "условно", "говоря",
    "какой-то", "какая-то", "кто-то", "где-то", "зачем-то", "почему-то",
    "чего-то", "как-нибудь", "чуть-чуть", "хотел", "хотели", "посмотреть",
    "спросить", "сказал", "понятно", "анна", "иван", "мария", "алексей",
    "юрий", "владислав", "сергей", "олег", "дмитрий", "екатерина",
    "через", "тот", "такие", "пользователю", "пользователя",
    "опять", "никакая", "никакой", "нужен", "нужна", "справедливее",
    "действительно", "важен", "важна", "важно", "описано", "пути",
    "вчера", "обсуждали", "базовой", "документами",
    "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять", "десять",
    "the", "and", "for", "with", "that", "this",
}

SPELLING_NORMALIZATIONS = {
    "дебет": "дебит",
    "демиридж": "демередж",
    "демередж": "демередж",
    "коносомент": "коносамент",
    "руберт": "RuBERT",
    "rubert": "RuBERT",
    "бетропик": "BERTopic",
    "bitropic": "BERTopic",
    "вискозити": "вязкость",
    "pvt": "PVT",
}
GENERIC_SHORT_TOPIC_NAMES = {
    "вкр",
    "документация",
    "модель",
    "параметры",
    "параметры модели",
    "дебит",
    "скин-фактор",
    "аппроксимация",
    "интерпретация",
    "график встреч",
    "качество",
    "проблемы",
}

OIL_GAS_TOPIC_KEYWORDS = {
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

DOMAIN_TOPIC_LABELS = {
    "ценовая формула": {"брент", "brent", "премия", "дифференциал", "формула"},
    "релиз": {"релиз", "сборка", "сервер", "деплой", "frontend", "backend"},
    "параметры пласта": {"скин", "пласт", "проницаемость", "дебит", "скважина"},
    "платежные условия": {"оплата", "отсрочка", "гарантия", "аккредитив", "предоплата"},
}

DOMAIN_TEXT_LABELS = {
    "объемы поставки": {"тонн", "парт", "объем", "поставк", "июл", "август"},
    "премия и дифференциал": {"прем", "дифференциал", "доллар", "баррел", "минус", "брент", "brent"},
    "ценовая формула": {"dated", "котировоч", "коносамент", "ценов", "формул", "брент", "brent"},
    "платежные условия": {"оплат", "платеж", "банковск", "гарант", "предоплат", "отсроч", "аккредитив"},
    "сроки поставки и оплаты": {"рабоч", "контракт", "отгруз", "поставк", "дат", "календар", "срок", "дней"},
    "сроки и платежи": {"рабоч", "календар", "срок", "оплат", "платеж", "отсроч", "подписан"},
    "логистика и судно": {"судн", "продавец", "номинир", "коносамент", "отгруз", "терминал", "труб"},
    "качество и инспекция": {"качест", "инспектор", "инспекц", "проба", "sgs", "shore", "лаборатор", "сер"},
    "риски и хеджирование": {"риск", "хедж", "страхов", "расход", "задерж", "фьючерс", "своп"},
    "фрахт и демередж": {"фрахт", "демередж", "чартер", "сталийн"},
    "дебит и динамика": {"дебит", "дебет", "динамик", "метрокуб", "сутк", "давлен"},
    "интерпретация данных": {"интерпретац", "данн", "эталон", "калькулирован", "аппроксим", "интерпол"},
    "параметры пласта": {"пласт", "проницаем", "скваж", "гидродинами", "параметр"},
    "скин-фактор и трещины": {"скин", "скин-фактор", "трещин", "мгрп"},
    "модель и аппроксимация": {"модель", "аппроксим", "интерпол", "r-квадрат", "r2", "точност"},
    "безразмерные кривые": {"безразмер", "крив", "формул", "коэффициент"},
    "учебная встреча": {"семестр", "учебн", "страниц", "документ", "вкр"},
    "организация следующей встречи": {"вторник", "четверг", "свободн", "встреч", "пары", "следующ"},
}

TOPIC_CANDIDATE_LABELS = {
    "задачи и поручения": {"задач", "поруч", "сделать", "подготов", "исполн", "action"},
    "сроки": {"срок", "дедлайн", "пятниц", "завтра", "недел", "четверг", "дата"},
    "ответственные": {"ответствен", "исполнитель", "кто", "назнач"},
    "риски": {"риск", "опасен", "проблем", "задерж", "огранич"},
    "проблемы": {"проблем", "ошибк", "сбой", "инцидент", "баг"},
    "статус работ": {"статус", "готов", "выполн", "открыт", "закрыт"},
    "план работ": {"план", "этап", "дорожн", "следующ", "спринт"},
    "бюджет": {"бюджет", "стоимост", "затрат", "финанс"},
    "ресурсы": {"ресурс", "команд", "нагрузк", "люди"},
    "клиент": {"клиент", "заказчик", "пользователь"},
    "презентация": {"презентац", "демо", "показ"},
    "договор": {"договор", "контракт", "term", "термшит"},
    "документация": {"документ", "документац", "отчет", "отчёт", "материал"},
    "релиз": {"релиз", "сборк", "деплой", "верси"},
    "тестирование": {"тест", "провер", "qa", "валидац"},
    "сервер": {"сервер", "backend", "api", "деплой"},
    "авторизация": {"авторизац", "auth", "login", "jwt", "доступ"},
    "аналитика": {"аналитик", "метрик", "дашборд", "отчет"},
    "интеграция": {"интеграц", "api", "webhook", "сервис"},
    "требования": {"требован", "спецификац", "критер"},
    "качество": {"качест", "проверка", "приемк"},
    "исправления": {"исправ", "фикс", "правк"},
    "модель": {"модель", "модел"},
    "параметры модели": {"параметр", "коэффициент", "формул"},
    "интерпретация данных": {"интерпретац", "данн", "эталон", "калькулирован"},
    "эксперимент": {"эксперимент", "вычислительн", "проверка"},
    "аппроксимация": {"аппроксим", "интерпол", "крив"},
    "точность модели": {"точност", "r-квадрат", "r2", "ошибк"},
    "устойчивость модели": {"устойчив", "стабильн", "чувствительн"},
    "данные и выборка": {"данн", "выборк", "датасет", "пример"},
    "расчетные параметры": {"расчет", "расчёт", "параметр", "дебит"},
    "алгоритм": {"алгоритм", "метод", "подход"},
    "архитектура системы": {"архитектур", "систем", "модул"},
    "производительность": {"производительн", "скорост", "время", "оптимизац"},
    "ограничения метода": {"огранич", "применим", "услов"},
    "метрики качества": {"метрик", "precision", "recall", "accuracy", "f1"},
    "формализация задачи": {"формализац", "постановк", "задач"},
    "математическая модель": {"математ", "уравнен", "формул", "модель"},
    "программная реализация": {"реализац", "код", "программ", "скрипт"},
    "сбор данных": {"сбор", "данн", "источник"},
    "безопасность": {"безопасн", "доступ", "санкцион", "комплаенс"},
    "надежность": {"надежн", "отказ", "стабильн"},
    "ВКР и документация": {"вкр", "диплом", "глава", "документ", "страниц"},
    "постановка задачи": {"постановк", "задач", "цель"},
    "материалы и методы": {"материал", "метод", "методик"},
    "критерии качества": {"критер", "качест", "оценк"},
    "ограничения работы": {"огранич", "работ", "рамк"},
    "глава ВКР": {"глава", "вкр", "раздел"},
    "защита работы": {"защит", "комисси", "доклад"},
    "требования комиссии": {"комисси", "требован", "защит"},
    "правки текста": {"правк", "текст", "черновик"},
    "научное обоснование": {"обоснован", "научн", "применим"},
    "консультация": {"консультац", "руководител", "встреч"},
    "коммерческие условия": {"услов", "коммерческ", "сделк"},
    "договоренности": {"договор", "согласов", "зафикс"},
    "обязательства сторон": {"обязательств", "сторон", "покупатель", "поставщик"},
    "платежные условия": {"оплат", "платеж", "предоплат", "отсроч"},
    "цена": {"цен", "стоимост", "прайс"},
    "скидка": {"скидк", "дисконт"},
    "премия": {"прем"},
    "дифференциал": {"дифференциал"},
    "объем поставки": {"объем", "тонн", "поставк"},
    "график поставки": {"график", "парт", "поставк", "отгруз"},
    "логистика": {"логист", "маршрут", "терминал", "отгруз"},
    "качество товара": {"качест", "товар", "сырь"},
    "инспекция": {"инспекц", "инспектор", "проба"},
    "комплаенс": {"комплаенс", "санкцион", "страхов"},
    "риски сделки": {"риск", "сделк", "задерж"},
    "опцион": {"опцион"},
    "банковская гарантия": {"банковск", "гарант"},
    "аккредитив": {"аккредитив"},
    "объемы поставки": {"объем", "тонн", "парт", "поставк"},
    "партии поставки": {"парт", "поставк", "окно"},
    "ценовая формула": {"формул", "brent", "брент", "dated", "ценов"},
    "Brent / Dated Brent": {"brent", "брент", "dated"},
    "премия и дифференциал": {"прем", "дифференциал", "баррел"},
    "фрахт и демередж": {"фрахт", "демередж", "чартер"},
    "качество сырья": {"качест", "сырь", "сера", "плотност"},
    "плотность и сера": {"плотност", "сера"},
    "инспекция и коносамент": {"инспекц", "коносамент", "проба"},
    "хеджирование": {"хедж", "фьючерс", "своп", "форвард"},
    "терминал и отгрузка": {"терминал", "отгруз", "труба"},
    "судно и номинация": {"судн", "номинац", "продавец"},
    "комплаенс и санкционные ограничения": {"комплаенс", "санкцион", "флаг", "порт"},
    "найм": {"найм", "нанять", "ваканси"},
    "кандидат": {"кандидат", "резюме"},
    "собеседование": {"собеседован", "интервью"},
    "адаптация": {"адаптац", "онбординг"},
    "команда": {"команд", "сотрудник"},
    "нагрузка": {"нагрузк", "перегруз"},
    "отпуск": {"отпуск"},
    "конфликт": {"конфликт"},
    "мотивация": {"мотивац"},
    "оценка сотрудника": {"оценк", "перформанс", "грейд"},
    "обучение сотрудников": {"обучен", "сотрудник", "персонал"},
    "инцидент": {"инцидент", "авар"},
    "ошибка": {"ошибк", "баг"},
    "сбой": {"сбой"},
    "логирование": {"лог", "логирован"},
    "восстановление сервиса": {"восстанов", "сервис"},
    "причина проблемы": {"причин", "проблем"},
    "workaround": {"workaround", "обходн"},
    "SLA": {"sla"},
    "обращение пользователя": {"обращен", "пользователь"},
    "приоритет инцидента": {"приоритет", "инцидент"},
    "стратегия": {"стратег"},
    "цели": {"цел"},
    "KPI": {"kpi"},
    "дорожная карта": {"дорожн", "карта"},
    "приоритеты": {"приоритет"},
    "рынок": {"рынок", "конкурент"},
    "развитие продукта": {"развит", "продукт"},
    "бизнес-метрики": {"бизнес", "метрик"},
    "управленческие решения": {"управлен", "решен"},
    "обращение": {"обращен"},
    "поздравление": {"поздрав"},
    "благодарность": {"благодар"},
    "информационное сообщение": {"информац", "сообщен"},
    "организационная информация": {"организац", "информац"},
}


def _fragment_id(fragment: dict, fallback: int) -> int:
    return int(fragment.get("fragment_index") or fragment.get("source_fragment") or fallback)


def _valid_fragments(fragments: list[dict]) -> list[dict]:
    valid = []
    for index, fragment in enumerate(fragments, start=1):
        text = re.sub(r"\s+", " ", str(fragment.get("text") or "")).strip()
        if len(text) < 20 or len(text.split()) < 3:
            continue
        item = dict(fragment)
        item["text"] = text
        item["_topic_fragment_id"] = _fragment_id(fragment, index)
        valid.append(item)
    return valid


def _normalize_topic_token(word: str) -> str:
    normalized = re.sub(r"^[^A-Za-zА-Яа-яЁё0-9]+|[^A-Za-zА-Яа-яЁё0-9]+$", "", str(word or "").lower())
    return SPELLING_NORMALIZATIONS.get(normalized, normalized)


def _normalize_topic_text(text: str) -> str:
    clean = str(text or "")
    for wrong, right in SPELLING_NORMALIZATIONS.items():
        clean = re.sub(rf"\b{re.escape(wrong)}\b", right, clean, flags=re.IGNORECASE)
    return clean


def _keywords(texts: list[str], limit: int = 5) -> list[str]:
    counter: Counter[str] = Counter()
    for text in texts:
        for word in re.findall(r"[A-Za-zА-Яа-яЁё0-9][A-Za-zА-Яа-яЁё0-9-]{2,}", _normalize_topic_text(text).lower()):
            word = _normalize_topic_token(word)
            if word.lower() in STOPWORDS or word.isdigit():
                continue
            counter[word] += 1
    return [word for word, _ in counter.most_common(limit)]


def _is_meaningful_term(term: str | None) -> bool:
    words = [_normalize_topic_token(word) for word in re.findall(r"[A-Za-zА-Яа-яЁё0-9-]{2,}", str(term or "").lower())]
    words = [word for word in words if word]
    if not words:
        return False
    meaningful = [word for word in words if word.lower() not in STOPWORDS and not word.isdigit()]
    return len(meaningful) >= 1 and len(meaningful) >= max(1, len(words) // 2)


def _meaningful_terms(texts: list[str], limit: int = 8) -> list[str]:
    cleaned_texts = [_normalize_topic_text(text) for text in texts if str(text or "").strip()]
    if not cleaned_texts:
        return []
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore

        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b[A-Za-zА-Яа-яЁё0-9][A-Za-zА-Яа-яЁё0-9-]{2,}\b",
            stop_words=sorted(STOPWORDS),
            max_features=120,
        )
        matrix = vectorizer.fit_transform(cleaned_texts)
        scores = matrix.sum(axis=0).A1
        terms = vectorizer.get_feature_names_out()
        ranked = sorted(zip(terms, scores, strict=False), key=lambda item: item[1], reverse=True)
        result = []
        for term, _score in ranked:
            normalized = " ".join(_normalize_topic_token(part) for part in term.split())
            normalized = re.sub(r"\s+", " ", normalized).strip()
            if normalized and _is_meaningful_term(normalized) and normalized not in result:
                result.append(normalized)
            if len(result) >= limit:
                break
        if result:
            return result
    except Exception:
        pass
    return _keywords(cleaned_texts, limit)


def _candidate_label_from_texts(texts: list[str], keywords: list[str], *, min_hits: int = 2) -> str | None:
    joined_text = _normalize_topic_text(" ".join(texts)).lower()
    joined_tokens = set(_keywords(texts, 120)) | {str(keyword).lower() for keyword in keywords}
    best_label = None
    best_score = 0
    label_sources = (
        [TOPIC_CANDIDATE_LABELS, DOMAIN_TEXT_LABELS]
        if min_hits <= 1
        else [DOMAIN_TEXT_LABELS, TOPIC_CANDIDATE_LABELS]
    )
    for labels in label_sources:
        for label, markers in labels.items():
            hits = 0
            for marker in markers:
                marker_lower = marker.lower()
                if marker_lower in joined_tokens or marker_lower in joined_text:
                    hits += 1
            if hits > best_score:
                best_label = label
                best_score = hits
    return best_label if best_score >= min_hits else None


COMMERCIAL_TOPIC_EXACT_MAP = {
    "даты рабочих контракте": "сроки поставки и оплаты",
    "рабочих контракте отгрузки поставки даты": "сроки поставки и оплаты",
    "даты платежей": "платежные условия",
}
COMMERCIAL_TIMING_TOPIC_MARKERS = (
    "рабоч",
    "контракт",
    "отгруз",
    "поставк",
    "дат",
    "календар",
    "платеж",
    "платёж",
    "оплат",
    "срок",
)
COMMERCIAL_PAYMENT_TOPIC_MARKERS = (
    "платеж",
    "платёж",
    "оплат",
    "предоплат",
    "отсроч",
    "гарант",
    "аккредитив",
)


def _topic_key(text: str | None) -> str:
    return re.sub(r"[^a-zа-яё0-9]+", " ", str(text or "").lower().replace("ё", "е")).strip()


def _commercial_topic_label(name: str | None, keywords: list[str] | None = None, texts: list[str] | None = None) -> str | None:
    haystack = " ".join([name or "", " ".join(keywords or []), " ".join(texts or [])]).lower().replace("ё", "е")
    exact_key = _topic_key(name)
    if exact_key in COMMERCIAL_TOPIC_EXACT_MAP:
        return COMMERCIAL_TOPIC_EXACT_MAP[exact_key]
    if "за 5 рабочих" in haystack and "отгруз" in haystack:
        return "сроки поставки и оплаты"
    timing_hits = sum(1 for marker in COMMERCIAL_TIMING_TOPIC_MARKERS if marker in haystack)
    payment_hits = sum(1 for marker in COMMERCIAL_PAYMENT_TOPIC_MARKERS if marker in haystack)
    if payment_hits >= 2 and payment_hits > timing_hits:
        return "платежные условия"
    if timing_hits >= 3:
        return "сроки поставки и оплаты"
    return None


def _is_low_quality_topic_name(name: str | None, keywords: list[str] | None = None) -> bool:
    clean = re.sub(r"\s+", " ", str(name or "").strip().lower())
    if not clean or clean in {"тема 1", "кластер 1", "прочее", "обсуждаемая тема"}:
        return True
    words = re.findall(r"[A-Za-zА-Яа-яЁё0-9-]{2,}", clean)
    if not words:
        return True
    filler_hits = sum(1 for word in words if _normalize_topic_token(word).lower() in STOPWORDS)
    meaningful = [
        _normalize_topic_token(word)
        for word in words
        if _normalize_topic_token(word).lower() not in STOPWORDS and not _normalize_topic_token(word).isdigit()
    ]
    normalized_keywords = [
        _normalize_topic_token(keyword)
        for keyword in (keywords or [])
        if _normalize_topic_token(keyword).lower() not in STOPWORDS and not _normalize_topic_token(keyword).isdigit()
    ]
    if filler_hits >= max(1, len(words) - 1):
        return True
    if len(words) > 5:
        return True
    if len(meaningful) < 2 and len(normalized_keywords) < 2 and clean not in TOPIC_CANDIDATE_LABELS:
        return True
    if re.fullmatch(r"[а-яё]{3,}(?:\s+[а-яё]{3,})?", clean) and all(word in STOPWORDS for word in words):
        return True
    return False


def _looks_like_phrase_topic_name(name: str | None) -> bool:
    clean = str(name or "").strip().lower()
    words = re.findall(r"[A-Za-zА-Яа-яЁё0-9-]{2,}", clean)
    if len(words) > 3:
        return True
    phrase_fillers = {
        "опять",
        "вводится",
        "проводятся",
        "никакая",
        "никакой",
        "нужен",
        "нужна",
        "важен",
        "важно",
        "описано",
        "обсуждали",
        "справедливее",
        "оставляем",
        "обновлю",
        "финансовым",
        "можем",
        "будем",
        "будут",
        "надо",
        "нужно",
        "можно",
    }
    number_words = {"один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять", "десять"}
    if len(words) >= 2 and any(word in phrase_fillers for word in words):
        return True
    if len(words) >= 2 and any(word in number_words for word in words):
        return True
    if len(words) >= 2 and any(
        word.endswith(("ется", "ится", "ются", "емся", "аем", "яем", "ируем", "ивают", "ывает", "ивают"))
        for word in words
    ):
        return True
    if len(words) >= 2 and any(
        marker in clean
        for marker in (
            "завтра",
            "сегодня",
            "недел",
            "пятниц",
            "часам",
            "вечера",
            "авторизац",
            "ошиб",
            "сборк",
            "верс",
            "договор",
            "рискам",
            "клиент",
            "отчет",
            "отчёт",
            "аналитик",
            "финаль",
            "находится",
            "обновить",
            "обновлю",
            "раздел",
            "согласовать",
            "поставк",
            "отгруз",
            "парт",
            "коносамент",
            "котировоч",
            "календарн",
            "базовой",
            "терминал",
            "инспектор",
            "документами",
            "потолком",
            "тонн",
            "тысяч",
            "доллар",
            "платеж",
            "прем",
            "дифференциал",
            "данных",
            "параметр",
            "скваж",
            "трещин",
            "аппроксим",
            "интерпол",
            "генерализ",
            "кейс",
            "gds",
            "макродан",
            "момент",
            "вопрос",
        )
    ):
        return True
    if len(words) >= 3 and any(
        marker in clean
        for marker in (
            "сегодня",
            "завтра",
            "обновить",
            "отправить",
            "подготовить",
            "финальная версия",
            "раздел",
            "клиенту",
            "договора",
        )
    ):
        return True
    return False


def _keyword_generated_topic_name(keywords: list[str]) -> str:
    clean_keywords = []
    for keyword in keywords:
        normalized = " ".join(_normalize_topic_token(part) for part in str(keyword).strip().lower().split())
        normalized = re.sub(r"\s+", " ", normalized).strip()
        if normalized and _is_meaningful_term(normalized) and normalized not in clean_keywords:
            clean_keywords.append(normalized)
    if not clean_keywords:
        return "обсуждаемая тема"
    phrase = next((keyword for keyword in clean_keywords if " " in keyword and not _looks_like_phrase_topic_name(keyword)), None)
    if phrase:
        return phrase
    return " ".join(clean_keywords[:3])


def _topic_name_from_keywords(keywords: list[str], texts: list[str] | None = None, *, prefer_mapping: bool = False) -> str:
    clean_keywords = []
    for keyword in keywords:
        normalized = _normalize_topic_token(str(keyword).strip())
        if normalized and _is_meaningful_term(normalized) and not normalized.isdigit():
            clean_keywords.append(normalized)

    domain_label = _domain_label_from_texts(texts or [], clean_keywords) if texts and prefer_mapping else None
    if domain_label:
        return domain_label

    name = _keyword_generated_topic_name(clean_keywords)
    if _is_low_quality_topic_name(name, clean_keywords):
        return "обсуждаемая тема"
    return name


def _domain_label_from_texts(texts: list[str], keywords: list[str]) -> str | None:
    return _candidate_label_from_texts(texts, keywords, min_hits=2)


def _meaningful_keywords(texts: list[str], limit: int = 6) -> list[str]:
    return _meaningful_terms(texts, limit)


def _aspect_frequencies(topics: list[dict]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for topic in topics:
        if topic.get("topic_name"):
            counts[str(topic.get("topic_name"))] += int(
                topic.get("count") or len(topic.get("fragment_ids") or topic.get("fragments") or []) or 1
            )
    return dict(counts)


def _cosine_similarity(left: Any, right: Any) -> float:
    try:
        import numpy as np  # type: ignore

        left_array = np.asarray(left, dtype=float)
        right_array = np.asarray(right, dtype=float)
        denominator = float(np.linalg.norm(left_array) * np.linalg.norm(right_array))
        return float(np.dot(left_array, right_array) / denominator) if denominator else 0.0
    except Exception:
        return 0.0


def _semantic_label_from_embeddings(model: Any, texts: list[str]) -> tuple[str | None, float]:
    if not texts:
        return None, 0.0
    try:
        cluster_text = " ".join(texts[:8])[:2000]
        labels = list(TOPIC_CANDIDATE_LABELS.keys())
        descriptions = [
            f"{label}: {', '.join(sorted(TOPIC_CANDIDATE_LABELS[label])[:8])}"
            for label in labels
        ]
        vectors = model.encode([cluster_text, *descriptions], show_progress_bar=False)
        cluster_vector = vectors[0]
        best_label = None
        best_score = 0.0
        for label, vector in zip(labels, vectors[1:], strict=False):
            score = _cosine_similarity(cluster_vector, vector)
            if score > best_score:
                best_label = label
                best_score = score
        return (best_label, best_score) if best_score >= 0.55 else (None, best_score)
    except Exception:
        return None, 0.0


def _topic_quality_score(topic: dict) -> float:
    keywords = list(topic.get("keywords") or [])
    keyword_score = min(1.0, len(keywords) / 5)
    count_score = min(1.0, int(topic.get("count") or 1) / 4)
    confidence = float(topic.get("confidence", 0.5) or 0.5)
    penalty = 0.45 if _is_low_quality_topic_name(topic.get("topic_name"), keywords) else 0.0
    return round(max(0.0, confidence * 0.5 + keyword_score * 0.3 + count_score * 0.2 - penalty), 3)


def _finalize_topics(topics: list[dict], max_topics: int) -> list[dict]:
    merged: dict[str, dict] = {}
    for topic in topics:
        generated_keywords = _meaningful_keywords([topic.get("source_text") or " ".join(topic.get("texts") or [])], 8)
        keywords = []
        for keyword in [*generated_keywords, *list(topic.get("keywords") or [])]:
            normalized = _normalize_topic_token(keyword)
            if normalized and normalized.lower() not in STOPWORDS and not normalized.isdigit() and normalized not in keywords:
                keywords.append(normalized)
            if len(keywords) >= 8:
                break
        name = str(topic.get("topic_name") or "").strip()
        texts = topic.get("texts") or [topic.get("source_text") or ""]
        semantic_label = topic.get("semantic_label")
        semantic_score = float(topic.get("semantic_label_score", 0.0) or 0.0)
        candidate_name = semantic_label if semantic_label and semantic_score >= 0.55 else None
        commercial_name = _commercial_topic_label(name, keywords, texts)
        mapping_name = _candidate_label_from_texts(texts, keywords, min_hits=2)
        if not mapping_name and _looks_like_phrase_topic_name(name):
            mapping_name = _candidate_label_from_texts(texts, keywords, min_hits=1)
        if int(topic.get("count") or 1) <= 1 and not candidate_name and name.lower() not in TOPIC_CANDIDATE_LABELS:
            continue
        if commercial_name:
            name = commercial_name
        elif candidate_name and (
            name.lower() in GENERIC_SHORT_TOPIC_NAMES
            or _is_low_quality_topic_name(name, keywords)
            or _looks_like_phrase_topic_name(name)
        ):
            name = candidate_name
        elif _is_low_quality_topic_name(name, keywords):
            name = mapping_name or _topic_name_from_keywords(keywords, texts)
        elif mapping_name and (name.lower() in GENERIC_SHORT_TOPIC_NAMES or _looks_like_phrase_topic_name(name)):
            name = mapping_name
        elif _looks_like_phrase_topic_name(name):
            continue
        if _is_low_quality_topic_name(name, keywords):
            continue
        topic["topic_name"] = name
        topic["keywords"] = keywords
        topic["quality_score"] = _topic_quality_score(topic)
        if topic["quality_score"] < 0.25:
            continue
        key = name.lower()
        if key not in merged:
            merged[key] = topic
        else:
            existing = merged[key]
            existing["count"] = int(existing.get("count") or 0) + int(topic.get("count") or 1)
            existing["fragment_ids"] = sorted(set(existing.get("fragment_ids") or []) | set(topic.get("fragment_ids") or []))
            existing["fragments"] = existing["fragment_ids"]
            existing["keywords"] = list(dict.fromkeys(list(existing.get("keywords") or []) + keywords))[:8]
            existing["confidence"] = max(float(existing.get("confidence", 0.0) or 0.0), float(topic.get("confidence", 0.0) or 0.0))
            existing["quality_score"] = max(float(existing.get("quality_score", 0.0) or 0.0), float(topic.get("quality_score", 0.0) or 0.0))
    return sorted(
        merged.values(),
        key=lambda item: (int(item.get("count") or 0), float(item.get("confidence", 0.0) or 0.0), float(item.get("quality_score", 0.0) or 0.0)),
        reverse=True,
    )[:max_topics]


def _normalize_topic(topic: dict, source: str, topic_id: int) -> dict:
    fragment_ids = topic.get("fragment_ids") or topic.get("fragments") or []
    count = int(topic.get("count") or len(fragment_ids) or 1)
    source_text = topic.get("source_text") or " ".join(topic.get("texts") or [])
    return {
        "topic_id": topic.get("topic_id", topic_id),
        "topic_name": topic.get("topic_name") or topic.get("name") or f"topic_{topic_id}",
        "keywords": list(topic.get("keywords") or [])[:8],
        "count": count,
        "fragment_ids": fragment_ids,
        "fragments": fragment_ids,
        "confidence": float(topic.get("confidence", 0.55)),
        "source": source,
        "source_text": source_text,
        "texts": list(topic.get("texts") or []),
        "semantic_label": topic.get("semantic_label"),
        "semantic_label_score": float(topic.get("semantic_label_score", 0.0) or 0.0),
    }


def is_oil_gas_meeting(fragments: list[dict]) -> bool:
    joined = " ".join(fragment.get("text", "") for fragment in fragments).lower()
    hits = 0
    for keywords in OIL_GAS_TOPIC_KEYWORDS.values():
        hits += sum(1 for keyword in keywords if keyword in joined)
    return hits >= 5


def build_rule_based_topics(fragments: list[dict]) -> list[dict]:
    topic_id = 0
    topics = []
    if is_oil_gas_meeting(fragments):
        keywords_map = OIL_GAS_TOPIC_KEYWORDS
    elif is_technical_meeting(fragments):
        keywords_map = TECHNICAL_ASPECT_KEYWORDS
    else:
        keywords_map = ASPECT_KEYWORDS

    for topic_name, keywords in keywords_map.items():
        matched_fragments = []
        matched_texts = []
        for index, fragment in enumerate(fragments, start=1):
            lower = fragment.get("text", "").lower()
            if any(keyword.lower() in lower for keyword in keywords):
                matched_fragments.append(_fragment_id(fragment, index))
                matched_texts.append(fragment.get("text", ""))
        if matched_fragments:
            topics.append(
                {
                    "topic_id": topic_id,
                    "topic_name": topic_name,
                    "keywords": _keywords(matched_texts, 5) or keywords[:5],
                    "count": len(matched_fragments),
                    "fragment_ids": matched_fragments,
                    "fragments": matched_fragments,
                    "confidence": 0.62,
                    "source": "rule_based_fallback",
                    "texts": matched_texts,
                    "source_text": " ".join(matched_texts),
                }
            )
            topic_id += 1

    if not topics:
        texts = [fragment.get("text", "") for fragment in _valid_fragments(fragments)]
        keywords = _keywords(texts, 8)
        if keywords:
            fallback_name = _candidate_label_from_texts(texts, keywords, min_hits=2) or _topic_name_from_keywords(keywords, texts)
            topics.append(
                {
                    "topic_id": 0,
                    "topic_name": fallback_name,
                    "keywords": keywords[:5],
                    "count": len(texts),
                    "fragment_ids": [_fragment_id(fragment, index) for index, fragment in enumerate(fragments, start=1) if fragment.get("text")],
                    "fragments": [_fragment_id(fragment, index) for index, fragment in enumerate(fragments, start=1) if fragment.get("text")],
                    "confidence": 0.4,
                    "source": "rule_based_fallback",
                    "texts": texts,
                    "source_text": " ".join(texts),
                }
            )
    return topics


def _fit_bertopic_topics(valid: list[dict], max_topics: int, allow_fit: bool = False, save_model: bool = False) -> list[dict]:
    if not settings.TOPIC_MODEL_PATH.exists() and not allow_fit:
        raise RuntimeError(f"BERTopic model is not found at {settings.TOPIC_MODEL_PATH}.")
    try:
        from bertopic import BERTopic  # type: ignore
        from sentence_transformers import SentenceTransformer  # type: ignore
        from hdbscan import HDBSCAN  # type: ignore
        from umap import UMAP  # type: ignore
    except Exception as exc:
        raise RuntimeError("BERTopic optional dependencies are not installed.") from exc

    texts = [fragment["text"] for fragment in valid]
    embedding_model = SentenceTransformer(settings.TOPIC_EMBEDDING_MODEL)
    if settings.TOPIC_MODEL_PATH.exists():
        model = BERTopic.load(str(settings.TOPIC_MODEL_PATH), embedding_model=embedding_model)
        topic_ids, _ = model.transform(texts)
    elif allow_fit:
        n_neighbors = max(2, min(15, len(texts) - 1))
        n_components = max(2, min(5, len(texts) - 2)) if len(texts) > 3 else 2
        min_cluster_size = max(2, min(settings.BERTOPIC_MIN_TOPIC_SIZE, max(2, len(texts) // 3)))
        model = BERTopic(
            language=getattr(settings, "BERTOPIC_LANGUAGE", "multilingual"),
            embedding_model=embedding_model,
            umap_model=UMAP(
                n_neighbors=n_neighbors,
                n_components=n_components,
                min_dist=0.0,
                metric="cosine",
                random_state=42,
            ),
            hdbscan_model=HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=1,
                metric="euclidean",
                cluster_selection_method="eom",
                prediction_data=True,
            ),
            min_topic_size=min_cluster_size,
        )
        topic_ids, _ = model.fit_transform(texts)
        if save_model:
            settings.TOPIC_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(settings.TOPIC_MODEL_PATH), serialization="pickle", save_embedding_model=False, save_ctfidf=True)

    grouped: dict[int, list[int]] = defaultdict(list)
    text_by_topic: dict[int, list[str]] = defaultdict(list)
    for index, topic_id in enumerate(topic_ids):
        if topic_id == -1:
            continue
        grouped[int(topic_id)].append(valid[index]["_topic_fragment_id"])
        text_by_topic[int(topic_id)].append(valid[index]["text"])

    topics = []
    for index, (topic_id, fragment_ids) in enumerate(sorted(grouped.items(), key=lambda pair: len(pair[1]), reverse=True)[:max_topics]):
        words = model.get_topic(topic_id) or []
        raw_keywords = [word for word, _ in words[:8]]
        keywords = _meaningful_keywords(text_by_topic[topic_id], 6)
        for keyword in raw_keywords:
            normalized = str(keyword).strip().lower()
            if normalized and normalized not in STOPWORDS and normalized not in keywords:
                keywords.append(normalized)
            if len(keywords) >= 6:
                break
        topics.append(
            _normalize_topic(
                {
                    "topic_id": topic_id,
                    "topic_name": _topic_name_from_keywords(keywords, text_by_topic[topic_id]),
                    "keywords": keywords,
                    "count": len(fragment_ids),
                    "fragment_ids": fragment_ids,
                    "confidence": 0.78,
                    "texts": text_by_topic[topic_id],
                    "source_text": " ".join(text_by_topic[topic_id]),
                },
                "bertopic",
                index,
            )
        )
    return topics


def _embedding_cluster_topics(valid: list[dict], max_topics: int) -> list[dict]:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        from sklearn.cluster import KMeans  # type: ignore
    except Exception as exc:
        raise RuntimeError("Embedding clustering optional dependencies are not installed.") from exc

    texts = [fragment["text"] for fragment in valid]
    model = SentenceTransformer(settings.TOPIC_EMBEDDING_MODEL)
    embeddings = model.encode(texts, show_progress_bar=False)
    clusters_count = max(1, min(max_topics, max(2, len(texts) // 4)))
    if clusters_count >= len(texts):
        clusters_count = max(1, len(texts) // 2)
    labels = KMeans(n_clusters=clusters_count, random_state=42, n_init="auto").fit_predict(embeddings)

    grouped: dict[int, list[dict]] = defaultdict(list)
    for label, fragment in zip(labels, valid, strict=False):
        grouped[int(label)].append(fragment)

    topics = []
    for topic_id, rows in sorted(grouped.items(), key=lambda pair: len(pair[1]), reverse=True)[:max_topics]:
        texts = [row["text"] for row in rows]
        keywords = _meaningful_terms(texts, 8)
        fragment_ids = [row["_topic_fragment_id"] for row in rows]
        semantic_label, semantic_score = _semantic_label_from_embeddings(model, texts)
        topics.append(
            _normalize_topic(
                {
                    "topic_id": topic_id,
                    "topic_name": _topic_name_from_keywords(keywords, texts),
                    "keywords": keywords,
                    "count": len(fragment_ids),
                    "fragment_ids": fragment_ids,
                    "confidence": max(0.68, semantic_score) if semantic_label else 0.68,
                    "semantic_label": semantic_label,
                    "semantic_label_score": semantic_score,
                    "texts": texts,
                    "source_text": " ".join(texts),
                },
                "embedding_clustering",
                len(topics),
            )
        )
    return topics


def extract_topics(
    fragments: list[dict],
    meeting_type: str | None = None,
    engine: str = "auto",
    max_topics: int = 12,
    allow_fit: bool = False,
    save_model: bool = False,
) -> dict[str, Any]:
    selected = (engine or settings.TOPIC_MODEL_ENGINE or "auto").lower()
    if selected == "fallback":
        selected = "rule_based"
    warnings: list[str] = []
    valid = _valid_fragments(fragments)
    min_fragments = max(1, settings.TOPIC_MIN_FRAGMENTS)

    def fallback(reason: str | None = None) -> dict[str, Any]:
        if reason:
            warnings.append(reason)
        topics = [
            _normalize_topic(topic, "rule_based_fallback", index)
            for index, topic in enumerate(build_rule_based_topics(fragments)[:max_topics])
        ]
        topics = _finalize_topics(topics, max_topics)
        return {
            "source": "rule_based_fallback",
            "topics": topics,
            "aspect_frequencies": _aspect_frequencies(topics),
            "warnings": warnings,
            "fragments_total": len(fragments),
            "fragments_used": len(valid),
        }

    if selected == "rule_based":
        return fallback()
    if len(valid) < min_fragments:
        return fallback(f"Not enough fragments for embedding topic modeling: {len(valid)} < {min_fragments}.")

    if selected in {"auto", "bertopic"}:
        try:
            topics = _finalize_topics(
                _fit_bertopic_topics(
                    valid,
                    max_topics=max_topics,
                    allow_fit=allow_fit,
                    save_model=save_model,
                ),
                max_topics,
            )
            if topics:
                result = {
                    "source": "bertopic",
                    "topics": topics,
                    "aspect_frequencies": _aspect_frequencies(topics),
                    "warnings": warnings,
                    "fragments_total": len(fragments),
                    "fragments_used": len(valid),
                }
                if save_model and settings.TOPIC_MODEL_PATH.exists():
                    result["model_saved_to"] = str(settings.TOPIC_MODEL_PATH)
                return result
            warnings.append("BERTopic returned no topics.")
        except Exception as exc:
            warnings.append(f"BERTopic unavailable, using fallback: {exc}")

    if selected in {"auto", "embedding", "bertopic"}:
        try:
            topics = _finalize_topics(_embedding_cluster_topics(valid, max_topics=max_topics), max_topics)
            if topics:
                return {
                    "source": "embedding_clustering",
                    "topics": topics,
                    "aspect_frequencies": _aspect_frequencies(topics),
                    "warnings": warnings,
                    "fragments_total": len(fragments),
                    "fragments_used": len(valid),
                }
            warnings.append("Embedding clustering returned no topics.")
        except Exception as exc:
            warnings.append(f"Embedding clustering unavailable, using rule_based_fallback: {exc}")

    return fallback()


def build_topics(fragments: list[dict], engine: str | None = None, allow_fit: bool = False) -> list[dict]:
    return extract_topics(
        fragments,
        engine=engine or settings.TOPIC_MODEL_ENGINE,
        max_topics=settings.TOPIC_MAX_TOPICS,
        allow_fit=allow_fit,
    )["topics"]
