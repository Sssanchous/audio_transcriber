# Классы

Классы MVP:

- `task` — задача или поручение;
- `question` — вопрос;
- `answer` — ответ или пояснение;
- `decision` — принятое решение;
- `deadline` — срок;
- `responsible` — ответственный;
- `aspect` — аспект обсуждения;
- `sentiment_positive` — позитивная оценка;
- `sentiment_neutral` — нейтральный рабочий фрагмент;
- `sentiment_negative` — негативная оценка;
- `other` — прочий полезный фрагмент.

Приоритет: `question`, `decision`, `deadline`, `responsible`, `task`, `answer`, `sentiment_negative`, `sentiment_positive`, `aspect`, `sentiment_neutral`, `other`.

Если сработало несколько классов, основной класс выбирается по приоритету, остальные сохраняются в `secondary_labels`.

## Текущее покрытие данными

В актуальном `datasets/pm_dataset.jsonl` представлены не все классы. Класс `task` представлен лучше остальных. `question`, `aspect`, `sentiment_positive` можно использовать для начальных экспериментов. `decision` и `deadline` пока слабые. `answer`, `responsible`, `sentiment_negative`, `sentiment_neutral`, `other` отсутствуют в строгой сборке из `transcripts/`.

Причина: текущие `.docx` в основном похожи на протоколы и конспекты, а не на полные диалоги встречи. Для отсутствующих классов нужна ручная добавочная разметка реальных реплик.

`datasets/manual_seed_examples.jsonl` временно закрывает отсутствующие классы минимальными стартовыми примерами. Эти записи помечены как `manual_seed` и `verified=false`; они нужны для проверки pipeline, merge, split и baseline classifier, но не считаются финальной обучающей разметкой.
