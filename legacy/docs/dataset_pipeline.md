# DOCX dataset pipeline

Источник `.docx`: `transcripts/`.

Pipeline:

1. найти `.docx`;
2. прочитать абзацы и таблицы;
3. нормализовать русский текст;
4. удалить служебные заголовки, мусор, mojibake, короткие строки;
5. разбить длинные абзацы на фрагменты;
6. классифицировать rule-based;
7. добавить `secondary_labels`, `matched_rules`, `confidence`;
8. проверить обязательные поля;
9. сохранить JSONL или CSV;
10. вывести статистику labels и dropped reasons.

Команда:

```bash
python scripts/build_dataset.py --input transcripts --output datasets/pm_dataset.jsonl --format jsonl --min-length 10 --stats
```

Актуальный итоговый файл: `datasets/pm_dataset.jsonl`.
Статистический отчёт: `datasets/dataset_report.md`.
Split-файлы создаются командой:

```bash
python scripts/split_dataset.py --input datasets/pm_dataset.jsonl --output-dir datasets --seed 42
```

Текущий датасет является MVP/seed dataset для проверки pipeline и начальных экспериментов. Он не является достаточным для полноценного обучения ML/RuBERT-классификатора: всего 216 обучающих примеров, классы сильно несбалансированы, `task` доминирует, `responsible`, `answer`, `sentiment_negative`, `sentiment_neutral`, `other` отсутствуют, а `decision` и `deadline` представлены слабо.

Val/test split используется только для технической проверки seed dataset, а не для надёжной научной оценки качества модели.

## Рекомендации по расширению датасета

- `task`: 300-500+ примеров
- `question`: 300-500+ примеров
- `answer`: 300-500+ примеров
- `decision`: 200-300+ примеров
- `deadline`: 200-300+ примеров
- `responsible`: 200-300+ примеров
- `aspect`: 300-500+ примеров
- `sentiment_positive`: 300+ примеров
- `sentiment_negative`: 300+ примеров
- `sentiment_neutral`: 300+ примеров
- `other`: 300+ примеров

Для BERT/RuBERT fine-tuning желательно иметь хотя бы сотни качественно размеченных примеров на каждый класс. При текущем объёме лучше использовать rule-based MVP, ручную доразметку и weak supervision.

## Доразметка и enriched dataset

`datasets/pm_dataset.jsonl` остаётся основным auto-labeled seed dataset. Для подготовки к обучению добавлены:

- `datasets/annotation_queue.jsonl` — очередь кандидатов для ручной разметки;
- `datasets/manual_labels_template.jsonl` и `datasets/manual_labels_template.csv` — шаблоны ручной разметки;
- `datasets/manual_seed_examples.jsonl` — небольшой ручной seed-набор для отсутствующих классов, помеченный `source=manual_seed` и `verified=false`;
- `datasets/manual_labels.jsonl` — будущий файл ручной разметки;
- `datasets/pm_dataset_enriched.jsonl` — объединённый датасет из auto-labeled и manual seed/manual labels.

Manual seed нужен для проверки pipeline и baseline training. Он не заменяет настоящую ручную разметку и не делает датасет достаточным для полноценного ML.
