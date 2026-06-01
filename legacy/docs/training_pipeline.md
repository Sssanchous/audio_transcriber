# Training pipeline

Текущий корпус PM Insights является MVP/seed dataset. Он нужен для проверки пайплайна, ручной доразметки и экспериментального baseline. Он не достаточен для полноценного RuBERT/BERT fine-tuning.

## Файлы

- `datasets/pm_dataset.jsonl` — основной auto-labeled seed dataset из `transcripts/`;
- `datasets/annotation_queue.jsonl` — очередь фрагментов для ручной разметки;
- `datasets/manual_seed_examples.jsonl` — небольшой ручной seed-набор для отсутствующих классов;
- `datasets/manual_labels.jsonl` — будущая ручная разметка;
- `datasets/pm_dataset_enriched.jsonl` — объединённый датасет;
- `datasets/enriched/train.jsonl`, `val.jsonl`, `test.jsonl` — split enriched dataset;
- `models/baseline_classifier/` — экспериментальная baseline-модель.

## Команды

```bash
python scripts/export_annotation_queue.py --input transcripts --dataset datasets/pm_dataset.jsonl --output datasets/annotation_queue.jsonl
python scripts/merge_datasets.py --base datasets/pm_dataset.jsonl --manual-seed datasets/manual_seed_examples.jsonl --manual-labels datasets/manual_labels.jsonl --output datasets/pm_dataset_enriched.jsonl
python scripts/split_dataset.py --input datasets/pm_dataset_enriched.jsonl --output-dir datasets/enriched --seed 42 --strategy seed
python scripts/check_dataset_readiness.py --input datasets/pm_dataset_enriched.jsonl --output datasets/enriched/readiness_report.json
python scripts/train_baseline_classifier.py --train datasets/enriched/train.jsonl --val datasets/enriched/val.jsonl --output models/baseline_classifier
```

## Baseline

Baseline использует `TfidfVectorizer` и `LogisticRegression`. Это не production-модель и не замена RuBERT. Метрики на маленьком validation split нестабильны и нужны только как smoke test ML-пайплайна.

## Следующий этап

Перед RuBERT fine-tuning нужно расширить и вручную проверить корпус: собрать сотни качественных примеров на каждый ключевой класс, сбалансировать классы и отдельно подготовить стабильный validation/test набор.
