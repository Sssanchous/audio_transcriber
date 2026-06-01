# PM Insights — статус проекта

## Готово

- FastAPI backend, React frontend, PostgreSQL через `DATABASE_URL`.
- Auth/user isolation: регистрация, вход, JWT, `users`, `meetings.user_id`, per-user archive/result/dashboard.
- Upload flow: согласие на обработку, обязательное название встречи, проект, дата, участники и роли, `.mp3/.wav/.m4a`.
- Whisper large-v3 / faster-whisper: локальная русская транскрибация с таймкодами.
- Natasha optional-safe + rule-based NLP: задачи, вопросы/ответы, решения, дедлайны, ответственные, аспекты, тональность.
- Semantic blocks + clean analysis layer: `semantic_blocks`, `meeting_type`, `clean_tasks`, `clean_questions_answers`, `review_items`, `analysis_summary`.
- Baseline classifier на 4 класса: `task`, `question`, `answer`, `other`.
- RuBERT tiny classifier обучен и подключается optional-safe через `TASK_CLASSIFIER_ENGINE=rubert`; активный путь: `models/rubert_classifier`.
- Candidate-модели сравниваются отдельно и не продвигаются, если `recommendation: keep_current`.
- Optional `rubert-tiny2-sentiment` и optional BERTopic с fallback.
- Dashboard KPI, JSON export, KPI scripts.
- Светлая монохромная тема frontend для демонстрации.

## Participants and roles

- Пользователь указывает участников и роли при загрузке.
- Эти данные сохраняются в `metadata.meeting_info.participants`.
- Responsible extraction использует список участников как whitelist.
- Система не назначает ответственного по фразам вида `я сделаю`, если speaker неизвестен.

## Structured protocol dataset

- Старый датасет строился по фрагментам и эвристикам.
- Новый `datasets/sources/protocol_dataset.jsonl` строится из структуры протоколов в `transcripts/`.
- Разделы `До следующей встречи подготовить`, `Поставленные задачи`, `Поручения` дают точные `task`.
- Разделы `Принятые решения`, `Решили`, `Согласовано` дают `decision`.
- Разделы `На встрече обсуждались`, `Повестка`, `Ход встречи` дают `discussion_item`.
- `discussion_item` для текущей 4-классовой модели мапится в `other`, а не в `question`.
- `eval_data/protocol_references.json` используется как структурная reference-база.
- Активные итоговые файлы обучения: `training_dataset.jsonl`, `train.jsonl`, `val.jsonl`, `test.jsonl`, `dataset_stats.json`, `split_summary.json`.
- Source-файлы перенесены в `datasets/sources/`; старые review/tmp/report артефакты перенесены в `legacy/datasets/`.
- `scripts/audit_training_dataset.py` проверяет labels, дубли, mojibake, короткие и служебные строки; спорные случаи сохраняются в `dataset_label_review.jsonl`.
- Последний аудит: `3378` примеров после чистки, `22` очевидных label-fix, `462` спорных примера вынесены в review.

## Feedback learning

- Исправления пользователя сохраняются в `analysis_feedback`.
- После анализа модель не переобучается автоматически.
- Feedback экспортируется в `datasets/sources/feedback_examples.jsonl`.
- Дообучение запускается отдельной командой и сохраняет модель в `models/rubert_classifier_candidate`.
- Candidate сравнивается с текущей моделью; активная модель меняется только ручной promotion-командой.
- Это предотвращает деградацию модели на случайных или ошибочных данных.

## BERTopic

- BERTopic отвечает только за `topics`, `main_topics`, `topic_frequencies`.
- Он не заменяет задачи, Q/A, дедлайны, ответственных или sentiment.
- Если `TOPIC_MODELING_ENGINE=bertopic`, но зависимость или модель недоступны, используется `rule_based_fallback`.
- В `topics[].source` записывается `bertopic` или `rule_based_fallback`.

## KPI

- Demo WER/Task/Sentiment checks работают только как проверка скриптов.
- Official WER требует независимой экспертной расшифровки.
- Official Task Precision/Recall требует экспертной разметки задач.
- Official Sentiment Accuracy требует экспертной разметки тональности.
- Business usefulness 30–50% требует пилотного опроса.

## Требование ТЗ | Реализация | Статус

| Требование | Реализация | Статус |
|---|---|---|
| Загрузка MP3/WAV/M4A | `/upload`, frontend UploadPage | готово |
| Плашка согласия | UploadPage consent | готово |
| Участники и роли | обязательные upload metadata | готово |
| ASR Whisper large-v3 | faster-whisper adapter | готово |
| Задачи | rule-based + clean layer + review | готово |
| Вопросы/ответы | semantic blocks + QA linker | готово |
| Ответственные | participant whitelist + Natasha/rules | готово |
| Дедлайны | rule-based, `meeting_time` kind | готово |
| Sentiment | rule-based + optional RuBERT sentiment | частично |
| Аспекты | domain dictionaries | готово |
| BERTopic/fallback | optional BERTopic + fallback | optional |
| Динамика | grouped by `meeting_key` | частично |
| Dashboard | KPI + aggregates + groups | готово |
| Архив пользователя | DB-only per-user meetings | готово |
| JSON export | `/meetings/{meeting_id}/export/json` | готово |
| PostgreSQL | SQLAlchemy + `create_all` + safe columns | готово |
| Auth/user isolation | JWT + `REQUIRE_AUTH=true` | готово |
| Feedback learning | `analysis_feedback`, export, candidate scripts | готово |
| KPI scripts | WER/tasks/sentiment | требует экспертной разметки |

## Команды

```powershell
.\.venv\Scripts\python.exe -m uvicorn app:app --reload
cd frontend
npm.cmd run dev
```

```powershell
.\.venv\Scripts\python.exe scripts\build_protocol_dataset.py --input transcripts --output datasets\sources\protocol_dataset.jsonl --references-output eval_data\protocol_references.json --stats-output datasets\sources\protocol_dataset_stats.json
.\.venv\Scripts\python.exe scripts\prepare_training_dataset.py
.\.venv\Scripts\python.exe scripts\split_dataset.py --input datasets\training_dataset.jsonl --output-dir datasets --seed 42 --strategy seed
.\.venv\Scripts\python.exe scripts\export_feedback_dataset.py --output datasets\sources\feedback_examples.jsonl
.\.venv\Scripts\python.exe scripts\audit_training_dataset.py --input datasets\training_dataset.jsonl --output datasets\dataset_audit_report.json --review-output datasets\dataset_label_review.jsonl --fix-output datasets\training_dataset_fixed.jsonl
```

```powershell
.\.venv\Scripts\python.exe scripts\retrain_rubert_from_feedback.py --output models\rubert_classifier_candidate
.\.venv\Scripts\python.exe scripts\evaluate_model_candidate.py --current models\rubert_classifier --candidate models\rubert_classifier_candidate
.\.venv\Scripts\python.exe scripts\promote_rubert_candidate.py --candidate models\rubert_classifier_candidate --target models\rubert_classifier
```

## Осталось после MVP

- Накопить проверенный feedback.
- Разметить независимые expert references.
- Запустить официальную KPI validation.
- Дообучить RuBERT candidate после накопления достаточного feedback.
- Провести пилот бизнес-пользы.

## Dashboard / Project Summary

- Dashboard now follows the original ТЗ as `Сводка по проекту`.
- Dashboard uses a project-only selector.
- Main dashboard content is limited to summary cards, sentiment trend, task/action-item trend, and aspect word cloud.
- KPI/debug blocks are hidden from user UI.
- Archive remains the place for the per-meeting list, opening results, and report downloads.
- Dashboard aggregation uses clean/report-layer data where available.

## Exports / Reports

- JSON export: done.
- Excel `.xlsx` export: done.
- Word `.docx` export: done.
- PDF export: done.
- PDF/DOCX/XLSX exports preserve Cyrillic text via Unicode strings and a system TTF font for PDF.
- JSON endpoint is kept for API/dev use and hidden from the user interface.
- All report exports use the normalized clean/report layer.
- Export endpoints are protected by the same meeting owner lookup as ResultPage.
