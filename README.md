# PM Insights

Сервис для анализа аудиозаписей встреч на русском языке: транскрибация речи, выделение задач, тем и тональности по репликам.

## Запуск

### База данных

Нужен PostgreSQL. Создать базу и применить схему:

```
createdb audio_transcriber_db
psql audio_transcriber_db < schema.sql
```

Скопировать `.env.example` в `.env` и поправить `DATABASE_URL` и `JWT_SECRET_KEY`.

### Backend

```
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt -r requirements-ml.txt
uvicorn app:app --reload
```

Для асинхронной обработки (опционально, `ASYNC_PROCESSING=true` в `.env`) нужен Redis и Celery worker.

### Frontend

```
cd frontend
npm install
npm run dev
```

## Технологии

Backend: FastAPI, SQLAlchemy, PostgreSQL (psycopg), Celery + Redis, faster-whisper (ASR), natasha, scikit-learn, transformers / sentence-transformers, BERTopic.

Frontend: React, React Router, Vite, Tailwind CSS, Recharts, Axios.

Деплой: nginx (конфиг в `deploy/nginx`).

## Структура

- `src/pm_insights/` — backend: api, asr, audio, nlp, analytics, meeting, export
- `frontend/` — веб-интерфейс
- `scripts/` — обучение и оценка моделей, служебные скрипты
- `schema.sql` — схема БД (генерируется из моделей SQLAlchemy)
- `tests/` — тесты
