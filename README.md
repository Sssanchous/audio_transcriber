# PM Insights

Система анализирует уже существующие аудиозаписи проектных встреч. Приложение не записывает встречи, не включает микрофон и не подключается к онлайн-конференциям.

## Запуск

```powershell
.\.venv\Scripts\python.exe -m uvicorn app:app --reload
```

Frontend:

```powershell
cd frontend
npm.cmd run dev
```

Тесты:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Сборка датасета:

```powershell
.\.venv\Scripts\python.exe scripts\build_dataset.py --input transcripts --output datasets\pm_dataset.jsonl --format jsonl --min-length 10 --stats
```

Настройки берутся из `.env`. Для работы backend нужен PostgreSQL через `DATABASE_URL`; Whisper загружается локально и лениво при запуске анализа.
