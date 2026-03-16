from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from pydub import AudioSegment
from faster_whisper import WhisperModel
from dotenv import load_dotenv
from pathlib import Path
from psycopg.types.json import Json
import psycopg
import os
import shutil
import json
import html

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
SESSION_SECRET_KEY = os.getenv("SESSION_SECRET_KEY", "super-secret-session-key")

if not DATABASE_URL:
    raise RuntimeError("Не найден DATABASE_URL в файле .env")

app = FastAPI()

app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET_KEY)

UPLOAD_FOLDER = "uploads"
CONVERTED_FOLDER = "converted"
RESULTS_FOLDER = "results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CONVERTED_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.mount("/results", StaticFiles(directory=RESULTS_FOLDER), name="results")

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a"}

model = WhisperModel("small", device="cpu", compute_type="int8")


def get_connection():
    return psycopg.connect(DATABASE_URL)


def convert_to_wav(input_path: str, output_path: str) -> None:
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)
    audio.export(output_path, format="wav")


def transcribe_audio(audio_path: str) -> dict:
    segments, info = model.transcribe(audio_path, beam_size=5)

    segment_list = []
    full_text_parts = []

    for segment in segments:
        text = segment.text.strip()
        if text:
            segment_list.append({
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": text
            })
            full_text_parts.append(text)

    full_text = " ".join(full_text_parts)

    return {
        "language": info.language if info.language else "unknown",
        "duration": round(info.duration, 2) if info.duration else 0,
        "text": full_text,
        "segments": segment_list
    }


def save_results(filename: str, result_data: dict) -> tuple[str, str, str, str]:
    base_name = Path(filename).stem

    txt_filename = f"{base_name}.txt"
    json_filename = f"{base_name}.json"

    txt_path = os.path.join(RESULTS_FOLDER, txt_filename)
    json_path = os.path.join(RESULTS_FOLDER, json_filename)

    with open(txt_path, "w", encoding="utf-8") as txt_file:
        txt_file.write(result_data["text"])

    json_data = {
        "filename": filename,
        "language": result_data["language"],
        "duration": result_data["duration"],
        "text": result_data["text"],
        "segments": result_data["segments"]
    }

    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(json_data, json_file, ensure_ascii=False, indent=4)

    txt_url = f"/results/{txt_filename}"
    json_url = f"/results/{json_filename}"

    return txt_path, json_path, txt_url, json_url


def save_transcription_to_db(
    filename: str,
    original_path: str,
    converted_path: str,
    txt_path: str,
    json_path: str,
    result_data: dict
) -> None:
    query = """
    INSERT INTO transcriptions (
        filename,
        original_path,
        converted_path,
        txt_path,
        json_path,
        language,
        duration,
        full_text,
        segments
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                query,
                (
                    filename,
                    original_path,
                    converted_path,
                    txt_path,
                    json_path,
                    result_data["language"],
                    result_data["duration"],
                    result_data["text"],
                    Json(result_data["segments"])
                )
            )
        conn.commit()


def render_history(history: list) -> str:
    if not history:
        return "<p>История пока пуста.</p>"

    items = []
    for item in reversed(history):
        safe_filename = html.escape(item["filename"])
        safe_language = html.escape(item["language"])
        safe_txt_url = html.escape(item["txt_url"])
        safe_json_url = html.escape(item["json_url"])

        items.append(f"""
        <div class="history-item">
            <p><b>Файл:</b> {safe_filename}</p>
            <p><b>Язык:</b> {safe_language}</p>
            <p><b>Длительность:</b> {item["duration"]} сек.</p>
            <p>
                <a href="{safe_txt_url}" target="_blank">Открыть TXT</a> |
                <a href="{safe_json_url}" target="_blank">Открыть JSON</a>
            </p>
        </div>
        """)

    return "".join(items)


@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    history = request.session.get("history", [])
    history_html = render_history(history)

    return f"""
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <title>Загрузка аудиофайла</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                background-color: #f8f9fa;
            }}

            .container {{
                max-width: 900px;
                margin: 0 auto;
                background: white;
                padding: 30px;
                border-radius: 12px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            }}

            h2, h3 {{
                margin-bottom: 15px;
            }}

            input[type="file"] {{
                margin-bottom: 15px;
            }}

            button {{
                background-color: #007bff;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 16px;
            }}

            button:disabled {{
                background-color: #999;
                cursor: not-allowed;
            }}

            .spinner-box {{
                display: none;
                text-align: center;
                margin-top: 25px;
            }}

            .spinner {{
                width: 50px;
                height: 50px;
                border: 6px solid #ddd;
                border-top: 6px solid #007bff;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin: 0 auto 15px auto;
            }}

            .hint {{
                color: #555;
                margin-top: 10px;
            }}

            .history {{
                margin-top: 35px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
            }}

            .history-item {{
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 15px;
                background-color: #fafafa;
            }}

            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Загрузить аудиофайл</h2>

            <form id="uploadForm" action="/upload/" method="post" enctype="multipart/form-data">
                <input name="file" type="file" accept=".mp3,.wav,.m4a" required>
                <br>
                <button id="submitBtn" type="submit">Загрузить и распознать</button>
                <p class="hint">Поддерживаются форматы: MP3, WAV, M4A</p>
            </form>

            <div class="spinner-box" id="spinnerBox">
                <div class="spinner"></div>
                <p><b>Идёт обработка файла...</b></p>
                <p>Пожалуйста, подождите.</p>
            </div>

            <div class="history">
                <h3>История загруженных файлов за сессию</h3>
                {history_html}
            </div>
        </div>

        <script>
            const form = document.getElementById("uploadForm");
            const spinnerBox = document.getElementById("spinnerBox");
            const submitBtn = document.getElementById("submitBtn");

            form.addEventListener("submit", function () {{
                spinnerBox.style.display = "block";
                submitBtn.disabled = true;
                submitBtn.textContent = "Обработка...";
            }});
        </script>
    </body>
    </html>
    """


@app.post("/upload/", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Файл не выбран")

    filename = Path(file.filename).name
    ext = Path(filename).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Допустимы только файлы MP3, WAV, M4A"
        )

    input_path = os.path.join(UPLOAD_FOLDER, filename)
    output_filename = Path(filename).stem + ".wav"
    output_path = os.path.join(CONVERTED_FOLDER, output_filename)

    try:
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        convert_to_wav(input_path, output_path)
        transcription_result = transcribe_audio(output_path)
        txt_path, json_path, txt_url, json_url = save_results(filename, transcription_result)

        save_transcription_to_db(
            filename=filename,
            original_path=input_path,
            converted_path=output_path,
            txt_path=txt_path,
            json_path=json_path,
            result_data=transcription_result
        )

        history = request.session.get("history", [])
        history.append({
            "filename": filename,
            "language": transcription_result["language"],
            "duration": transcription_result["duration"],
            "txt_url": txt_url,
            "json_url": json_url
        })
        request.session["history"] = history

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при обработке аудио: {str(e)}"
        )
    finally:
        file.file.close()

    safe_filename = html.escape(filename)
    safe_output_path = html.escape(output_path)
    safe_txt_path = html.escape(txt_path)
    safe_json_path = html.escape(json_path)
    safe_language = html.escape(transcription_result["language"])
    safe_text = html.escape(transcription_result["text"])
    safe_txt_url = html.escape(txt_url)
    safe_json_url = html.escape(json_url)

    return f"""
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <title>Результат обработки</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                background-color: #f8f9fa;
            }}

            .container {{
                max-width: 900px;
                margin: 0 auto;
                background: white;
                padding: 30px;
                border-radius: 12px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            }}

            .result-box {{
                white-space: pre-wrap;
                border: 1px solid #ccc;
                padding: 15px;
                margin-top: 15px;
                border-radius: 8px;
                background-color: #fafafa;
            }}

            a {{
                display: inline-block;
                margin-top: 15px;
                margin-right: 15px;
                text-decoration: none;
                color: #007bff;
            }}

            p {{
                line-height: 1.6;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Файл успешно загружен, конвертирован, распознан и сохранён в PostgreSQL</h2>

            <p><b>Исходный файл:</b> {safe_filename}</p>
            <p><b>Путь к WAV-файлу:</b> {safe_output_path}</p>
            <p><b>Язык:</b> {safe_language}</p>
            <p><b>Длительность:</b> {transcription_result["duration"]} сек.</p>
            <p><b>TXT сохранён в:</b> {safe_txt_path}</p>
            <p><b>JSON сохранён в:</b> {safe_json_path}</p>

            <p>
                <a href="{safe_txt_url}" target="_blank">Открыть TXT</a>
                <a href="{safe_json_url}" target="_blank">Открыть JSON</a>
            </p>

            <h3>Текст распознавания:</h3>
            <div class="result-box">
                {safe_text if safe_text else "Текст не распознан"}
            </div>

            <a href="/">Назад на главную</a>
        </div>
    </body>
    </html>
    """