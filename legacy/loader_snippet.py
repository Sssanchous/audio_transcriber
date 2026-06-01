from pathlib import Path

def load_lines(path: str) -> list[str]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

TASK_PROTOTYPES = load_lines("prototypes/tasks.txt")
QUESTION_PROTOTYPES = load_lines("prototypes/questions.txt")
ANSWER_PROTOTYPES = load_lines("prototypes/answers.txt")
OTHER_PROTOTYPES = load_lines("prototypes/other.txt")  # пригодится позже для класса other