import json
import re

from pathlib import Path

# Lấy thư mục hiện tại của file clean.py
BASE_DIR = Path(__file__).resolve().parent

# Đi lên 1 cấp → tới root project, rồi vào data/
DATA_DIR = BASE_DIR.parent / "data"

file1_path = DATA_DIR / "file1.txt"
file2_path = DATA_DIR / "file2.txt"

# ===== CLEAN TEXT =====
def clean_text(text):
    # 2. Remove LaTeX
    text = re.sub(r'\$.*?\$', '', text)

    # 3. Remove ::: (section separator)
    text = re.sub(r'\s*:::\s*', ' - ', text)

    # 4. Remove weird characters
    text = re.sub(r'[^\w\s.,;:!?()\-]', '', text)

    # 5. Remove space BEFORE punctuation
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)

    # 6. Normalize multiple punctuation
    text = re.sub(r'[:]{2,}', ':', text)

    # 7. Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# ===== LOAD + CLEAN =====
def load_and_clean(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.readlines()

    cleaned = ". ".join([clean_text(line) for line in raw_text])
    print(cleaned)
    return cleaned


# ===== MAIN =====
if __name__ == "__main__":
    cleaned_text = load_and_clean(file1_path)

    with open(file1_path, "w", encoding="utf-8") as f:
        f.write(cleaned_text)

    cleaned_text = load_and_clean(file2_path)

    with open(file2_path, "w", encoding="utf-8") as f:
        f.write(cleaned_text)