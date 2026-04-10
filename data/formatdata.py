import json

# Đọc file txt
with open("file1.txt", "r", encoding="utf-8") as f:
    data = json.load(f)

# Lấy phần paragraphs và nối thành text
all_text = []

for item in data:
    if item == "section_name":
        all_text.extend(data[item])
    elif item == "paragraphs":
        for paragraph in data[item]:
            all_text.extend(paragraph)
            # print(paragraph)

all_text = "\n".join(all_text)
print(all_text)

with open("file1.txt", "w", encoding="utf-8") as f:
    f.write(all_text)