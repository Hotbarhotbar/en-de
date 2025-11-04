import os
import re
from pathlib import Path

# 修改为相对于src目录的路径
RAW_DIR = "../data/raw"
OUT_DIR = "../data/en-de"
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

TAG_PATTERN = re.compile(r"^<.*?>$")

def load_train(file):
    lines = []
    try:
        with open(file, encoding="utf-8") as f:
            for l in f:
                l = l.strip()
                if not l:
                    continue
                if TAG_PATTERN.match(l):
                    continue
                lines.append(l)
    except FileNotFoundError:
        print(f"Error: File {file} not found!")
        return []
    return lines

def load_xml(file):
    try:
        text = open(file, encoding="utf-8").read()
        segs = re.findall(r"<seg id=\"\d+\">(.*?)</seg>", text)
        return [s.strip() for s in segs]
    except FileNotFoundError:
        print(f"Error: File {file} not found!")
        return []

print("Loading training data...")
train_en = load_train(f"{RAW_DIR}/train.tags.en-de.en")
train_de = load_train(f"{RAW_DIR}/train.tags.en-de.de")

if not train_en or not train_de:
    print("Error: Training data not loaded!")
    exit(1)

min_len = min(len(train_en), len(train_de))
train_en = train_en[:min_len]
train_de = train_de[:min_len]

print("Loading validation data...")
valid_en = load_xml(f"{RAW_DIR}/IWSLT17.TED.tst2014.en-de.en.xml")
valid_de = load_xml(f"{RAW_DIR}/IWSLT17.TED.tst2014.en-de.de.xml")

print("Loading test data...")
test_en = load_xml(f"{RAW_DIR}/IWSLT17.TED.tst2015.en-de.en.xml")
test_de = load_xml(f"{RAW_DIR}/IWSLT17.TED.tst2015.en-de.de.xml")

print("Train size:", len(train_en))
print("Valid size:", len(valid_en))
print("Test size:", len(test_de))

def save(lines, path):
    with open(path, "w", encoding="utf-8") as f:
        for l in lines:
            f.write(l + "\n")

save(train_en, f"{OUT_DIR}/train.en")
save(train_de, f"{OUT_DIR}/train.de")
save(valid_en, f"{OUT_DIR}/valid.en")
save(valid_de, f"{OUT_DIR}/valid.de")
save(test_en, f"{OUT_DIR}/test.en")
save(test_de, f"{OUT_DIR}/test.de")

print("✅ Data preparation completed!")
