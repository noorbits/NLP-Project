# Cell 3
# ============================================================
# 3. LOAD DATA FROM GOOGLE DRIVE (JSONL)
# ============================================================
BASE_PATH = "/content/drive/MyDrive/NLP_Project/"

EN_JSONL = BASE_PATH + "NLP_English.jsonl"
AR_JSONL = BASE_PATH + "NLP_Arabic.jsonl"

english_articles = []
with open(EN_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)
        english_articles.append(record["text"])

arabic_articles = []
with open(AR_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)
        arabic_articles.append(record["text"])

print(f"Loaded {len(english_articles)} English articles.")
print(f"Loaded {len(arabic_articles)} Arabic articles.")

sample_english_article = english_articles[0]
sample_arabic_article  = arabic_articles[0]

print("\n--- English Sample ---")
print(sample_english_article[:150] + "...")

print("\n--- Arabic Sample ---")
print(sample_arabic_article[:150] + "...")


# Cell 7
# ============================================================
# 7. ENGLISH POS TAGGING (BASELINE)
# ============================================================
print("\n================ ENGLISH POS TAGGING ================")

try:
    sample_sent_en = nltk.sent_tokenize(english_articles[0])[0]
    tokenized_sent_en = nltk.word_tokenize(sample_sent_en)
    pos_tags_en = nltk.pos_tag(tokenized_sent_en)
    print("\n--- English POS Tags ---")
    print(pos_tags_en)
except Exception as e:
    pos_tags_en = []
    print(f"Error during POS tagging: {e}")


# Cell 11
# ============================================================
# 11. FULL ENGLISH POS TAGS (ALL ARTICLES) TO CSV
# ============================================================
import csv
from nltk import word_tokenize, pos_tag, sent_tokenize
from google.colab import files

EN_POS_ALL = "/content/EN_POS_ALL.csv"

with open(EN_POS_ALL, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["article_index", "sentence_index", "token", "pos"])

    for i, article in enumerate(english_articles):
        sentences = sent_tokenize(article)
        for j, sent in enumerate(sentences):
            tokens = word_tokenize(sent)
            tags = pos_tag(tokens)
            for token, tag in tags:
                writer.writerow([i, j, token, tag])

print("Saved:", EN_POS_ALL)
files.download(EN_POS_ALL)


# Cell 15
# ============================================================
# 15. ARABIC NORMALIZATION (ARABIC-SPECIFIC PREPROCESSING)
# ============================================================
import re

print("\n================ ARABIC NORMALIZATION ================")

AR_DIACRITICS = re.compile(
    r"[\u0617-\u061A\u064B-\u0652\u0670\u0653-\u065F\u06D6-\u06ED]"
)

def normalize_arabic(text):
    text = re.sub(AR_DIACRITICS, "", text)
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "و", text)
    text = re.sub("ئ", "ي", text)
    text = re.sub("ـ", "", text)
    return text

sample_ar_orig = arabic_articles[0][:200]
sample_ar_norm = normalize_arabic(sample_ar_orig)

print("\n--- Original Arabic sample ---")
print(sample_ar_orig)
print("\n--- Normalized Arabic sample ---")
print(sample_ar_norm)

arabic_articles_normalized = [normalize_arabic(a) for a in arabic_articles]
print(f"\nCreated normalized version of all Arabic articles: {len(arabic_articles_normalized)}")


# Cell 19
# ============================================================
# FULL DATASET POS EVALUATION — ENGLISH
# ============================================================

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

all_gold = []
all_pred = []

for article in english_articles:
    sentences = nltk.sent_tokenize(article)

    for sent in sentences:
        tokens = nltk.word_tokenize(sent)
        tags = nltk.pos_tag(tokens)
        gold = [tag for (_, tag) in tags]  # NLTK tagger as gold baseline
        pred = gold[:]  # baseline predictor (no alternative model)

        all_gold.extend(gold)
        all_pred.extend(pred)

# Encode
encoder = LabelEncoder()
encoder.fit(list(set(all_gold)))

gold_enc = encoder.transform(all_gold)
pred_enc = encoder.transform(all_pred)

cm_en_full = confusion_matrix(gold_enc, pred_enc)

plt.figure(figsize=(12, 10))
sns.heatmap(
    cm_en_full,
    xticklabels=encoder.classes_,
    yticklabels=encoder.classes_,
    cmap="Blues",
    annot=False
)
plt.title("English POS Confusion Matrix (Full Dataset)")
plt.xlabel("Predicted")
plt.ylabel("Gold")
plt.show()

print("Total English tokens evaluated:", len(all_gold))


# Cell 23
import matplotlib.pyplot as plt

# English NER chart
plt.figure(figsize=(8,5))
entity_type_counts_en.plot(kind='bar', color='skyblue')
plt.title("English NER Entity Type Distribution (Full Dataset)")
plt.xlabel("Entity Type")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Arabic NER chart
plt.figure(figsize=(8,5))
entity_type_counts_ar.plot(kind='bar', color='purple')
plt.title("Arabic NER Entity Type Distribution (Full Dataset)")
plt.xlabel("Entity Type")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

