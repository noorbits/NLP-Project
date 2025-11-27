# Cell 2
# ============================================================
# 2. IMPORT LIBRARIES
# ============================================================
import os
import json
import nltk
import pandas as pd

from nltk.tokenize import word_tokenize, sent_tokenize, WordPunctTokenizer
from nltk.util import ngrams
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import Laplace

from camel_tools.tokenizers.word import simple_word_tokenize as camel_simple_tokenize

# NLTK resources (IMPORTANT: includes punkt_tab)
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

# Spark / Java env
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"

import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.sql import SparkSession
import pyspark.sql.functions as F


# Cell 6
# ============================================================
# 6. ARABIC N-GRAM LANGUAGE MODEL (BIGRAM)
# ============================================================
print("\n================ ARABIC N-GRAM LM (BIGRAM) ================")

arabic_tokenizer = WordPunctTokenizer()
def simple_word_tokenize_nltk(text):
    return arabic_tokenizer.tokenize(text)

print("Tokenizing all Arabic articles using NLTK WordPunctTokenizer...")
all_arabic_sentences = []
for article in arabic_articles:
    sents = nltk.sent_tokenize(article)
    all_arabic_sentences.extend(sents)

tokenized_sentences_ar = [simple_word_tokenize_nltk(sent) for sent in all_arabic_sentences]
print(f"Total AR sentences: {len(tokenized_sentences_ar)}")

split_point_ar = int(len(tokenized_sentences_ar) * 0.8)
train_sents_ar = tokenized_sentences_ar[:split_point_ar]
test_sents_ar  = tokenized_sentences_ar[split_point_ar:]

print(f"Train AR sentences: {len(train_sents_ar)}")
print(f"Test AR sentences:  {len(test_sents_ar)}")

if train_sents_ar and test_sents_ar:
    n_ar = 2
    train_data_ar, padded_sents_ar = padded_everygram_pipeline(n_ar, train_sents_ar)

    model_ar = Laplace(n_ar)
    model_ar.fit(train_data_ar, padded_sents_ar)

    print("\n--- Arabic Bigram Model (NLTK Baseline) ---")
    print(f"Vocabulary size: {len(model_ar.vocab)}")

    print("Calculating AR perplexity...")
    test_bigrams_ar = [ngrams(
        sent, n_ar,
        pad_left=True, pad_right=True,
        left_pad_symbol="<s>", right_pad_symbol="</s>"
    ) for sent in test_sents_ar]
    test_data_ar = [gram for sent in test_bigrams_ar for gram in sent]

    perplexity_ar = model_ar.perplexity(test_data_ar)
    print(f"Arabic Model Perplexity on Test Set: {perplexity_ar:.2f}")

    # Text generation
    try:
        generated_text_ar = model_ar.generate(10, text_seed=['العالم'])
        print("Sample AR generation:", " ".join(['العالم'] + generated_text_ar))
    except Exception as e:
        print(f"Could not generate text (seed may not be in vocab): {e}")
else:
    print("Could not train Arabic N-gram model. Not enough data.")


# Cell 10
# ============================================================
# 10. SAVE CORE RESULTS TO FILES IN GOOGLE DRIVE
# ============================================================
import csv
from datetime import datetime

RESULTS_PATH = BASE_PATH

summary_lines = []

summary_lines.append(f"Run timestamp: {datetime.now()}")
summary_lines.append("")
summary_lines.append("=== DATA STATS ===")
summary_lines.append(f"Total English articles: {len(english_articles)}")
summary_lines.append(f"Total Arabic articles: {len(arabic_articles)}")
summary_lines.append("")
summary_lines.append("=== ENGLISH N-GRAM LM (BIGRAM) ===")
summary_lines.append(f"English vocab size: {len(model_en.vocab)}")
summary_lines.append(f"English perplexity: {perplexity_en}")
summary_lines.append(f"English sample generation: {' '.join(generated_text_en)}")

summary_lines.append("")
summary_lines.append("=== ARABIC N-GRAM LM (BIGRAM) ===")
summary_lines.append(f"Arabic vocab size: {len(model_ar.vocab)}")
summary_lines.append(f"Arabic perplexity: {perplexity_ar}")
summary_lines.append(f"Arabic sample generation: {' '.join(['العالم'] + generated_text_ar)}")

summary_lines.append("")
summary_lines.append("=== ENGLISH POS SAMPLE (FIRST SENTENCE) ===")
summary_lines.append(str(pos_tags_en))
summary_lines.append("")
summary_lines.append("=== NOTE ===")
summary_lines.append("Full Arabic POS (Spark NLP) is saved separately in AR_POS_spark.csv")

summary_path = RESULTS_PATH + "NLP_Results_Summary.txt"
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))

print(f"Saved summary to: {summary_path}")

# English POS sample CSV
en_pos_path = RESULTS_PATH + "EN_POS_sample.csv"
with open(en_pos_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["token", "pos"])
    for tok, tag in pos_tags_en:
        writer.writerow([tok, tag])

print(f"Saved English POS sample to: {en_pos_path}")

# Arabic POS sample CSV (Spark NLP, one article)
ar_pos_path = RESULTS_PATH + "AR_POS_spark.csv"
result_df.toPandas().to_csv(ar_pos_path, index=False)
print(f"Saved Arabic POS (Spark NLP) to: {ar_pos_path}")

# ZIP important outputs
import zipfile

zip_path = "/content/NLP_outputs.zip"
with zipfile.ZipFile(zip_path, 'w') as z:
    z.write(summary_path, "NLP_Results_Summary.txt")
    z.write(en_pos_path, "EN_POS_sample.csv")
    z.write(ar_pos_path, "AR_POS_spark.csv")

print("Created ZIP:", zip_path)

from google.colab import files
files.download(zip_path)


# Cell 14
# ============================================================
# 14. ARABIC NER WITH XLM-R (MULTILINGUAL)
# ============================================================
!pip install -q transformers accelerate

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import re

model_name = "Davlan/xlm-roberta-base-ner-hrl"  # multilingual, supports Arabic

tokenizer_xlmr = AutoTokenizer.from_pretrained(model_name)
model_xlmr = AutoModelForTokenClassification.from_pretrained(model_name)

ner_xlmr = pipeline(
    "token-classification",
    model=model_xlmr,
    tokenizer=tokenizer_xlmr,
    aggregation_strategy="simple"
)

def split_ar_sentences(text):
    sents = re.split(r"[\.!\؟\n]+", text)
    return [s.strip() for s in sents if s.strip()]

records = []
MAX_AR_ARTICLES_FOR_XLMR = 200

for i, article in enumerate(arabic_articles[:MAX_AR_ARTICLES_FOR_XLMR]):
    sentences = split_ar_sentences(article)
    for s_idx, sent in enumerate(sentences):
        ents = ner_xlmr(sent)
        for ent in ents:
            records.append({
                "article_index": i,
                "sentence_index": s_idx,
                "entity": ent["word"],
                "label": ent["entity_group"],
                "start": ent["start"],
                "end": ent["end"],
                "sentence": sent
            })
    if (i + 1) % 10 == 0:
        print(f"Processed {i+1} Arabic articles with XLM-R NER...")

ar_ner_xlmr_df = pd.DataFrame(records)
print(ar_ner_xlmr_df.head(20))

XLMR_NER_PATH = BASE_PATH + "AR_NER_XLMR_ALL.csv"
ar_ner_xlmr_df.to_csv(XLMR_NER_PATH, index=False, encoding="utf-8")
print("Saved:", XLMR_NER_PATH)

ar_ner_xlmr_df.to_excel(BASE_PATH + "AR_NER_XLMR_ALL.xlsx", index=False)
print("Saved:", BASE_PATH + "AR_NER_XLMR_ALL.xlsx")

# Cell 18
# ============================================================
# ARABIC POS CONFUSION MATRIX (Spark NLP) — FIXED VERSION
# ============================================================

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import numpy as np

# -------- 1) Use your Arabic sentence --------
arabic_sentence = "زار محمد بن سلمان مدينة الرياض اليوم."

# -------- 2) Run Spark NLP pipeline --------
df_tmp = spark.createDataFrame([[arabic_sentence]]).toDF("text")

pipeline_tmp = Pipeline(stages=[
    document_assembler,
    sentence_detector,
    tokenizer_spark,
    pos_tagger
])

model_tmp = pipeline_tmp.fit(df_tmp)
result_tmp = model_tmp.transform(df_tmp)

# Extract tokens + predictions
df_exploded = result_tmp.withColumn("zipped", F.arrays_zip("token", "pos"))
df_exploded = df_exploded.withColumn("exploded", F.explode("zipped"))

pred_tokens = df_exploded.select("exploded.token.result").rdd.flatMap(lambda x: x).collect()
pred_tags   = df_exploded.select("exploded.pos.result").rdd.flatMap(lambda x: x).collect()

print("TOKENS:", pred_tokens)
print("PREDICTED TAGS:", pred_tags)

# -------- 3) Correct GOLD TAGS (length must be 8) --------
gold_tags_ar = [
    "VERB",   # زار
    "PROPN",  # محمد
    "PROPN",  # بن
    "PROPN",  # سلمان
    "NOUN",   # مدينة
    "PROPN",  # الرياض
    "NOUN",   # اليوم
    "PUNCT"   # .
]

# Check lengths match
assert len(gold_tags_ar) == len(pred_tags), "Gold and Pred must be same length."

# -------- 4) Encode tags --------
encoder_ar = LabelEncoder()
encoder_ar.fit(list(set(pred_tags + gold_tags_ar)))

gold_enc = encoder_ar.transform(gold_tags_ar)
pred_enc = encoder_ar.transform(pred_tags)

# -------- 5) Compute confusion matrix --------
cm_ar = confusion_matrix(gold_enc, pred_enc)

# -------- 6) Plot confusion matrix --------
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm_ar,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=encoder_ar.classes_,
    yticklabels=encoder_ar.classes_
)
plt.xlabel("Predicted")
plt.ylabel("Gold (True)")
plt.title("Arabic POS Tagging Confusion Matrix")
plt.show()


# Cell 22
# ============================================================
# FULL-DATASET NER STATISTICS — ARABIC (XLM-R)
# ============================================================

df_ar = ar_ner_xlmr_df  # from your earlier code

# Count entity categories
entity_type_counts_ar = df_ar['label'].value_counts()
print("=== Arabic NER: Entity Type Counts ===")
print(entity_type_counts_ar)

# Top 20 entity spans
top_entities_ar = df_ar['entity'].value_counts().head(20)
print("\n=== Top 20 Arabic Entities ===")
print(top_entities_ar)

# Save for report
entity_type_counts_ar.to_csv(BASE_PATH + "NER_AR_entity_type_counts.csv")
top_entities_ar.to_csv(BASE_PATH + "NER_AR_top_entities.csv")

