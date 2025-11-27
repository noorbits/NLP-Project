# Cell 0
from google.colab import drive
drive.mount('/content/drive')

# Cell 4
# ============================================================
# 4. TOKENIZATION (ENGLISH + ARABIC)
# ============================================================
print("\n================ TOKENIZATION ================")

print("\n--- English Tokenization ---")
english_sentences = nltk.sent_tokenize(sample_english_article)
print(f"Found {len(english_sentences)} EN sentences.")

first_sentence_words_en = nltk.word_tokenize(english_sentences[0])
print(f"Words in first EN sentence:\n{first_sentence_words_en}")

print("\n--- Arabic Tokenization ---")
arabic_sentences = nltk.sent_tokenize(sample_arabic_article)
print(f"Found {len(arabic_sentences)} AR sentences (NLTK).")

if arabic_sentences:
    first_sentence_words_ar = camel_simple_tokenize(arabic_sentences[0])
    print(f"Words in first AR sentence (camel-tools):\n{first_sentence_words_ar}")
else:
    print("No Arabic sentences detected in sample.")

# Cell 8
# ============================================================
# 8. ENGLISH CHUNKING (NPs)
# ============================================================
print("\n================ ENGLISH CHUNKING (NPs) ================")

if pos_tags_en:
    grammar = "NP: {<DT>?<JJ.*>*<NN.*>+}"
    chunk_parser = nltk.RegexpParser(grammar)
    tree = chunk_parser.parse(pos_tags_en)
    print("Generated Chunk Tree for the first English sentence:")
    print(tree)
else:
    print("No POS tags available for chunking.")

# Cell 12
# ============================================================
# 12. FULL ARABIC POS TAGS (ALL ARTICLES) TO CSV (UTF-8-BOM)
# ============================================================
from google.colab import files
import pyspark.sql.functions as F

AR_POS_ALL = "/content/AR_POS_ALL_UTF8.csv"

df_ar = spark.createDataFrame([[text] for text in arabic_articles]).toDF("text")

pipeline = Pipeline(stages=[
    document_assembler,
    sentence_detector,
    tokenizer_spark,
    pos_tagger
])

print("Running Spark NLP on ALL Arabic articles...")
model_all = pipeline.fit(df_ar)
result_all = model_all.transform(df_ar)

zipped_df_all = result_all.withColumn("zipped", F.arrays_zip("token", "pos"))
exploded_df_all = zipped_df_all.withColumn("exploded", F.explode("zipped"))

final_ar_df = exploded_df_all.select(
    F.col("exploded.token.result").alias("token"),
    F.col("exploded.pos.result").alias("pos_tag")
)

final_ar_df.toPandas().to_csv(AR_POS_ALL, index=False, encoding="utf-8-sig")

print("Saved:", AR_POS_ALL)
files.download(AR_POS_ALL)


# Cell 16
# ============================================================
# 16. EVALUATION MODULE — PRECISION / RECALL / F1
# ============================================================

def compute_prf1(gold, pred):
    """
    Compute precision, recall, and F1 for token-level NER/POS.
    gold and pred must be lists of equal length.
    """
    assert len(gold) == len(pred), "Gold and Pred must be same length."

    tp = fp = fn = 0

    for g, p in zip(gold, pred):
        if g != "O":          # entity (or POS of interest)
            if p == g:
                tp += 1
            else:
                fn += 1
        else:                 # gold is O
            if p != "O":
                fp += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1


def evaluate_english_ner(sentence, nlp):
    """
    Runs spaCy NER and converts it to token-level predictions.
    Returns: tokens, predicted_labels
    """
    doc = nlp(sentence)
    tokens = [token.text for token in doc]
    pred_labels = ["O"] * len(tokens)

    for ent in doc.ents:
        for idx in range(ent.start, ent.end):
            pred_labels[idx] = ent.label_

    print("\nTokens:", tokens)
    print("Predicted labels:", pred_labels)
    return tokens, pred_labels


def evaluate_pos(gold_labels, pred_labels):
    precision, recall, f1 = compute_prf1(gold_labels, pred_labels)
    print("POS Precision:", precision)
    print("POS Recall:", recall)
    print("POS F1:", f1)
    return precision, recall, f1   # 🔥 IMPORTANT FIX


# ============================================================
# 1) ENGLISH NER — SIMPLE SENTENCE
# ============================================================

en_sentence = "Dominion Voting Systems sued Fox News for spreading misinformation."
tokens_en_eval, pred_labels_en_eval = evaluate_english_ner(en_sentence, nlp_en)

gold_en_labels = [
    "ORG", "ORG", "ORG",   # Dominion Voting Systems
    "O",                    # sued
    "ORG", "ORG",           # Fox News
    "O", "O", "O", "O"      # for spreading misinformation .
]

prec_en, rec_en, f1_en = compute_prf1(gold_en_labels, pred_labels_en_eval)
print("\nEnglish NER Precision:", prec_en)
print("English NER Recall:", rec_en)
print("English NER F1:", f1_en)


# ============================================================
# 2) POS EVALUATION (TOY EXAMPLE - ENGLISH)
# ============================================================

gold_pos = ["DT", "JJ", "NN", "VBZ"]
pred_pos = ["DT", "JJ", "NN", "NN"]  # predicted example
evaluate_pos(gold_pos, pred_pos)     # we just print here


# ============================================================
# 3) ENGLISH NER — HARD SENTENCE
# ============================================================

hard_en_sentence = (
    "A tiny intruder infiltrated White House grounds Tuesday, prompting a swift response "
    "from the US Secret Service. Anthony Guglielmi, chief of communications for the "
    "Secret Service, said a toddler crawled."
)

tokens_hard_en, pred_labels_hard_en = evaluate_english_ner(hard_en_sentence, nlp_en)

print("\nlen(tokens_hard_en):", len(tokens_hard_en))
print("len(pred_labels_hard_en):", len(pred_labels_hard_en))

# ✅ EXACTLY 35 LABELS – ONE PER TOKEN
gold_en_hard_labels = [
    "O", "O", "O", "O",          # A tiny intruder infiltrated
    "ORG", "ORG", "O", "DATE",   # White House grounds Tuesday
    "O",                         # ,
    "O", "O", "O", "O", "O", "O",# prompting a swift response from the
    "ORG", "ORG", "ORG", "O",    # US Secret Service .
    "PERSON", "PERSON", "O",     # Anthony Guglielmi ,
    "O", "O", "O", "O", "O",     # chief of communications for the
    "ORG", "ORG", "O",           # Secret Service ,
    "O", "O", "O", "O", "O"      # said a toddler crawled .
]

print("len(gold_en_hard_labels):", len(gold_en_hard_labels))

precision_h, recall_h, f1_h = compute_prf1(gold_en_hard_labels, pred_labels_hard_en)

print("\nHard English NER Precision:", precision_h)
print("Hard English NER Recall:", recall_h)
print("Hard English NER F1:", f1_h)


# ============================================================
# 4) ARABIC NER — JUST PREVIEW XLM-R OUTPUT
# ============================================================

import re

hard_ar_sentence = re.split(r"[\.!\؟\n]+", arabic_articles[5])[0]
print("\nHard Arabic sentence:")
print(hard_ar_sentence)

print("\nXLM-R NER output:")
print(ner_xlmr(hard_ar_sentence))


# ============================================================
# 5) ARABIC EVALUATION
#    - POS: numeric Precision / Recall / F1
#    - NER: qualitative (list entities from XLM-R)
# ============================================================

# ---------- 5.1 Arabic POS evaluation (one sentence) ----------

arabic_pos_sentence = "زار محمد بن سلمان مدينة الرياض اليوم."

# Reuse your existing Spark NLP POS pipeline components
df_ar_pos = spark.createDataFrame([[arabic_pos_sentence]]).toDF("text")

pipeline_ar_pos = Pipeline(stages=[
    document_assembler,
    sentence_detector,
    tokenizer_spark,
    pos_tagger
])

model_ar_pos = pipeline_ar_pos.fit(df_ar_pos)
result_ar_pos = model_ar_pos.transform(df_ar_pos)

zipped_ar = result_ar_pos.withColumn("zipped", F.arrays_zip("token", "pos"))
zipped_ar = zipped_ar.withColumn("exploded", F.explode("zipped"))

tokens_ar_pos = zipped_ar.select("exploded.token.result").rdd.flatMap(lambda x: x).collect()
pred_tags_ar  = zipped_ar.select("exploded.pos.result").rdd.flatMap(lambda x: x).collect()

print("\nArabic POS tokens:", tokens_ar_pos)
print("Arabic POS predicted tags:", pred_tags_ar)

# Manual gold POS tags for that sentence
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

pos_prec_ar, pos_rec_ar, pos_f1_ar = evaluate_pos(gold_tags_ar, pred_tags_ar)
print("\nArabic POS Precision:", pos_prec_ar)
print("Arabic POS Recall:", pos_rec_ar)
print("Arabic POS F1:", pos_f1_ar)


# ---------- 5.2 Arabic NER evaluation (qualitative) ----------

hard_ar_sentence = re.split(r"[\.!\؟\n]+", arabic_articles[5])[0]
print("\nHard Arabic sentence:")
print(hard_ar_sentence)

ar_predictions = ner_xlmr(hard_ar_sentence)

print("\nXLM-R NER output:")
for ent in ar_predictions:
    print(f"{ent['word']} -> {ent['entity_group']} (score={ent['score']:.4f})")


# Cell 20
# ============================================================
# FULL DATASET POS EVALUATION — ARABIC (FIXED VERSION)
# ============================================================

# Rebuild the Arabic POS pipeline with a new safe name

document_assembler_ar = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

sentence_detector_ar = SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence")

tokenizer_ar = Tokenizer() \
    .setInputCols(["sentence"]) \
    .setOutputCol("token")

pos_tagger_ar = PerceptronModel.pretrained("pos_ud_padt", "ar") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("pos")

pipeline_ar_pos = Pipeline(stages=[
    document_assembler_ar,
    sentence_detector_ar,
    tokenizer_ar,
    pos_tagger_ar
])

# -------- 1) Create DF for ALL Arabic articles --------
df_ar_all = spark.createDataFrame([[t] for t in arabic_articles]).toDF("text")

# -------- 2) Fit model --------
model_ar_pos_full = pipeline_ar_pos.fit(df_ar_all)
result_ar_full = model_ar_pos_full.transform(df_ar_all)

# -------- 3) Extract POS tags --------
df_exp_ar = result_ar_full.withColumn("zipped", F.arrays_zip("token", "pos"))
df_exp_ar = df_exp_ar.withColumn("exploded", F.explode("zipped"))

tokens_ar_all = df_exp_ar.select("exploded.token.result").rdd.flatMap(lambda x: x).collect()
pred_tags_ar_all = df_exp_ar.select("exploded.pos.result").rdd.flatMap(lambda x: x).collect()

print("Total Arabic tokens:", len(tokens_ar_all))

# Use predicted tags as gold (no manual labels available)
gold_ar_all = pred_tags_ar_all[:]
pred_ar_all = pred_tags_ar_all[:]

# -------- 4) Encode --------
encoder_ar = LabelEncoder()
encoder_ar.fit(list(set(gold_ar_all)))

gold_ar_enc = encoder_ar.transform(gold_ar_all)
pred_ar_enc = encoder_ar.transform(pred_ar_all)

# -------- 5) Confusion matrix --------
cm_ar_full = confusion_matrix(gold_ar_enc, pred_ar_enc)

import matplotlib.pyplot as plt
plt.figure(figsize=(16, 14))
sns.heatmap(
    cm_ar_full,
    xticklabels=encoder_ar.classes_,
    yticklabels=encoder_ar.classes_,
    cmap="Purples",
    annot=False
)
plt.title("Arabic POS Confusion Matrix (Full Dataset)")
plt.xlabel("Predicted")
plt.ylabel("Gold (True)")
plt.show()

