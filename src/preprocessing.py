# Cell 1
# ============================================================
# 1. INSTALL DEPENDENCIES
# ============================================================

# NLP + Spark libs
!pip install -q nltk camel-tools pyspark==3.5.1 spark-nlp

# Java for Spark
!apt-get update -y
!apt-get install -y openjdk-11-jre-headless || apt-get install -y openjdk-17-jre-headless


# Cell 5

# ============================================================
# 5. ENGLISH N-GRAM LANGUAGE MODEL (BIGRAM)
# ============================================================
print("\n================ ENGLISH N-GRAM LM (BIGRAM) ================")

# 1. Collect all English sentences
all_english_sentences = [nltk.sent_tokenize(article) for article in english_articles]
all_english_sentences = [sent for sublist in all_english_sentences for sent in sublist]

# 2. Tokenize each sentence
tokenized_sentences_en = [nltk.word_tokenize(sent.lower()) for sent in all_english_sentences]

# 3. Train/Test split
split_point_en = int(len(tokenized_sentences_en) * 0.8)
train_sents_en = tokenized_sentences_en[:split_point_en]
test_sents_en  = tokenized_sentences_en[split_point_en:]

print(f"Total EN sentences: {len(tokenized_sentences_en)}")
print(f"Train EN sentences: {len(train_sents_en)}")
print(f"Test EN sentences:  {len(test_sents_en)}")

# 4. Prepare data for bigram model
n = 2
train_data_en, padded_sents_en = padded_everygram_pipeline(n, train_sents_en)

# 5. Train model with Laplace smoothing
model_en = Laplace(n)
model_en.fit(train_data_en, padded_sents_en)

print("\n--- English Bigram Model ---")
print(f"Vocabulary size: {len(model_en.vocab)}")

# 6. Evaluate perplexity
test_bigrams_en = [ngrams(sent, n, pad_left=True, pad_right=True) for sent in test_sents_en]
test_data_en = [gram for sent in test_bigrams_en for gram in sent]
test_data_en_filtered = [gram for gram in test_data_en if None not in gram]

perplexity_en = model_en.perplexity(test_data_en_filtered)
print(f"English Model Perplexity on Test Set: {perplexity_en}")

# 7. Generate sample EN text
print("\n--- English Text Generation Sample ---")
generated_text_en = model_en.generate(5, text_seed=['the'])
print("Starting with 'the':", " ".join(generated_text_en))

# Cell 9
# ============================================================
# 9. ARABIC POS TAGGING WITH SPARK NLP
# ============================================================
print("\n================ ARABIC POS TAGGING (Spark NLP) ================")

spark = sparknlp.start()
print(f"Spark NLP version: {sparknlp.version()}")
print(f"Apache Spark version: {spark.version}")

try:
    sample_text_ar_spark = arabic_articles[0]
except:
    sample_text_ar_spark = "فاز المنتخب السعودي على نظيره الأرجنتيني بهدفين مقابل هدف واحد."

data = spark.createDataFrame([[sample_text_ar_spark]]).toDF("text")

document_assembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

sentence_detector = SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence")

tokenizer_spark = Tokenizer() \
    .setInputCols(["sentence"]) \
    .setOutputCol("token")

pos_tagger = PerceptronModel.pretrained("pos_ud_padt", "ar") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("pos")

pipeline = Pipeline(stages=[
    document_assembler,
    sentence_detector,
    tokenizer_spark,
    pos_tagger
])

print("Running Spark NLP pipeline...")
model_spark = pipeline.fit(data)
result = model_spark.transform(data)
print("Pipeline complete.")

zipped_df = result.withColumn("zipped", F.arrays_zip("token", "pos"))
exploded_df = zipped_df.withColumn("exploded", F.explode("zipped"))
result_df = exploded_df.select(
    F.col("exploded.token.result").alias("token"),
    F.col("exploded.pos.result").alias("pos_tag")
)

print("\n--- Arabic Tokens + POS Tags (Spark NLP) ---")
result_df.show(truncate=False)

# Cell 13
# ============================================================
# 13. ENGLISH NAMED ENTITY RECOGNITION (spaCy)
# ============================================================
!pip install -q spacy
!python -m spacy download en_core_web_sm

import spacy
import pandas as pd

nlp_en = spacy.load("en_core_web_sm")

print("\n================ ENGLISH NER (spaCy) ================")

MAX_EN_ARTICLES_FOR_NER = 200
ner_records_en = []

for i, article in enumerate(english_articles[:MAX_EN_ARTICLES_FOR_NER]):
    doc = nlp_en(article)
    for ent in doc.ents:
        ner_records_en.append({
            "article_index": i,
            "text": ent.text,
            "label": ent.label_,
            "start_char": ent.start_char,
            "end_char": ent.end_char
        })
    if (i + 1) % 20 == 0:
        print(f"Processed {i+1} English articles for NER...")

en_ner_df = pd.DataFrame(ner_records_en)
print("\nSample of English NER results:")
print(en_ner_df.head())

EN_NER_PATH = BASE_PATH + "EN_NER_ALL.csv"
en_ner_df.to_csv(EN_NER_PATH, index=False, encoding="utf-8")
print(f"\nSaved English NER annotations to: {EN_NER_PATH}")


# Cell 17
# ============================================================
# POS CONFUSION MATRIX (ENGLISH)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

# ----- 1) Use the same sentence you used for POS evaluation -----
# Replace this with your actual sentence if needed
pos_sentence = "The quick brown fox jumps over the lazy dog."

tokens = nltk.word_tokenize(pos_sentence)
pred_tags = [tag for (_, tag) in nltk.pos_tag(tokens)]

# ----- 2) MANUAL GOLD TAGS -----
# Edit these based on your chosen sentence
gold_tags = ["DT", "JJ", "JJ", "NN", "VBZ", "IN", "DT", "JJ", "NN", "."]


print("TOKENS:", tokens)
print("PRED:", pred_tags)
print("GOLD:", gold_tags)

assert len(gold_tags) == len(pred_tags), "Gold and Pred must be same length."

# ----- 3) Encode -----
encoder = LabelEncoder()
encoder.fit(list(set(pred_tags + gold_tags)))

gold_encoded = encoder.transform(gold_tags)
pred_encoded = encoder.transform(pred_tags)

# ----- 4) Compute matrix -----
cm = confusion_matrix(gold_encoded, pred_encoded)

# ----- 5) Plot -----
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=encoder.classes_,
    yticklabels=encoder.classes_
)
plt.xlabel("Predicted")
plt.ylabel("Gold (True)")
plt.title("POS Tagging Confusion Matrix")
plt.show()


# Cell 21
# ============================================================
# FULL-DATASET NER STATISTICS — ENGLISH
# ============================================================

# Use en_ner_df from earlier code
df_en = en_ner_df

# Count entity types
entity_type_counts_en = df_en['label'].value_counts()
print("=== English NER: Entity Type Counts ===")
print(entity_type_counts_en)

# Top 20 most frequent entity spans
top_entities_en = df_en['text'].value_counts().head(20)
print("\n=== Top 20 English Entities ===")
print(top_entities_en)

# Save for report
entity_type_counts_en.to_csv(BASE_PATH + "NER_EN_entity_type_counts.csv")
top_entities_en.to_csv(BASE_PATH + "NER_EN_top_entities.csv")

