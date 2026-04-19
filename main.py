import os
import re
import math
import random
import warnings

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity

from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

warnings.filterwarnings("ignore")


# -----------------------------
# basic settings
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DATA_DIR = "Data"
FAKE_PATH = os.path.join(DATA_DIR, "Fake.csv")
TRUE_PATH = os.path.join(DATA_DIR, "True.csv")

# use a smaller BERT-family model so it is faster and more practical for grading
MODEL_NAME = "distilbert-base-uncased"

# this keeps runtime reasonable
MAX_LEN = 256
BATCH_SIZE = 16

# use only part of data by default for faster runs.
# set to None if you want full dataset.
MAX_SAMPLES_PER_CLASS = 3000

# number of article segments
NUM_SEGMENTS = 3


# -----------------------------
# helper functions
# -----------------------------
def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text)

    # remove some weird spaces / line breaks
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()

    return text


def sentence_split(text):
    """
    simple sentence splitter so the code stays portable.
    avoids forcing the professor to download extra NLP models.
    """
    text = clean_text(text)

    if not text:
        return []

    # split on punctuation followed by whitespace
    pieces = re.split(r'(?<=[.!?])\s+', text)
    pieces = [p.strip() for p in pieces if p.strip()]

    return pieces


def split_into_segments(text, num_segments=3):
    """
    split article into 3 sentence-based chunks:
    intro, middle, end
    """
    sentences = sentence_split(text)

    if len(sentences) == 0:
        return [""] * num_segments

    if len(sentences) <= num_segments:
        # pad with repeats if article is too short
        out = sentences[:]
        while len(out) < num_segments:
            out.append(sentences[-1])
        return out

    chunk_size = math.ceil(len(sentences) / num_segments)
    segments = []

    for i in range(num_segments):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, len(sentences))
        seg = " ".join(sentences[start:end]).strip()
        if not seg:
            seg = sentences[-1]
        segments.append(seg)

    while len(segments) < num_segments:
        segments.append(segments[-1])

    return segments[:num_segments]


def mean_pooling(last_hidden_state, attention_mask):
    """
    masked mean pooling
    """
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked_embeddings = last_hidden_state * mask
    summed = torch.sum(masked_embeddings, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def encode_texts(texts, tokenizer, model, device, batch_size=16, max_len=256):
    """
    encode a list of texts into dense embeddings
    """
    all_embeddings = []

    model.eval()

    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding", leave=False):
        batch_texts = texts[i:i + batch_size]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        )

        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)
            pooled = mean_pooling(outputs.last_hidden_state, encoded["attention_mask"])

        all_embeddings.append(pooled.cpu().numpy())

    return np.vstack(all_embeddings)


def build_segment_feature_matrix(df, tokenizer, model, device):
    """
    for each article:
    1. split into 3 segments
    2. get BERT embedding for each segment
    3. add cosine similarity features
    4. concatenate into final feature vector

    final feature =
      [seg1_emb, seg2_emb, seg3_emb, sim12, sim23, sim13]
    """
    seg1_texts = []
    seg2_texts = []
    seg3_texts = []

    for txt in df["text"].tolist():
        segments = split_into_segments(txt, NUM_SEGMENTS)
        seg1_texts.append(segments[0])
        seg2_texts.append(segments[1])
        seg3_texts.append(segments[2])

    print("\nEncoding segment 1...")
    emb1 = encode_texts(seg1_texts, tokenizer, model, device, BATCH_SIZE, MAX_LEN)

    print("Encoding segment 2...")
    emb2 = encode_texts(seg2_texts, tokenizer, model, device, BATCH_SIZE, MAX_LEN)

    print("Encoding segment 3...")
    emb3 = encode_texts(seg3_texts, tokenizer, model, device, BATCH_SIZE, MAX_LEN)

    sim12 = []
    sim23 = []
    sim13 = []

    for i in range(len(df)):
        s12 = cosine_similarity(emb1[i].reshape(1, -1), emb2[i].reshape(1, -1))[0][0]
        s23 = cosine_similarity(emb2[i].reshape(1, -1), emb3[i].reshape(1, -1))[0][0]
        s13 = cosine_similarity(emb1[i].reshape(1, -1), emb3[i].reshape(1, -1))[0][0]

        sim12.append(s12)
        sim23.append(s23)
        sim13.append(s13)

    sim12 = np.array(sim12).reshape(-1, 1)
    sim23 = np.array(sim23).reshape(-1, 1)
    sim13 = np.array(sim13).reshape(-1, 1)

    X = np.hstack([emb1, emb2, emb3, sim12, sim23, sim13])

    return X


def build_baseline_feature_matrix(df, tokenizer, model, device):
    """
    whole-article baseline:
    title + text -> one embedding
    """
    combined_texts = (df["title"].fillna("") + " " + df["text"].fillna("")).tolist()

    print("\nEncoding whole articles for baseline...")
    X = encode_texts(combined_texts, tokenizer, model, device, BATCH_SIZE, MAX_LEN)

    return X


def evaluate_model(name, y_true, y_pred):
    print(f"\n{'=' * 60}")
    print(f"{name} RESULTS")
    print(f"{'=' * 60}")
    print("Accuracy :", round(accuracy_score(y_true, y_pred), 4))
    print("Precision:", round(precision_score(y_true, y_pred), 4))
    print("Recall   :", round(recall_score(y_true, y_pred), 4))
    print("F1 Score :", round(f1_score(y_true, y_pred), 4))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))


# -----------------------------
# main
# -----------------------------
def main():
    print("Loading data...")

    if not os.path.exists(FAKE_PATH):
        raise FileNotFoundError(f"Could not find {FAKE_PATH}")

    if not os.path.exists(TRUE_PATH):
        raise FileNotFoundError(f"Could not find {TRUE_PATH}")

    fake_df = pd.read_csv(FAKE_PATH)
    true_df = pd.read_csv(TRUE_PATH)

    fake_df["label"] = 1
    true_df["label"] = 0

    keep_cols = ["title", "text", "label"]
    fake_df = fake_df[keep_cols].copy()
    true_df = true_df[keep_cols].copy()

    fake_df["title"] = fake_df["title"].apply(clean_text)
    fake_df["text"] = fake_df["text"].apply(clean_text)

    true_df["title"] = true_df["title"].apply(clean_text)
    true_df["text"] = true_df["text"].apply(clean_text)

    # drop empty rows
    fake_df = fake_df[(fake_df["text"].str.len() > 50)]
    true_df = true_df[(true_df["text"].str.len() > 50)]

    # optional sampling for speed
    if MAX_SAMPLES_PER_CLASS is not None:
        fake_df = fake_df.sample(min(MAX_SAMPLES_PER_CLASS, len(fake_df)), random_state=SEED)
        true_df = true_df.sample(min(MAX_SAMPLES_PER_CLASS, len(true_df)), random_state=SEED)

    df = pd.concat([fake_df, true_df], ignore_index=True)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    print(f"Total samples being used: {len(df)}")
    print(df["label"].value_counts())

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=SEED,
        stratify=df["label"]
    )

    print(f"\nTrain size: {len(train_df)}")
    print(f"Test size : {len(test_df)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.to(device)

    # -----------------------------
    # baseline
    # -----------------------------
    print("\nBuilding baseline features...")
    X_train_base = build_baseline_feature_matrix(train_df, tokenizer, model, device)
    X_test_base = build_baseline_feature_matrix(test_df, tokenizer, model, device)

    y_train = train_df["label"].values
    y_test = test_df["label"].values

    base_clf = LogisticRegression(
        max_iter=2000,
        random_state=SEED
    )
    base_clf.fit(X_train_base, y_train)
    base_preds = base_clf.predict(X_test_base)

    evaluate_model("BASELINE (WHOLE ARTICLE BERT)", y_test, base_preds)

    # -----------------------------
    # segmentation-based model
    # -----------------------------
    print("\nBuilding segmentation-based features...")
    X_train_seg = build_segment_feature_matrix(train_df, tokenizer, model, device)
    X_test_seg = build_segment_feature_matrix(test_df, tokenizer, model, device)

    seg_clf = LogisticRegression(
        max_iter=2000,
        random_state=SEED
    )
    seg_clf.fit(X_train_seg, y_train)
    seg_preds = seg_clf.predict(X_test_seg)

    evaluate_model("SEGMENTATION + BERT + CONSISTENCY FEATURES", y_test, seg_preds)

    # quick comparison
    print(f"\n{'=' * 60}")
    print("FINAL COMPARISON")
    print(f"{'=' * 60}")

    base_acc = accuracy_score(y_test, base_preds)
    seg_acc = accuracy_score(y_test, seg_preds)

    base_f1 = f1_score(y_test, base_preds)
    seg_f1 = f1_score(y_test, seg_preds)

    print(f"Baseline Accuracy: {base_acc:.4f}")
    print(f"Segment  Accuracy: {seg_acc:.4f}")
    print(f"Baseline F1      : {base_f1:.4f}")
    print(f"Segment  F1      : {seg_f1:.4f}")

    if seg_f1 > base_f1:
        print("\nSegment-based model performed better on F1.")
    elif seg_f1 < base_f1:
        print("\nBaseline performed better on F1.")
    else:
        print("\nBoth models performed the same on F1.")


if __name__ == "__main__":
    main()