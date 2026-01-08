"""
Main entry point for the NLP pipeline on Italian substandard texts.

The pipeline performs:
- text loading and cleaning
- tokenization
- lexical richness analysis (STTR)
- semantic segmentation using SBERT and cosine similarity
- keyword extraction via TF-IDF
"""

import os
import string
import nltk
import spacy
import torch
import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.chunk import RegexpParser
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer


# =========================
# CONFIGURATION
# =========================

TEXT_PATH = "data/-----.txt"
OUTPUT_DIR = "output"

MIN_TOKENS_SEGMENT = 1000
SEMANTIC_THRESHOLD = 0.75
CONTEXT_WINDOW = 50

SPACY_MODEL = "it_core_news_lg"
SBERT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# =========================
# PREPROCESSING
# =========================

def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def clean_text(text: str) -> str:
    text = " ".join(text.split())
    text = text.replace("'", " ")
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.lower()


def tokenize(text: str) -> list[str]:
    return word_tokenize(text, language="italian")


# =========================
# LEXICAL ANALYSIS
# =========================

def compute_sttr(tokens: list[str], segment_size: int = 1000) -> list[float]:
    """
    Compute Standardized Type–Token Ratio (STTR)
    using fixed-size token chunks.
    """
    values = []
    for i in range(0, len(tokens), segment_size):
        segment = tokens[i:i + segment_size]
        if len(segment) < segment_size:
            break
        values.append(len(set(segment)) / len(segment))
    return values


# =========================
# SEMANTIC SEGMENTATION
# =========================

def semantic_segmentation(
    text: str,
    nlp,
    sbert,
    min_tokens: int,
    threshold: float,
    ctx_window: int
) -> list[list[str]]:
    """
    Segment text based on semantic shifts detected
    via SBERT cosine similarity.
    """
    grammar = r"""
      NP: {<DET>?<ADJ>*<NOUN>+}
      VP: {<VERB>+<ADV>+}
    """

    chunker = RegexpParser(grammar)
    doc = nlp(text)

    pos_idx = [(t.text, t.pos_, t.i) for t in doc if not t.is_space]
    tree = chunker.parse(pos_idx)

    chunk_ends = {
        leaves[-1][2]
        for sub in tree.subtrees()
        if sub.label() in ("NP", "VP")
        for leaves in [sub.leaves()]
    }

    segments, buffer, count = [], [], 0

    for tok, _, idx in pos_idx:
        buffer.append(tok)
        count += 1

        if count >= min_tokens and idx in chunk_ends:
            prev_ctx = " ".join(buffer[-ctx_window:])
            next_ctx = " ".join(
                [t.text for t in doc if t.i > idx][:ctx_window]
            )

            sim = torch.nn.functional.cosine_similarity(
                sbert.encode(prev_ctx, convert_to_tensor=True).unsqueeze(0),
                sbert.encode(next_ctx, convert_to_tensor=True).unsqueeze(0),
                dim=1
            ).item()

            if sim < threshold:
                segments.append(buffer.copy())
                buffer, count = [], 0

    if buffer:
        segments.append(buffer)

    return segments


# =========================
# TF-IDF
# =========================

def extract_tfidf(text: str, nlp) -> pd.DataFrame:
    """
    Extract keywords using TF-IDF
    after POS-based filtering.
    """
    allowed_pos = {"NOUN", "PROPN", "PRON"}
    allowed_pronouns = {"io", "tu", "noi", "voi"}

    sw = set(stopwords.words("italian"))
    tokens = []

    for tok in nlp(text):
        lemma = tok.lemma_.lower()
        if lemma in sw:
            continue
        if tok.pos_ in allowed_pos and tok.is_alpha:
            if tok.pos_ == "PRON" and lemma not in allowed_pronouns:
                continue
            tokens.append(lemma)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform([" ".join(tokens)])

    return pd.DataFrame({
        "term": vectorizer.get_feature_names_out(),
        "tfidf": X.toarray()[0]
    }).sort_values("tfidf", ascending=False)


# =========================
# MAIN PIPELINE
# =========================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    nlp = spacy.load(SPACY_MODEL)
    sbert = SentenceTransformer(SBERT_MODEL)

    raw_text = load_text(TEXT_PATH)
    clean = clean_text(raw_text)
    tokens = tokenize(clean)

    sttr = compute_sttr(tokens)
    segments = semantic_segmentation(
        clean, nlp, sbert,
        MIN_TOKENS_SEGMENT,
        SEMANTIC_THRESHOLD,
        CONTEXT_WINDOW
    )

    tfidf = extract_tfidf(clean, nlp)

    tfidf.to_csv(os.path.join(OUTPUT_DIR, "keywords_tfidf.csv"), index=False)

    print("Pipeline completed.")
    print(f"STTR mean: {sum(sttr)/len(sttr):.4f}")
    print(f"Semantic segments: {len(segments)}")


if __name__ == "__main__":
    main()
