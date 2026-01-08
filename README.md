
# NLP for Italian Substandard Text

This repository contains a complete NLP pipeline for the analysis of **Italian substandard texts**, such as autobiographical narratives, oral histories, and non-standard or spoken-like language.

The project combines linguistic preprocessing, lexical statistics, and semantic modeling to study texts that deviate from standard written Italian, with particular attention to thematic structure and lexical behavior.

---

## Project Goals

The main objectives of this project are:

- to preprocess and normalize Italian substandard text
- to measure lexical richness in a statistically robust way
- to segment texts according to semantic coherence rather than formal structure
- to extract meaningful keywords using linguistic and statistical filters

The pipeline is designed for **research and academic use**, especially in linguistics, digital humanities, and NLP applied to non-standard language.

---

## Pipeline Overview

The pipeline is composed of the following stages:

1. Text loading and normalization  
2. Tokenization  
3. Lexical richness analysis (STTR)  
4. Semantic segmentation using SBERT and cosine similarity  
5. Linguistic filtering and keyword extraction (TF-IDF)  
6. Output generation

Each stage is described in detail below.

---

## 1. Text Loading and Preprocessing

The input text is loaded from a UTF-8 encoded file and normalized to reduce noise and inconsistencies.

Preprocessing includes:
- whitespace normalization
- lowercasing
- punctuation removal
- apostrophe handling (e.g. “l’amico” → “l amico”)

This step ensures that downstream analyses are not affected by orthographic or formatting variation.

---

## 2. Tokenization

The cleaned text is tokenized using NLTK’s Italian tokenizer.

The resulting token list is used consistently across:
- lexical analysis (STTR)
- semantic segmentation
- keyword extraction

---

## 3. Lexical Richness Analysis (STTR)

Lexical richness is measured using the **Standardized Type–Token Ratio (STTR)**.

Procedure:
- the token list is divided into fixed-size chunks of 1000 tokens
- for each chunk, the Type–Token Ratio is computed
- the final STTR value is obtained by averaging across chunks

This method avoids the length bias of the traditional TTR and provides a stable measure of vocabulary variation.

Important note:
STTR chunks are purely statistical units and do not correspond to semantic or thematic segments.

---

## 4. Semantic Segmentation

Semantic segmentation aims to identify **topic shifts** in the text without relying on predefined sections or chapters.

The process combines linguistic and semantic constraints:

### Linguistic constraint
Candidate segmentation points are restricted to the end of grammatical chunks:
- Noun Phrases (NP)
- Verb Phrases (VP)

This prevents breaks in the middle of syntactic units.

### Semantic constraint
Semantic similarity is computed using **Sentence-BERT (SBERT)** embeddings.

Procedure:
- two contextual windows are extracted around a potential boundary
  - last 50 tokens of the current segment
  - first 50 tokens of the following text
- both windows are converted into dense semantic vectors
- cosine similarity between the vectors is calculated

If the similarity score falls below a threshold (default: 0.75), a semantic boundary is detected and a new segment is created.

This approach produces segments that are coherent from a thematic and narrative perspective.

---

## 5. Linguistic Filtering

Before keyword extraction, the text undergoes linguistic filtering using spaCy.

The pipeline retains only:
- NOUN (nouns)
- PROPN (proper nouns)
- selected PRON (io, tu, noi, voi)

Stopwords are removed, and lemmatized forms are used.

This filtering focuses the analysis on semantically meaningful content, which is particularly important in spoken or autobiographical texts.

---

## 6. Keyword Extraction (TF-IDF)

Keywords are extracted using **TF-IDF (Term Frequency–Inverse Document Frequency)**.

TF-IDF assigns higher weights to terms that:
- are frequent in the text
- but not overly common in general language use

This highlights words that are distinctive of the analyzed text rather than merely frequent.

The output is a ranked list of terms with their corresponding TF-IDF scores.

---

## 7. Output

The pipeline produces:
- STTR statistics (printed to console)
- number of semantic segments detected
- a CSV file containing TF-IDF keywords and scores

Example output file:
- `output/keywords_tfidf.csv`

---

## Repository Structure
