# Lab 4 – Text Feature Engineering Pipeline (Azure ML)
**Delson Fernandes – 60302101**

---

## Overview

This project implements an end-to-end feature engineering pipeline using Azure Machine Learning for the Amazon Electronics review dataset.

The objective of Lab 4 is to transform raw review data into structured, reusable machine learning features and prepare fully processed datasets for downstream model training (Assignment 2).

The pipeline is built using modular Azure ML components and executed as a pipeline job on Azure compute. It produces **four merged datasets (train, validation, test, deployment)** containing all engineered features.

---

## Objective

The goal of this lab is to:

- Convert raw text reviews into meaningful numerical features
- Ensure reproducible and modular feature engineering using Azure ML
- Avoid feature engineering during model training (required for Assignment 2)
- Generate clean, merged datasets with:
  - one row per review
  - preserved label (`overall`)
  - multiple feature types

---

## Pipeline Architecture

The final pipeline consists of the following stages:

1. Dataset Splitting (4-way)
2. Text Normalization
3. Review Length Feature Extraction
4. Sentiment Feature Extraction
5. TF-IDF Feature Generation
6. SBERT Embedding Generation
7. Feature Merging (for each split)

Pipeline execution command:

```bash
az ml job create --file pipelines/feature_pipeline.yml
```

---

## 1. Dataset Splitting

**Component:** `split_dataset`

The dataset is split into four partitions:

- Train (60%)
- Validation (15%)
- Test (15%)
- Deployment (10%)

This structure is required for Assignment 2.

Where possible, `review_year` is used to ensure the deployment set represents more recent data, helping simulate real-world production scenarios.

**Outputs:**
- `train`
- `val`
- `test`
- `deploy`

---

## 2. Text Normalization

**Component:** `normalize_text`

This step standardizes review text to improve feature quality.

Processing includes:
- lowercasing
- removing punctuation
- cleaning whitespace
- removing noise

Applied to:
- train
- validation
- test
- deployment

---

## 3. Review Length Features

**Component:** `review_length`

Generated features:
- `review_length_chars`
- `review_length_words`

These capture the size and verbosity of each review.

To ensure safe merging, the output is reduced to one row per:
- `asin`
- `reviewerID`

---

## 4. Sentiment Features

**Component:** `sentiment`

Sentiment is computed using VADER sentiment analysis.

Generated features:
- `sentiment_pos`
- `sentiment_neg`
- `sentiment_neu`
- `sentiment_compound`

Important design decision:
- This component **does NOT include the `overall` label**
- This prevents duplicate columns (`overall_x`, `overall_y`) during merging

---

## 5. TF-IDF Features

**Component:** `tfidf_features`

TF-IDF features are generated using `TfidfVectorizer`.

Configuration:
- stop words removed
- n-grams: (1,2)
- max features: 500

Important design:
- TF-IDF is **fit only on the training set**
- then applied to validation, test, and deployment sets
- avoids data leakage

Output format:
- **wide format (one row per review)**
- feature columns like:
  - `tfidf_battery`
  - `tfidf_quality`
  - `tfidf_sound quality`

---

## 6. SBERT Embedding Features

**Component:** `sbert_embeddings`

Semantic features are generated using:
- `all-MiniLM-L6-v2`

To ensure efficiency:
- text is truncated before encoding
- embeddings reduced to **32 dimensions**

Generated features:
- `sbert_0` to `sbert_31`

These embeddings capture deeper meaning beyond simple word frequencies.

---

## 7. Feature Merging

**Component:** `merge_features`

This is the most critical step.

The following datasets are merged:
- base split (contains `overall`)
- review length features
- sentiment features
- TF-IDF features
- SBERT features

Merge keys:
- `asin`
- `reviewerID`

Merge type:
- **inner join**

---

## Critical Merge Fixes

Several issues were identified and resolved:

### 1. Duplicate key problem
Feature components were producing multiple rows per review.

Fix:
```python
df.drop_duplicates(subset=["asin", "reviewerID"])
```

### 2. Duplicate label columns
Previously:
- `overall_x`
- `overall_y`

Fix:
- Only base dataset contains `overall`
- feature components exclude label

### 3. Many-to-many joins
Caused repeated rows after merging.

Fix:
- enforce one row per key before merging

### 4. TF-IDF format issue
Previously long format caused merge explosion.

Fix:
- converted to wide format

---

## Final Output Structure

Each merged dataset contains:

### Keys
- `asin`
- `reviewerID`

### Label
- `overall`

### Features

**Length**
- review_length_chars
- review_length_words

**Sentiment**
- sentiment_pos
- sentiment_neg
- sentiment_neu
- sentiment_compound

**TF-IDF**
- tfidf_... (hundreds of columns)

**SBERT**
- sbert_0 → sbert_31

Each row represents one unique review.

---

## Output Datasets

The pipeline produces four datasets:

- Train dataset
- Validation dataset
- Test dataset
- Deployment dataset

Each is stored as:

```bash
data.parquet
```

These datasets are ready to be used directly in Assignment 2.

---

## Why These Features Matter

Each feature type contributes differently:

- **Length features** → review structure
- **Sentiment features** → emotional tone
- **TF-IDF** → important keywords
- **SBERT embeddings** → semantic meaning

Together, they create a strong representation for machine learning models.

---

## Repository Structure

```bash
components/
  split_dataset/
  normalize_text/
  review_length/
  sentiment/
  tfidf_features/
  sbert_embeddings/
  merge_features/

pipelines/
  feature_pipeline.yml
```

---

## Key Improvements from Initial Version

- upgraded from 3-way split → 4-way split  
- applied features to ALL splits  
- fixed duplicate merge errors  
- preserved label correctly  
- integrated TF-IDF into final dataset  
- added SBERT embeddings  
- ensured one row per review  

---

## Final Outcome

The pipeline successfully:

- builds modular Azure ML components  
- processes raw data into structured features  
- produces four clean datasets  
- avoids data leakage  
- ensures merge correctness  
- prepares data for Assignment 2 training  

---

## Conclusion

This Lab 4 pipeline provides a complete, reusable feature engineering solution using Azure Machine Learning. The final datasets are fully prepared for training machine learning models without requiring any additional preprocessing.

The design ensures scalability, reproducibility, and compatibility with real-world ML workflows.
