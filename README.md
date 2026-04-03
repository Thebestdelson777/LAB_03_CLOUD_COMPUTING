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

* Convert raw text reviews into meaningful numerical features
* Ensure reproducible and modular feature engineering using Azure ML
* Avoid feature engineering during model training (required for Assignment 2)
* Generate clean, merged datasets with:

  * one row per review
  * preserved label (`overall`)
  * multiple feature types

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

* Train (60%)
* Validation (15%)
* Test (15%)
* Deployment (10%)

### 🔥 Key Fix

A unique identifier is created:

```python
df["record_id"] = df.index.astype(str)
```

This ensures:

* one unique row per review
* correct alignment across all feature components

**Outputs:**

* `train`
* `val`
* `test`
* `deploy`

---

## 2. Text Normalization

**Component:** `normalize_text`

This step standardizes review text to improve feature quality.

Processing includes:

* lowercasing
* removing punctuation
* cleaning whitespace

Applied to all splits:

* train
* validation
* test
* deployment

---

## 3. Review Length Features

**Component:** `review_length`

Generated features:

* `review_length_chars`
* `review_length_words`

These capture review size and verbosity.

### ✅ Important Fix

* Uses `record_id` to preserve row-level alignment
* No deduplication is performed

---

## 4. Sentiment Features

**Component:** `sentiment`

Sentiment is computed using VADER.

Generated features:

* `sentiment_pos`
* `sentiment_neg`
* `sentiment_neu`
* `sentiment_compound`

### ✅ Important Design


* Prevents duplicate columns (`overall_x`, `overall_y`)

---

## 5. TF-IDF Features

**Component:** `tfidf_features`

TF-IDF features are generated using `TfidfVectorizer`.

Configuration:

* stop words removed
* n-grams: (1,1)
* max features: ~300–500

### 🔥 Critical Fix

* Previously reduced dataset incorrectly using `drop_duplicates`
* Now preserves full dataset using `record_id`

### Output format:

* wide format (one row per review)
* columns like:

  * `tfidf_battery`
  * `tfidf_quality`

---

## 6. SBERT Embedding Features

**Component:** `sbert_embeddings`

Semantic features generated using:

* `all-MiniLM-L6-v2`

Optimizations:

* text truncation
* reduced embedding size → **32 dimensions**

Generated features:

* `sbert_0` → `sbert_31`

---

## 7. Feature Merging

**Component:** `merge_features`

This is the most critical stage.

### 🔥 FINAL FIX (IMPORTANT)

Merge key:

```python
record_id
```

Merge type:

```python
left join
```

### Why this matters:

* preserves ALL rows from split
* prevents dataset shrinkage
* ensures correct feature alignment

Merged datasets include:

* base dataset (with `overall`)
* length features
* sentiment features
* TF-IDF features
* SBERT features

---

## Critical Issues Identified & Fixed

### ❌ Problem 1: Dataset Shrinking (~9k rows)

Cause:

* `drop_duplicates(["asin", "reviewerID"])`

Fix:

* replaced with `record_id`

---

### ❌ Problem 2: Wrong Merge Key

Cause:

* using non-unique keys (`asin`, `reviewerID`)

Fix:

* introduced `record_id`

---

### ❌ Problem 3: Inner Join Loss

Cause:

* `how="inner"`

Fix:

* changed to `how="left"`

---

### ❌ Problem 4: TF-IDF Misalignment

Cause:

* inconsistent row indexing

Fix:

* aligned all outputs using `record_id`

---

## Final Output Structure

Each dataset contains:

### Keys

* `record_id`
* `asin`
* `reviewerID`

### Label

* `overall`

### Features

**Length**

* review_length_chars
* review_length_words

**Sentiment**

* sentiment_pos
* sentiment_neg
* sentiment_neu
* sentiment_compound

**TF-IDF**

* tfidf_* (hundreds of columns)

**SBERT**

* sbert_0 → sbert_31

---

## Output Datasets

The pipeline produces four datasets:

* Training dataset (~180k rows)
* Validation dataset (~45k rows)
* Test dataset (~45k rows)
* Deployment dataset (~30k rows)

Each stored as:

```bash
data.parquet
```

---

## Why These Features Matter

| Feature Type | Purpose            |
| ------------ | ------------------ |
| Length       | review structure   |
| Sentiment    | emotional tone     |
| TF-IDF       | keyword importance |
| SBERT        | semantic meaning   |

This combination creates a strong feature space for machine learning.

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

* Added **record_id for correct row alignment**
* Fixed merge shrinking issue
* Switched from inner → left joins
* Removed incorrect deduplication
* Ensured full dataset propagation
* Added SBERT embeddings
* Integrated TF-IDF correctly
* Produced 4 final datasets

---

## Final Outcome

The pipeline successfully:

* processes raw data into structured features
* preserves full dataset size
* avoids data leakage
* ensures correct feature alignment
* produces 4 clean datasets ready for training

---

## Conclusion

This Lab 4 pipeline provides a complete and production-style feature engineering workflow using Azure Machine Learning.

The final datasets are fully prepared for Assignment 2 and can be used directly for training without additional preprocessing.

