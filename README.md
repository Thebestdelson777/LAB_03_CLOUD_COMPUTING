# Lab 4 – Text Feature Engineering Pipeline (Azure ML)
#Delson fernandes -60302101
## Overview

This project implements an end-to-end feature engineering pipeline using Azure Machine Learning for the Amazon Electronics review dataset.  

The objective is to transform raw review data into structured, reusable features and register them in Azure ML Feature Store for downstream modeling and training workflows.

The pipeline was built using modular Azure ML components and executed as a pipeline job on Azure compute.

---

## Pipeline Architecture

The pipeline consists of the following stages:

1. Dataset Split
2. Text Normalization
3. Review Length Feature Extraction
4. Sentiment Feature Extraction
5. TF-IDF Feature Generation
6. Feature Merging
7. Feature Store Registration

Pipeline execution command:

```bash
az ml job create --file pipelines/feature_pipeline.yml --stream
```

---

## 1. Dataset Splitting

Component: `split_dataset`

The sampled dataset (`features_v1_sampled`) is split into:

- Train (70%)
- Validation (15%)
- Test (15%)

This ensures proper separation for modeling and evaluation while preventing data leakage.

Outputs:
- `train`
- `val`
- `test`

---

## 2. Text Normalization

Component: `normalize_text`

This step standardizes review text by:

- Converting text to lowercase
- Cleaning and formatting text fields

This improves consistency before downstream feature extraction.

Applied to:
- Train
- Validation
- Test splits

---

## 3. Review Length Features

Component: `review_length`

Generated features:

- `review_length_chars`
- `review_length_words`

These features capture review verbosity and structural characteristics of the text.

---

## 4. Sentiment Features

Component: `sentiment`

Generated features:

- `sentiment_pos`
- `sentiment_neg`
- `sentiment_neu`
- `sentiment_compound`

These numerical scores represent the emotional polarity of the review text.

---

## 5. TF-IDF Features

Component: `tfidf_features`

TF-IDF vectors were generated from normalized review text to capture important term frequencies.

Due to memory and storage considerations, TF-IDF vectors were not merged into the final feature set registered in the Feature Store.

---

## 6. Feature Merging

Component: `merge_features`

This step combines structured features into a single dataset.

Final merged dataset includes:

### Entity Keys
- `reviewerID`
- `asin`

### Engineered Features
- `review_length_chars`
- `review_length_words`
- `sentiment_pos`
- `sentiment_neg`
- `sentiment_neu`
- `sentiment_compound`

### Contextual Features
- `price`
- `overall`
- `review_year`

Output:
- `data.parquet`

The output artifact was retrieved from:

```
azureml://.../datastores/workspaceblobstore/paths/azureml/<job-id>/out/
```

---

## Azure ML Feature Store Integration

### Feature Store
```
amazon-electronics-fs-60302101
```

### Entities Created

1. Reviewer Entity
   - Index Column: `reviewerID`
   - Type: string

2. Product Entity
   - Index Column: `asin`
   - Type: string

Entity creation command:

```bash
az ml feature-store-entity create \
  --file feature_store/entities/<entity_file>.yml \
  --resource-group rg-60302101 \
  --feature-store-name amazon-electronics-fs-60302101
```

---

## Feature Specification

File:
```
feature_store/FeatureSetSpec.yaml
```

Defines:

- Source Parquet location
- Index columns
- Feature schema
- Data types

---

## Feature Set Definition

File:
```
feature_store/feature_set_amazon_review_text_features.yml
```

Feature set name:
```
amazon_review_text_features
```

Version:
```
1
```

Registration command:

```bash
az ml feature-set create \
  --file feature_store/feature_set_amazon_review_text_features.yml \
  --resource-group rg-60302101 \
  --feature-store-name amazon-electronics-fs-60302101
```

Verification command:

```bash
az ml feature-set show \
  --name amazon_review_text_features \
  --version 1 \
  --resource-group rg-60302101 \
  --feature-store-name amazon-electronics-fs-60302101
```

---

## Repository Structure

```
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

feature_store/
  FeatureSetSpec.yaml
  entities/
    reviewer_entity.yml
    asin_entity.yml
  feature_set_amazon_review_text_features.yml
```

---

## Final Outcome

- Modular Azure ML components implemented
- End-to-end feature engineering pipeline executed successfully
- Feature-enriched dataset generated
- Feature entities defined
- Feature set registered in Azure ML Feature Store
- Versioned and tracked via GitHub
