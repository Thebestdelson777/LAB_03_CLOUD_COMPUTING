#Delson fernandes-60302101
# Assignment 2 — Model Training, Automation, Hyperparameter Tuning, and Deployment with Azure ML

## Overview

This assignment extends the Lab 4 feature engineering pipeline into a complete MLOps workflow using Azure Machine Learning and Azure DevOps.

The main goal was to reuse the merged feature datasets generated in Lab 4, train a machine learning model on Azure ML compute, track experiments with MLflow, automate training through Azure DevOps, tune hyperparameters using a sweep job, register model versions in the Azure ML Model Registry, deploy the final model to a Managed Online Endpoint, and invoke the deployed endpoint using the deployment split.

This project follows a practical end-to-end workflow:

**feature datasets → training job → MLflow tracking → hyperparameter tuning → model registry → DevOps automation → endpoint deployment → deployment evaluation**

---

## Repository Branch

This work was completed on the following branch:

`assignment2_model_training`

---

## Azure ML Data Assets Used

The following merged feature datasets from Lab 4 were registered as Azure ML Data Assets and reused in this assignment:

- `amazon_review_merged_features_train`
- `amazon_review_merged_features_val`
- `amazon_review_merged_features_test`
- `amazon_review_merged_features_deploy`

---

## Model Choice

The selected model was **Logistic Regression**.

### Why Logistic Regression?

- simple and efficient  
- fast to train on Azure ML compute  
- easy to debug  
- suitable for high-dimensional features (SBERT + TF-IDF)  
- works well for automated runs and sweep jobs  

---

## Label Definition

The original `overall` rating was converted into a binary target:

- `1` if `overall >= 4`  
- `0` otherwise  

This converts the task into binary classification (positive vs not-positive).

---

## Features Used

The merged datasets already contained engineered features:

- **SBERT embeddings** (semantic text representation)  
- **TF-IDF features** (important word patterns)  
- **Sentiment features**  
  - sentiment_pos  
  - sentiment_neg  
  - sentiment_neu  
  - sentiment_compound  
- **Length features**  
  - review_length_chars  
  - review_length_words  

---

## Training Script

File: `src/train.py`

Steps:

1. Load train, validation, test datasets  
2. Create binary labels from `overall`  
3. Build feature matrix  
4. Train Logistic Regression model  
5. Evaluate on all splits  
6. Log metrics using MLflow  
7. Save model as `model.pkl`  

---

## Training Environment

Defined in: `env/conda.yml`

Includes:

- pandas  
- pyarrow  
- scikit-learn  
- joblib  
- mlflow  
- azureml-mlflow  

---

## Manual Training Job

Defined in: `jobs/train_job.yml`

### Results (All Features)

- Train accuracy ≈ 0.801  
- Validation accuracy ≈ 0.804  
- Test accuracy ≈ 0.800  

Model showed stable performance with no major overfitting.

---

## Hyperparameter Tuning (Sweep)

File: `jobs/sweep_job.yml`

### Tuned Parameters

- `c_value`  
- `max_iter`  

### Best Configuration

- `c_value = 0.89`  
- `max_iter = 300`  

Chosen for best balance between performance and efficiency.

---

## Feature Experiments

Three configurations tested:

1. All features (SBERT + TF-IDF + sentiment + length)  
2. SBERT only  
3. SBERT + TF-IDF  

### Results

- All features: validation accuracy ≈ 0.804  
- SBERT only: validation accuracy ≈ 0.806  
- SBERT + TF-IDF: validation accuracy ≈ 0.864  

### Best Model

**SBERT + TF-IDF performed the best**

Reason:
- SBERT → semantic meaning  
- TF-IDF → word importance  
- Combined → stronger representation  

---

## Azure DevOps Automation

Pipeline: `azure-pipelines.yml`

### What it does:

- triggers on push  
- authenticates Azure  
- submits training job  
- streams logs  

✔ Successful CI/CD automation achieved

---

## Model Registry

Model name:

`amazon-review-sentiment-model`

### Versions:

- v1 → initial model  
- v2 → tuned model  
- v3 → final best model  

---

## Model Deployment

Deployment uses Azure ML Managed Online Endpoint.

### Components:

- Model: latest registered version  
- Script: `src/score.py`  
- Env: `env/inference_conda.yml`  
- Config: `jobs/deployment.yml`  

---

## Endpoint Invocation

Script: `src/invoke_endpoint.py`

Steps:

1. Load deployment dataset  
2. Build features  
3. Send requests to endpoint  
4. Get predictions  
5. Compare with true labels  

### Result

- **Deployment accuracy: 0.8652**

---

## Final Model Summary

- Model: Logistic Regression  
- Features: SBERT + TF-IDF  
- Hyperparameters:
  - C = 0.89  
  - max_iter = 300  

### Final Performance

- Validation ≈ 0.864  
- Test ≈ 0.862  
- Deployment ≈ 0.8652  

---

## Deliverables Completed

- Azure ML training job  
- MLflow tracking  
- Hyperparameter sweep  
- Model registration  
- Azure DevOps pipeline  
- Endpoint deployment  
- Deployment evaluation  

---

## Conclusion

This project implemented a complete end-to-end Azure ML workflow:

- training  
- tracking  
- tuning  
- automation  
- deployment  

The final model using **SBERT + TF-IDF** achieved the best performance and was successfully deployed.
