# LAB_03_CLOUD_COMPUTING
"""
DSAI3202 – Lab 3
Data Preprocessing in Azure Databricks
Lakehouse Architecture (Bronze → Silver → Gold)

Overview
--------
In this lab, we implemented a data preprocessing pipeline using Azure Databricks
and Apache Spark. The goal was to transform Amazon Electronics review data
into a curated dataset ready for analytics and machine learning.

Architecture
------------
We followed the Medallion (Lakehouse) architecture:

Bronze Layer:
    - Raw JSON files stored in Azure Data Lake Storage Gen2

Silver Layer:
    - Reviews converted to Parquet
    - Cleaned reviews
    - Enriched reviews (joined with metadata)

Gold Layer:
    - Final curated dataset: features_v1
    - Stored in Parquet format for analytics/ML use

Technologies Used
-----------------
- Azure Databricks
- Apache Spark (PySpark)
- Azure Data Lake Storage Gen2
- Parquet format
- Databricks Jobs (pipeline orchestration)

Notebook Structure (ETL Process)
---------------------------------

1) 01_load_and_clean_reviews
    - Connect to ADLS Gen2
    - Load Parquet reviews from Silver layer
    - Remove null critical fields (asin, reviewerID, overall)
    - Enforce rating between 1 and 5
    - Trim review text and remove short reviews
    - Write cleaned data to processed/clean_reviews/

2) 02_enrich_with_metadata
    - Load cleaned reviews
    - Load metadata JSON from Bronze layer
    - Select relevant columns (asin, title, brand, price)
    - Perform left join on asin
    - Write enriched data to processed/enriched_reviews/

3) 03_write_gold_features_v1
    - Load enriched reviews
    - Select final feature columns:
        asin, title, brand, price, reviewerID,
        overall, summary, reviewText,
        helpful, reviewTime, review_year
    - Write final dataset to curated/features_v1/

Pipeline Orchestration
----------------------
All three notebooks were orchestrated using a Databricks Job:

load_and_clean_reviews
        ↓
enrich_with_metadata
        ↓
write_gold_features_v1

Each task depends on the previous one to ensure proper ETL flow.

Final Data Lake Structure
--------------------------
raw/                → Bronze (JSON files)
processed/          → Silver (cleaned + enriched Parquet)
curated/features_v1 → Gold (ML-ready dataset)

Conclusion
----------
This lab demonstrates how to build an end-to-end preprocessing pipeline
using Azure Databricks and Spark. The final output is a validated,
enriched, and curated Gold dataset ready for analytics and machine learning.
""""""
DSAI3202 – Lab 3
Data Preprocessing in Azure Databricks
Lakehouse Architecture (Bronze → Silver → Gold)

Overview
--------
In this lab, we implemented a data preprocessing pipeline using Azure Databricks
and Apache Spark. The goal was to transform Amazon Electronics review data
into a curated dataset ready for analytics and machine learning.

Architecture
------------
We followed the Medallion (Lakehouse) architecture:

Bronze Layer:
    - Raw JSON files stored in Azure Data Lake Storage Gen2

Silver Layer:
    - Reviews converted to Parquet
    - Cleaned reviews
    - Enriched reviews (joined with metadata)

Gold Layer:
    - Final curated dataset: features_v1
    - Stored in Parquet format for analytics/ML use

Technologies Used
-----------------
- Azure Databricks
- Apache Spark (PySpark)
- Azure Data Lake Storage Gen2
- Parquet format
- Databricks Jobs (pipeline orchestration)

Notebook Structure (ETL Process)
---------------------------------

1) 01_load_and_clean_reviews
    - Connect to ADLS Gen2
    - Load Parquet reviews from Silver layer
    - Remove null critical fields (asin, reviewerID, overall)
    - Enforce rating between 1 and 5
    - Trim review text and remove short reviews
    - Write cleaned data to processed/clean_reviews/

2) 02_enrich_with_metadata
    - Load cleaned reviews
    - Load metadata JSON from Bronze layer
    - Select relevant columns (asin, title, brand, price)
    - Perform left join on asin
    - Write enriched data to processed/enriched_reviews/

3) 03_write_gold_features_v1
    - Load enriched reviews
    - Select final feature columns:
        asin, title, brand, price, reviewerID,
        overall, summary, reviewText,
        helpful, reviewTime, review_year
    - Write final dataset to curated/features_v1/

Pipeline Orchestration
----------------------
All three notebooks were orchestrated using a Databricks Job:

load_and_clean_reviews
        ↓
enrich_with_metadata
        ↓
write_gold_features_v1

Each task depends on the previous one to ensure proper ETL flow.

Final Data Lake Structure
--------------------------
raw/                → Bronze (JSON files)
processed/          → Silver (cleaned + enriched Parquet)
curated/features_v1 → Gold (ML-ready dataset)

Conclusion
----------
This lab demonstrates how to build an end-to-end preprocessing pipeline
using Azure Databricks and Spark. The final output is a validated,
enriched, and curated Gold dataset ready for analytics and machine learning.
"""
