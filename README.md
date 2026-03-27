# Challenge 1: Product Recommendation Engine

## Overview

A regional bank wants to identify which existing customers are most likely to adopt an additional banking product (e.g., open a new savings account, take out a mortgage, start an investment account). Your task is to build a recommendation scoring model that ranks customers by their adoption likelihood so the bank can target its next cross-sell campaign efficiently.

This is a **classification + ranking** problem. A model that can merely classify adopt/not-adopt is not enough -- the bank needs a ranked list of customers sorted by probability so it can prioritize outreach.

---

## Description

You are given three related tables containing customer demographics, their existing product holdings, and their digital banking activity. These tables must be **joined on `customer_id`** and cleaned before modeling.

The target variable is `adopted_new_product` (1 = adopted, 0 = did not adopt), provided in `train.csv` for training customers only.

### Key challenges:
- Tables must be joined and aggregated (e.g., count products per customer, flag product types, compute average balances)
- Missing values, inconsistent casing, duplicate rows, and orphaned foreign keys exist across all tables
- Some features are noise -- not every column is predictive
- The ranked output matters as much as the binary prediction

---

## Data Dictionary

### customers.csv (~252k rows)

| Column | Type | Description |
|--------|------|-------------|
| `customer_id` | string | **Primary key.** Unique customer identifier |
| `first_name` | string | Customer first name |
| `last_name` | string | Customer last name |
| `email` | string | Email address |
| `phone` | string | Phone number |
| `city` | string | City of residence |
| `state` | string | State abbreviation |
| `date_of_birth` | date | Customer date of birth |
| `account_open_date` | date | Date the customer first opened an account |
| `annual_income` | float | Self-reported annual income ($) |
| `credit_score` | int | Credit score (300-850 range) |
| `customer_satisfaction_score` | int | Latest survey satisfaction score (1-10) |

### products.csv (~834k rows)

| Column | Type | Description |
|--------|------|-------------|
| `account_id` | string | **Primary key.** Unique account identifier |
| `customer_id` | string | **Foreign key** to customers.csv |
| `product_type` | string | Product category: checking, savings, credit_card, mortgage, investment, auto_loan |
| `balance` | float | Current account balance ($) |
| `open_date` | date | Date the product was opened |
| `status` | string | Account status: active or closed |

**Note:** Each customer may have 1-6 products. You will likely need to aggregate this table (count products, flag types, compute average balances, etc.).

### digital_activity.csv (~252k rows)

| Column | Type | Description |
|--------|------|-------------|
| `customer_id` | string | **Foreign key** to customers.csv |
| `avg_monthly_logins` | int | Average monthly login count |
| `mobile_app_sessions_30d` | int | Mobile app sessions in the last 30 days |
| `online_transactions_30d` | int | Online transactions in the last 30 days |
| `preferred_channel` | string | Preferred banking channel: branch, online, mobile, phone |

### train.csv (175k rows)

| Column | Type | Description |
|--------|------|-------------|
| `customer_id` | string | Customer identifier |
| `adopted_new_product` | int | Target: 1 = adopted a new product, 0 = did not |

### eval.csv (75k rows)

| Column | Type | Description |
|--------|------|-------------|
| `customer_id` | string | Customer identifier (predict for these) |

---

## Evaluation Criteria

Your submission will be scored on **multiple metrics** to capture both classification accuracy and ranking quality:

| Metric | Weight | Description |
|--------|--------|-------------|
| **AUC-ROC** | Primary | Area under the ROC curve -- overall discrimination ability |
| **Precision@K** (K=100, 500, 1000) | Secondary | Of the top K customers you recommend, what fraction actually adopted? |
| **MAP@1000** | Secondary | Mean Average Precision over the top 1000 ranked customers |

### Baseline Benchmarks (your floor to beat)

See `benchmark.txt` for exact numbers. A Logistic Regression baseline achieves ~0.73 AUC-ROC. Your goal is to substantially exceed this.

---

## Submission Format

Submit a CSV file with **exactly two columns**, one row per `customer_id` in `eval.csv`:

```csv
customer_id,adoption_probability
C12345-67890,0.82
C98765-43210,0.15
...
```

- `customer_id` must match the IDs in `eval.csv`
- `adoption_probability` must be a float between 0 and 1
- Higher values = more likely to adopt

---

## Tips

1. **Start with EDA.** Understand the distributions, check for nulls, and look at how tables relate before jumping to modeling.
2. **Aggregate products.csv carefully.** Features like "number of active products," "has_mortgage," and "average balance across accounts" are likely informative.
3. **Handle messy categoricals.** You will find inconsistent casing and whitespace in several columns -- standardize before encoding.
4. **Watch for orphaned keys.** Some `customer_id` values in products.csv and digital_activity.csv do not exist in customers.csv. Decide how to handle them.
5. **Not every feature is useful.** Some columns may be noise. Feature selection or regularization can help.
6. **Think about ranking, not just classification.** Optimizing for AUC and Precision@K may require different approaches than optimizing for accuracy.
