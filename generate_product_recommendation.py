#!/usr/bin/env python3
"""
Dataset 1 — Product Recommendation
Problem type: Classification + Ranking
Tables: customers.csv, products.csv, digital_activity.csv
Target: adopted_new_product (0/1, ~35 % positive)
Evaluation: AUC-ROC, Precision@K, MAP@K
"""

import pandas as pd
import numpy as np
import random
import os

from synthetic_utils import (
    generate_base_classification, map_to_range, map_to_int_range,
    map_to_categories, generate_ids, sample_pool, inject_nulls,
    inject_messiness, add_orphaned_keys, train_eval_split,
    benchmark_recommendation, write_recommendation_benchmark, fake,
)

RANDOM_STATE = 42
N_CUSTOMERS = 250_000
OUTPUT_DIR = "output/product_recommendation"

PRODUCT_TYPES_BY_COUNT = {
    1: ["checking"],
    2: ["checking", "savings"],
    3: ["checking", "savings", "credit_card"],
    4: ["checking", "savings", "credit_card", "mortgage"],
    5: ["checking", "savings", "credit_card", "mortgage", "investment"],
    6: ["checking", "savings", "credit_card", "mortgage", "investment",
        "auto_loan"],
}


def main():
    print("=" * 60)
    print("  Product Recommendation Dataset Generator")
    print("=" * 60)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.random.seed(RANDOM_STATE)
    random.seed(RANDOM_STATE)

    # ------------------------------------------------------------------
    # 1. Base predictive features
    # ------------------------------------------------------------------
    print("\n[1/7] Generating base features (make_classification)…")
    X, y = generate_base_classification(
        n_samples=N_CUSTOMERS, n_features=10,
        weights=[0.65, 0.35], flip_y=0.08, class_sep=0.5,
        n_informative=7, random_state=RANDOM_STATE,
    )
    pos_rate = y.mean()
    print(f"      Samples: {N_CUSTOMERS:,}  |  Positive rate: {pos_rate:.1%}")

    # ------------------------------------------------------------------
    # 2. Map features to realistic domains
    # ------------------------------------------------------------------
    print("[2/7] Mapping features to realistic domains…")
    customer_ids = generate_ids(N_CUSTOMERS, prefix="C")

    annual_income     = map_to_range(pd.Series(X[:, 0]), 20_000, 300_000)
    credit_score      = map_to_int_range(pd.Series(X[:, 1]), 300, 850)
    satisfaction      = map_to_int_range(pd.Series(X[:, 2]), 1, 10)
    num_products_raw  = map_to_int_range(pd.Series(X[:, 3]), 1, 6)
    balance_signal    = map_to_range(pd.Series(X[:, 4]), 500, 150_000)
    # X[:,5] reserved for product-type diversity (used indirectly)
    monthly_logins    = map_to_int_range(pd.Series(X[:, 6]), 0, 60)
    mobile_sessions   = map_to_int_range(pd.Series(X[:, 7]), 0, 120)
    online_txns       = map_to_int_range(pd.Series(X[:, 8]), 0, 80)
    preferred_channel = map_to_categories(
        pd.Series(X[:, 9]), ["branch", "online", "mobile", "phone"])

    # ------------------------------------------------------------------
    # 3. Build customers.csv
    # ------------------------------------------------------------------
    print("[3/7] Building customers.csv…")
    dates_pool = pd.date_range("1945-01-01", "2007-01-01", freq="D")
    acct_dates_pool = pd.date_range("2010-01-01", "2025-06-01", freq="D")

    customers = pd.DataFrame({
        "customer_id": customer_ids,
        "first_name": sample_pool("first_names", N_CUSTOMERS),
        "last_name": sample_pool("last_names", N_CUSTOMERS),
        "email": sample_pool("emails", N_CUSTOMERS),
        "phone": sample_pool("phones", N_CUSTOMERS),
        "city": sample_pool("cities", N_CUSTOMERS),
        "state": sample_pool("states", N_CUSTOMERS),
        "date_of_birth": np.random.choice(dates_pool, size=N_CUSTOMERS),
        "account_open_date": np.random.choice(acct_dates_pool,
                                               size=N_CUSTOMERS),
        "annual_income": annual_income,
        "credit_score": credit_score,
        "customer_satisfaction_score": satisfaction,
    })
    print(f"      {len(customers):,} rows")

    # ------------------------------------------------------------------
    # 4. Build products.csv (expand 1-to-many)
    # ------------------------------------------------------------------
    print("[4/7] Building products.csv…")
    product_frames = []
    open_dates_pool = pd.date_range("2012-01-01", "2025-11-01", freq="D")

    for rank, ptype in enumerate(
            ["checking", "savings", "credit_card",
             "mortgage", "investment", "auto_loan"]):
        mask = num_products_raw.values >= (rank + 1)
        n = mask.sum()
        if n == 0:
            continue
        noise = np.random.normal(1.0, 0.30, size=n)
        balances = np.round(np.maximum(0, balance_signal.values[mask] * noise), 2)
        frame = pd.DataFrame({
            "account_id": generate_ids(n, prefix="A"),
            "customer_id": customer_ids[mask],
            "product_type": ptype,
            "balance": balances,
            "open_date": np.random.choice(open_dates_pool, size=n),
            "status": np.random.choice(
                ["active", "closed"], size=n, p=[0.85, 0.15]),
        })
        product_frames.append(frame)

    products = (pd.concat(product_frames, ignore_index=True)
                .sample(frac=1, random_state=RANDOM_STATE)
                .reset_index(drop=True))
    print(f"      {len(products):,} rows  "
          f"(avg {len(products)/N_CUSTOMERS:.1f} products/customer)")

    # ------------------------------------------------------------------
    # 5. Build digital_activity.csv
    # ------------------------------------------------------------------
    print("[5/7] Building digital_activity.csv…")
    digital = pd.DataFrame({
        "customer_id": customer_ids,
        "avg_monthly_logins": monthly_logins,
        "mobile_app_sessions_30d": mobile_sessions,
        "online_transactions_30d": online_txns,
        "preferred_channel": preferred_channel,
    })
    print(f"      {len(digital):,} rows")

    # ------------------------------------------------------------------
    # 6. Inject data-quality issues
    # ------------------------------------------------------------------
    print("[6/7] Injecting data-quality issues…")
    pk_cols_cust = ["customer_id"]
    customers = inject_nulls(customers, pct=0.12,
                             exclude_cols=pk_cols_cust)
    customers = inject_messiness(
        customers,
        categorical_cols=["state"],
        string_cols=["first_name", "last_name", "email", "city"],
        numeric_cols=["annual_income", "credit_score"],
    )

    products = inject_nulls(products, pct=0.10,
                            exclude_cols=["account_id", "customer_id"])
    products = inject_messiness(
        products,
        categorical_cols=["product_type", "status"],
        numeric_cols=["balance"],
    )
    products = add_orphaned_keys(products, "customer_id", pct=0.005)

    digital = inject_nulls(digital, pct=0.10,
                           exclude_cols=["customer_id"])
    digital = inject_messiness(
        digital,
        categorical_cols=["preferred_channel"],
        numeric_cols=["avg_monthly_logins", "mobile_app_sessions_30d",
                      "online_transactions_30d"],
    )
    digital = add_orphaned_keys(digital, "customer_id", pct=0.005)

    # ------------------------------------------------------------------
    # 7. Train/eval split + benchmarks
    # ------------------------------------------------------------------
    print("[7/7] Splitting train/eval and running benchmarks…")
    target_df = pd.DataFrame({
        "customer_id": customer_ids, "adopted_new_product": y})
    train, evaluation, answer_key, train_ids, eval_ids = train_eval_split(
        target_df, "customer_id", "adopted_new_product")

    # Benchmark on the clean flat feature matrix
    train_mask = np.isin(customer_ids, train_ids)
    eval_mask = np.isin(customer_ids, eval_ids)
    bench = benchmark_recommendation(
        X[train_mask], y[train_mask], X[eval_mask], y[eval_mask])
    write_recommendation_benchmark(bench, OUTPUT_DIR, pos_rate)

    # ------------------------------------------------------------------
    # Write CSVs
    # ------------------------------------------------------------------
    print("\nWriting CSVs…")
    customers.to_csv(f"{OUTPUT_DIR}/customers.csv", index=False)
    products.to_csv(f"{OUTPUT_DIR}/products.csv", index=False)
    digital.to_csv(f"{OUTPUT_DIR}/digital_activity.csv", index=False)
    train.to_csv(f"{OUTPUT_DIR}/train.csv", index=False)
    evaluation.to_csv(f"{OUTPUT_DIR}/eval.csv", index=False)
    answer_key.to_csv(f"{OUTPUT_DIR}/answer_key.csv", index=False)

    print(f"\nDone!  Files in {OUTPUT_DIR}/")
    print(f"  customers.csv          : {len(customers):>10,} rows")
    print(f"  products.csv           : {len(products):>10,} rows")
    print(f"  digital_activity.csv   : {len(digital):>10,} rows")
    print(f"  train.csv              : {len(train):>10,} rows")
    print(f"  eval.csv               : {len(evaluation):>10,} rows")
    print(f"  answer_key.csv         : {len(answer_key):>10,} rows")


if __name__ == "__main__":
    main()
