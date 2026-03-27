"""
Shared utilities for generating synthetic banking competition datasets.
Core technique from make_synthetic.py: make_classification/make_regression -> pd.cut mapping
preserves predictive signal through monotonic binning.
"""

import warnings
import pandas as pd
import numpy as np
import random
import os

from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_recall_curve, auc,
    mean_squared_error, mean_absolute_error, r2_score, roc_curve,
)
from sklearn.preprocessing import StandardScaler

from faker import Faker

fake = Faker()

# ---------------------------------------------------------------------------
# Faker value pools -- pre-generate to avoid per-row faker calls at scale
# ---------------------------------------------------------------------------
POOL_SIZE = 10000

def _build_pools(seed=42):
    Faker.seed(seed)
    random.seed(seed)
    return {
        "first_names": [fake.first_name() for _ in range(POOL_SIZE)],
        "last_names": [fake.last_name() for _ in range(POOL_SIZE)],
        "cities": [fake.city() for _ in range(5000)],
        "states": list({fake.state_abbr() for _ in range(500)}),
        "companies": [fake.company() for _ in range(POOL_SIZE)],
        "emails": ["fake_" + fake.free_email() for _ in range(POOL_SIZE)],
        "phones": [
            f"(555){random.randint(100,999)}-{random.randint(1000,9999)}"
            for _ in range(POOL_SIZE)
        ],
        "street_addresses": [
            f"{fake.building_number()} {fake.street_name()} Fake St."
            for _ in range(5000)
        ],
    }

POOLS = _build_pools()


def sample_pool(pool_name, n):
    return np.random.choice(POOLS[pool_name], size=n, replace=True)


# ---------------------------------------------------------------------------
# ID generation (vectorized)
# ---------------------------------------------------------------------------
def generate_ids(n, prefix=""):
    """Generate *n* unique IDs.  Wider range avoids collisions at scale."""
    oversample = max(n + 1000, int(n * 1.05))
    p1 = np.random.randint(100, 99_999, size=oversample)
    p2 = np.random.randint(10_000, 999_999, size=oversample)
    seen = set()
    unique = []
    for a, b in zip(p1, p2):
        tag = f"{prefix}{a}-{b}"
        if tag not in seen:
            seen.add(tag)
            unique.append(tag)
            if len(unique) == n:
                break
    return np.array(unique)


# ---------------------------------------------------------------------------
# Base data generation
# ---------------------------------------------------------------------------
def generate_base_classification(n_samples, n_features, weights,
                                  flip_y=0.03, class_sep=1.0,
                                  n_informative=None, random_state=42):
    if n_informative is None:
        n_informative = n_features
    n_redundant = n_features - n_informative
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=0,
        weights=weights,
        n_classes=2,
        flip_y=flip_y,
        class_sep=class_sep,
        random_state=random_state,
    )
    return X, y


def generate_base_regression(n_samples, n_features, n_informative=None,
                              noise=30.0, random_state=42):
    if n_informative is None:
        n_informative = n_features
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=noise,
        random_state=random_state,
    )
    return X, y


# ---------------------------------------------------------------------------
# Feature mapping (pd.cut preserves monotonic ordering → predictive signal)
# ---------------------------------------------------------------------------
def map_to_range(series, low, high, n_bins=1000, decimals=2):
    """Map continuous sklearn feature to a realistic numeric range."""
    max_unique = int((high - low) * (10 ** decimals)) + 1
    n_bins = min(n_bins, max_unique)
    labels = np.round(np.linspace(low, high, n_bins), decimals)
    # Ensure labels are strictly unique after rounding
    labels = np.unique(labels)
    n_bins = len(labels)
    result = pd.cut(series, bins=n_bins, labels=labels, include_lowest=True)
    result = result.astype("float64")
    step = (high - low) / n_bins
    jitter = np.random.uniform(-step / 3, step / 3, size=len(result))
    result = np.round(result + jitter, decimals)
    return np.clip(result, low, high)


def map_to_int_range(series, low, high):
    """Map continuous sklearn feature to an integer range."""
    n_bins = high - low + 1
    labels = list(range(low, high + 1))
    result = pd.cut(series, bins=n_bins, labels=labels, include_lowest=True)
    return result.astype("int64")


def map_to_categories(series, categories):
    """Map continuous sklearn feature to categorical labels."""
    result = pd.cut(series, bins=len(categories), labels=categories,
                    include_lowest=True)
    return result.astype("str")


def map_to_binary(series, threshold_pct=50):
    """Map continuous feature to binary based on percentile threshold."""
    threshold = np.percentile(series, threshold_pct)
    return (series > threshold).astype(int)


# ---------------------------------------------------------------------------
# Data quality injection
# ---------------------------------------------------------------------------
def inject_nulls(df, pct, exclude_cols=None):
    """Scatter random NaN values across eligible columns."""
    df = df.copy()
    if exclude_cols is None:
        exclude_cols = []
    eligible = [c for c in df.columns if c not in exclude_cols]
    if not eligible:
        return df

    n_nulls = int(len(df) * pct)
    row_idx = np.random.randint(0, len(df), size=n_nulls)
    col_idx = np.random.randint(0, len(eligible), size=n_nulls)

    for ci in range(len(eligible)):
        rows_for_col = row_idx[col_idx == ci]
        if len(rows_for_col) > 0:
            df.iloc[rows_for_col, df.columns.get_loc(eligible[ci])] = np.nan
    return df


def inject_messiness(df, categorical_cols=None, string_cols=None,
                     numeric_cols=None, duplicate_pct=0.01,
                     outlier_pct=0.005, whitespace_pct=0.05,
                     case_pct=0.10):
    """Add realistic data quality problems."""
    df = df.copy()
    n = len(df)

    if categorical_cols:
        transforms = [str.upper, str.lower, str.title,
                      lambda x: " " + x, lambda x: x + " "]
        for col in categorical_cols:
            mask = np.random.random(n) < case_pct
            choices = np.random.randint(0, len(transforms), size=n)
            for ti, fn in enumerate(transforms):
                apply_mask = mask & (choices == ti)
                if apply_mask.any():
                    df.loc[apply_mask, col] = (
                        df.loc[apply_mask, col]
                        .apply(lambda x: fn(str(x)) if pd.notna(x) else x)
                    )

    if string_cols:
        for col in string_cols:
            mask = np.random.random(n) < whitespace_pct
            df.loc[mask, col] = (
                df.loc[mask, col]
                .apply(lambda x: " " + str(x) + " " if pd.notna(x) else x)
            )

    if numeric_cols:
        for col in numeric_cols:
            mask = np.random.random(n) < outlier_pct
            if mask.any():
                col_std = df[col].std()
                col_mean = df[col].mean()
                signs = np.random.choice([-1, 1], size=mask.sum())
                df.loc[mask, col] = col_mean + signs * col_std * 10

    n_dupes = int(n * duplicate_pct)
    if n_dupes > 0:
        dupes = df.iloc[np.random.choice(n, size=n_dupes, replace=True)].copy()
        df = pd.concat([df, dupes], ignore_index=True)

    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def add_orphaned_keys(df, fk_col, pct=0.005):
    """Replace a fraction of FK values with IDs that won't match the parent."""
    df = df.copy()
    mask = np.random.random(len(df)) < pct
    n_orphans = mask.sum()
    if n_orphans > 0:
        df.loc[mask, fk_col] = generate_ids(n_orphans, prefix="ORPHAN-")
    return df


# ---------------------------------------------------------------------------
# Train / eval split (Kaggle-style)
# ---------------------------------------------------------------------------
def train_eval_split(df, id_col, target_cols, eval_pct=0.30,
                     random_state=42):
    """Return train (with labels), eval (no labels), answer_key DataFrames."""
    if isinstance(target_cols, str):
        target_cols = [target_cols]

    ids = df[id_col].unique()
    rng = np.random.RandomState(random_state)
    rng.shuffle(ids)
    split = int(len(ids) * (1 - eval_pct))

    train_ids = set(ids[:split])
    eval_ids = set(ids[split:])

    train_df = (df[df[id_col].isin(train_ids)][[id_col] + target_cols]
                .drop_duplicates())
    eval_df = (df[df[id_col].isin(eval_ids)][[id_col]]
               .drop_duplicates())
    answer_key = (df[df[id_col].isin(eval_ids)][[id_col] + target_cols]
                  .drop_duplicates())

    return train_df, eval_df, answer_key, np.array(list(train_ids)), np.array(list(eval_ids))


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------
def _precision_at_k(y_true, y_scores, k):
    top_k = np.argsort(y_scores)[::-1][:k]
    return np.mean(y_true[top_k])


def _average_precision_at_k(y_true, y_scores, k):
    top_k = np.argsort(y_scores)[::-1][:k]
    hits = y_true[top_k]
    n_rel = 0
    precisions = []
    for i, hit in enumerate(hits):
        if hit == 1:
            n_rel += 1
            precisions.append(n_rel / (i + 1))
    return np.mean(precisions) if precisions else 0.0


def _recall_at_fpr(y_true, y_scores, target_fpr=0.05):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    valid = np.where(fpr <= target_fpr)[0]
    return tpr[valid[-1]] if len(valid) > 0 else 0.0


def _fit_and_predict(model, X_train, y_train, X_eval):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xev = scaler.transform(X_eval)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model.fit(Xtr, y_train)
    return model, Xtr, Xev


# ---------------------------------------------------------------------------
# Problem-specific benchmarks
# ---------------------------------------------------------------------------
def benchmark_recommendation(X_train, y_train, X_eval, y_eval):
    """AUC-ROC + Precision@K + MAP@K."""
    y_eval = np.asarray(y_eval)
    results = {}
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000,
                                                   random_state=42),
        "Random Forest (n=100)": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1),
    }
    for name, model in models.items():
        model, _, Xev = _fit_and_predict(model, X_train, y_train, X_eval)
        y_prob = model.predict_proba(Xev)[:, 1]
        results[name] = {
            "AUC-ROC": roc_auc_score(y_eval, y_prob),
            "Precision@100": _precision_at_k(y_eval, y_prob, 100),
            "Precision@500": _precision_at_k(y_eval, y_prob, 500),
            "Precision@1000": _precision_at_k(y_eval, y_prob, 1000),
            "MAP@1000": _average_precision_at_k(y_eval, y_prob, 1000),
        }
    return results


def benchmark_fraud(X_train, y_train, X_eval, y_eval):
    """AUC-PR + F1 + Recall@5%FPR."""
    y_eval = np.asarray(y_eval)
    results = {}
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000,
                                                   random_state=42),
        "Random Forest (n=100)": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1),
    }
    for name, model in models.items():
        model, _, Xev = _fit_and_predict(model, X_train, y_train, X_eval)
        y_prob = model.predict_proba(Xev)[:, 1]
        y_pred = model.predict(Xev)
        prec, rec, _ = precision_recall_curve(y_eval, y_prob)
        results[name] = {
            "AUC-PR": auc(rec, prec),
            "F1": f1_score(y_eval, y_pred),
            "Recall@5%FPR": _recall_at_fpr(y_eval, y_prob, 0.05),
        }
    return results


def benchmark_cashflow(X_train_reg, y_train_reg, X_eval_reg, y_eval_reg,
                       X_train_cls, y_train_cls, X_eval_cls, y_eval_cls):
    """Regression: RMSE/MAE/R2.  Classification: AUC/F1."""
    y_eval_reg = np.asarray(y_eval_reg)
    y_eval_cls = np.asarray(y_eval_cls)
    results = {"regression": {}, "classification": {}}

    for name, model in [
        ("Linear Regression", LinearRegression()),
        ("RF Regressor (n=100)",
         RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
    ]:
        model, _, Xev = _fit_and_predict(model, X_train_reg, y_train_reg,
                                         X_eval_reg)
        y_pred = model.predict(Xev)
        results["regression"][name] = {
            "RMSE": np.sqrt(mean_squared_error(y_eval_reg, y_pred)),
            "MAE": mean_absolute_error(y_eval_reg, y_pred),
            "R2": r2_score(y_eval_reg, y_pred),
        }

    for name, model in [
        ("Logistic Regression",
         LogisticRegression(max_iter=1000, random_state=42)),
        ("RF Classifier (n=100)",
         RandomForestClassifier(n_estimators=100, random_state=42,
                                n_jobs=-1)),
    ]:
        model, _, Xev = _fit_and_predict(model, X_train_cls, y_train_cls,
                                         X_eval_cls)
        y_prob = model.predict_proba(Xev)[:, 1]
        y_pred = model.predict(Xev)
        results["classification"][name] = {
            "AUC-ROC": roc_auc_score(y_eval_cls, y_prob),
            "F1": f1_score(y_eval_cls, y_pred),
        }

    return results


# ---------------------------------------------------------------------------
# Benchmark report writers
# ---------------------------------------------------------------------------
def write_recommendation_benchmark(results, output_path, positive_rate):
    path = os.path.join(output_path, "benchmark.txt")
    with open(path, "w") as f:
        f.write("--- Product Recommendation Benchmark ---\n")
        f.write("Train/Eval split: 70/30\n")
        f.write(f"Positive class rate: {positive_rate:.1%}\n\n")
        for name, m in results.items():
            f.write(f"{name}:\n")
            f.write(f"  AUC-ROC:        {m['AUC-ROC']:.4f}\n")
            f.write(f"  Precision@100:  {m['Precision@100']:.4f}\n")
            f.write(f"  Precision@500:  {m['Precision@500']:.4f}\n")
            f.write(f"  Precision@1000: {m['Precision@1000']:.4f}\n")
            f.write(f"  MAP@1000:       {m['MAP@1000']:.4f}\n\n")
    print(f"  Benchmark written to {path}")


def write_fraud_benchmark(results, output_path, fraud_rate):
    path = os.path.join(output_path, "benchmark.txt")
    with open(path, "w") as f:
        f.write("--- Fraud Detection Benchmark ---\n")
        f.write("Train/Eval split: 70/30\n")
        f.write(f"Fraud rate: {fraud_rate:.1%}\n\n")
        for name, m in results.items():
            f.write(f"{name}:\n")
            f.write(f"  AUC-PR:        {m['AUC-PR']:.4f}\n")
            f.write(f"  F1:            {m['F1']:.4f}\n")
            f.write(f"  Recall@5%FPR:  {m['Recall@5%FPR']:.4f}\n\n")
    print(f"  Benchmark written to {path}")


def write_cashflow_benchmark(results, output_path, shortfall_rate):
    path = os.path.join(output_path, "benchmark.txt")
    with open(path, "w") as f:
        f.write("--- Cash Flow Shortfall Benchmark ---\n")
        f.write("Train/Eval split: 70/30\n")
        f.write(f"Shortfall rate: {shortfall_rate:.1%}\n\n")
        f.write("Regression (cashflow_shortfall_amount):\n")
        for name, m in results["regression"].items():
            f.write(f"  {name}:\n")
            f.write(f"    RMSE: ${m['RMSE']:,.2f}\n")
            f.write(f"    MAE:  ${m['MAE']:,.2f}\n")
            f.write(f"    R2:   {m['R2']:.4f}\n\n")
        f.write("Classification (shortfall_flag):\n")
        for name, m in results["classification"].items():
            f.write(f"  {name}:\n")
            f.write(f"    AUC-ROC: {m['AUC-ROC']:.4f}\n")
            f.write(f"    F1:      {m['F1']:.4f}\n\n")
    print(f"  Benchmark written to {path}")


def benchmark_loan_default(X_train_reg, y_train_reg, X_eval_reg, y_eval_reg,
                            X_train_cls, y_train_cls, X_eval_cls, y_eval_cls):
    """Regression: RMSE/MAE/R2 on days_to_default.  Classification: AUC/F1 on default_flag."""
    y_eval_reg = np.asarray(y_eval_reg)
    y_eval_cls = np.asarray(y_eval_cls)
    results = {"regression": {}, "classification": {}}

    for name, model in [
        ("Linear Regression", LinearRegression()),
        ("RF Regressor (n=100)",
         RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
    ]:
        model, _, Xev = _fit_and_predict(model, X_train_reg, y_train_reg,
                                         X_eval_reg)
        y_pred = model.predict(Xev)
        results["regression"][name] = {
            "RMSE": np.sqrt(mean_squared_error(y_eval_reg, y_pred)),
            "MAE": mean_absolute_error(y_eval_reg, y_pred),
            "R2": r2_score(y_eval_reg, y_pred),
        }

    for name, model in [
        ("Logistic Regression",
         LogisticRegression(max_iter=1000, random_state=42)),
        ("RF Classifier (n=100)",
         RandomForestClassifier(n_estimators=100, random_state=42,
                                n_jobs=-1)),
    ]:
        model, _, Xev = _fit_and_predict(model, X_train_cls, y_train_cls,
                                         X_eval_cls)
        y_prob = model.predict_proba(Xev)[:, 1]
        y_pred = model.predict(Xev)
        results["classification"][name] = {
            "AUC-ROC": roc_auc_score(y_eval_cls, y_prob),
            "F1": f1_score(y_eval_cls, y_pred),
        }

    return results


def write_loan_default_benchmark(results, output_path, default_rate):
    path = os.path.join(output_path, "benchmark.txt")
    with open(path, "w") as f:
        f.write("--- Loan Default Benchmark ---\n")
        f.write("Train/Eval split: 70/30\n")
        f.write(f"Default rate: {default_rate:.1%}\n\n")
        f.write("Regression (days_to_early_default):\n")
        for name, m in results["regression"].items():
            f.write(f"  {name}:\n")
            f.write(f"    RMSE: {m['RMSE']:.1f} days\n")
            f.write(f"    MAE:  {m['MAE']:.1f} days\n")
            f.write(f"    R2:   {m['R2']:.4f}\n\n")
        f.write("Classification (default_flag):\n")
        for name, m in results["classification"].items():
            f.write(f"  {name}:\n")
            f.write(f"    AUC-ROC: {m['AUC-ROC']:.4f}\n")
            f.write(f"    F1:      {m['F1']:.4f}\n\n")
    print(f"  Benchmark written to {path}")
