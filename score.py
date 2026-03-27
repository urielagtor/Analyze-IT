#!/usr/bin/env python3
"""
Batch Scoring & Leaderboard Generator
======================================
Scans submissions/ for team folders, scores each challenge's CSV against
the answer keys, and writes leaderboard.csv + leaderboard.md.

Expected layout:
    submissions/
        team_alpha/
            product_recommendation.csv   → customer_id, adoption_probability
            fraud_detection.csv          → transaction_id, fraud_probability
            cashflow_shortfall.csv       → business_id, predicted_shortfall_amount, predicted_shortfall_flag
            loan_default.csv             → loan_id, predicted_days_to_default, predicted_default_flag
        team_beta/
            ...

Usage:
    python score.py                          # default dirs
    python score.py --submissions ./subs     # custom submissions folder
    python score.py --output ./results       # custom output folder
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
    roc_curve,
)

REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = REPO_ROOT / "output"

CHALLENGES = {
    "product_recommendation": {
        "answer_key": OUTPUT_DIR / "product_recommendation" / "answer_key.csv",
        "id_col": "customer_id",
    },
    "fraud_detection": {
        "answer_key": OUTPUT_DIR / "fraud_detection" / "answer_key.csv",
        "id_col": "transaction_id",
    },
    "cashflow_shortfall": {
        "answer_key": OUTPUT_DIR / "cashflow_shortfall" / "answer_key.csv",
        "id_col": "business_id",
    },
    "loan_default": {
        "answer_key": OUTPUT_DIR / "loan_default" / "answer_key.csv",
        "id_col": "loan_id",
    },
}

# ── Answer key caches ───────────────────────────────────────────────────────

_answer_keys: dict[str, pd.DataFrame] = {}


def _load_answer_key(challenge: str) -> pd.DataFrame:
    if challenge not in _answer_keys:
        path = CHALLENGES[challenge]["answer_key"]
        _answer_keys[challenge] = pd.read_csv(path)
    return _answer_keys[challenge]


# ── Per-challenge scoring ───────────────────────────────────────────────────


def _validate_submission(sub: pd.DataFrame, challenge: str, required_cols: list[str]):
    """Return list of error strings (empty = valid)."""
    errors = []
    for col in required_cols:
        if col not in sub.columns:
            errors.append(f"missing column '{col}'")
    if errors:
        return errors

    id_col = CHALLENGES[challenge]["id_col"]
    ak = _load_answer_key(challenge)
    sub_ids = set(sub[id_col].dropna())
    ak_ids = set(ak[id_col].dropna())
    missing = ak_ids - sub_ids
    if missing:
        errors.append(f"{len(missing)} IDs in answer key not found in submission")

    for col in required_cols:
        if col == id_col:
            continue
        nan_count = sub[col].isna().sum()
        if nan_count > 0:
            errors.append(f"{nan_count} NaN values in '{col}'")

    return errors


def score_product_recommendation(sub: pd.DataFrame) -> dict:
    challenge = "product_recommendation"
    id_col = "customer_id"
    prob_col = "adoption_probability"

    errors = _validate_submission(sub, challenge, [id_col, prob_col])
    if errors:
        return {"_errors": errors}

    ak = _load_answer_key(challenge)
    merged = sub.merge(ak, on=id_col, how="inner")
    y_true = merged["adopted_new_product"].values
    y_prob = merged[prob_col].values

    auc_roc = roc_auc_score(y_true, y_prob)

    results = {"AUC-ROC": round(auc_roc, 4)}

    ranked_idx = np.argsort(y_prob)[::-1]
    for k in [100, 500, 1000]:
        top_k = ranked_idx[:k]
        prec = np.mean(y_true[top_k]) if len(top_k) > 0 else 0.0
        results[f"Prec@{k}"] = round(prec, 4)

    hits = 0.0
    relevant = 0
    for i, idx in enumerate(ranked_idx[:1000], 1):
        if y_true[idx] == 1:
            relevant += 1
            hits += relevant / i
    map_1000 = hits / min(1000, y_true.sum()) if y_true.sum() > 0 else 0.0
    results["MAP@1000"] = round(map_1000, 4)

    return results


def score_fraud_detection(sub: pd.DataFrame) -> dict:
    challenge = "fraud_detection"
    id_col = "transaction_id"
    prob_col = "fraud_probability"

    errors = _validate_submission(sub, challenge, [id_col, prob_col])
    if errors:
        return {"_errors": errors}

    ak = _load_answer_key(challenge)
    merged = sub.merge(ak, on=id_col, how="inner")
    y_true = merged["is_fraud"].values
    y_prob = merged[prob_col].values
    y_pred = (y_prob >= 0.5).astype(int)

    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    auc_pr = auc(rec, prec)

    f1 = f1_score(y_true, y_pred)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    valid = np.where(fpr <= 0.05)[0]
    recall_5fpr = tpr[valid[-1]] if len(valid) > 0 else 0.0

    return {
        "AUC-PR": round(auc_pr, 4),
        "F1": round(f1, 4),
        "Recall@5%FPR": round(recall_5fpr, 4),
    }


def score_cashflow_shortfall(sub: pd.DataFrame) -> dict:
    challenge = "cashflow_shortfall"
    id_col = "business_id"
    amount_col = "predicted_shortfall_amount"
    flag_col = "predicted_shortfall_flag"

    errors = _validate_submission(sub, challenge, [id_col, amount_col, flag_col])
    if errors:
        return {"_errors": errors}

    ak = _load_answer_key(challenge)
    merged = sub.merge(ak, on=id_col, how="inner")

    y_true_reg = merged["cashflow_shortfall_amount"].values
    y_pred_reg = merged[amount_col].values

    rmse = np.sqrt(mean_squared_error(y_true_reg, y_pred_reg))
    mae = mean_absolute_error(y_true_reg, y_pred_reg)
    r2 = r2_score(y_true_reg, y_pred_reg)

    y_true_cls = merged["shortfall_flag"].values
    y_pred_cls = merged[flag_col].values.astype(int)

    try:
        cls_auc = roc_auc_score(y_true_cls, y_pred_cls)
    except ValueError:
        cls_auc = 0.0
    cls_f1 = f1_score(y_true_cls, y_pred_cls)

    return {
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2),
        "R²": round(r2, 4),
        "AUC-ROC": round(cls_auc, 4),
        "F1": round(cls_f1, 4),
    }


def score_loan_default(sub: pd.DataFrame) -> dict:
    challenge = "loan_default"
    id_col = "loan_id"
    days_col = "predicted_days_to_default"
    flag_col = "predicted_default_flag"

    errors = _validate_submission(sub, challenge, [id_col, days_col, flag_col])
    if errors:
        return {"_errors": errors}

    ak = _load_answer_key(challenge)
    merged = sub.merge(ak, on=id_col, how="inner")

    y_true_reg = merged["days_to_early_default"].values
    y_pred_reg = merged[days_col].values.astype(float)

    rmse = np.sqrt(mean_squared_error(y_true_reg, y_pred_reg))
    mae = mean_absolute_error(y_true_reg, y_pred_reg)
    r2 = r2_score(y_true_reg, y_pred_reg)

    y_true_cls = merged["default_flag"].values
    y_pred_cls = merged[flag_col].values.astype(int)

    try:
        cls_auc = roc_auc_score(y_true_cls, y_pred_cls)
    except ValueError:
        cls_auc = 0.0
    cls_f1 = f1_score(y_true_cls, y_pred_cls)

    return {
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2),
        "R²": round(r2, 4),
        "AUC-ROC": round(cls_auc, 4),
        "F1": round(cls_f1, 4),
    }


SCORERS = {
    "product_recommendation": score_product_recommendation,
    "fraud_detection": score_fraud_detection,
    "cashflow_shortfall": score_cashflow_shortfall,
    "loan_default": score_loan_default,
}

# ── Composite score (for ranking) ──────────────────────────────────────────

# Primary metric per challenge, and direction (higher is better except RMSE)
PRIMARY_METRICS = {
    "product_recommendation": ("AUC-ROC", "higher"),
    "fraud_detection": ("AUC-PR", "higher"),
    "cashflow_shortfall": ("R²", "higher"),
    "loan_default": ("R²", "higher"),
}


def composite_score(team_results: dict) -> float | None:
    """Average of primary metrics over challenges with valid submissions."""
    scores = []
    for challenge, (metric, _) in PRIMARY_METRICS.items():
        if challenge in team_results and "_errors" not in team_results[challenge]:
            scores.append(team_results[challenge].get(metric, 0.0))
    return round(np.mean(scores), 4) if scores else None


# ── Batch processing ───────────────────────────────────────────────────────


def discover_teams(submissions_dir: Path) -> list[str]:
    teams = []
    if not submissions_dir.exists():
        return teams
    for entry in sorted(submissions_dir.iterdir()):
        if entry.is_dir() and not entry.name.startswith("."):
            teams.append(entry.name)
    return teams


def score_team(team_dir: Path) -> dict:
    results = {}
    for challenge, scorer in SCORERS.items():
        sub_file = team_dir / f"{challenge}.csv"
        if not sub_file.exists():
            results[challenge] = {"_errors": ["no submission file found"]}
            continue
        try:
            sub = pd.read_csv(sub_file)
            results[challenge] = scorer(sub)
        except Exception as e:
            results[challenge] = {"_errors": [str(e)]}
    return results


# ── Output formatting ──────────────────────────────────────────────────────

# Fixed column order for the flat leaderboard
FLAT_COLUMNS = [
    ("product_recommendation", "AUC-ROC"),
    ("product_recommendation", "Prec@100"),
    ("product_recommendation", "Prec@500"),
    ("product_recommendation", "Prec@1000"),
    ("product_recommendation", "MAP@1000"),
    ("fraud_detection", "AUC-PR"),
    ("fraud_detection", "F1"),
    ("fraud_detection", "Recall@5%FPR"),
    ("cashflow_shortfall", "RMSE"),
    ("cashflow_shortfall", "MAE"),
    ("cashflow_shortfall", "R²"),
    ("cashflow_shortfall", "AUC-ROC"),
    ("cashflow_shortfall", "F1"),
    ("loan_default", "RMSE"),
    ("loan_default", "MAE"),
    ("loan_default", "R²"),
    ("loan_default", "AUC-ROC"),
    ("loan_default", "F1"),
]


def _metric_display(challenge: str, metric: str) -> str:
    prefix = {
        "product_recommendation": "PR",
        "fraud_detection": "FD",
        "cashflow_shortfall": "CF",
        "loan_default": "LD",
    }
    return f"{prefix[challenge]}_{metric}"


def build_leaderboard_df(all_results: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for team, team_results in all_results.items():
        row = {"Team": team, "Composite": composite_score(team_results)}
        for challenge, metric in FLAT_COLUMNS:
            col_name = _metric_display(challenge, metric)
            if challenge in team_results and "_errors" not in team_results[challenge]:
                row[col_name] = team_results[challenge].get(metric, "--")
            else:
                row[col_name] = "--"
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("Composite", ascending=False, na_position="last")
    df.insert(0, "Rank", range(1, len(df) + 1))
    return df


def write_csv(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False)
    print(f"  CSV  → {path}")


def write_markdown(df: pd.DataFrame, path: Path, errors_by_team: dict):
    lines = ["# Leaderboard", ""]

    lines.append(f"*{len(df)} teams scored*")
    lines.append("")

    header = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    lines.append(header)
    lines.append(sep)
    for _, row in df.iterrows():
        cells = []
        for col in df.columns:
            val = row[col]
            if isinstance(val, float) and not np.isnan(val):
                cells.append(f"{val}")
            elif val is None or (isinstance(val, float) and np.isnan(val)):
                cells.append("--")
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")

    if errors_by_team:
        lines.extend(["", "## Submission Errors", ""])
        for team, team_errors in sorted(errors_by_team.items()):
            lines.append(f"### {team}")
            for challenge, errs in sorted(team_errors.items()):
                for err in errs:
                    lines.append(f"- **{challenge}**: {err}")
            lines.append("")

    path.write_text("\n".join(lines) + "\n")
    print(f"  MD   → {path}")


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Batch score submissions and build a leaderboard.")
    parser.add_argument(
        "--submissions", type=str, default=str(REPO_ROOT / "submissions"),
        help="Path to submissions folder (default: ./submissions)",
    )
    parser.add_argument(
        "--output", type=str, default=str(REPO_ROOT),
        help="Directory for leaderboard output files (default: repo root)",
    )
    args = parser.parse_args()

    submissions_dir = Path(args.submissions).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Submissions dir: {submissions_dir}")
    print(f"Output dir:      {output_dir}")
    print()

    teams = discover_teams(submissions_dir)
    if not teams:
        print(f"No team folders found in {submissions_dir}")
        print("Expected layout: submissions/<team_name>/<challenge>.csv")
        sys.exit(1)

    print(f"Found {len(teams)} team(s): {', '.join(teams)}\n")

    all_results: dict[str, dict] = {}
    errors_by_team: dict[str, dict] = {}

    for team in teams:
        print(f"Scoring {team} ...")
        team_dir = submissions_dir / team
        results = score_team(team_dir)
        all_results[team] = results

        team_errors = {}
        for challenge, metrics in results.items():
            if "_errors" in metrics:
                team_errors[challenge] = metrics["_errors"]
                for err in metrics["_errors"]:
                    print(f"  ⚠  {challenge}: {err}")
            else:
                primary_metric, _ = PRIMARY_METRICS[challenge]
                val = metrics.get(primary_metric, "?")
                print(f"  ✓  {challenge}: {primary_metric}={val}")

        if team_errors:
            errors_by_team[team] = team_errors
        print()

    df = build_leaderboard_df(all_results)

    print("=" * 60)
    print("LEADERBOARD")
    print("=" * 60)
    print(df.to_string(index=False))
    print()

    write_csv(df, output_dir / "leaderboard.csv")
    write_markdown(df, output_dir / "leaderboard.md", errors_by_team)


if __name__ == "__main__":
    main()
