"""
Challenge 1: Product Recommendation Engine -- Sample Starter
Pipeline: load → join → clean → feature engineer → train → predict → submit → score
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings("ignore")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 1. Load ─────────────────────────────────────────────────────────────────

customers = pd.read_csv(os.path.join(DATA_DIR, "customers.csv"))
products = pd.read_csv(os.path.join(DATA_DIR, "products.csv"))
digital = pd.read_csv(os.path.join(DATA_DIR, "digital_activity.csv"))
train_labels = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
eval_ids = pd.read_csv(os.path.join(DATA_DIR, "eval.csv"))
answer_key = pd.read_csv(os.path.join(DATA_DIR, "answer_key.csv"))

print(f"Customers:  {customers.shape}")
print(f"Products:   {products.shape}")
print(f"Digital:    {digital.shape}")
print(f"Train:      {train_labels.shape}")
print(f"Eval:       {eval_ids.shape}")

# ── 2. Quick EDA ────────────────────────────────────────────────────────────

print("\n--- Null counts ---")
print(customers.isnull().sum())
print("\n--- Product types (raw) ---")
print(products["product_type"].value_counts().head(10))
orphans = products[~products["customer_id"].isin(customers["customer_id"])]
print(f"\nOrphaned rows in products: {len(orphans)} ({len(orphans)/len(products):.1%})")

# ── 3. Clean ────────────────────────────────────────────────────────────────


def clean_string(s):
    if pd.isna(s):
        return s
    return str(s).strip().lower()


customers["state"] = customers["state"].apply(clean_string)
products["product_type"] = products["product_type"].apply(clean_string)
products["status"] = products["status"].apply(clean_string)
digital["preferred_channel"] = digital["preferred_channel"].apply(clean_string)

valid_cust_ids = set(customers["customer_id"].dropna())
products = products[products["customer_id"].isin(valid_cust_ids)]
digital = digital[digital["customer_id"].isin(valid_cust_ids)]

customers = customers.drop_duplicates(subset=["customer_id"], keep="first")
digital = digital.drop_duplicates(subset=["customer_id"], keep="first")

print(f"\nCleaned customers: {len(customers):,}")
print(f"Cleaned products:  {len(products):,}")
print(f"Cleaned digital:   {len(digital):,}")

# ── 4. Feature Engineering ──────────────────────────────────────────────────

prod_agg = products.groupby("customer_id").agg(
    num_products=("account_id", "count"),
    avg_balance=("balance", "mean"),
    total_balance=("balance", "sum"),
    num_active=("status", lambda x: (x == "active").sum()),
).reset_index()

for ptype in ["mortgage", "credit_card", "investment", "auto_loan"]:
    flag = (
        products[products["product_type"] == ptype]
        .groupby("customer_id")
        .size()
        .reset_index(name=f"has_{ptype}")
    )
    flag[f"has_{ptype}"] = 1
    prod_agg = prod_agg.merge(
        flag[["customer_id", f"has_{ptype}"]], on="customer_id", how="left"
    )
    prod_agg[f"has_{ptype}"] = prod_agg[f"has_{ptype}"].fillna(0).astype(int)

df = customers[
    ["customer_id", "annual_income", "credit_score", "customer_satisfaction_score"]
].copy()
df = df.merge(prod_agg, on="customer_id", how="left")
df = df.merge(
    digital[
        [
            "customer_id",
            "avg_monthly_logins",
            "mobile_app_sessions_30d",
            "online_transactions_30d",
            "preferred_channel",
        ]
    ],
    on="customer_id",
    how="left",
)

df["preferred_channel"] = df["preferred_channel"].fillna("unknown")
le = LabelEncoder()
df["channel_encoded"] = le.fit_transform(df["preferred_channel"])

print(f"\nJoined dataset: {df.shape}")

# ── 5. Train / Eval Split ──────────────────────────────────────────────────

feature_cols = [
    "annual_income", "credit_score", "customer_satisfaction_score",
    "num_products", "avg_balance", "total_balance", "num_active",
    "has_mortgage", "has_credit_card", "has_investment", "has_auto_loan",
    "avg_monthly_logins", "mobile_app_sessions_30d",
    "online_transactions_30d", "channel_encoded",
]

train_df = df.merge(train_labels, on="customer_id", how="inner")
eval_df = df.merge(eval_ids, on="customer_id", how="inner")

X_train = train_df[feature_cols].fillna(0)
y_train = train_df["adopted_new_product"]
X_eval = eval_df[feature_cols].fillna(0)

print(f"Train: {X_train.shape}  |  Eval: {X_eval.shape}")
print(f"Target rate: {y_train.mean():.1%}")

# ── 6. Train Baseline ──────────────────────────────────────────────────────

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_eval_s = scaler.transform(X_eval)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_s, y_train)

train_proba = model.predict_proba(X_train_s)[:, 1]
print(f"\nTrain AUC-ROC: {roc_auc_score(y_train, train_proba):.4f}")

# ── 7. Generate Submission ──────────────────────────────────────────────────

eval_proba = model.predict_proba(X_eval_s)[:, 1]

submission = pd.DataFrame({
    "customer_id": eval_df["customer_id"],
    "adoption_probability": eval_proba,
})
submission.to_csv(os.path.join(DATA_DIR, "submission.csv"), index=False)
print(f"Submission written: {len(submission):,} rows")

# ── 8. Score Against Answer Key ─────────────────────────────────────────────

scored = submission.merge(answer_key, on="customer_id", how="inner")
y_true = scored["adopted_new_product"].values
y_prob = scored["adoption_probability"].values

auc_roc = roc_auc_score(y_true, y_prob)

for k in [100, 500, 1000]:
    top_k = np.argsort(y_prob)[::-1][:k]
    prec = np.mean(y_true[top_k])
    print(f"Precision@{k}: {prec:.4f}")

print(f"\nFinal AUC-ROC: {auc_roc:.4f}")
print("Compare to benchmark.txt -- can you beat it?")
