import os
import warnings

import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Bank Cross-Sell Recommendation Dashboard",
    page_icon="🏦",
    layout="wide",
)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "files")

REQUIRED_FILES = {
    "customers.csv": "Customer demographics and account tenure",
    "products.csv": "Existing product holdings by customer",
    "digital_activity.csv": "Digital engagement signals",
    "train.csv": "Training labels",
    "eval.csv": "Customers to rank for the next campaign",
    "submission.csv": "Precomputed ranked customer recommendations",
}


def norm(series):
    return series.str.strip().str.lower() if hasattr(series, "str") else series


@st.cache_data(show_spinner=False)
def load_local_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def get_repo_file_paths() -> dict:
    return {name: os.path.join(DATA_DIR, name) for name in REQUIRED_FILES}


def validate_repo_files(file_paths: dict) -> list:
    return [name for name, path in file_paths.items() if not os.path.exists(path)]


@st.cache_data(show_spinner=False)
def prepare_dashboard_data(customers, products, digital, train_labels, eval_ids, submission):
    customers = customers.copy()
    products = products.copy()
    digital = digital.copy()
    train_labels = train_labels.copy()
    eval_ids = eval_ids.copy()
    submission = submission.copy()

    customers["state"] = norm(customers["state"])
    products["product_type"] = norm(products["product_type"])
    products["status"] = norm(products["status"])
    digital["preferred_channel"] = norm(digital["preferred_channel"])

    valid_ids = set(customers["customer_id"].dropna())
    products = products[products["customer_id"].isin(valid_ids)]
    digital = digital[digital["customer_id"].isin(valid_ids)]

    customers = customers.drop_duplicates(subset=["customer_id"], keep="first")
    digital = digital.drop_duplicates(subset=["customer_id"], keep="first")

    today = pd.Timestamp.today().normalize()
    for col in ("date_of_birth", "account_open_date"):
        customers[col] = pd.to_datetime(customers[col], errors="coerce")
    products["open_date"] = pd.to_datetime(products["open_date"], errors="coerce")

    customers["age"] = (today - customers["date_of_birth"]).dt.days / 365.25
    customers["tenure_years"] = (today - customers["account_open_date"]).dt.days / 365.25
    customers["age_bucket"] = pd.cut(
        customers["age"], bins=[0, 25, 40, 60, 999], labels=["18-25", "26-40", "41-60", "60+"]
    )

    ptype_pivot = (
        products.groupby(["customer_id", "product_type"])
        .size()
        .unstack(fill_value=0)
        .add_prefix("cnt_")
        .reset_index()
    )

    active_products = products[products["status"] == "active"]
    active_agg = (
        active_products.groupby("customer_id")
        .agg(
            num_active=("account_id", "count"),
            active_balance=("balance", "sum"),
            avg_active_balance=("balance", "mean"),
        )
        .reset_index()
    )

    prod_agg = (
        products.groupby("customer_id")
        .agg(
            num_products=("account_id", "count"),
            avg_balance=("balance", "mean"),
            total_balance=("balance", "sum"),
            max_balance=("balance", "max"),
            min_balance=("balance", "min"),
            std_balance=("balance", "std"),
            days_since_last_product=(
                "open_date",
                lambda x: (today - x.max()).days if pd.notna(x.max()) else pd.NA,
            ),
        )
        .reset_index()
    )

    prod_agg = prod_agg.merge(active_agg, on="customer_id", how="left")
    prod_agg = prod_agg.merge(ptype_pivot, on="customer_id", how="left")

    expected_products = ["checking", "savings", "credit_card", "mortgage", "investment", "auto_loan"]
    for ptype in expected_products:
        col = f"cnt_{ptype}"
        if col not in prod_agg.columns:
            prod_agg[col] = 0
        prod_agg[f"has_{ptype}"] = (prod_agg[col] > 0).astype(int)

    count_cols = [c for c in prod_agg.columns if c.startswith("cnt_")]
    prod_agg["product_diversity"] = (prod_agg[count_cols] > 0).sum(axis=1)
    prod_agg["closed_ratio"] = 1 - (
        prod_agg["num_active"].fillna(0) / prod_agg["num_products"].clip(lower=1)
    )

    digital["total_digital_activity"] = (
        digital["avg_monthly_logins"].fillna(0)
        + digital["mobile_app_sessions_30d"].fillna(0)
        + digital["online_transactions_30d"].fillna(0)
    )
    digital["mobile_ratio"] = (
        digital["mobile_app_sessions_30d"].fillna(0)
        / digital["total_digital_activity"].replace(0, pd.NA)
    )

    channel_map = {"branch": 0, "phone": 1, "online": 2, "mobile": 3}
    digital["channel_encoded"] = digital["preferred_channel"].map(channel_map).fillna(-1).astype(int)

    cust_cols = [
        "customer_id",
        "annual_income",
        "credit_score",
        "customer_satisfaction_score",
        "age",
        "tenure_years",
        "age_bucket",
        "state",
    ]
    master = customers[cust_cols].merge(prod_agg, on="customer_id", how="left")
    master = master.merge(
        digital[
            [
                "customer_id",
                "avg_monthly_logins",
                "mobile_app_sessions_30d",
                "online_transactions_30d",
                "total_digital_activity",
                "mobile_ratio",
                "channel_encoded",
            ]
        ],
        on="customer_id",
        how="left",
    )

    master["income_per_product"] = master["annual_income"] / master["num_products"].clip(lower=1)
    master["products_per_tenure"] = master["num_products"] / master["tenure_years"].clip(lower=0.1)
    master["high_value"] = (
        (master["credit_score"] > 700) & (master["annual_income"] > 75000)
    ).astype(int)

    state_means = (
        master.merge(train_labels, on="customer_id", how="inner")
        .groupby("state")["adopted_new_product"]
        .mean()
        .rename("state_adopt_rate")
    )
    master = master.merge(state_means, on="state", how="left")

    ranked = submission.merge(eval_ids, on="customer_id", how="inner")
    ranked = ranked.merge(master, on="customer_id", how="left")
    ranked = ranked.sort_values("adoption_probability", ascending=False).reset_index(drop=True)
    ranked["campaign_rank"] = ranked.index + 1

    return master, ranked


st.title("🏦 Bank Cross-Sell Recommendation Dashboard")
st.caption("This dashboard uses the precomputed submission.csv ranking and enriches it with the same customer-level features used in the starter workflow. No model training runs in the app.")

file_paths = get_repo_file_paths()
missing_files = validate_repo_files(file_paths)
if missing_files:
    st.error("Missing required repo files: " + ", ".join(missing_files))
    st.stop()

customers = load_local_csv(file_paths["customers.csv"])
products = load_local_csv(file_paths["products.csv"])
digital = load_local_csv(file_paths["digital_activity.csv"])
train_labels = load_local_csv(file_paths["train.csv"])
eval_ids = load_local_csv(file_paths["eval.csv"])
submission = load_local_csv(file_paths["submission.csv"])

master_df, ranked_customers = prepare_dashboard_data(
    customers=customers,
    products=products,
    digital=digital,
    train_labels=train_labels,
    eval_ids=eval_ids,
    submission=submission,
)

with st.sidebar:
    st.header("Campaign Filters")
    min_prob = st.slider("Minimum adoption probability", 0.0, 1.0, 0.5, 0.01)
    states = sorted([s for s in ranked_customers["state"].dropna().unique().tolist()])
    selected_states = st.multiselect("State filter", states)
    high_value_only = st.checkbox("High-value only", value=False)
    top_cut = st.number_input("Rows to display", min_value=25, max_value=5000, value=250, step=25)

filtered = ranked_customers[ranked_customers["adoption_probability"] >= min_prob].copy()
if selected_states:
    filtered = filtered[filtered["state"].isin(selected_states)]
if high_value_only:
    filtered = filtered[filtered["high_value"] == 1]
filtered = filtered.sort_values("adoption_probability", ascending=False).head(int(top_cut))

summary_tab, ranking_tab, segment_tab, data_tab = st.tabs(
    ["Executive Summary", "Ranked Customers", "Segments", "Data Explorer"]
)

with summary_tab:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Eval customers", f"{len(ranked_customers):,}")
    c2.metric("Filtered customers", f"{len(filtered):,}")
    c3.metric("Avg probability", f"{ranked_customers['adoption_probability'].mean():.2%}")
    c4.metric("High-value share", f"{ranked_customers['high_value'].mean():.1%}")

    d1, d2, d3 = st.columns(3)
    d1.metric("Top 100 avg prob", f"{ranked_customers.head(100)['adoption_probability'].mean():.2%}")
    d2.metric("Top 500 avg prob", f"{ranked_customers.head(500)['adoption_probability'].mean():.2%}")
    d3.metric("Median probability", f"{ranked_customers['adoption_probability'].median():.2%}")

    st.subheader("Top ranked customers")
    preview_cols = [
        "campaign_rank",
        "customer_id",
        "adoption_probability",
        "annual_income",
        "credit_score",
        "customer_satisfaction_score",
        "num_products",
        "product_diversity",
        "total_digital_activity",
        "high_value",
        "state",
    ]
    preview_cols = [c for c in preview_cols if c in ranked_customers.columns]
    st.dataframe(ranked_customers[preview_cols].head(100), use_container_width=True, height=420)

    st.download_button(
        label="Download ranked submission.csv",
        data=ranked_customers[["customer_id", "adoption_probability"]].to_csv(index=False).encode("utf-8"),
        file_name="submission.csv",
        mime="text/csv",
    )

with ranking_tab:
    st.subheader("Campaign targeting list")
    display_cols = [
        "campaign_rank",
        "customer_id",
        "adoption_probability",
        "state",
        "annual_income",
        "credit_score",
        "num_products",
        "product_diversity",
        "avg_monthly_logins",
        "total_digital_activity",
        "high_value",
        "has_checking",
        "has_savings",
        "has_credit_card",
        "has_mortgage",
        "has_investment",
        "has_auto_loan",
    ]
    display_cols = [c for c in display_cols if c in filtered.columns]
    st.dataframe(filtered[display_cols], use_container_width=True, height=560)

with segment_tab:
    st.subheader("Priority segment summaries")

    top100 = ranked_customers.head(100)
    top500 = ranked_customers.head(500)

    seg1, seg2 = st.columns(2)
    with seg1:
        state_summary = (
            filtered.groupby("state", dropna=False)
            .agg(
                customers=("customer_id", "count"),
                avg_probability=("adoption_probability", "mean"),
                avg_income=("annual_income", "mean"),
                avg_credit_score=("credit_score", "mean"),
            )
            .reset_index()
            .sort_values("avg_probability", ascending=False)
        )
        st.write("Top states by average score")
        st.dataframe(state_summary.head(15), use_container_width=True)

    with seg2:
        value_summary = pd.DataFrame(
            {
                "segment": ["Top 100", "Top 500", "All Eval"],
                "avg_probability": [
                    top100["adoption_probability"].mean(),
                    top500["adoption_probability"].mean(),
                    ranked_customers["adoption_probability"].mean(),
                ],
                "avg_income": [
                    top100["annual_income"].mean(),
                    top500["annual_income"].mean(),
                    ranked_customers["annual_income"].mean(),
                ],
                "high_value_share": [
                    top100["high_value"].mean(),
                    top500["high_value"].mean(),
                    ranked_customers["high_value"].mean(),
                ],
            }
        )
        st.write("Priority cohort comparison")
        st.dataframe(value_summary, use_container_width=True)

    product_cols = [c for c in [
        "has_checking", "has_savings", "has_credit_card", "has_mortgage", "has_investment", "has_auto_loan"
    ] if c in ranked_customers.columns]
    if product_cols:
        product_mix = top100[product_cols].mean().sort_values(ascending=False).reset_index()
        product_mix.columns = ["product_flag", "share_of_top_100"]
        st.write("Top 100 current product mix")
        st.dataframe(product_mix, use_container_width=True)

with data_tab:
    st.subheader("Dataset viewer")
    dataset_name = st.selectbox(
        "Choose dataset",
        ["ranked_output", "submission", "customers", "products", "digital_activity", "train", "eval"],
    )
    dataset_map = {
        "ranked_output": ranked_customers,
        "submission": submission,
        "customers": customers,
        "products": products,
        "digital_activity": digital,
        "train": train_labels,
        "eval": eval_ids,
    }
    selected_df = dataset_map[dataset_name]
    st.write(f"Shape: {selected_df.shape[0]:,} rows × {selected_df.shape[1]:,} columns")
    st.dataframe(selected_df.head(1000), use_container_width=True, height=560)
