import os
import warnings
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

st.set_page_config(
    page_title='Bank Cross-Sell Recommendation Dashboard',
    page_icon='🏦',
    layout='wide'
)

DEFAULT_DIR = os.path.dirname(os.path.abspath(__file__))
REQUIRED_FILES = {
    'customers.csv': 'Customer demographics and account tenure',
    'products.csv': 'Existing product holdings by customer',
    'digital_activity.csv': 'Digital engagement signals',
    'train.csv': 'Training labels: adopted_new_product',
    'eval.csv': 'Customers to rank for the next campaign',
}
OPTIONAL_FILES = {
    'answer_key.csv': 'Optional holdout labels for scoring the eval set'
}


def norm(series):
    return series.str.strip().str.lower() if hasattr(series, 'str') else series


@st.cache_data(show_spinner=False)
def load_csv(file_obj):
    return pd.read_csv(file_obj)


@st.cache_data(show_spinner=False)
def load_local_csv(path):
    return pd.read_csv(path)


@st.cache_resource(show_spinner=False)
def get_lightgbm_available():
    return lgb is not None


def discover_local_files(base_dir: str):
    files_dir = os.path.join(base_dir, 'files')
    search_dirs = [base_dir, files_dir]
    found = {}
    for folder in search_dirs:
        if not os.path.isdir(folder):
            continue
        for filename in list(REQUIRED_FILES) + list(OPTIONAL_FILES):
            full_path = os.path.join(folder, filename)
            if os.path.exists(full_path) and filename not in found:
                found[filename] = full_path
    return found


@st.cache_data(show_spinner=False)
def prepare_features(customers, products, digital, train_labels=None, eval_ids=None):
    customers = customers.copy()
    products = products.copy()
    digital = digital.copy()
    train_labels = None if train_labels is None else train_labels.copy()
    eval_ids = None if eval_ids is None else eval_ids.copy()

    customers['state'] = norm(customers['state'])
    products['product_type'] = norm(products['product_type'])
    products['status'] = norm(products['status'])
    digital['preferred_channel'] = norm(digital['preferred_channel'])

    valid_ids = set(customers['customer_id'].dropna())
    products = products[products['customer_id'].isin(valid_ids)]
    digital = digital[digital['customer_id'].isin(valid_ids)]

    customers = customers.drop_duplicates(subset=['customer_id'], keep='first')
    digital = digital.drop_duplicates(subset=['customer_id'], keep='first')

    today = pd.Timestamp.today().normalize()
    for col in ('date_of_birth', 'account_open_date'):
        customers[col] = pd.to_datetime(customers[col], errors='coerce')
    products['open_date'] = pd.to_datetime(products['open_date'], errors='coerce')

    customers['age'] = (today - customers['date_of_birth']).dt.days / 365.25
    customers['tenure_years'] = (today - customers['account_open_date']).dt.days / 365.25
    customers['age_bucket'] = pd.cut(
        customers['age'], bins=[0, 25, 40, 60, 999], labels=[0, 1, 2, 3]
    ).astype(float)

    ptype_pivot = (
        products.groupby(['customer_id', 'product_type'])
        .size()
        .unstack(fill_value=0)
        .add_prefix('cnt_')
        .reset_index()
    )

    active_prods = products[products['status'] == 'active']
    active_agg = (
        active_prods.groupby('customer_id')
        .agg(
            num_active=('account_id', 'count'),
            active_balance=('balance', 'sum'),
            avg_active_balance=('balance', 'mean'),
        )
        .reset_index()
    )

    prod_agg = (
        products.groupby('customer_id')
        .agg(
            num_products=('account_id', 'count'),
            avg_balance=('balance', 'mean'),
            total_balance=('balance', 'sum'),
            max_balance=('balance', 'max'),
            min_balance=('balance', 'min'),
            std_balance=('balance', 'std'),
            days_since_last_product=(
                'open_date',
                lambda x: (today - x.max()).days if pd.notna(x.max()) else np.nan,
            ),
            days_since_first_product=(
                'open_date',
                lambda x: (today - x.min()).days if pd.notna(x.min()) else np.nan,
            ),
        )
        .reset_index()
    )

    prod_agg = prod_agg.merge(active_agg, on='customer_id', how='left')
    prod_agg = prod_agg.merge(ptype_pivot, on='customer_id', how='left')

    for ptype in ['checking', 'savings', 'credit_card', 'mortgage', 'investment', 'auto_loan']:
        col = f'cnt_{ptype}'
        if col not in prod_agg.columns:
            prod_agg[col] = 0
        prod_agg[f'has_{ptype}'] = (prod_agg[col] > 0).astype(int)

    count_cols = [c for c in prod_agg.columns if c.startswith('cnt_')]
    prod_agg['product_diversity'] = (prod_agg[count_cols] > 0).sum(axis=1)
    prod_agg['closed_ratio'] = 1 - (
        prod_agg['num_active'].fillna(0) / prod_agg['num_products'].clip(lower=1)
    )
    prod_agg['balance_spread'] = prod_agg['max_balance'] - prod_agg['min_balance']

    digital['total_digital_activity'] = (
        digital['avg_monthly_logins'].fillna(0)
        + digital['mobile_app_sessions_30d'].fillna(0)
        + digital['online_transactions_30d'].fillna(0)
    )
    digital['mobile_ratio'] = (
        digital['mobile_app_sessions_30d'].fillna(0)
        / digital['total_digital_activity'].replace(0, np.nan)
    )
    channel_map = {'branch': 0, 'phone': 1, 'online': 2, 'mobile': 3}
    digital['channel_encoded'] = digital['preferred_channel'].map(channel_map).fillna(-1).astype(int)

    cust_cols = [
        'customer_id', 'annual_income', 'credit_score',
        'customer_satisfaction_score', 'age', 'tenure_years', 'age_bucket', 'state'
    ]
    df = customers[cust_cols].merge(prod_agg, on='customer_id', how='left')
    df = df.merge(
        digital[[
            'customer_id', 'avg_monthly_logins', 'mobile_app_sessions_30d',
            'online_transactions_30d', 'channel_encoded',
            'total_digital_activity', 'mobile_ratio'
        ]],
        on='customer_id', how='left'
    )

    df['income_per_product'] = df['annual_income'] / df['num_products'].clip(lower=1)
    df['balance_per_income'] = df['total_balance'] / df['annual_income'].replace(0, np.nan)
    df['credit_x_income'] = df['credit_score'] * df['annual_income'] / 1e6
    df['logins_per_product'] = df['avg_monthly_logins'] / df['num_products'].clip(lower=1)
    df['activity_per_product'] = df['total_digital_activity'] / df['num_products'].clip(lower=1)
    df['products_per_tenure'] = df['num_products'] / df['tenure_years'].clip(lower=0.1)
    df['satisfaction_x_activity'] = df['customer_satisfaction_score'] * df['total_digital_activity']
    df['recency_x_diversity'] = df['days_since_last_product'] * df['product_diversity']
    df['high_value'] = (
        (df['credit_score'] > 700) & (df['annual_income'] > 75000)
    ).astype(int)

    if train_labels is not None:
        state_means = (
            df.merge(train_labels, on='customer_id', how='inner')
            .groupby('state')['adopted_new_product']
            .mean()
            .rename('state_adopt_rate')
        )
        global_state_mean = float(train_labels['adopted_new_product'].mean())
        df = df.merge(state_means, on='state', how='left')
        df['state_adopt_rate'] = df['state_adopt_rate'].fillna(global_state_mean)
    else:
        df['state_adopt_rate'] = np.nan

    base_features = [
        'annual_income', 'credit_score', 'customer_satisfaction_score',
        'age', 'tenure_years', 'age_bucket',
        'num_products', 'avg_balance', 'total_balance', 'max_balance',
        'std_balance', 'balance_spread', 'num_active', 'active_balance',
        'avg_active_balance', 'product_diversity', 'closed_ratio',
        'days_since_last_product', 'days_since_first_product',
        *[c for c in prod_agg.columns if c.startswith('cnt_')],
        'has_checking', 'has_savings', 'has_credit_card',
        'has_mortgage', 'has_investment', 'has_auto_loan',
        'avg_monthly_logins', 'mobile_app_sessions_30d',
        'online_transactions_30d', 'channel_encoded',
        'total_digital_activity', 'mobile_ratio',
        'income_per_product', 'balance_per_income', 'credit_x_income',
        'logins_per_product', 'activity_per_product', 'products_per_tenure',
        'satisfaction_x_activity', 'recency_x_diversity', 'high_value',
        'state_adopt_rate',
    ]
    feature_cols = [c for c in base_features if c in df.columns]

    train_df = None if train_labels is None else df.merge(train_labels, on='customer_id', how='inner')
    eval_df = None if eval_ids is None else df.merge(eval_ids, on='customer_id', how='inner')

    return {
        'master_df': df,
        'train_df': train_df,
        'eval_df': eval_df,
        'feature_cols': feature_cols,
        'product_count_cols': count_cols,
    }


@st.cache_data(show_spinner=False)
def train_and_score(train_df, eval_df, feature_cols, params, n_folds, n_rounds, early_stop):
    X_train = train_df[feature_cols].fillna(-999)
    y_train = train_df['adopted_new_product'].values
    X_eval = eval_df[feature_cols].fillna(-999)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X_train))
    eval_preds = np.zeros(len(X_eval))
    fold_aucs = []
    importance_frames = []
    best_iterations = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), start=1):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_cols)
        dval = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols, reference=dtrain)

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=n_rounds,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(early_stop, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        oof_preds[val_idx] = val_pred
        eval_preds += model.predict(X_eval, num_iteration=model.best_iteration) / n_folds
        fold_auc = roc_auc_score(y_val, val_pred)
        fold_aucs.append(fold_auc)
        best_iterations.append(model.best_iteration)

        fold_importance = pd.DataFrame({
            'feature': feature_cols,
            'gain': model.feature_importance(importance_type='gain'),
            'fold': fold,
        })
        importance_frames.append(fold_importance)

    oof_auc = roc_auc_score(y_train, oof_preds)
    importances = (
        pd.concat(importance_frames, ignore_index=True)
        .groupby('feature', as_index=False)['gain']
        .mean()
        .sort_values('gain', ascending=False)
    )

    submission = eval_df[['customer_id']].copy()
    submission['adoption_probability'] = eval_preds
    submission = submission.sort_values('adoption_probability', ascending=False).reset_index(drop=True)

    metrics = {
        'oof_auc': oof_auc,
        'mean_fold_auc': float(np.mean(fold_aucs)),
        'std_fold_auc': float(np.std(fold_aucs)),
        'fold_aucs': fold_aucs,
        'best_iterations': best_iterations,
        'positive_rate': float(y_train.mean()),
        'train_rows': int(len(train_df)),
        'eval_rows': int(len(eval_df)),
    }
    return submission, importances, metrics


def precision_at_k(y_true, y_prob, k):
    ranked = np.argsort(y_prob)[::-1][:k]
    return float(np.mean(y_true[ranked]))



def map_at_k(y_true, y_prob, k=1000):
    ranked = np.argsort(y_prob)[::-1][:k]
    ap_sum = 0.0
    hits = 0
    for i, idx in enumerate(ranked, start=1):
        if y_true[idx] == 1:
            hits += 1
            ap_sum += hits / i
    denom = min(int(np.sum(y_true)), k)
    return float(ap_sum / denom) if denom > 0 else 0.0


st.title('🏦 Bank Cross-Sell Recommendation Dashboard')
st.caption('Train a ranking model, score customers by adoption probability, and surface the best cross-sell targets.')

with st.sidebar:
    st.header('Data Sources')
    st.write('Upload the challenge files or let the app auto-detect local CSVs in the same folder or a `/files` subfolder.')

    local_files = discover_local_files(DEFAULT_DIR)
    use_local_files = st.toggle('Use auto-detected local CSVs when available', value=True)

    uploads = {}
    for filename, desc in REQUIRED_FILES.items():
        uploads[filename] = st.file_uploader(f'{filename}', type=['csv'], help=desc)
    for filename, desc in OPTIONAL_FILES.items():
        uploads[filename] = st.file_uploader(f'{filename} (optional)', type=['csv'], help=desc)

    st.header('Model Settings')
    n_folds = st.slider('CV folds', min_value=3, max_value=7, value=5, step=1)
    n_rounds = st.slider('Boosting rounds', min_value=200, max_value=2000, value=1000, step=100)
    early_stop = st.slider('Early stopping rounds', min_value=20, max_value=200, value=50, step=10)
    learning_rate = st.select_slider('Learning rate', options=[0.03, 0.05, 0.07, 0.1], value=0.05)
    num_leaves = st.select_slider('Num leaves', options=[31, 63, 127, 255], value=127)


loaded = {}
for filename in list(REQUIRED_FILES) + list(OPTIONAL_FILES):
    if uploads.get(filename) is not None:
        loaded[filename] = load_csv(uploads[filename])
    elif use_local_files and filename in local_files:
        loaded[filename] = load_local_csv(local_files[filename])

missing_required = [f for f in REQUIRED_FILES if f not in loaded]

if not get_lightgbm_available():
    st.error('LightGBM is not installed in this environment. Install it with `pip install lightgbm` before running the dashboard.')
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric('Required files loaded', f"{len(REQUIRED_FILES) - len(missing_required)}/{len(REQUIRED_FILES)}")
col2.metric('Optional files loaded', f"{sum(1 for f in OPTIONAL_FILES if f in loaded)}/{len(OPTIONAL_FILES)}")
col3.metric('Local files found', len(local_files))
col4.metric('Model engine', 'LightGBM')

with st.expander('Detected files', expanded=False):
    if local_files:
        for name, path in local_files.items():
            st.write(f'• {name}: `{path}`')
    else:
        st.write('No local CSVs detected next to the app.')

if missing_required:
    st.info('Upload or place these CSVs next to the app to begin: ' + ', '.join(missing_required))
    st.stop()

customers = loaded['customers.csv']
products = loaded['products.csv']
digital = loaded['digital_activity.csv']
train_labels = loaded['train.csv']
eval_ids = loaded['eval.csv']
answer_key = loaded.get('answer_key.csv')

feature_bundle = prepare_features(customers, products, digital, train_labels, eval_ids)
train_df = feature_bundle['train_df']
eval_df = feature_bundle['eval_df']
feature_cols = feature_bundle['feature_cols']
master_df = feature_bundle['master_df']

params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': num_leaves,
    'learning_rate': learning_rate,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 30,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'n_jobs': -1,
    'verbose': -1,
    'random_state': 42,
}

with st.spinner('Training ranking model and scoring eval customers...'):
    submission, importances, metrics = train_and_score(
        train_df,
        eval_df,
        feature_cols,
        params,
        n_folds,
        n_rounds,
        early_stop,
    )

ranked_customers = submission.merge(master_df, on='customer_id', how='left')

if answer_key is not None:
    scored = submission.merge(answer_key, on='customer_id', how='inner')
    y_true = scored['adopted_new_product'].values
    y_prob = scored['adoption_probability'].values
    eval_metrics = {
        'AUC-ROC': roc_auc_score(y_true, y_prob),
        'Precision@100': precision_at_k(y_true, y_prob, 100),
        'Precision@500': precision_at_k(y_true, y_prob, 500),
        'Precision@1000': precision_at_k(y_true, y_prob, 1000),
        'MAP@1000': map_at_k(y_true, y_prob, 1000),
    }
else:
    eval_metrics = None

summary_tab, ranking_tab, feature_tab, data_tab = st.tabs([
    'Executive Summary', 'Campaign Ranking', 'Model Diagnostics', 'Data Explorer'
])

with summary_tab:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Train rows', f"{metrics['train_rows']:,}")
    c2.metric('Eval rows', f"{metrics['eval_rows']:,}")
    c3.metric('Positive rate', f"{metrics['positive_rate']:.1%}")
    c4.metric('Features used', len(feature_cols))

    d1, d2, d3 = st.columns(3)
    d1.metric('OOF AUC', f"{metrics['oof_auc']:.4f}")
    d2.metric('Mean CV AUC', f"{metrics['mean_fold_auc']:.4f}")
    d3.metric('CV AUC std', f"{metrics['std_fold_auc']:.4f}")

    if eval_metrics:
        st.subheader('Holdout performance')
        e1, e2, e3, e4, e5 = st.columns(5)
        e1.metric('AUC-ROC', f"{eval_metrics['AUC-ROC']:.4f}")
        e2.metric('P@100', f"{eval_metrics['Precision@100']:.4f}")
        e3.metric('P@500', f"{eval_metrics['Precision@500']:.4f}")
        e4.metric('P@1000', f"{eval_metrics['Precision@1000']:.4f}")
        e5.metric('MAP@1000', f"{eval_metrics['MAP@1000']:.4f}")

    st.subheader('Recommended campaign list')
    top_n = st.slider('Preview top N ranked customers', 10, 500, 100, 10)
    preview_cols = [
        'customer_id', 'adoption_probability', 'annual_income', 'credit_score',
        'customer_satisfaction_score', 'num_products', 'product_diversity',
        'total_digital_activity', 'high_value', 'state'
    ]
    preview_cols = [c for c in preview_cols if c in ranked_customers.columns]
    st.dataframe(ranked_customers[preview_cols].head(top_n), use_container_width=True)

    csv_bytes = submission.to_csv(index=False).encode('utf-8')
    st.download_button(
        label='Download ranked submission.csv',
        data=csv_bytes,
        file_name='submission.csv',
        mime='text/csv'
    )

with ranking_tab:
    st.subheader('Campaign targeting workbench')
    left, right = st.columns([1, 3])
    with left:
        min_prob = st.slider('Minimum adoption probability', 0.0, 1.0, 0.5, 0.01)
        states = sorted([s for s in ranked_customers['state'].dropna().unique().tolist()])
        selected_states = st.multiselect('State filter', states)
        high_value_only = st.checkbox('High-value customers only', value=False)
        top_cut = st.number_input('Rows to display', min_value=25, max_value=5000, value=250, step=25)

    filtered = ranked_customers[ranked_customers['adoption_probability'] >= min_prob].copy()
    if selected_states:
        filtered = filtered[filtered['state'].isin(selected_states)]
    if high_value_only and 'high_value' in filtered.columns:
        filtered = filtered[filtered['high_value'] == 1]

    filtered = filtered.sort_values('adoption_probability', ascending=False).head(int(top_cut))

    with right:
        st.dataframe(filtered, use_container_width=True, height=500)

    fig = plt.figure(figsize=(10, 4))
    plt.hist(ranked_customers['adoption_probability'], bins=40)
    plt.xlabel('Adoption probability')
    plt.ylabel('Customer count')
    plt.title('Score distribution across eval customers')
    st.pyplot(fig)

with feature_tab:
    st.subheader('Top feature importances')
    top_features = importances.head(20).sort_values('gain', ascending=True)
    fig = plt.figure(figsize=(10, 8))
    plt.barh(top_features['feature'], top_features['gain'])
    plt.xlabel('Average gain across CV folds')
    plt.ylabel('Feature')
    plt.title('Top 20 LightGBM features')
    st.pyplot(fig)

    fold_df = pd.DataFrame({
        'fold': list(range(1, len(metrics['fold_aucs']) + 1)),
        'auc': metrics['fold_aucs'],
        'best_iteration': metrics['best_iterations'],
    })
    c1, c2 = st.columns(2)
    with c1:
        st.dataframe(fold_df, use_container_width=True)
    with c2:
        fig2 = plt.figure(figsize=(8, 4))
        plt.plot(fold_df['fold'], fold_df['auc'], marker='o')
        plt.xlabel('Fold')
        plt.ylabel('Validation AUC')
        plt.title('Cross-validation stability')
        st.pyplot(fig2)

with data_tab:
    st.subheader('Loaded datasets')
    dataset_name = st.selectbox('Choose dataset', ['customers', 'products', 'digital_activity', 'train', 'eval', 'ranked_output'])
    dataset_map = {
        'customers': customers,
        'products': products,
        'digital_activity': digital,
        'train': train_labels,
        'eval': eval_ids,
        'ranked_output': ranked_customers,
    }
    selected_df = dataset_map[dataset_name]
    st.write(f'Shape: {selected_df.shape[0]:,} rows × {selected_df.shape[1]:,} columns')
    st.dataframe(selected_df.head(1000), use_container_width=True, height=500)
