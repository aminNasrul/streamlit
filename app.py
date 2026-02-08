import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="UNILINK | Student Performance Dashboard", layout="wide")

# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("student_data_merged.csv")

df = load_data()

# Basic checks
required_cols = ["id_student", "final_result_binary", "avg_score"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Your CSV is missing required columns: {missing}")
    st.stop()

# -----------------------------
# Title
# -----------------------------
st.title("UNILINK – Early Prediction of Student Performance")
st.caption("CSV-only Streamlit prototype: trains lightweight models at runtime + shows decision-support visuals.")

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Controls")

# Student selector
student_id = st.sidebar.selectbox(
    "Select id_student",
    sorted(df["id_student"].unique())
)

# Sampling for faster plots
sample_n = st.sidebar.slider("Sample size for plots (speed)", 500, 10000, 2000, 500)
df_plot = df.sample(n=min(sample_n, len(df)), random_state=42).copy()

# -----------------------------
# Train lightweight models (no joblib)
# -----------------------------
@st.cache_resource
def train_models(df_train: pd.DataFrame):
    # Targets
    y_clf = df_train["final_result_binary"]
    y_reg = df_train["avg_score"]

    # Features = everything except targets
    X = df_train.drop(columns=["final_result_binary", "avg_score"], errors="ignore")

    # Pipeline models (scaled)
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ])

    reg = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])

    clf.fit(X, y_clf)
    reg.fit(X, y_reg)

    return clf, reg, X.columns.tolist()

clf_model, reg_model, feature_cols = train_models(df)

# -----------------------------
# Student row + predictions
# -----------------------------
row = df[df["id_student"] == student_id].iloc[0]
X_row = pd.DataFrame([row[feature_cols]])

pred_class = int(clf_model.predict(X_row)[0])
pred_proba = float(clf_model.predict_proba(X_row)[0][1])
pred_score = float(reg_model.predict(X_row)[0])

# -----------------------------
# SUMMARY (Total + Pass + Fail) in one section
# -----------------------------
total_students = len(df)
pass_count = int((df["final_result_binary"] == 1).sum())
fail_count = int((df["final_result_binary"] == 0).sum())
pass_rate = pass_count / total_students if total_students > 0 else np.nan

# KPI cards row
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Total Students", f"{total_students:,}")
with k2:
    st.metric("Passed", f"{pass_count:,}")
with k3:
    st.metric("Failed", f"{fail_count:,}")
with k4:
    st.metric("Pass Rate", f"{pass_rate:.3f}" if not np.isnan(pass_rate) else "N/A")

# One combined chart (bar) under KPIs
st.subheader("Overall Outcome Summary (Actual)")
plt.figure()
plt.bar(["Fail", "Pass"], [fail_count, pass_count])
plt.ylabel("Number of Students")
plt.title("Pass vs Fail Distribution")
st.pyplot(plt.gcf())
plt.clf()

st.divider()

# -----------------------------
# Student Snapshot + Prediction Panel
# -----------------------------
cA, cB = st.columns([1.2, 1.0])

with cA:
    st.subheader("Student Snapshot")
    # show some key columns if they exist
    key_signals = [
        "early_avg_score",
        "total_clicks_vle",
        "num_vle_interactions_days",
        "early_total_clicks_vle",
        "early_num_vle_interactions_days",
        "avg_score",
        "final_result_binary",
        "cluster_label",
    ]
    snap = {k: row[k] for k in key_signals if k in df.columns}
    st.dataframe(pd.DataFrame([snap]) if snap else pd.DataFrame([row.to_dict()]))

with cB:
    st.subheader("Model Predictions (Demo)")
    st.metric("Predicted Outcome", "PASS" if pred_class == 1 else "FAIL")
    st.metric("Pass Probability", f"{pred_proba:.3f}")
    st.metric("Predicted avg_score", f"{pred_score:.2f}")
    if "cluster_label" in df.columns:
        st.metric("cluster_label", int(row["cluster_label"]))
        st.caption("Clustering is exploratory: used for segmentation, not final decisions.")
    else:
        st.caption("No cluster_label column found in your CSV.")

st.divider()

# -----------------------------
# Chart 1: Engagement vs Performance
# -----------------------------
st.subheader("Engagement vs Performance (Sample)")

x_col = "total_clicks_vle"
y_col = "avg_score"
color_col = "final_result_binary"

if x_col in df.columns and y_col in df.columns:
    tmp = df_plot[[x_col, y_col] + ([color_col] if color_col in df_plot.columns else [])].dropna()

    plt.figure()
    if color_col in tmp.columns:
        pass_pts = tmp[tmp[color_col] == 1]
        fail_pts = tmp[tmp[color_col] == 0]
        plt.scatter(fail_pts[x_col], fail_pts[y_col], alpha=0.5, label="Fail")
        plt.scatter(pass_pts[x_col], pass_pts[y_col], alpha=0.5, label="Pass")
        plt.legend()
    else:
        plt.scatter(tmp[x_col], tmp[y_col], alpha=0.5)

    plt.xlabel("total_clicks_vle")
    plt.ylabel("avg_score")
    plt.title("total_clicks_vle vs avg_score")
    st.pyplot(plt.gcf())
    plt.clf()
else:
    st.info("Columns total_clicks_vle and/or avg_score not found. Export them from your ipynb if needed.")

st.divider()

# -----------------------------
# Chart 2: Correlation Heatmap (key features)
# -----------------------------
st.subheader("Correlation Heatmap (Key Features)")

corr_features = [
    c for c in [
        "early_avg_score",
        "total_clicks_vle",
        "num_vle_interactions_days",
        "early_total_clicks_vle",
        "early_num_vle_interactions_days",
        "avg_score",
        "final_result_binary"
    ]
    if c in df.columns
]

if len(corr_features) >= 3:
    corr_df = df[corr_features].dropna().sample(n=min(sample_n, df[corr_features].dropna().shape[0]), random_state=42)
    corr = corr_df.corr(numeric_only=True)

    plt.figure(figsize=(8, 4))
    plt.imshow(corr.values)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)
    plt.colorbar()
    plt.title("Correlation Heatmap (Sample)")
    st.pyplot(plt.gcf())
    plt.clf()
else:
    st.info("Not enough numeric columns found for correlation heatmap.")

st.divider()

# -----------------------------
# Chart 3: Regression Quality (Actual vs Predicted) on sample
# -----------------------------
st.subheader("Regression Check (Sample): Actual vs Predicted avg_score")

try:
    X_batch = df_plot[feature_cols]
    pred_batch = reg_model.predict(X_batch)
    actual = df_plot["avg_score"].values

    plt.figure()
    plt.scatter(actual, pred_batch, alpha=0.5)
    plt.xlabel("Actual avg_score")
    plt.ylabel("Predicted avg_score")
    plt.title("Actual vs Predicted avg_score")

    mn = float(np.nanmin([np.nanmin(actual), np.nanmin(pred_batch)]))
    mx = float(np.nanmax([np.nanmax(actual), np.nanmax(pred_batch)]))
    plt.plot([mn, mx], [mn, mx])  # diagonal reference
    st.pyplot(plt.gcf())
    plt.clf()
except Exception:
    st.info("Could not compute regression check (feature mismatch). Make sure your CSV matches your training features.")

st.divider()

# -----------------------------
# Chart 4: Cluster Profiles (if cluster_label exists)
# -----------------------------
st.subheader("Cluster Profiles (Exploratory)")

if "cluster_label" in df.columns:
    prof_cols = [c for c in ["avg_score", "total_clicks_vle", "num_vle_interactions_days", "early_avg_score"] if c in df.columns]
    if prof_cols:
        cluster_means = df.groupby("cluster_label")[prof_cols].mean(numeric_only=True)

        plt.figure(figsize=(9, 4))
        cluster_means.plot(kind="bar", ax=plt.gca())
        plt.title("Mean Key Features per Cluster")
        plt.xlabel("cluster_label")
        plt.ylabel("Mean Value")
        plt.xticks(rotation=0)
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        st.pyplot(plt.gcf())
        plt.clf()
    else:
        st.info("No profile columns found to summarize clusters.")
else:
    st.info("No cluster_label column found in your CSV (optional).")

# -----------------------------
# Debug expander (optional)
# -----------------------------
with st.expander("Debug: Features Used"):
    st.write("Total features used:", len(feature_cols))
    st.write("First 25 features:", feature_cols[:25])
