# app.py (FULL) - CSV-only + Cooler UI + KPI boxed + Student selector inside "Selected Student"
# + NEW ADD-ONS:
#   1) Model Performance Panel (Accuracy/F1 + MAE/RMSE/R2 + Confusion Matrix)
#   2) Feature Importance (LogReg + LinReg coefficients)
#   3) Risk Threshold Labels (High/Medium/Low)
#   4) Student vs Cohort Benchmark (percentiles + pass/fail group means)
#   5) Dark Matplotlib styling + nicer graphs

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="UNILINK | Student Performance Dashboard", layout="wide")

# -----------------------------
# Custom CSS (cooler UI)
# -----------------------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"]{
  background: radial-gradient(circle at 15% 15%, rgba(99,102,241,0.10), transparent 40%),
              radial-gradient(circle at 85% 10%, rgba(16,185,129,0.10), transparent 35%),
              radial-gradient(circle at 70% 85%, rgba(236,72,153,0.08), transparent 40%),
              #0b1220;
}
.block-container { padding-top: 1.2rem; }
h1, h2, h3, h4 { color: #e5e7eb; }
p, span, div { color: #cbd5e1; }

.card {
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 16px;
  padding: 16px 18px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.25);
}

.kpi-grid{
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
}
@media (max-width: 1100px){
  .kpi-grid{ grid-template-columns: repeat(2, 1fr); }
}
@media (max-width: 600px){
  .kpi-grid{ grid-template-columns: 1fr; }
}

.kpi {
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 18px;
  padding: 14px 16px;
  box-shadow: 0 10px 25px rgba(0,0,0,0.25);
  position: relative;
  overflow: hidden;
}
.kpi:before{
  content:"";
  position:absolute;
  inset:0;
  background: linear-gradient(135deg, rgba(99,102,241,0.18), rgba(16,185,129,0.10), rgba(236,72,153,0.08));
  filter: blur(20px);
  opacity: 0.65;
}
.kpi * { position: relative; z-index: 1; }
.kpi-label{
  font-size: 0.82rem;
  opacity: 0.85;
  letter-spacing: 0.02em;
}
.kpi-value{
  font-size: 1.6rem;
  font-weight: 800;
  margin-top: 4px;
  color: #f8fafc;
}
.kpi-sub{
  margin-top: 2px;
  font-size: 0.80rem;
  opacity: 0.85;
}

.hr {
  height: 1px;
  background: rgba(255,255,255,0.10);
  margin: 16px 0 8px 0;
}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Matplotlib dark styling
# -----------------------------
def style_dark_matplotlib():
    plt.rcParams.update({
        "figure.facecolor": (0, 0, 0, 0),   # transparent
        "axes.facecolor": (0, 0, 0, 0),
        "axes.edgecolor": (1, 1, 1, 0.15),
        "axes.labelcolor": "#e5e7eb",
        "text.color": "#e5e7eb",
        "xtick.color": "#cbd5e1",
        "ytick.color": "#cbd5e1",
        "grid.color": (1, 1, 1, 0.10),
        "grid.linestyle": "-",
        "font.size": 12,
        "axes.titleweight": "bold",
    })

style_dark_matplotlib()

# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("student_data_merged.csv")

df = load_data()

# Required columns check
required_cols = ["id_student", "final_result_binary", "avg_score"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Your CSV is missing required columns: {missing}")
    st.stop()

# -----------------------------
# Header
# -----------------------------
st.title("UNILINK – Early Prediction of Student Performance")

# -----------------------------
# Sidebar controls (speed + evaluation)
# -----------------------------
st.sidebar.header("Controls")
sample_n = st.sidebar.slider("Sample size for plots (speed)", 500, 10000, 2000, 500)
test_size = st.sidebar.slider("Test split (for metrics)", 0.10, 0.40, 0.20, 0.05)
random_state = st.sidebar.number_input("Random seed", value=42, step=1)

df_plot = df.sample(n=min(sample_n, len(df)), random_state=int(random_state)).copy()

# -----------------------------
# Prepare numeric feature matrix (robust)
# -----------------------------
def build_feature_matrix(df_in: pd.DataFrame):
    """
    Keep ONLY numeric columns (required for StandardScaler + linear models).
    Drop targets and id columns.
    """
    # Numeric-only
    numeric = df_in.select_dtypes(include=[np.number]).copy()

    # Remove targets and id if numeric
    drop_cols = ["final_result_binary", "avg_score"]
    if "id_student" in numeric.columns:
        drop_cols.append("id_student")
    numeric = numeric.drop(columns=drop_cols, errors="ignore")

    return numeric

X_all = build_feature_matrix(df)
if X_all.shape[1] < 2:
    st.error("Not enough numeric feature columns found. Encode categorical columns in your notebook before exporting CSV.")
    st.stop()

# Targets
y_clf_all = df["final_result_binary"].astype(int)
y_reg_all = df["avg_score"].astype(float)

# -----------------------------
# Train models + compute metrics (cached)
# -----------------------------
@st.cache_resource
def train_models_and_metrics(df_in: pd.DataFrame, test_size: float, random_state: int):
    X = build_feature_matrix(df_in)
    y_clf = df_in["final_result_binary"].astype(int)
    y_reg = df_in["avg_score"].astype(float)

    # Split once and reuse for both tasks (consistent evaluation)
    X_train, X_test, y_clf_train, y_clf_test, y_reg_train, y_reg_test = train_test_split(
        X, y_clf, y_reg, test_size=test_size, random_state=random_state, stratify=y_clf
    )

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000))
    ])
    reg = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])

    clf.fit(X_train, y_clf_train)
    reg.fit(X_train, y_reg_train)

    # Predictions
    yhat_clf = clf.predict(X_test)
    yhat_proba = clf.predict_proba(X_test)[:, 1]
    yhat_reg = reg.predict(X_test)

    # Metrics (classification)
    acc = float(accuracy_score(y_clf_test, yhat_clf))
    f1 = float(f1_score(y_clf_test, yhat_clf, zero_division=0))
    cm = confusion_matrix(y_clf_test, yhat_clf, labels=[0, 1])

    # Metrics (regression)
    mae = float(mean_absolute_error(y_reg_test, yhat_reg))
    rmse = float(np.sqrt(mean_squared_error(y_reg_test, yhat_reg)))
    r2 = float(r2_score(y_reg_test, yhat_reg))

    # Coefficients (after scaler): coefficients correspond to scaled features
    feat_cols = X.columns.tolist()
    clf_coef = clf.named_steps["model"].coef_.ravel()
    reg_coef = reg.named_steps["model"].coef_.ravel()

    return {
        "clf": clf,
        "reg": reg,
        "feat_cols": feat_cols,
        "metrics": {
            "acc": acc,
            "f1": f1,
            "cm": cm,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        },
        "coefs": {
            "clf_coef": clf_coef,
            "reg_coef": reg_coef,
        },
        "splits": {
            "X_test": X_test,
            "y_clf_test": y_clf_test,
            "y_reg_test": y_reg_test,
            "yhat_proba": yhat_proba,
            "yhat_reg": yhat_reg,
        }
    }

bundle = train_models_and_metrics(df, float(test_size), int(random_state))
clf_model = bundle["clf"]
reg_model = bundle["reg"]
feature_cols = bundle["feat_cols"]
metrics = bundle["metrics"]

# -----------------------------
# KPI Summary (boxed) - dataset level
# -----------------------------
total_students = len(df)
pass_count = int((df["final_result_binary"] == 1).sum())
fail_count = int((df["final_result_binary"] == 0).sum())
pass_rate = pass_count / total_students if total_students else np.nan

st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi">
    <div class="kpi-label">Total Students</div>
    <div class="kpi-value">{total_students:,}</div>
    <div class="kpi-sub">Merged & engineered dataset</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Passed</div>
    <div class="kpi-value">{pass_count:,}</div>
    <div class="kpi-sub">final_result_binary = 1</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Failed</div>
    <div class="kpi-value">{fail_count:,}</div>
    <div class="kpi-sub">final_result_binary = 0</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Pass Rate</div>
    <div class="kpi-value">{pass_rate:.3f}</div>
    <div class="kpi-sub">Overall outcome distribution</div>
  </div>
</div>
<div class="hr"></div>
""", unsafe_allow_html=True)

# -----------------------------
# Row: Summary chart + Selected student (with selector inside)
# -----------------------------
left, right = st.columns([1, 1])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Overall Outcome Summary (Actual)")

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["Fail", "Pass"], [fail_count, pass_count])
    ax.set_ylabel("Number of Students")
    ax.set_title("Pass vs Fail Distribution")
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{int(b.get_height()):,}", ha="center", va="bottom", fontsize=11)

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Selected Student")

    # Student selector INSIDE the card
    student_id = st.selectbox(
        "Choose Student ID",
        sorted(df["id_student"].unique()),
        key="student_selector_main"
    )

    # Student row + predictions (defined AFTER selector)
    row = df[df["id_student"] == student_id].iloc[0]

    # Build numeric X_row aligned to feature_cols
    # (use df row -> dataframe -> numeric selection -> reindex to training features)
    X_row_num = pd.DataFrame([row.to_dict()]).select_dtypes(include=[np.number])
    # drop targets/id if present
    X_row_num = X_row_num.drop(columns=["final_result_binary", "avg_score", "id_student"], errors="ignore")
    X_row = X_row_num.reindex(columns=feature_cols).fillna(0.0)

    pred_class = int(clf_model.predict(X_row)[0])
    pred_proba = float(clf_model.predict_proba(X_row)[0][1])
    pred_score = float(reg_model.predict(X_row)[0])

    pred_out = "PASS" if pred_class == 1 else "FAIL"

    # Risk thresholds (decision-support)
    if pred_proba < 0.40:
        risk_label = "HIGH RISK"
        risk_note = "Immediate intervention recommended"
    elif pred_proba < 0.60:
        risk_label = "MEDIUM RISK"
        risk_note = "Monitor + light intervention"
    else:
        risk_label = "LOW RISK"
        risk_note = "Normal monitoring"

    st.markdown(f"""
    <div class="kpi-grid" style="grid-template-columns: repeat(4, 1fr); margin-top:10px;">
      <div class="kpi">
        <div class="kpi-label">Predicted Outcome</div>
        <div class="kpi-value">{pred_out}</div>
        <div class="kpi-sub">Classification</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Pass Probability</div>
        <div class="kpi-value">{pred_proba:.3f}</div>
        <div class="kpi-sub">Model confidence</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Risk Label</div>
        <div class="kpi-value">{risk_label}</div>
        <div class="kpi-sub">{risk_note}</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Predicted Avg Score</div>
        <div class="kpi-value">{pred_score:.2f}</div>
        <div class="kpi-sub">Regression output</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# -----------------------------
# Student Snapshot + Benchmark + Notes
# -----------------------------
cA, cB, cC = st.columns([1.05, 1.05, 0.90])

with cA:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Student Snapshot")

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
    st.markdown('</div>', unsafe_allow_html=True)

with cB:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Student vs Cohort Benchmark")

    bench_features = [c for c in [
        "early_avg_score",
        "early_total_clicks_vle",
        "early_num_vle_interactions_days",
        "total_clicks_vle",
        "num_vle_interactions_days",
        "avg_score"
    ] if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

    if bench_features:
        pass_df = df[df["final_result_binary"] == 1]
        fail_df = df[df["final_result_binary"] == 0]

        rows = []
        for f in bench_features:
            val = row.get(f, np.nan)
            # percentile within overall dataset
            pct = float((df[f].rank(pct=True) * 100).get(row.name, np.nan)) if f in df else np.nan
            rows.append({
                "feature": f,
                "student_value": val,
                "percentile_overall(%)": np.nan if np.isnan(val) else round(pct, 1),
                "mean_pass": float(pass_df[f].mean()) if len(pass_df) else np.nan,
                "mean_fail": float(fail_df[f].mean()) if len(fail_df) else np.nan,
            })
        st.dataframe(pd.DataFrame(rows))
    else:
        st.info("No benchmark-ready numeric columns found.")
    st.markdown('</div>', unsafe_allow_html=True)

with cC:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Notes")
    st.write("- Dashboard supports intervention decisions, not automatic decisions.")
    st.write("- Models are lightweight (runtime training) for deployment simplicity.")
    st.write("- Use charts to explain drivers (engagement, early score, etc.).")
    st.write("- Evaluation metrics use a test split set in the sidebar.")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# -----------------------------
# NEW: Model Performance Panel (Trust / Confidence)
# -----------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Model Performance (Test Split)")

m1, m2, m3, m4, m5, m6 = st.columns(6)
with m1: st.metric("Accuracy", f"{metrics['acc']:.3f}")
with m2: st.metric("F1 Score", f"{metrics['f1']:.3f}")
with m3: st.metric("MAE", f"{metrics['mae']:.3f}")
with m4: st.metric("RMSE", f"{metrics['rmse']:.3f}")
with m5: st.metric("R²", f"{metrics['r2']:.3f}")
with m6: st.metric("Test Size", f"{test_size:.2f}")

# Confusion Matrix plot
cm = metrics["cm"]
fig, ax = plt.subplots(figsize=(5.5, 3.8))
im = ax.imshow(cm, aspect="auto")
ax.set_title("Confusion Matrix (Test)")
ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred Fail", "Pred Pass"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual Fail", "Actual Pass"])
for (i, j), v in np.ndenumerate(cm):
    ax.text(j, i, str(v), ha="center", va="center", fontsize=12)
fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
fig.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# -----------------------------
# Engagement vs Performance
# -----------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Engagement vs Performance (Sample)")

x_col = "total_clicks_vle"
y_col = "avg_score"
color_col = "final_result_binary"

if x_col in df.columns and y_col in df.columns:
    tmp = df_plot[[x_col, y_col] + ([color_col] if color_col in df_plot.columns else [])].dropna()

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    if color_col in tmp.columns:
        pass_pts = tmp[tmp[color_col] == 1]
        fail_pts = tmp[tmp[color_col] == 0]
        ax.scatter(fail_pts[x_col], fail_pts[y_col], alpha=0.35, s=18, label="Fail", edgecolors="none")
        ax.scatter(pass_pts[x_col], pass_pts[y_col], alpha=0.35, s=18, label="Pass", edgecolors="none")
        ax.legend(frameon=False)
    else:
        ax.scatter(tmp[x_col], tmp[y_col], alpha=0.35, s=18, edgecolors="none")

    ax.set_xlabel("total_clicks_vle")
    ax.set_ylabel("avg_score")
    ax.set_title("Relationship between engagement and performance")
    ax.grid(True, alpha=0.20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
else:
    st.info("Missing total_clicks_vle and/or avg_score. Export them in your CSV if needed.")

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# -----------------------------
# Correlation Heatmap
# -----------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Correlation Heatmap (Key Features)")

corr_features = [
    c for c in [
        "early_avg_score",
        "total_clicks_vle",
        "num_vle_interactions_days",
        "early_total_clicks_vle",
        "early_num_vle_interactions_days",
        "avg_score",
        "final_result_binary",
    ]
    if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
]

if len(corr_features) >= 3:
    corr_df = df[corr_features].dropna().sample(
        n=min(sample_n, df[corr_features].dropna().shape[0]),
        random_state=int(random_state)
    )
    corr = corr_df.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(corr.values, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    ax.set_title("Correlation Heatmap (Sample)")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
else:
    st.info("Not enough numeric columns found for correlation heatmap.")

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# -----------------------------
# Regression Check: Actual vs Predicted
# -----------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Regression Check (Sample): Actual vs Predicted avg_score")

try:
    # Use numeric feature matrix aligned with training
    X_batch = build_feature_matrix(df_plot).reindex(columns=feature_cols).fillna(0.0)
    pred_batch = reg_model.predict(X_batch)
    actual = df_plot["avg_score"].values

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.scatter(actual, pred_batch, alpha=0.35, s=18, edgecolors="none")
    ax.set_xlabel("Actual avg_score")
    ax.set_ylabel("Predicted avg_score")
    ax.set_title("Actual vs Predicted (Diagonal = Perfect)")
    ax.grid(True, alpha=0.20)

    mn = float(np.nanmin([np.nanmin(actual), np.nanmin(pred_batch)]))
    mx = float(np.nanmax([np.nanmax(actual), np.nanmax(pred_batch)]))
    ax.plot([mn, mx], [mn, mx])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
except Exception:
    st.info("Could not compute regression check. Ensure your features are numeric and exist in the CSV.")

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# -----------------------------
# NEW: Feature Importance (Coefficients)
# -----------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Model Drivers (Feature Importance via Coefficients)")

top_k = st.slider("Top K features to show", 5, 30, 12, 1)

clf_coef = pd.Series(bundle["coefs"]["clf_coef"], index=feature_cols).sort_values()
reg_coef = pd.Series(bundle["coefs"]["reg_coef"], index=feature_cols).sort_values()

c1, c2 = st.columns(2)

with c1:
    st.markdown("**Classification Drivers (Logistic Regression)**")
    # show strongest positive and negative
    strongest = pd.concat([clf_coef.head(top_k//2), clf_coef.tail(top_k - top_k//2)])
    fig, ax = plt.subplots(figsize=(6, 4.6))
    ax.barh(strongest.index, strongest.values)
    ax.set_title("Top Coefficients (Pass Probability)")
    ax.set_xlabel("Coefficient (scaled features)")
    ax.grid(True, axis="x", alpha=0.20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with c2:
    st.markdown("**Regression Drivers (Linear Regression)**")
    strongest_r = pd.concat([reg_coef.head(top_k//2), reg_coef.tail(top_k - top_k//2)])
    fig, ax = plt.subplots(figsize=(6, 4.6))
    ax.barh(strongest_r.index, strongest_r.values)
    ax.set_title("Top Coefficients (Predicted avg_score)")
    ax.set_xlabel("Coefficient (scaled features)")
    ax.grid(True, axis="x", alpha=0.20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# -----------------------------
# Cluster Profiles (optional)
# -----------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Cluster Profiles (Exploratory)")

if "cluster_label" in df.columns:
    prof_cols = [c for c in ["avg_score", "total_clicks_vle", "num_vle_interactions_days", "early_avg_score"] if c in df.columns]
    if prof_cols:
        cluster_means = df.groupby("cluster_label")[prof_cols].mean(numeric_only=True)

        fig, ax = plt.subplots(figsize=(10, 4))
        cluster_means.plot(kind="bar", ax=ax)
        ax.set_title("Mean Key Features per Cluster")
        ax.set_xlabel("cluster_label")
        ax.set_ylabel("Mean Value")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.grid(True, axis="y", alpha=0.20)
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1), frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.info("No profile columns found to summarize clusters.")
else:
    st.info("No cluster_label column found in your CSV (optional).")

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# What-if Simulator (optional but powerful)
# -----------------------------
st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("What-if Simulator (Intervention Planning)")

sim_features = [c for c in ["early_avg_score", "early_total_clicks_vle", "early_num_vle_interactions_days"] if c in df.columns]
if sim_features:
    st.caption("Adjust early indicators and see how predictions change (decision-support simulation).")

    sim_row = row.copy()
    cols = st.columns(len(sim_features))
    for i, f in enumerate(sim_features):
        fmin = float(np.nanpercentile(df[f].dropna(), 5))
        fmax = float(np.nanpercentile(df[f].dropna(), 95))
        fcur = float(sim_row[f]) if pd.notna(sim_row[f]) else float(np.nanmedian(df[f].dropna()))
        with cols[i]:
            sim_row[f] = st.slider(f, min_value=fmin, max_value=fmax, value=min(max(fcur, fmin), fmax))

    # Build simulated feature vector
    sim_df = pd.DataFrame([sim_row.to_dict()]).select_dtypes(include=[np.number])
    sim_df = sim_df.drop(columns=["final_result_binary", "avg_score", "id_student"], errors="ignore")
    sim_X = sim_df.reindex(columns=feature_cols).fillna(0.0)

    sim_pred_class = int(clf_model.predict(sim_X)[0])
    sim_proba = float(clf_model.predict_proba(sim_X)[0][1])
    sim_score = float(reg_model.predict(sim_X)[0])
    sim_out = "PASS" if sim_pred_class == 1 else "FAIL"

    st.markdown(f"""
    <div class="kpi-grid" style="grid-template-columns: repeat(3, 1fr); margin-top:10px;">
      <div class="kpi">
        <div class="kpi-label">Simulated Outcome</div>
        <div class="kpi-value">{sim_out}</div>
        <div class="kpi-sub">After adjusting early indicators</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Simulated Pass Probability</div>
        <div class="kpi-value">{sim_proba:.3f}</div>
        <div class="kpi-sub">Change vs current: {sim_proba - pred_proba:+.3f}</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Simulated Predicted Avg Score</div>
        <div class="kpi-value">{sim_score:.2f}</div>
        <div class="kpi-sub">Change vs current: {sim_score - pred_score:+.2f}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.info("No early_* columns found for simulation. Add early indicators (recommended) to your CSV export.")

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Debug expander (optional)
# -----------------------------
with st.expander("Debug: Features Used"):
    st.write("Total numeric features used:", len(feature_cols))
    st.write("First 30 features:", feature_cols[:30])
    st.write("Note: Non-numeric columns are automatically excluded in this version.")
