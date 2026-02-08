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
# Custom CSS (cooler UI)
# -----------------------------
st.markdown("""
<style>
/* App background */
[data-testid="stAppViewContainer"]{
  background: radial-gradient(circle at 15% 15%, rgba(99,102,241,0.10), transparent 40%),
              radial-gradient(circle at 85% 10%, rgba(16,185,129,0.10), transparent 35%),
              radial-gradient(circle at 70% 85%, rgba(236,72,153,0.08), transparent 40%),
              #0b1220;
}

/* Reduce top padding */
.block-container { padding-top: 1.2rem; }

/* Headings color */
h1, h2, h3, h4 { color: #e5e7eb; }
p, span, div { color: #cbd5e1; }

/* Card style */
.card {
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 16px;
  padding: 16px 18px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.25);
}

/* KPI grid */
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

/* KPI card */
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

/* Section divider */
.hr {
  height: 1px;
  background: rgba(255,255,255,0.10);
  margin: 16px 0 8px 0;
}

/* Hide Streamlit footer */
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("student_data_merged.csv")

df = load_data()

required_cols = ["id_student", "final_result_binary", "avg_score"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Your CSV is missing required columns: {missing}")
    st.stop()

# -----------------------------
# Header
# -----------------------------
st.title("UNILINK – Early Prediction of Student Performance")
st.caption("Prototype decision-support dashboard (CSV-only). Lightweight models are trained at runtime for demo.")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Controls")
student_id = st.sidebar.selectbox("Select id_student", sorted(df["id_student"].unique()))
sample_n = st.sidebar.slider("Sample size for plots (speed)", 500, 10000, 2000, 500)
df_plot = df.sample(n=min(sample_n, len(df)), random_state=42).copy()

# -----------------------------
# Train lightweight models
# -----------------------------
@st.cache_resource
def train_models(df_train: pd.DataFrame):
    y_clf = df_train["final_result_binary"]
    y_reg = df_train["avg_score"]
    X = df_train.drop(columns=["final_result_binary", "avg_score"], errors="ignore")

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

# Student row + predictions
row = df[df["id_student"] == student_id].iloc[0]
X_row = pd.DataFrame([row[feature_cols]])

pred_class = int(clf_model.predict(X_row)[0])
pred_proba = float(clf_model.predict_proba(X_row)[0][1])
pred_score = float(reg_model.predict(X_row)[0])

# -----------------------------
# KPI Summary (boxed)
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
# Layout: charts in cards
# -----------------------------
c1, c2 = st.columns([1, 1])

with c1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Overall Outcome Summary (Actual)")

    plt.figure()
    plt.bar(["Fail", "Pass"], [fail_count, pass_count])
    plt.ylabel("Number of Students")
    plt.title("Pass vs Fail Distribution")
    st.pyplot(plt.gcf())
    plt.close()

    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Selected Student – Prediction (Demo)")

    # prediction “cards”
    pred_out = "PASS" if pred_class == 1 else "FAIL"
    st.markdown(f"""
    <div class="kpi-grid" style="grid-template-columns: repeat(3, 1fr);">
      <div class="kpi">
        <div class="kpi-label">Predicted Outcome</div>
        <div class="kpi-value">{pred_out}</div>
        <div class="kpi-sub">Classifier (runtime)</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Pass Probability</div>
        <div class="kpi-value">{pred_proba:.3f}</div>
        <div class="kpi-sub">Higher = safer</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Predicted avg_score</div>
        <div class="kpi-value">{pred_score:.2f}</div>
        <div class="kpi-sub">Regression (runtime)</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# -----------------------------
# Student Snapshot + Key Signals
# -----------------------------
cA, cB = st.columns([1.15, 0.85])

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
    st.subheader("Notes")
    st.write("- Dashboard is for decision support, not automatic decisions.")
    st.write("- Models are lightweight and trained at runtime for demo.")
    st.write("- Use the scatter/heatmap to explain key drivers.")
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

    plt.figure()
    if color_col in tmp.columns:
        pass_pts = tmp[tmp[color_col] == 1]
        fail_pts = tmp[tmp[color_col] == 0]
        plt.scatter(fail_pts[x_col], fail_pts[y_col], alpha=0.45, label="Fail")
        plt.scatter(pass_pts[x_col], pass_pts[y_col], alpha=0.45, label="Pass")
        plt.legend()
    else:
        plt.scatter(tmp[x_col], tmp[y_col], alpha=0.45)

    plt.xlabel("total_clicks_vle")
    plt.ylabel("avg_score")
    plt.title("Relationship between engagement and performance")
    st.pyplot(plt.gcf())
    plt.close()
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
    if c in df.columns
]

if len(corr_features) >= 3:
    corr_df = df[corr_features].dropna().sample(
        n=min(sample_n, df[corr_features].dropna().shape[0]),
        random_state=42
    )
    corr = corr_df.corr(numeric_only=True)

    plt.figure(figsize=(9, 4))
    plt.imshow(corr.values)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)
    plt.colorbar()
    plt.title("Correlation Heatmap (Sample)")
    st.pyplot(plt.gcf())
    plt.close()
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
    X_batch = df_plot[feature_cols]
    pred_batch = reg_model.predict(X_batch)
    actual = df_plot["avg_score"].values

    plt.figure()
    plt.scatter(actual, pred_batch, alpha=0.45)
    plt.xlabel("Actual avg_score")
    plt.ylabel("Predicted avg_score")
    plt.title("Actual vs Predicted (Diagonal = Perfect)")

    mn = float(np.nanmin([np.nanmin(actual), np.nanmin(pred_batch)]))
    mx = float(np.nanmax([np.nanmax(actual), np.nanmax(pred_batch)]))
    plt.plot([mn, mx], [mn, mx])  # reference line (default color)
    st.pyplot(plt.gcf())
    plt.close()
except Exception:
    st.info("Could not compute regression check. Ensure features exist and are numeric.")

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

        plt.figure(figsize=(10, 4))
        cluster_means.plot(kind="bar", ax=plt.gca())
        plt.title("Mean Key Features per Cluster")
        plt.xlabel("cluster_label")
        plt.ylabel("Mean Value")
        plt.xticks(rotation=0)
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        st.pyplot(plt.gcf())
        plt.close()
    else:
        st.info("No profile columns found to summarize clusters.")
else:
    st.info("No cluster_label column found in your CSV (optional).")

st.markdown('</div>', unsafe_allow_html=True)

# Debug (optional)
with st.expander("Debug: Features Used"):
    st.write("Total features used:", len(feature_cols))
    st.write("First 25 features:", feature_cols[:25])
