import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="UNILINK | Student Performance", layout="wide")

# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("student_data_merged.csv")

df = load_data()

st.title("UNILINK – Student Performance Dashboard")
st.caption("CSV-only Streamlit prototype (models trained at runtime)")

# -----------------------------
# Train lightweight models
# -----------------------------
@st.cache_resource
def train_models(df):
    y_clf = df["final_result_binary"]
    y_reg = df["avg_score"]

    X = df.drop(columns=["final_result_binary", "avg_score"], errors="ignore")

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

    return clf, reg, X.columns

clf_model, reg_model, feature_cols = train_models(df)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Select Student")

student_id = st.sidebar.selectbox(
    "id_student",
    sorted(df["id_student"].unique())
)

row = df[df["id_student"] == student_id].iloc[0]
X_row = pd.DataFrame([row[feature_cols]])

# Predictions
pred_class = int(clf_model.predict(X_row)[0])
pred_proba = float(clf_model.predict_proba(X_row)[0][1])
pred_score = float(reg_model.predict(X_row)[0])

# -----------------------------
# KPI Cards
# -----------------------------
c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Predicted Outcome", "PASS" if pred_class == 1 else "FAIL")

with c2:
    st.metric("Pass Probability", f"{pred_proba:.3f}")

with c3:
    st.metric("Predicted Avg Score", f"{pred_score:.2f}")

st.divider()

# -----------------------------
# Pass vs Fail distribution
# -----------------------------
st.subheader("Pass vs Fail Distribution")

counts = df["final_result_binary"].value_counts().sort_index()
plt.figure()
plt.bar(["Fail", "Pass"], counts.values)
plt.ylabel("Count")
st.pyplot(plt.gcf())
plt.clf()

# -----------------------------
# Engagement vs Performance
# -----------------------------
if "total_clicks_vle" in df.columns:
    st.subheader("Engagement vs Performance")
    plt.figure()
    plt.scatter(df["total_clicks_vle"], df["avg_score"], alpha=0.5)
    plt.xlabel("Total VLE Clicks")
    plt.ylabel("Average Score")
    st.pyplot(plt.gcf())
    plt.clf()
