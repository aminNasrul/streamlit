# UNILINK – Early Prediction of Student Performance Dashboard

UNILINK is a lightweight, CSV-based Streamlit dashboard designed to support early prediction and monitoring of student academic performance. The system applies machine learning models to student engagement and assessment data to assist educators and academic advisors in identifying at-risk students and planning timely interventions.

> ⚠️ This system is a **decision-support tool**, not an automated decision-making system.

---

## 📌 Project Overview

The dashboard predicts:
- **Pass / Fail outcome** (classification)
- **Average score** (regression)

using early indicators such as:
- Early assessment scores  
- Virtual Learning Environment (VLE) engagement metrics  
- Engineered early-week features  

The application trains lightweight models at runtime and visualizes both predictions and explanatory insights to support academic decision-making.

---

## 🚀 Key Features

- 📊 **Dataset-level KPIs**
  - Total students, pass/fail counts, pass rate
- 👤 **Individual student prediction**
  - Predicted outcome (Pass/Fail)
  - Pass probability
  - Predicted average score
  - Risk level (High / Medium / Low)
- 🧠 **Model performance evaluation**
  - Accuracy, F1-score
  - MAE, RMSE, R²
  - Confusion matrix
- 🔍 **Explainability**
  - Feature importance via model coefficients
- 📈 **Visual analytics**
  - Engagement vs performance scatter plots
  - Correlation heatmap
  - Actual vs predicted regression plot
- 🧪 **What-if simulation**
  - Simulate improvements in early engagement and scores
- 🧩 **Optional clustering insights**
  - Cluster-level behavioral profiles (if provided)

---

## 🛠️ Technologies Used

- **Python**
- **Streamlit**
- **Pandas / NumPy**
- **Matplotlib**
- **Scikit-learn**
  - Logistic Regression (classification)
  - Linear Regression (regression)
  - StandardScaler, Pipelines

---

## 📁 Project Structure

├── app.py # Main Streamlit application
├── student_data_merged.csv # Input dataset (CSV-only)
├── README.md # Project documentation


---

## 📊 Dataset Requirements

The input CSV must contain at least the following columns:

Required:
- `id_student`
- `final_result_binary` (0 = Fail, 1 = Pass)
- `avg_score`

Recommended (for full functionality):
- `early_avg_score`
- `total_clicks_vle`
- `early_total_clicks_vle`
- `num_vle_interactions_days`
- `early_num_vle_interactions_days`
- `cluster_label` (optional)

> ⚠️ All features used for modeling must be **numeric**.  
Categorical variables should be encoded before exporting the CSV.

---

## ▶️ How to Run the Dashboard

### 1. Install dependencies
```bash
pip install streamlit pandas numpy matplotlib scikit-learn
2. Place your dataset

Ensure student_data_merged.csv is in the same directory as app.py.

3. Run the app
streamlit run app.py
