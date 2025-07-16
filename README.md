# 🧠 Banking Customer Churn Prediction | Hard + Soft Data Fusion

This project implements a powerful **churn prediction system** for the banking industry by combining **machine learning algorithms (hard data)** with **business-rule logic (soft data)**, inspired by the paper:

📄 *"Development of a Customer Churn Model for Banking Industry Based on Hard and Soft Data Fusion"*

---

## 🚀 Project Highlights

- ✅ End-to-end churn prediction pipeline (EDA → ML → Fusion → Dashboard)
- ✅ Combines **quantitative (LMF)** and **qualitative (soft rules)** data
- ✅ Real-time churn prediction using **Streamlit Web App**
- ✅ Business-friendly segmentation: **Low / Medium / High Risk**

---

## 📁 Files Included

| File / Folder | Description |
|---------------|-------------|
| `Banking_Customer_Churn_Project.ipynb` | Full Jupyter notebook pipeline |
| `final_churn_fused.csv` | Final dataset with all engineered features |
| `app.py` | Streamlit app for live churn prediction |
| `.streamlit/config.toml` | Custom theme config for dark/luxury mode |
| `README.md` | Project overview and documentation |

---

## 🧱 ML Models (Supervised Learning)

- Logistic Regression (`class_weight=balanced`)
- Decision Tree (CART, C4.5, C5.0)
- Random Forest (with `RandomizedSearchCV`)
- CHAID-style Chi-Square scoring

📌 Evaluated on:
- Accuracy
- Precision, Recall, F1-score
- AUC ROC

---

## 🧩 Fusion Strategy

### 🔸 Hard Data – LMF (Latent Modeling Features)
- Features used: `tenure`, `products_number`, `balance`
- Clustering: **KMeans (k=3)**
- Output: `lfm_cluster`

### 🔸 Soft Data – Business Logic Rules
Custom behavioral flags:
- Low Trust (credit score < 550)
- Dissatisfaction (inactive + short tenure)
- Flight Risk (high salary, low balance)

Output: `soft_score` (0 to 3)

### 🔸 Final Churn Risk Score
```python
final_churn_risk_score = 0.5 * lfm_score_norm + 0.5 * soft_score_norm
