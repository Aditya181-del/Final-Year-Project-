# 🧠 Final Project Summary – **Banking Customer Churn Prediction Using Hard + Soft Data Fusion**

This project presents a robust, end-to-end **churn prediction framework** for the banking industry by combining:

👉 **Machine Learning models (Hard Data)**  
👉 **Business Logic–driven insights (Soft Data)**

📄 *Inspired by the research paper:*  
**"Development of a Customer Churn Model for Banking Industry Based on Hard and Soft Data Fusion"**

---

## 📍 **Workflow Overview**

---

### 🔹 1. Data Exploration & Cleaning

- Loaded the **banking customer churn dataset**
- Handled:
  - **Missing values**
  - **Duplicate records**
  - **Outliers** in `age`, `credit_score`, etc. using **IQR**
- Performed **descriptive analysis** and correct **data typing**

---

### 🔹 2. Exploratory Data Analysis (EDA)

- Visualized patterns using:
  - `histplot`, `boxplot`, `heatmap`, `barplot`
- Discovered strong **churn indicators** like:
  - Inactive membership  
  - Low product usage  
  - High balance with low engagement  

---

### 🔹 3. Preprocessing

- **Encoded** categorical features (`gender`, `country`) using `LabelEncoder`
- Applied `train_test_split`
- Defined:
  - `X` = input features  
  - `y` = target (`churn`)  

---

### 🔹 4. Supervised Machine Learning Models (Hard Data)

Trained and evaluated several models:

- ✅ **Logistic Regression**  
- ✅ **Decision Tree** (CART, C4.5, C5.0 via XGBoost)  
- ✅ **Random Forest** (Basic & Hyperparameter-Tuned)  
- ✅ **Feature Selection** using Chi-Square (CHAID-style logic)

📊 **Evaluation Metrics:**
- Accuracy
- F1-Score
- ROC-AUC

🎯 **Best Model:**  
**Tuned Random Forest** via `RandomizedSearchCV` achieved **~86% AUC**

---

### 🔹 5. Hard Data Fusion – **Latent Modeling Features (LMF)**

- Selected behavioral indicators:  
  `tenure`, `products_number`, `balance`

- Applied:
  - `StandardScaler`  
  - **K-Means clustering** (`k=3`)  
  → Created `lfm_cluster`  

---

### 🔹 6. Soft Data – **Business Rule Modeling**

Defined 3 behavioral churn flags:

1. `low_trust_flag`: **credit score < 550**  
2. `dissatisfied_flag`: **inactive & tenure < 3**  
3. `flight_risk_flag`: **high salary but low balance**

🎯 These were summed to form a **`soft_score`** (range: **0–3**)

---

### 🔹 7. Fusion: Combining Hard + Soft Scores

- **Normalized** both `lfm_cluster` and `soft_score` to 0–1 range
- Calculated final churn risk score:

```python
df['final_churn_risk_score'] = 0.5 * lfm_score_norm + 0.5 * soft_score_norm

--- 





