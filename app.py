import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(page_title="Banking Churn Risk Fusion App", layout="wide")

# ------------------
# HEADER
# ------------------
st.title("🏦 Banking Customer Churn Risk Analyzer")
st.markdown("""
Welcome to the interactive churn prediction tool.
This app demonstrates **Hard + Soft Data Fusion** for predicting **customer churn risk** using:
- LFM Clustering (KMeans)
- Soft Behavioral Rules (Trust, Dissatisfaction, Flight Risk)
- Combined Risk Score and Tagging
""")

# ------------------
# LOAD DATA
# ------------------
def load_data():
    df = pd.read_csv("final_churn_fused.csv")  # <- You must export your processed df to this file
    return df

try:
    df = load_data()
    st.success("✅ Data loaded successfully.")
except Exception as e:
    st.error("❌ Failed to load data. Make sure 'final_churn_fused.csv' is in the same folder.")
    st.stop()

# ------------------
# SIDEBAR
# ------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=70)
st.sidebar.markdown("#### **Built by Aditya T.**")
st.sidebar.caption("Final Year Project - 2025")

st.sidebar.header("📂 Filter by Risk Category")
risk_filter = st.sidebar.multiselect(
    label="Select Risk Categories",
    options=df['risk_category'].unique(),
    default=df['risk_category'].unique()
)

# ------------------
# TABLE OUTPUT
# ------------------
st.subheader("📋 Customer Churn Risk Table")
filtered_df = df[df['risk_category'].isin(risk_filter)]
st.dataframe(
    filtered_df[['customer_id', 'lfm_cluster', 'soft_score', 'final_churn_risk_score', 'risk_category', 'churn']].sort_values(by='final_churn_risk_score', ascending=False),
    use_container_width=True
)

# ------------------
# PLOTS
# ------------------
st.subheader("📊 Risk Category Distribution")
risk_counts = df['risk_category'].value_counts().reset_index()
risk_counts.columns = ['Risk Category', 'Count']
st.plotly_chart(
    px.bar(risk_counts, x='Risk Category', y='Count', color='Risk Category', title="Customer Distribution by Risk")
)

st.subheader("🔍 Churn Rate by Risk Category")
churn_rate = df.groupby('risk_category')['churn'].value_counts(normalize=True).unstack()
fig, ax = plt.subplots()
churn_rate.plot(kind='bar', stacked=True, ax=ax, colormap='Set2')
ax.set_title("Churn Distribution per Risk Category")
ax.set_ylabel("Proportion")
ax.legend(title="Churn", loc='upper right')
st.pyplot(fig)

# ------------------
# DOWNLOAD
# ------------------
st.sidebar.download_button(
    label="📥 Download Filtered Data",
    data=filtered_df.to_csv(index=False),
    file_name="filtered_churn_data.csv",
    mime='text/csv'
)

# ------------------
# LIVE PREDICTION
# ------------------
st.markdown("---")
st.header("🔮 Live Customer Churn Prediction")

with st.form("churn_form"):
    st.subheader("Enter New Customer Details")

    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
    tenure = st.slider("Tenure (Years)", 0, 10, 3)
    active_member = st.selectbox("Is Active Member?", [0, 1])
    products_number = st.slider("Number of Products", 1, 4, 2)
    balance = st.number_input("Account Balance", min_value=0.0, value=50000.0)
    estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=70000.0)

    submit = st.form_submit_button("Predict Churn Risk")

if submit:
    # Soft Score
    soft_score = 0
    if credit_score < 550:
        soft_score += 1
    if active_member == 0 and tenure < 3:
        soft_score += 1
    if estimated_salary > balance and balance < 10000:
        soft_score += 1
    soft_score_norm = soft_score / 3

    # Hard Score
    scaler = StandardScaler()
    kmeans = KMeans(n_clusters=3, random_state=42)
    scaler.fit(df[['tenure', 'products_number', 'balance']])
    kmeans.fit(scaler.transform(df[['tenure', 'products_number', 'balance']]))

    scaled_input = scaler.transform([[tenure, products_number, balance]])
    lfm_cluster = kmeans.predict(scaled_input)[0]
    lfm_score_norm = lfm_cluster / df['lfm_cluster'].max()

    final_score = 0.5 * soft_score_norm + 0.5 * lfm_score_norm

    if final_score < 0.33:
        risk_cat = "Low Risk"
        st.success(f"✅ Predicted Churn Risk: **{risk_cat}**")
    elif final_score < 0.66:
        risk_cat = "Medium Risk"
        st.warning(f"⚠️ Predicted Churn Risk: **{risk_cat}**")
    else:
        risk_cat = "High Risk"
        st.error(f"🛑 Predicted Churn Risk: **{risk_cat}**")

    st.info(f"📈 Final Risk Score: `{final_score:.2f}`  |  LFM Cluster: `{lfm_cluster}`  |  Soft Score: `{soft_score}`")

    with st.expander("📘 Why this prediction?"):
        st.markdown(f"""
        - **LFM Cluster**: Based on tenure, products, and balance
        - **Soft Score**: Trust/Dissatisfaction/Flight Risk flags
        - **Final Score**: `{final_score:.2f}` (0 = Safe, 1 = High Risk)
        """)

# ------------------
# FOOTER
# ------------------
st.markdown("---")
st.caption("© 2025 | Banking Churn Fusion App | Built by Aditya T.")