import pandas as pd
import streamlit as st
import plotly.express as px
import sqlite3

# ML
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# -------------------------------
# Page Config (UI)
# -------------------------------
st.set_page_config(page_title="O2C Dashboard", layout="wide")

st.markdown("""
    <style>
    body {
        background-color: #0E1117;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Load Data
# -------------------------------
df = pd.read_csv("data.csv")

df["invoice_date"] = pd.to_datetime(df["invoice_date"])
df["due_date"] = pd.to_datetime(df["due_date"])
df["payment_date"] = pd.to_datetime(df["payment_date"])

# -------------------------------
# Feature Engineering
# -------------------------------
df["delay_days"] = (df["payment_date"] - df["due_date"]).dt.days
df["delay_days"] = df["delay_days"].fillna(0)

df["status"] = df["payment_date"].apply(lambda x: "Unpaid" if pd.isna(x) else "Paid")
df["late_flag"] = df["delay_days"].apply(lambda x: "Late" if x > 0 else "On Time")

# Risk
df["risk_flag"] = df["delay_days"].apply(lambda x: "High Risk" if x > 5 else "Low Risk")

# -------------------------------
# SQL Integration
# -------------------------------
conn = sqlite3.connect("o2c.db")
df.to_sql("invoices", conn, if_exists="replace", index=False)

query = """
SELECT customer, SUM(amount) as total_revenue
FROM invoices
GROUP BY customer
ORDER BY total_revenue DESC
"""
result = pd.read_sql(query, conn)

# -------------------------------
# ML Model
# -------------------------------
df_ml = df.copy()
df_ml["late_flag"] = df_ml["late_flag"].map({"Late":1, "On Time":0})

X = df_ml[["amount"]]
y = df_ml["late_flag"]

# Handle small dataset safely
if len(df_ml) > 2:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    df["predicted_late"] = model.predict(X)
else:
    df["predicted_late"] = 0

# -------------------------------
# UI Title
# -------------------------------
st.title("📊 Order-to-Cash Analytics Dashboard")

# -------------------------------
# KPIs
# -------------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Total Revenue", int(df["amount"].sum()))
col2.metric("Avg Delay (Days)", round(df["delay_days"].mean(), 2))
col3.metric("Total Invoices", len(df))

# -------------------------------
# Insights
# -------------------------------
st.subheader("📌 Insights")

late_percent = (df["late_flag"] == "Late").mean() * 100

st.write(f"🔹 {round(late_percent,2)}% invoices are delayed, impacting cash flow")
st.write("🔹 Revenue is concentrated among key customers")
st.write("🔹 Delay patterns indicate need for better payment tracking")

# -------------------------------
# Charts
# -------------------------------
col4, col5 = st.columns(2)

fig1 = px.bar(result, x="customer", y="total_revenue",
              title="Revenue by Customer (SQL)")
col4.plotly_chart(fig1, use_container_width=True)

fig2 = px.pie(df, names="late_flag", title="Payment Status")
col5.plotly_chart(fig2, use_container_width=True)

# Risk chart
fig3 = px.pie(df, names="risk_flag", title="Customer Payment Risk")
st.plotly_chart(fig3, use_container_width=True)

# ML Prediction chart
st.subheader("🤖 ML Prediction")
fig4 = px.pie(df, names="predicted_late", title="Predicted Late Payments")
st.plotly_chart(fig4, use_container_width=True)

# -------------------------------
# Filter
# -------------------------------
st.subheader("🔍 Filter Data")

customer = st.selectbox("Select Customer", df["customer"].unique())
filtered_df = df[df["customer"] == customer]

st.dataframe(filtered_df, use_container_width=True)

# -------------------------------
# Full Data
# -------------------------------
st.subheader("📋 Full Dataset")
st.dataframe(df, use_container_width=True)