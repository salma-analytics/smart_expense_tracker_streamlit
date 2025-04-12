import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Set up Streamlit page
st.set_page_config(page_title="Smart Expense Tracker", layout="wide")
st.title("🧠 Smart Expense Tracker")

# Load ML training data
@st.cache_data
def train_model():
    try:
        df_train = pd.read_csv("category_train.csv")
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(df_train["Description"])
        y = df_train["Category"]
        model = MultinomialNB()
        model.fit(X, y)
        return model, vectorizer
    except:
        return None, None

model, vectorizer = train_model()

# Session state
if "expenses" not in st.session_state:
    st.session_state.expenses = pd.DataFrame(columns=["Date", "Category", "Amount"])

# --- Sidebar: Add New Expense ---
st.sidebar.header("➕ Add New Expense")

# Description + ML prediction
description = st.sidebar.text_input("Description (e.g., Zomato Order)")
predicted_category = ""
if description and model:
    X_test = vectorizer.transform([description])
    predicted_category = model.predict(X_test)[0]
    st.sidebar.markdown(f"**Predicted Category:** `{predicted_category}`")
else:
    predicted_category = "Food"

# Category dropdown (with auto-selection)
category = st.sidebar.selectbox("Or Choose Category", 
    ["Food", "Transport", "Utilities", "Entertainment", "Health", "Others"],
    index=0 if predicted_category == "Food" else
          1 if predicted_category == "Transport" else
          2 if predicted_category == "Utilities" else
          3 if predicted_category == "Entertainment" else
          4 if predicted_category == "Health" else 5
)

# Amount & Date
amount = st.sidebar.number_input("Amount", min_value=0.0, format="%.2f", step=0.01)
date = st.sidebar.date_input("Date", value=datetime.today())

if st.sidebar.button("Add Expense"):
    new_expense = pd.DataFrame({
        "Date": [pd.to_datetime(date)],
        "Category": [category],
        "Amount": [amount]
    })
    st.session_state.expenses = pd.concat([st.session_state.expenses, new_expense], ignore_index=True)
    st.success("Expense added successfully!")
    st.rerun()

# --- Expense History ---
st.subheader("📋 Expense History")
st.dataframe(st.session_state.expenses)

# Total spent
total = st.session_state.expenses["Amount"].sum()
st.metric("💰 Total Spent (₹)", f"{total:.2f}")

# --- Smart Insights ---
st.subheader("🧠 Smart Insights")
if not st.session_state.expenses.empty:
    df = st.session_state.expenses.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Month"] = df["Date"].dt.to_period("M").astype(str)
    df["DayName"] = df["Date"].dt.day_name()

    # Top Category
    top_category = df.groupby("Category")["Amount"].sum().idxmax()
    top_cat_amt = df.groupby("Category")["Amount"].sum().max()
    total_amt = df["Amount"].sum()
    top_cat_pct = (top_cat_amt / total_amt) * 100

    # Top Month
    top_month = df.groupby("Month")["Amount"].sum().idxmax()
    top_month_amt = df.groupby("Month")["Amount"].sum().max()

    # Weekend Spike
    df["IsWeekend"] = df["DayName"].isin(["Saturday", "Sunday"])
    weekend_spend = df[df["IsWeekend"]]["Amount"].sum()
    weekday_spend = df[~df["IsWeekend"]]["Amount"].sum()
    weekend_diff_pct = ((weekend_spend - weekday_spend) / weekday_spend) * 100 if weekday_spend > 0 else 0

    # Show Insights
    st.markdown(f"• Most spending is on **{top_category}** (₹{top_cat_amt:.2f}) — **{top_cat_pct:.1f}%** of total.")
    st.markdown(f"• Highest spending month: **{top_month}** (₹{top_month_amt:.2f})")
    st.markdown(f"• Weekend spending is **{abs(weekend_diff_pct):.1f}% {'higher' if weekend_diff_pct > 0 else 'lower'}** than weekdays.")
else:
    st.info("Add expenses to see smart insights.")

# --- Summary Cards ---
st.subheader("📌 Quick Summary")
if not st.session_state.expenses.empty:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Spent (₹)", f"{total:.2f}")
    with col2:
        st.metric("Most Spent On", top_category, f"₹{top_cat_amt:.2f}")
    with col3:
        st.metric("Top Month", top_month, f"₹{top_month_amt:.2f}")
    with col4:
        label = "Higher" if weekend_diff_pct > 0 else "Lower"
        st.metric("Weekend Spike", f"{abs(weekend_diff_pct):.1f}%", label)
else:
    st.info("Add expenses to view summary.")

# --- Category-wise Pie Chart ---
st.subheader("📊 Expense by Category")
if not st.session_state.expenses.empty:
    pie_data = st.session_state.expenses.groupby("Category")["Amount"].sum()
    fig1, ax1 = plt.subplots()
    ax1.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    st.pyplot(fig1)

# --- Monthly Bar Chart ---
st.subheader("📅 Monthly Expense Trend")
if not st.session_state.expenses.empty:
    bar_df = st.session_state.expenses.copy()
    bar_df["Date"] = pd.to_datetime(bar_df["Date"], errors="coerce")
    bar_df["Month"] = bar_df["Date"].dt.to_period("M").astype(str)
    monthly = bar_df.groupby("Month")["Amount"].sum()
    st.bar_chart(monthly)

# --- Filter + CSV Export ---
st.subheader("🔍 Filter Your Expenses")
with st.expander("🧰 Filter Options"):
    filter_category = st.selectbox("Filter by Category", options=["All"] + list(st.session_state.expenses["Category"].unique()))
    start_date = st.date_input("Start Date", value=datetime.today())
    end_date = st.date_input("End Date", value=datetime.today())

    filtered_data = st.session_state.expenses.copy()
    filtered_data["Date"] = pd.to_datetime(filtered_data["Date"], errors="coerce")
    filtered_data = filtered_data[
        (filtered_data["Date"] >= pd.to_datetime(start_date)) &
        (filtered_data["Date"] <= pd.to_datetime(end_date))
    ]

    if filter_category != "All":
        filtered_data = filtered_data[filtered_data["Category"] == filter_category]

    filtered_data["Month"] = filtered_data["Date"].dt.to_period("M").astype(str)

    st.dataframe(filtered_data)
    st.write(f"**Total in Selected Range:** ₹{filtered_data['Amount'].sum():.2f}")

    csv = filtered_data.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Filtered CSV", csv, "filtered_expenses.csv", "text/csv")





