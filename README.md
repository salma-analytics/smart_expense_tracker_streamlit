# 💼 Smart Expense Tracker with ML-based Category Prediction

This Streamlit app helps users **track, manage, and analyze expenses**, powered by **machine learning** for category prediction and **behavioral analytics** for smart insights.

---

## 🚀 Features

- Add and view expenses manually or via CSV upload
- **Smart Category Predictor**: Predicts expense category from free-text descriptions (e.g., "Zomato", "Uber ride")
- **Behavioral Insights**:
  - Top spending category
  - Highest spending month
  - Weekend vs Weekday spending analysis
  - Budget overrun alert
  - Smart spike detection across months
- **Visualizations**:
  - Monthly trend line chart
  - Category-wise pie chart
  - Category-wise bar chart for current month
- Filter & export expense data (date range + category)

---

## 🧠 Tech Stack

- **Frontend**: Streamlit
- **ML Model**: Logistic Regression (joblib + TF-IDF)
- **Backend**: Python (Pandas, Plotly, Joblib)
- **Visualization**: Altair, Matplotlib

---

## 📁 Folder Structure
## **ML Workflow (Auto-Categorization)**

- **Model**: Multinomial Naive Bayes
- **Vectorizer**: CountVectorizer
- **Training Data**: Manually labeled CSV (`category_train.csv`)
- **Input**: Description (e.g., Zomato Order)
- **Output**: Predicted Category (e.g., Food, Health)

---

---

## 📝 How to Run

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Run the app
streamlit run app.py