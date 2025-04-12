# Smart Expense Tracker with ML-based Category Prediction

This Streamlit app helps users track personal expenses, auto-categorize entries using machine learning (Naive Bayes + CountVectorizer), and gain smart behavioral insights.

---

## **Features**

- Add and view expenses
- Predicts category from free-text descriptions using ML
- **Smart Insights**:
  - Top spending category
  - Highest spending month
  - Weekend vs Weekday spike analysis
- Visualizations:
  - Pie Chart of category-wise spend
  - Monthly Trend Graph
- Filter & Export your data (CSV)
- Clean and intuitive UI built with **Streamlit**

---

## **ML Workflow (Auto-Categorization)**

- **Model**: Multinomial Naive Bayes
- **Vectorizer**: CountVectorizer
- **Training Data**: Manually labeled CSV (`category_train.csv`)
- **Input**: Description (e.g., Zomato Order)
- **Output**: Predicted Category (e.g., Food, Health)

---

## **How to Run**

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Run the app
streamlit run app.py
