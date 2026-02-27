# RedFlag AI - Customer Churn Predictor

Identify customers at risk of leaving before it's too late.

Live Demo: http://3.216.91.189:8501

---

## Overview

RedFlag AI is a machine learning web application that predicts whether a customer is likely to churn based on their account and billing information. Businesses can use this tool to proactively identify at-risk customers and take action before losing them.

Customer churn is one of the most costly problems across industries. Retaining an existing customer is 5x cheaper than acquiring a new one. RedFlag AI helps businesses act on data, not guesswork.

---

## Features

- Predicts customer churn with 81.55% accuracy
- Interactive web interface
- Real-time churn risk score with probability gauge
- Actionable business recommendations based on prediction
- Deployed on AWS EC2

---

## Tech Stack

| Layer | Technology |
|---|---|
| Machine Learning | Scikit-learn, XGBoost |
| Data Processing | Pandas, NumPy |
| Web App | Streamlit |
| Deployment | AWS EC2 |
| Language | Python 3.12 |

---

## Model Performance

Three models were trained and evaluated on the Telco Customer Churn dataset:

| Model | Accuracy |
|---|---|
| Logistic Regression | 81.55% |
| Random Forest | 79.63% |
| XGBoost | 79.42% |

Logistic Regression outperformed more complex models, demonstrating that simpler interpretable models can be highly effective on structured business data.

---

## Dataset

- Source: Telco Customer Churn (Kaggle)
- Size: 7,043 customers, 21 features
- Target: Churn (Yes/No)

---

## Run Locally

Clone the repo:
```bash
git clone https://github.com/yrufai/redFlagAi.git
cd redFlagAi
```

Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the app:
```bash
streamlit run app.py
```

Open in browser:
```
http://localhost:8501
```

---

## How It Works

1. User inputs customer details such as contract type, tenure and billing info
2. Inputs are encoded and scaled to match training data format
3. Logistic Regression model predicts churn probability
4. App displays prediction with risk score and recommended business actions

---

## Author

Rufai - ML Engineer
GitHub: https://github.com/yrufai

---

## License

MIT License