# Gold Price Forecasting Dashboard

This project was developed during my internship at **Dabur India Ltd.** to explore how machine learning can be used to forecast gold prices using historical market data and macroeconomic indicators.

The application collects market data, trains prediction models, evaluates their performance, and presents everything through an interactive Streamlit dashboard.

---

## What does this project do?

The dashboard allows users to:

- View historical gold price trends
- Predict gold prices for the next **1, 7, and 30 days**
- Compare model predictions with actual values
- Analyze model performance using different evaluation metrics
- Visualize moving averages and market trends
- Interact with an AI assistant to ask questions about the forecasts and dashboard

---

## Data Used

The model is trained using publicly available financial and economic data.

The following datasets are used:

- Gold Futures (GC=F)
- USD/INR Exchange Rate
- US Dollar Index (DXY)
- US Federal Interest Rate
- Consumer Price Index (CPI)

Data is fetched from **Yahoo Finance** and **FRED (Federal Reserve Economic Data)**.

---

## Feature Engineering

Instead of relying only on the gold price, the model uses several engineered features to better capture market behaviour.

| Feature | Why it is used |
|----------|----------------|
| Close | Latest gold futures closing price |
| USDINR | Reflects currency movement |
| DXY | Indicates the strength of the US Dollar |
| Interest Rate | Represents monetary policy changes |
| CPI | Captures inflation trends |
| MA7 | Short-term market trend |
| MA30 | Medium-term market trend |
| Lag1 | Previous day's closing price |
| Lag7 | Price from one week earlier |
| Return | Daily percentage change |
| Volatility | Measures market uncertainty |

---

## Machine Learning Model

The forecasting model is built using **Linear Regression**.

Three separate models are trained to predict:

- 1 Day Ahead
- 7 Days Ahead
- 30 Days Ahead

The models are evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## Dashboard Features

The Streamlit dashboard includes:

- Current gold price
- Historical price visualization
- Moving average analysis
- Forecast visualization
- Model performance metrics
- Residual analysis
- Prediction comparison table
- AI-powered assistant

---

## Project Structure

```text
gold_price_prediction/

├── app.py                  # Streamlit dashboard
├── notebook.py             # Data collection, preprocessing and model training
├── ai_advisor.py           # AI Assistant

├── model_1day.pkl
├── model_7day.pkl
├── model_30day.pkl

├── latest_features.csv
├── history.csv

├── data/
│   ├── gold.csv
│   ├── results_1day.csv
│   ├── results_7day.csv
│   └── results_30day.csv

├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/gold_price_prediction.git
cd gold_price_prediction
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate it

**macOS / Linux**

```bash
source .venv/bin/activate
```

**Windows**

```bash
.venv\Scripts\activate
```

Install all required packages

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Step 1

Train the models

```bash
python notebook.py
```

This generates:

- Trained model files
- Latest feature dataset
- Historical dataset
- Validation results

### Step 2

Launch the dashboard

```bash
streamlit run app.py
```

---

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Matplotlib
- Joblib
- Yahoo Finance API
- FRED API

---

## Current Limitations

Some limitations of the current implementation are:

- Uses Linear Regression, which may not capture complex nonlinear market behaviour.
- Uses publicly available financial data only.
- Does not currently include news sentiment or geopolitical events.
- Forecast accuracy may reduce during periods of unusually high market volatility.

---

## Possible Future Improvements

Some ideas for extending the project include:

- Using models such as XGBoost or LSTM
- Incorporating news and sentiment analysis
- Adding real-time market updates
- Automated model retraining
- Explainable AI using SHAP values
- Email or notification alerts for major price movements

---

## About

This project was built as part of my summer internship at **Dabur India Ltd.**

The goal was to understand the complete machine learning pipeline—from collecting financial data and engineering features to training predictive models and presenting the results through an interactive dashboard.

It also helped me gain hands-on experience with data preprocessing, model evaluation, visualization, and deploying ML applications using Streamlit.
