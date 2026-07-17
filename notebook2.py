#!/usr/bin/env python
# coding: utf-8

# # Gold Price Prediction: End-to-End Machine Learning Project
# 
# This notebook builds a complete gold price forecasting workflow using market data, macroeconomic indicators, feature engineering, model training, evaluation, and dashboard artifact generation.
# 
# The final outputs are used by the Streamlit dashboard:
# 
# - `model_1day.pkl`
# - `model_7day.pkl`
# - `model_30day.pkl`
# - `latest_features.csv`
# - `data/gold.csv`

# ## 1. Project Introduction
# 
# ### Problem Statement
# 
# Gold prices are influenced by currency movements, inflation, interest rates, investor sentiment, and broader macroeconomic conditions. The goal of this project is to forecast future gold futures prices using historical gold market data and related economic indicators.
# 
# ### Business Objective
# 
# The business objective is to build a simple, interpretable forecasting system that can estimate gold prices over three horizons:
# 
# - 1 trading day ahead
# - 7 trading days ahead
# - 30 trading days ahead
# 
# These forecasts can support dashboards, market monitoring, portfolio analysis, and educational analysis of how financial and macroeconomic variables relate to gold prices.
# 
# ### Why Gold Price Forecasting Matters
# 
# Gold is widely used as a store of value, inflation hedge, safe-haven asset, and portfolio diversifier. Forecasting gold prices helps investors, analysts, jewelers, and commodity observers understand potential short-term price movement and the effect of economic indicators such as USDINR, DXY, CPI, and interest rates.

# ## 2. Environment Setup
# 
# This section imports the required libraries, defines project constants, and lists the model features and targets used throughout the notebook.

# In[143]:


from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Plot styling for a clean project notebook.
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (12, 5)
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 11

# Project paths.
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Data sources.
START_DATE = "2015-01-01"
GOLD_TICKER = "GC=F"
USDINR_TICKER = "INR=X"
DXY_TICKER = "DX-Y.NYB"

# Unit conversion: GC=F is USD per troy ounce; Indian_Gold is INR per 10 grams.
GRAMS_PER_TROY_OUNCE = 31.1034768
GRAMS_IN_INDIAN_QUOTE = 10

# Model input features. Future target columns are intentionally excluded.
FEATURES = [
    "Indian_Gold",
    "USDINR",
    "DXY",
    "Interest_Rate",
    "CPI",
    "MA7",
    "MA30",
    "Lag1",
    "Lag7",
    "Return",
    "Volatility",
]

TARGETS = ["Target_1", "Target_7", "Target_30"]
MODEL_FILES = {
    "Target_1": "model_1day.pkl",
    "Target_7": "model_7day.pkl",
    "Target_30": "model_30day.pkl",
}


# ## 3. Data Collection
# 
# This section downloads gold futures prices from Yahoo Finance, along with USDINR, DXY, CPI, and interest-rate data. Yahoo Finance provides daily market data, while FRED provides macroeconomic series through public CSV endpoints.

# In[144]:


def flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with one-level Yahoo Finance column names."""
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def download_yahoo_ohlcv(ticker: str, start: str) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance and normalize the index/columns."""
    df = yf.download(ticker, start=start, auto_adjust=True, progress=True)
    if df.empty:
        raise ValueError(f"No Yahoo Finance data returned for {ticker}")

    df = flatten_yfinance_columns(df)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index()
    return df


def download_yahoo_close(ticker: str, start: str, column_name: str) -> pd.Series:
    """Download one Yahoo Finance close series and give it a stable name."""
    df = download_yahoo_ohlcv(ticker, start)
    return df["Close"].rename(column_name)


def download_fred_series(series_id: str, start: str, column_name: str) -> pd.Series:
    """Download one FRED series from FRED's public CSV endpoint."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df = pd.read_csv(url, parse_dates=["observation_date"])
    df = df.rename(columns={"observation_date": "Date", series_id: column_name})
    df[column_name] = pd.to_numeric(df[column_name], errors="coerce")
    df = df.dropna(subset=[column_name])
    df = df[df["Date"] >= pd.to_datetime(start)]

    series = df.set_index("Date")[column_name]
    series.index = pd.to_datetime(series.index).tz_localize(None)
    return series.sort_index()


# ### Download Raw Data
# 
# The latest available trading day is downloaded from Yahoo Finance. CPI and interest-rate data may update less frequently because they are macroeconomic series rather than daily market instruments.

# In[145]:


gold = download_yahoo_ohlcv(GOLD_TICKER, START_DATE)
usdinr = download_yahoo_close(USDINR_TICKER, START_DATE, "USDINR")
dxy = download_yahoo_close(DXY_TICKER, START_DATE, "DXY")

# FRED macro series:
# FEDFUNDS = Effective Federal Funds Rate
# CPIAUCSL = Consumer Price Index for All Urban Consumers
data_interest_rate = download_fred_series("FEDFUNDS", START_DATE, "Interest_Rate")
data_cpi = download_fred_series("CPIAUCSL", START_DATE, "CPI")

print("Latest Yahoo gold date:", gold.index.max())
print("Latest USDINR date:", usdinr.index.max())
print("Latest DXY date:", dxy.index.max())
print("Latest interest-rate observation:", data_interest_rate.index.max())
print("Latest CPI observation:", data_cpi.index.max())


# ### Align Market and Economic Data
# 
# All external series are aligned to gold trading dates. Missing indicator values are forward-filled so each gold trading day has the most recently available value. A backward fill is applied only to handle rare leading gaps at the beginning of the dataset.

# In[146]:


gold = gold.join(usdinr, how="left")
gold = gold.join(dxy, how="left")
gold = gold.join(data_interest_rate, how="left")
gold = gold.join(data_cpi, how="left")

gold[["USDINR", "DXY", "Interest_Rate", "CPI"]] = (
    gold[["USDINR", "DXY", "Interest_Rate", "CPI"]]
    .ffill()
    .bfill()
)

# Indian gold estimate: USD/oz * INR/USD * 10 grams / troy ounce.
gold["Indian_Gold"] = (
    gold["Close"]
    * gold["USDINR"]
    * GRAMS_IN_INDIAN_QUOTE
    / GRAMS_PER_TROY_OUNCE
)

gold[["Indian_Gold", "USDINR", "DXY", "Interest_Rate", "CPI", "Indian_Gold"]].tail()


# ## 4. Exploratory Data Analysis (EDA)
# 
# EDA helps us understand the dataset before modeling. The analysis includes dataset shape, missing values, descriptive statistics, trend plots, distributions, correlations, and economic indicator behavior.

# ### Dataset Overview
# 
# This cell shows the size of the dataset, date range, available columns, and the most recent rows after aligning all data sources.

# In[147]:


print("Dataset shape:", gold.shape)
print("Start date:", gold.index.min())
print("End date:", gold.index.max())
print("Columns:", list(gold.columns))

gold.tail()


# ### Missing Value Analysis
# 
# Missing values are inspected before feature engineering. Market holidays, macroeconomic reporting frequency, and rolling-window calculations can all create missing values.

# In[148]:


missing_summary = pd.DataFrame({
    "Missing_Count": gold.isna().sum(),
    "Missing_Percent": (gold.isna().mean() * 100).round(2),
}).sort_values("Missing_Count", ascending=False)

missing_summary


# ### Descriptive Statistics
# 
# Descriptive statistics summarize the central tendency, spread, and range of the core market and macroeconomic variables.

# In[149]:


descriptive_columns = ["Close", "Indian_Gold", "USDINR", "DXY", "Interest_Rate", "CPI", "Volume"]
gold[descriptive_columns].describe().T


# ### Gold Price Trend Visualization
# 
# This plot shows how the gold futures closing price and INR-converted gold estimate changed over time.

# In[150]:


fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

gold["Close"].plot(ax=axes[0], color="goldenrod", linewidth=1.5)
axes[0].set_title("Gold Futures Close Price (GC=F)")
axes[0].set_ylabel("USD per Troy Ounce")

gold["Indian_Gold"].plot(ax=axes[1], color="darkgreen", linewidth=1.5)
axes[1].set_title("Estimated Indian Gold Price")
axes[1].set_ylabel("USD per Troy Ounce")
axes[1].set_xlabel("Date")

plt.tight_layout()
plt.show()


# ### Distribution Plots
# 
# Distribution plots show the spread and skewness of gold prices, daily returns, and major economic indicators.

# In[151]:


distribution_columns = ["Close",  "Indian_Gold", "USDINR", "DXY", "Interest_Rate", "CPI"]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.ravel()

for ax, column in zip(axes, distribution_columns):
    sns.histplot(gold[column].dropna(), kde=True, ax=ax, color="steelblue")
    ax.set_title(f"Distribution of {column}")

plt.tight_layout()
plt.show()


# ### Economic Indicators Visualization
# 
# This section visualizes USDINR, DXY, CPI, and interest rates to understand the macroeconomic environment around gold prices.

# In[152]:


economic_columns = ["USDINR", "DXY", "CPI", "Interest_Rate"]
fig, axes = plt.subplots(2, 2, figsize=(15, 8), sharex=True)
axes = axes.ravel()

for ax, column in zip(axes, economic_columns):
    gold[column].plot(ax=ax, linewidth=1.4)
    ax.set_title(column)
    ax.set_xlabel("Date")

plt.tight_layout()
plt.show()


# ## 5. Feature Engineering
# 
# The model uses market, technical, and macroeconomic features. All features are created on the full dataframe so the dashboard can always use the latest available market row.
# 
# Feature explanations:
# 
# - **Indian_Gold**: Gold price converted from USD per troy ounce into INR per 10 grams.
# - **USDINR**: Exchange rate between the US dollar and Indian rupee.
# - **DXY**: US Dollar Index, a measure of USD strength against major currencies.
# - **Interest_Rate**: Effective federal funds rate from FRED.
# - **CPI**: Consumer Price Index from FRED, used as an inflation indicator.
# - **MA7**: 7-day moving average of gold close price, capturing short-term trend.
# - **MA30**: 30-day moving average of gold close price, capturing medium-term trend.
# - **Lag1**: Previous trading day's close price.
# - **Lag7**: Close price from seven trading days earlier.
# - **Return**: Daily percentage change in gold close price.
# - **Volatility**: 30-day rolling standard deviation of close price.

# In[153]:


# Moving averages capture trend information from current and historical prices.
gold["MA7"] = gold["Close"].rolling(window=7).mean()
gold["MA30"] = gold["Close"].rolling(window=30).mean()

# Lag features capture recent historical price levels.
gold["Lag1"] = gold["Close"].shift(1)
gold["Lag7"] = gold["Close"].shift(7)

# Return captures daily percentage movement; volatility captures recent price dispersion.
gold["Return"] = gold["Close"].pct_change()
gold["Volatility"] = gold["Close"].rolling(window=30).std()

gold[FEATURES].tail()


# ### Feature Visualization
# 
# Moving averages and lag values are plotted against the gold close price to visually inspect the engineered technical features.

# In[154]:


plot_start = gold.index.max() - pd.DateOffset(years=2)
recent_gold = gold.loc[gold.index >= plot_start]

plt.figure(figsize=(14, 6))
plt.plot(recent_gold.index, recent_gold["Close"], label="Close", linewidth=1.6)
plt.plot(recent_gold.index, recent_gold["MA7"], label="MA7", linewidth=1.3)
plt.plot(recent_gold.index, recent_gold["MA30"], label="MA30", linewidth=1.3)
plt.title("Gold Close Price with Moving Averages")
plt.ylabel("USD per Troy Ounce")
plt.xlabel("Date")
plt.legend()
plt.tight_layout()
plt.show()


# ### Target Creation
# 
# The targets are future gold futures closing prices. These shifts intentionally create missing values at the end of the dataframe because future prices are not available yet.
# 
# Important: the full dataframe is not overwritten with `dropna()`. Only `train_data` drops missing values for model training.

# In[155]:


gold["Target_1"] = gold["Close"].shift(-1)
gold["Target_7"] = gold["Close"].shift(-7)
gold["Target_30"] = gold["Close"].shift(-30)

print("Latest full data date:", gold.index.max())
print("Missing values after feature and target creation:")
print(gold[FEATURES + TARGETS].isna().sum())

gold[["Close", "Target_1", "Target_7", "Target_30"]].tail(35)


# ### Correlation Heatmap
# 
# The correlation heatmap helps identify relationships among model features and targets. Strong correlations are expected between current gold prices, lagged prices, moving averages, and future gold prices.

# In[156]:


correlation_columns = FEATURES + TARGETS
corr = gold[correlation_columns].corr()

plt.figure(figsize=(12, 9))
sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, linewidths=0.5)
plt.title("Feature and Target Correlation Heatmap")
plt.tight_layout()
plt.show()


# ## 6. Model Development
# 
# A chronological train/test split is used because this is a time-series forecasting problem. Random shuffling would allow future market regimes to influence past training examples, creating look-ahead bias.
# 
# The notebook evaluates each model on the latest 20% of trainable rows and then refits the final saved model on all rows where the target is available.

# In[157]:


# Save the full engineered dataframe, including latest rows whose future targets are unknown.
gold.to_csv(DATA_DIR / "gold.csv")

# Only the training dataframe drops missing values.
train_data = gold.dropna()

print("Latest full data date:", gold.index.max())
print("Latest training date:", train_data.index.max())
print("Full dataframe shape:", gold.shape)
print("Training dataframe shape:", train_data.shape)

assert gold.index.max() > train_data.index.max(), (
    "Expected training data to end earlier than full data because future targets are unavailable."
)
assert not train_data[FEATURES + TARGETS].isna().any().any(), "Training data still contains NaNs."


# ### Train/Test Split and Linear Regression Training
# 
# Linear Regression is used because it is interpretable, fast, and suitable as a baseline regression model. Each horizon receives its own separate model.

# In[158]:


def chronological_train_test_split(df: pd.DataFrame, target_col: str, test_size: float = 0.2):
    """Split time-series data without shuffling to avoid look-ahead bias."""
    split_idx = int(len(df) * (1 - test_size))
    X = df[FEATURES]
    y = df[target_col]

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    return X_train, X_test, y_train, y_test


def evaluate_and_train_final_model(df: pd.DataFrame, target_col: str) -> tuple[LinearRegression, dict, pd.DataFrame]:
    """Evaluate chronologically, compare with a naive baseline, then refit on all valid rows."""
    X_train, X_test, y_train, y_test = chronological_train_test_split(df, target_col)

    validation_model = LinearRegression()
    validation_model.fit(X_train, y_train)
    model_preds = validation_model.predict(X_test)

    # Naive baseline: assume the future close equals the current close.
    naive_preds = df.loc[X_test.index, "Close"]

    metrics = {
        "Target": target_col,
        "Horizon": target_col.replace("Target_", "") + " day(s)",
        "Model_MAE": mean_absolute_error(y_test, model_preds),
        "Model_RMSE": mean_squared_error(y_test, model_preds) ** 0.5,
        "Model_R2": r2_score(y_test, model_preds),
        "Naive_MAE": mean_absolute_error(y_test, naive_preds),
        "Naive_RMSE": mean_squared_error(y_test, naive_preds) ** 0.5,
        "Naive_R2": r2_score(y_test, naive_preds),
        "Train_Start": X_train.index.min(),
        "Train_End": X_train.index.max(),
        "Test_Start": X_test.index.min(),
        "Test_End": X_test.index.max(),
    }

    validation_predictions = pd.DataFrame({
        "Actual": y_test,
        "Predicted": model_preds,
        "Naive": naive_preds,
    }, index=X_test.index)

    # Refit on every row with a valid target before saving for dashboard predictions.
    final_model = LinearRegression()
    final_model.fit(df[FEATURES], df[target_col])

    return final_model, metrics, validation_predictions


# ## 7. Model Evaluation
# 
# The models are evaluated using:
# 
# - **MAE**: Average absolute forecast error.
# - **RMSE**: Penalizes larger errors more strongly than MAE.
# - **R²**: Measures variance explained by the model.
# - **Naive baseline**: Assumes the future close equals the current close.

# In[159]:


models = {}
metrics_rows = []
prediction_frames = {}

for target_col, model_file in MODEL_FILES.items():
    model, metrics, validation_predictions = evaluate_and_train_final_model(train_data, target_col)
    models[target_col] = model
    metrics_rows.append(metrics)
    prediction_frames[target_col] = validation_predictions
    joblib.dump(model, model_file)
    print(f"Saved {model_file}")

metrics_df = pd.DataFrame(metrics_rows)
metrics_df


# ### Results Summary Table
# 
# This table compares Linear Regression against the naive baseline for the 1-day, 7-day, and 30-day forecasting horizons.

# In[160]:


summary_columns = [
    "Target",
    "Horizon",
    "Model_MAE",
    "Model_RMSE",
    "Model_R2",
    "Naive_MAE",
    "Naive_RMSE",
    "Naive_R2",
    "Train_End",
    "Test_Start",
    "Test_End",
]

results_summary = metrics_df[summary_columns].copy()
results_summary[["Model_MAE", "Model_RMSE", "Model_R2", "Naive_MAE", "Naive_RMSE", "Naive_R2"]] = (
    results_summary[["Model_MAE", "Model_RMSE", "Model_R2", "Naive_MAE", "Naive_RMSE", "Naive_R2"]]
    .round(3)
)

results_summary


# ## 8. Model Visualizations
# 
# This section visualizes actual vs predicted values and feature coefficients for interpretability.

# ### Actual vs Predicted Plots
# 
# Actual vs predicted plots show how closely each model follows the test-period gold futures price.

# In[161]:


fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=False)

for ax, target_col in zip(axes, TARGETS):
    predictions = prediction_frames[target_col]
    ax.plot(predictions.index, predictions["Actual"], label="Actual", linewidth=1.6)
    ax.plot(predictions.index, predictions["Predicted"], label="Linear Regression", linewidth=1.3)
    ax.plot(predictions.index, predictions["Naive"], label="Naive Baseline", linewidth=1.1, linestyle="--")
    ax.set_title(f"Actual vs Predicted: {target_col}")
    ax.set_ylabel("Gold Close (USD/oz)")
    ax.legend()

plt.tight_layout()
plt.show()


# ### Feature Coefficient Plots
# 
# Linear Regression coefficients show the direction and magnitude of each feature's contribution. Because the features are not standardized, coefficient magnitudes should be interpreted carefully and mainly used as a directional diagnostic.

# In[162]:


fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, target_col in zip(axes, TARGETS):
    coefficient_df = pd.DataFrame({
        "Feature": FEATURES,
        "Coefficient": models[target_col].coef_,
    }).sort_values("Coefficient", key=lambda s: s.abs(), ascending=False)

    sns.barplot(data=coefficient_df, y="Feature", x="Coefficient", ax=ax, color="steelblue")
    ax.set_title(f"Feature Coefficients: {target_col}")
    ax.set_xlabel("Coefficient")
    ax.set_ylabel("")

plt.tight_layout()
plt.show()


# ## 9. Dashboard Artifact Generation
# 
# The dashboard must always use the newest available market data, even though the training dataframe ends earlier because future targets are unavailable.
# 
# This section saves `latest_features.csv` from the full dataframe, not from `train_data`.

# In[163]:


latest = gold[FEATURES].tail(1)

print("Latest full data date:", gold.index.max())
print("Latest training date:", train_data.index.max())
print("Latest dashboard feature date:", latest.index.max())

assert latest.index.max() == gold.index.max(), "latest_features.csv is not using the newest full-data row."
assert not latest.isna().any().any(), "Latest feature row contains NaNs; dashboard predictions would fail."

# Save date as a column so Streamlit can display the data freshness.
latest.to_csv("latest_features.csv", index=True, index_label="Date")
latest


# ### End-to-End Prediction Check
# 
# This final check loads the exact dashboard feature row and confirms all three saved models can generate predictions from it.

# In[164]:


latest_from_csv = pd.read_csv("latest_features.csv")
X_latest = latest_from_csv[FEATURES]

print("Dashboard date:", latest_from_csv["Date"].iloc[0])
print("Current Indian gold estimate:", latest_from_csv["Indian_Gold"].iloc[0])
print("1-day forecast, GC=F Close:", models["Target_1"].predict(X_latest)[0])
print("7-day forecast, GC=F Close:", models["Target_7"].predict(X_latest)[0])
print("30-day forecast, GC=F Close:", models["Target_30"].predict(X_latest)[0])


# ## 10. Leakage and Bias Review
# 
# This review documents the modeling safeguards used in the notebook.

# In[165]:


leakage_review = [
    "The full dataframe is preserved after feature engineering.",
    "train_data = gold.dropna() is the only dropna step used for training.",
    "Future target columns are never included in FEATURES.",
    "The train/test split is chronological and never shuffled.",
    "No scaler is used for LinearRegression, so there is no scaling leakage.",
    "Rolling and lag features use current/past prices only.",
    "latest_features.csv is generated from gold[FEATURES].tail(1), so Streamlit uses the latest market row.",
    "FRED monthly values are forward-filled onto trading dates; for production-grade macro modeling, release-date vintages should be considered.",
]

pd.DataFrame({"Review_Item": leakage_review})


# ## 11. Conclusion
# 
# ### Key Findings
# 
# - The full dataset can contain the latest Yahoo Finance trading date while the training dataset ends earlier because future target values are unavailable.
# - Linear Regression provides an interpretable baseline for 1-day, 7-day, and 30-day gold futures forecasting.
# - Lagged prices, moving averages, USDINR, DXY, CPI, and interest rates provide a useful combination of market and macroeconomic context.
# - The Streamlit dashboard should always read the latest full-data feature row, not the truncated training dataset.
# 
# ### Limitations
# 
# - Linear Regression may not capture nonlinear market behavior.
# - FRED macroeconomic data is forward-filled and does not model exact publication lag or vintage revisions.
# - The targets forecast `GC=F` close prices, while dashboard INR values require conversion using the latest USDINR.
# - Financial markets are noisy, and short-term gold price movements can be affected by news events not present in the dataset.
# 
# ### Future Improvements
# 
# - Add walk-forward validation for more robust time-series evaluation.
# - Compare Linear Regression with Ridge, Lasso, Random Forest, XGBoost, and time-series models.
# - Add prediction intervals to communicate uncertainty.
# - Use release-date-aware macroeconomic data to reduce macro look-ahead risk.
# - Automate scheduled retraining and dashboard artifact refresh.
