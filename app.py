from pathlib import Path
import numpy as np
import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import subprocess
import sys
import matplotlib.pyplot as plt
import shap
# from ai_advisor import get_procurement_advice
from ai_advisor import ask_ai

# ==========================================
# CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Gold Price Forecasting Platform",
    page_icon="🥇",
    layout="wide",
)

FEATURES = [
    "Close",
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
    "1 Day": "model_1day.pkl",
    "7 Days": "model_7day.pkl",
    "30 Days": "model_30day.pkl",
}

TARGET_BY_HORIZON = {
    "1 Day": "Target_1",
    "7 Days": "Target_7",
    "30 Days": "Target_30",
}

GRAMS_PER_TROY_OUNCE = 31.1034768
GRAMS_IN_INDIAN_QUOTE = 10

LATEST_FEATURES_PATH = Path("latest_features.csv")
HISTORY_PATH = Path("data/gold.csv")


results_1 = pd.read_csv(
    "data/results_1day.csv",
    parse_dates=["Date"]
)

results_7 = pd.read_csv(
    "data/results_7day.csv",
    parse_dates=["Date"]
)

results_30 = pd.read_csv(
    "data/results_30day.csv",
    parse_dates=["Date"]
)



# ==========================================
# PREMIUM DARK THEME
# ==========================================

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root {
            --bg: #070b13;
            --panel: #0d1422;
            --panel-2: #111b2e;
            --panel-3: #0a101c;
            --border: rgba(148, 163, 184, 0.18);
            --text: #f8fafc;
            --muted: #94a3b8;
            --gold: #f5c451;
            --gold-2: #d99b22;
            --green: #22c55e;
            --red: #ef4444;
            --blue: #38bdf8;
        }

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(245, 196, 81, 0.13), transparent 28rem),
                radial-gradient(circle at top right, rgba(56, 189, 248, 0.10), transparent 28rem),
                linear-gradient(180deg, #070b13 0%, #090f1b 45%, #070b13 100%);
            color: var(--text);
        }

        .block-container {
            padding-top: 1.25rem;
            padding-bottom: 2rem;
            max-width: 1440px;
        }

        h1, h2, h3, p, span, div {
            letter-spacing: 0;
        }

        div[data-testid="stMetric"] {
            background: linear-gradient(145deg, rgba(17, 27, 46, 0.96), rgba(9, 15, 27, 0.96));
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1rem 1.05rem;
            box-shadow: 0 18px 45px rgba(0, 0, 0, 0.25);
            min-height: 116px;
        }

        div[data-testid="stMetricLabel"] p {
            color: var(--muted);
            font-size: 0.82rem;
            font-weight: 600;
        }

        div[data-testid="stMetricValue"] {
            color: var(--text);
            font-size: 1.65rem;
            font-weight: 800;
        }

        .hero-card {
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(245, 196, 81, 0.22);
            border-radius: 22px;
            padding: 1.6rem;
            background:
                linear-gradient(135deg, rgba(245, 196, 81, 0.16), rgba(56, 189, 248, 0.08) 42%, rgba(15, 23, 42, 0.92)),
                linear-gradient(145deg, #111827, #07101f);
            box-shadow: 0 24px 70px rgba(0, 0, 0, 0.34);
            margin-bottom: 1.1rem;
        }

        .hero-card:after {
            content: "";
            position: absolute;
            right: -8rem;
            top: -8rem;
            width: 18rem;
            height: 18rem;
            background: radial-gradient(circle, rgba(245,196,81,0.22), transparent 65%);
        }

        .hero-eyebrow {
            color: var(--gold);
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.45rem;
        }

        .hero-title {
            color: var(--text);
            font-size: clamp(2rem, 4vw, 3.3rem);
            line-height: 1.03;
            font-weight: 800;
            margin: 0;
        }

        .hero-subtitle {
            color: #cbd5e1;
            max-width: 760px;
            margin: 0.7rem 0 1.2rem 0;
            font-size: 1rem;
        }

        .hero-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.75rem;
        }

        .hero-stat, .glass-card, .kpi-card, .indicator-card, .workflow-card, .feature-card, .accuracy-card {
            background: rgba(10, 16, 28, 0.78);
            border: 1px solid var(--border);
            border-radius: 16px;
            box-shadow: 0 18px 45px rgba(0, 0, 0, 0.24);
        }

        .hero-stat {
            padding: 0.85rem;
        }

        .label {
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            margin-bottom: 0.3rem;
        }

        .value {
            color: var(--text);
            font-size: 1.05rem;
            font-weight: 800;
        }

        .section-title {
            color: var(--text);
            font-size: 1.28rem;
            font-weight: 800;
            margin: 1.35rem 0 0.7rem 0;
        }

        .section-subtitle {
            color: var(--muted);
            font-size: 0.92rem;
            margin-top: -0.35rem;
            margin-bottom: 0.8rem;
        }

        .kpi-card {
            padding: 1rem;
            min-height: 154px;
        }

        .kpi-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.8rem;
        }

        .kpi-name {
            color: var(--muted);
            font-size: 0.82rem;
            font-weight: 800;
            text-transform: uppercase;
        }

        .pill {
            border-radius: 999px;
            padding: 0.25rem 0.55rem;
            font-size: 0.76rem;
            font-weight: 800;
        }

        .pill-up {
            color: #bbf7d0;
            background: rgba(34, 197, 94, 0.14);
            border: 1px solid rgba(34, 197, 94, 0.28);
        }

        .pill-down {
            color: #fecaca;
            background: rgba(239, 68, 68, 0.14);
            border: 1px solid rgba(239, 68, 68, 0.28);
        }

        .pill-flat {
            color: #cbd5e1;
            background: rgba(148, 163, 184, 0.12);
            border: 1px solid rgba(148, 163, 184, 0.22);
        }

        .kpi-value {
            color: var(--text);
            font-size: clamp(1.45rem, 2vw, 2rem);
            font-weight: 800;
            margin-bottom: 0.55rem;
        }

        .kpi-change {
            font-size: 0.9rem;
            font-weight: 700;
        }

        .positive { color: var(--green); }
        .negative { color: var(--red); }
        .neutral { color: var(--muted); }

        .glass-card {
            padding: 1rem;
        }

        .indicator-card, .accuracy-card, .feature-card {
            padding: 1rem;
            min-height: 132px;
        }

        .indicator-value, .accuracy-value {
            color: var(--text);
            font-size: 1.45rem;
            font-weight: 800;
            margin: 0.2rem 0;
        }

        .indicator-desc, .feature-desc, .accuracy-desc {
            color: var(--muted);
            font-size: 0.82rem;
            line-height: 1.35;
        }

        .workflow-grid {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 0.75rem;
            align-items: stretch;
        }

        .workflow-card {
            padding: 1rem;
            min-height: 124px;
            border-color: rgba(245, 196, 81, 0.18);
        }

        .workflow-step {
            color: var(--gold);
            font-weight: 800;
            font-size: 0.78rem;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }

        .workflow-title {
            color: var(--text);
            font-size: 1rem;
            font-weight: 800;
            margin-bottom: 0.35rem;
        }

        .workflow-desc {
            color: var(--muted);
            font-size: 0.82rem;
            line-height: 1.35;
        }

        .feature-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
        }

        .footer-card {
            margin-top: 1.4rem;
            padding: 1rem;
            border-radius: 16px;
            border: 1px solid var(--border);
            background: rgba(10, 16, 28, 0.74);
            color: var(--muted);
            text-align: center;
            box-shadow: 0 18px 45px rgba(0, 0, 0, 0.22);
        }

        div[data-testid="stTabs"] button {
            color: #cbd5e1;
            font-weight: 700;
        }

        div[data-testid="stTabs"] button[aria-selected="true"] {
            color: var(--gold);
        }

        div[data-testid="stExpander"] {
            border: 1px solid var(--border);
            border-radius: 16px;
            background: rgba(10, 16, 28, 0.55);
        }

        @media (max-width: 980px) {
            .hero-grid, .workflow-grid, .feature-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }

        @media (max-width: 640px) {
            .hero-grid, .workflow-grid, .feature-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# HELPERS
# ==========================================

def format_inr(value):
    return f"${value:,.2f}"

def format_number(value: float, decimals: int = 2) -> str:
    return f"{value:,.{decimals}f}"


def format_pct(value: float) -> str:
    return f"{value:+.2f}%"


def close_usd_to_indian_gold(close_price):
    return close_price 

def direction_class(value: float) -> str:
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "neutral"


def direction_pill(value: float) -> str:
    if value > 0:
        return "pill pill-up"
    if value < 0:
        return "pill pill-down"
    return "pill pill-flat"


def direction_arrow(value: float) -> str:
    if value > 0:
        return "↑"
    if value < 0:
        return "↓"
    return "→"


def performance_label(mae: float, mean_actual: float) -> str:
    error_pct = mae / mean_actual
    if error_pct <= 0.02:
        return "Excellent"
    if error_pct <= 0.05:
        return "Good"
    return "Moderate"


def dark_layout(height: int = 420, yaxis_title: str = "") -> dict:
    return dict(
        height=height,
        margin=dict(l=20, r=20, t=35, b=25),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(7,11,19,0.35)",
        font=dict(color="#cbd5e1", family="Inter"),
        xaxis=dict(
            gridcolor="rgba(148,163,184,0.12)",
            zerolinecolor="rgba(148,163,184,0.16)",
        ),
        yaxis=dict(
            title=yaxis_title,
            gridcolor="rgba(148,163,184,0.12)",
            zerolinecolor="rgba(148,163,184,0.16)",
        ),
        hoverlabel=dict(bgcolor="#111827", bordercolor="rgba(245,196,81,0.35)", font_color="#f8fafc"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )


@st.cache_resource
def load_models() -> dict:
    return {
        horizon: joblib.load(model_file)
        for horizon, model_file in MODEL_FILES.items()
    }


@st.cache_data
def load_latest_features() -> pd.DataFrame:
    latest = pd.read_csv(LATEST_FEATURES_PATH)
    missing_features = [feature for feature in FEATURES if feature not in latest.columns]
    if missing_features:
        raise ValueError(f"latest_features.csv is missing: {missing_features}")
    return latest


@st.cache_data
def load_history() -> pd.DataFrame:
    history = pd.read_csv(HISTORY_PATH, parse_dates=["Date"])
    return history.sort_values("Date")


@st.cache_data
def evaluate_models(history: pd.DataFrame) -> pd.DataFrame:
    train_data = history.dropna(subset=FEATURES + TARGETS).copy()
    rows = []

    for horizon, target_col in TARGET_BY_HORIZON.items():
        split_idx = int(len(train_data) * 0.8)
        X = train_data[FEATURES]
        y = train_data[target_col]

        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]

        validation_model = LinearRegression()
        validation_model.fit(X_train, y_train)
        preds = validation_model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds) ** 0.5
        r2 = r2_score(y_test, preds)

        rows.append({
            "Horizon": horizon,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "Interpretation": performance_label(mae, y_test.mean()),
        })

    return pd.DataFrame(rows)


def build_forecast_table(latest: pd.DataFrame, models: dict) -> pd.DataFrame:
    X_latest = latest[FEATURES]
    current_price = latest["Close"].iloc[0]
    usdinr = latest["USDINR"].iloc[0]

    rows = [{
        "Horizon": "Current",
        "Prediction": current_price,
        "Change": 0.0,
        "Pct_Change": 0.0,
        "Model": "Market Close",
    }]

    for horizon, model in models.items():
        predicted_close_usd = model.predict(X_latest)[0]
        predicted_price = predicted_close_usd
        rows.append({
            "Horizon": horizon,
            "Prediction": predicted_price,
            "Change": predicted_price - current_price,
            "Pct_Change": ((predicted_price - current_price) / current_price) * 100,
            "Model": "Linear Regression",
        })

    return pd.DataFrame(rows)


def kpi_card(title: str, value: float, change: float, pct_change: float) -> str:
    css_class = direction_class(change)
    pill_class = direction_pill(change)
    arrow = direction_arrow(change)
    change_text = "Baseline" if title == "Current Price" else f"{format_inr(change)} | {format_pct(pct_change)}"
    return f"""
        <div class="kpi-card">
            <div class="kpi-top">
                <div class="kpi-name">{title}</div>
                <div class="{pill_class}">{arrow}</div>
            </div>
            <div class="kpi-value">{format_inr(value)}</div>
            <div class="kpi-change {css_class}">{change_text}</div>
        </div>
    """


def indicator_card(name: str, value: str, description: str) -> str:
    return f"""
        <div class="indicator-card">
            <div class="label">{name}</div>
            <div class="indicator-value">{value}</div>
            <div class="indicator-desc">{description}</div>
        </div>
    """


def feature_card(name: str, description: str) -> str:
    return f"""
        <div class="feature-card">
            <div class="label">{name}</div>
            <div class="feature-desc">{description}</div>
        </div>
    """


def workflow_card(step: str, title: str, description: str) -> str:
    return f"""
        <div class="workflow-card">
            <div class="workflow-step">{step}</div>
            <div class="workflow-title">{title}</div>
            <div class="workflow-desc">{description}</div>
        </div>
    """


def forecast_summary_html(table_df: pd.DataFrame) -> str:
    rows = []
    for row in table_df.itertuples(index=False):
        css_class = direction_class(row.Change)
        arrow = direction_arrow(row.Change)
        rows.append(
            f"""
            <tr>
                <td>{row.Horizon}</td>
                <td>{format_inr(row.Prediction)}</td>
                <td class="{css_class}">{arrow} {format_inr(row.Change)}</td>
                <td class="{css_class}">{format_pct(row.Pct_Change)}</td>
                <td>{row.Model}</td>
            </tr>
            """
        )

    return f"""
        <div class="glass-card">
            <table style="width:100%; border-collapse:collapse;">
                <thead>
                    <tr style="color:#94a3b8; text-align:left; font-size:0.78rem; text-transform:uppercase;">
                        <th style="padding:0.7rem; border-bottom:1px solid rgba(148,163,184,0.18);">Horizon</th>
                        <th style="padding:0.7rem; border-bottom:1px solid rgba(148,163,184,0.18);">Prediction</th>
                        <th style="padding:0.7rem; border-bottom:1px solid rgba(148,163,184,0.18);">Change</th>
                        <th style="padding:0.7rem; border-bottom:1px solid rgba(148,163,184,0.18);">Percentage Change</th>
                        <th style="padding:0.7rem; border-bottom:1px solid rgba(148,163,184,0.18);">Model</th>
                    </tr>
                </thead>
                <tbody style="color:#f8fafc; font-weight:650;">
                    {''.join(rows)}
                </tbody>
            </table>
        </div>
    """


def stop_with_setup_message(error: Exception) -> None:
    st.error("The dashboard could not load its required files.")
    st.caption(str(error))
    st.info(
        "Run the notebook from top to bottom to regenerate "
        "`latest_features.csv`, `data/gold.csv`, and the model `.pkl` files."
    )
    st.stop()


# ==========================================
# LOAD DATA
# ==========================================

try:
    latest = load_latest_features()
    history = load_history()
    models = load_models()
    performance_df = evaluate_models(history)
except Exception as exc:
    stop_with_setup_message(exc)

forecast_df = build_forecast_table(latest, models)

latest_date = latest["Date"].iloc[0] if "Date" in latest.columns else "Latest available"
current_price = forecast_df.loc[forecast_df["Horizon"] == "Current", "Prediction"].iloc[0]

pred_1 = forecast_df.loc[
    forecast_df["Horizon"] == "1 Day",
    "Prediction"
].iloc[0]

pred_7 = forecast_df.loc[
    forecast_df["Horizon"] == "7 Days",
    "Prediction"
].iloc[0]

pred_30 = forecast_df.loc[
    forecast_df["Horizon"] == "30 Days",
    "Prediction"
].iloc[0]


# ==========================================
# SHAP EXPLAINER
# ==========================================

explainer = shap.LinearExplainer(
    models["1 Day"],
    history[FEATURES]
)

latest_features = latest[FEATURES]

shap_values = explainer(latest_features)

print(models["1 Day"])




# ==========================================
# HERO
# ==========================================

st.markdown(
    f"""
    <div class="hero-card">
        <div class="hero-eyebrow">AI-Powered Gold Price Intelligence Dashboard</div>
        <h1 class="hero-title">Gold Price Forecasting Platform</h1>
        <p class="hero-subtitle">
            Premium financial analytics dashboard for gold price monitoring, market indicators,
            engineered features, and Linear Regression forecasts.
        </p>
        <div class="hero-grid">
            <div class="hero-stat">
                <div class="label">Latest Market Date</div>
                <div class="value">{latest_date}</div>
            </div>
            <div class="hero-stat">
                <div class="label">Current Gold Price</div>
                <div class="value">{format_inr(current_price)} / troy ounce(31.10g)</div>
            </div>
            <div class="hero-stat">
                <div class="label">Model Type</div>
                <div class="value">Linear Regression</div>
            </div>
            <div class="hero-stat">
                <div class="label">Forecast Horizons</div>
                <div class="value">1D | 7D | 30D</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

dashboard_tab, analytics_tab, history_tab, pipeline_tab = st.tabs([
    " Dashboard ",
    " Analytics ",
    " Historical Lookup ",
    " Model Pipeline "
])

col1, col2 = st.columns([1,5])

with col1:

    if st.button("🔄 Refresh & Retrain", use_container_width=True):

        with st.spinner("Downloading latest data and retraining models..."):

            process = subprocess.run(
                [sys.executable, "notebook.py"],   # Replace with your Python filename
                capture_output=True,
                text=True
            )

        if process.returncode == 0:

            st.success("✅ Market data updated!")

            st.cache_data.clear()
            st.cache_resource.clear()

            st.rerun()

        else:

            st.error("❌ Refresh Failed")

            st.code(process.stderr)

# ==========================================
# DASHBOARD TAB
# ==========================================

with dashboard_tab:
    st.markdown('<div class="section-title">Forecast KPIs</div>', unsafe_allow_html=True)
    kpi_cols = st.columns(4)
    titles = ["Current Price", "1 Day Forecast", "7 Day Forecast", "30 Day Forecast"]
    for col, title, row in zip(kpi_cols, titles, forecast_df.to_dict("records")):
        with col:
            st.markdown(
                kpi_card(title, row["Prediction"], row["Change"], row["Pct_Change"]),
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-title">Forecast Visualization</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Current price and model forecasts converted to INR per 10 grams.</div>',
        unsafe_allow_html=True,
    )

    forecast_fig = go.Figure()
    forecast_fig.add_trace(
        go.Scatter(
            x=forecast_df["Horizon"],
            y=forecast_df["Prediction"],
            mode="lines+markers",
            line=dict(color="#f5c451", width=4, shape="spline"),
            marker=dict(size=11, color="#f8fafc", line=dict(color="#f5c451", width=3)),
            hovertemplate="<b>%{x}</b><br>Price: $%{y:,.0f}<extra></extra>",
        )
    )
    forecast_fig.update_layout(**dark_layout(height=430, yaxis_title="USD per Troy Ounce"))
    forecast_fig.update_yaxes(tickprefix="$", separatethousands=True)
    st.plotly_chart(forecast_fig, use_container_width=True)

    st.markdown('<div class="section-title">Forecast Summary</div>', unsafe_allow_html=True)
    table_df = forecast_df.loc[forecast_df["Horizon"] != "Current"].copy()
    summary = table_df.copy()

    summary["Prediction"] = summary["Prediction"].apply(lambda x: f"${x:,.0f}")
    summary["Change"] = summary["Change"].apply(lambda x: f"${x:,.0f}")
    summary["Pct_Change"] = summary["Pct_Change"].apply(lambda x: f"{x:+.2f}%")

    st.dataframe(
    summary,
    use_container_width=True,
    hide_index=True
    )

    st.markdown('<div class="section-title">Model Accuracy</div>', unsafe_allow_html=True)
    perf_cols = st.columns(3)
    for col, row in zip(perf_cols, performance_df.itertuples(index=False)):
        with col:
            st.markdown(
                f"""
                <div class="accuracy-card">
                    <div class="label">{row.Horizon} Model</div>
                    <div class="accuracy-value">{row.Interpretation}</div>
                    <div class="accuracy-desc">MAE: {format_number(row.MAE)} | RMSE: {format_number(row.RMSE)} | R²: {row.R2:.3f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-title">Market Indicators</div>', unsafe_allow_html=True)
    indicator_cols = st.columns(4)
    indicators = [
        ("USDINR", format_number(latest["USDINR"].iloc[0]), "Indian rupees required to buy one US dollar."),
        ("DXY", format_number(latest["DXY"].iloc[0]), "Strength of US Dollar against major currencies."),
        ("Interest Rate", f"{latest['Interest_Rate'].iloc[0]:.2f}%", "Federal funds rate used as a macro liquidity signal."),
        ("CPI", format_number(latest["CPI"].iloc[0]), "Inflation indicator from the consumer price index."),
    ]
    for col, indicator in zip(indicator_cols, indicators):
        with col:
            st.markdown(indicator_card(*indicator), unsafe_allow_html=True)


# ==========================================
# EDA ANALYTICS TAB
# ==========================================

with analytics_tab:
    st.markdown('<div class="section-title">Historical Gold Price Trend</div>', unsafe_allow_html=True)
    trend_fig = go.Figure()
    trend_fig.add_trace(
        go.Scatter(
            x=history["Date"],
            y=history["Close"],
            mode="lines",
            name="Indian Gold",
            line=dict(color="#f5c451", width=2.4),
            hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>",
        )
    )
    trend_fig.update_layout(**dark_layout(height=430, yaxis_title="USD per Troy Ounce"))
    trend_fig.update_yaxes(tickprefix="$", separatethousands=True)
    st.plotly_chart(trend_fig, use_container_width=True)

    st.markdown('<div class="section-title">Correlation Heatmap</div>', unsafe_allow_html=True)
    corr_cols = [col for col in FEATURES + TARGETS if col in history.columns]
    corr = history[corr_cols].corr()
    heatmap_fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale="RdBu",
            zmid=0,
            hovertemplate="%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>",
        )
    )
    heatmap_fig.update_layout(**dark_layout(height=620))
    st.plotly_chart(heatmap_fig, use_container_width=True)

    chart_col_1, chart_col_2 = st.columns(2)

    with chart_col_1:
        st.markdown('<div class="section-title">Feature Importance</div>', unsafe_allow_html=True)
        coef_df = pd.DataFrame({
            "Feature": FEATURES,
            "Coefficient": models["1 Day"].coef_,
        })
        coef_df["Abs"] = coef_df["Coefficient"].abs()
        coef_df = coef_df.sort_values("Abs", ascending=True)
        coef_fig = go.Figure(
            go.Bar(
                x=coef_df["Coefficient"],
                y=coef_df["Feature"],
                orientation="h",
                marker_color="#38bdf8",
                hovertemplate="%{y}<br>Coefficient: %{x:.5f}<extra></extra>",
            )
        )
        coef_fig.update_layout(**dark_layout(height=430, yaxis_title=""))
        st.plotly_chart(coef_fig, use_container_width=True)
        st.markdown(
    '<div class="section-title">SHAP Explanation</div>',
    unsafe_allow_html=True,
)
       



    with chart_col_2:
        st.markdown('<div class="section-title">Return Distribution</div>', unsafe_allow_html=True)
        returns = history["Return"].dropna()
        return_fig = go.Figure(
            go.Histogram(
                x=returns,
                nbinsx=70,
                marker_color="#22c55e",
                opacity=0.82,
                hovertemplate="Return: %{x:.3%}<br>Count: %{y}<extra></extra>",
            )
        )
        return_fig.update_layout(**dark_layout(height=430, yaxis_title="Frequency"))
        return_fig.update_xaxes(tickformat=".1%")
        st.plotly_chart(return_fig, use_container_width=True)




        st.markdown(
    '<div class="section-title">SHAP Feature Importance</div>',
    unsafe_allow_html=True,
)

    fig, ax = plt.subplots(figsize=(9,5))

    shap.plots.waterfall(shap_values[0])

    st.pyplot(fig)

    st.markdown('<div class="section-title">Moving Average Visualization</div>', unsafe_allow_html=True)
    ma_history = history.dropna(subset=["MA7", "MA30", "USDINR"]).copy()
    ma_fig = go.Figure()
    ma_fig.add_trace(
        go.Scatter(
            x=ma_history["Date"],
            y=ma_history["Close"],
            mode="lines",
            name="Indian Gold",
            line=dict(color="#f5c451", width=2.2),
            hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>",
        )
    )

# ==========================================
# MODEL VALIDATION
# ==========================================

st.markdown(
    '<div class="section-title">Model Validation</div>',
    unsafe_allow_html=True,
)

selected_model = st.selectbox(
    "Validation Horizon",
    ["1 Day", "7 Day", "30 Day"],
    key="validation_model"
)

if selected_model == "1 Day":
    validation_df = results_1.copy()
elif selected_model == "7 Day":
    validation_df = results_7.copy()
else:
    validation_df = results_30.copy()

# Actual vs Predicted
validation_fig = go.Figure()

validation_fig.add_trace(
    go.Scatter(
        x=validation_df["Date"],
        y=validation_df["Actual"],
        mode="lines",
        name="Actual",
        line=dict(color="#f5c451", width=3),
    )
)

validation_fig.add_trace(
    go.Scatter(
        x=validation_df["Date"],
        y=validation_df["Predicted"],
        mode="lines",
        name="Predicted",
        line=dict(color="#38bdf8", width=2, dash="dash"),
    )
)

validation_fig.update_layout(
    **dark_layout(height=500),
    yaxis_title="Gold Price (USD)"
)

st.plotly_chart(validation_fig, use_container_width=True)

mae = mean_absolute_error(validation_df["Actual"], validation_df["Predicted"])
rmse = np.sqrt(mean_squared_error(validation_df["Actual"], validation_df["Predicted"]))
r2 = r2_score(validation_df["Actual"], validation_df["Predicted"])

c1, c2, c3 = st.columns(3)

c1.metric("MAE", f"${mae:.2f}")
c2.metric("RMSE", f"${rmse:.2f}")
c3.metric("R²", f"{r2:.3f}")

validation_df["Residual"] = (
    validation_df["Actual"] - validation_df["Predicted"]
)

residual_fig = go.Figure()

residual_fig.add_trace(
    go.Scatter(
        x=validation_df["Date"],
        y=validation_df["Residual"],
        mode="markers",
        marker=dict(color="#ef4444", size=7),
        name="Residual",
    )
)

residual_fig.add_hline(y=0, line_dash="dash")

residual_fig.update_layout(
    **dark_layout(height=350),
    yaxis_title="Prediction Error"
)

st.plotly_chart(residual_fig, use_container_width=True)

st.markdown(
    '<div class="section-title">Prediction Comparison</div>',
    unsafe_allow_html=True,
)

table = validation_df.copy()
table["Absolute Error"] = table["Residual"].abs()

st.dataframe(table, use_container_width=True, hide_index=True)
ma_fig.add_trace(
        go.Scatter(
            x=ma_history["Date"],
            y=close_usd_to_indian_gold(ma_history["MA7"]),
            mode="lines",
            name="MA7",
            line=dict(color="#38bdf8", width=1.7),
            hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>",
        )
    )
ma_fig.add_trace(
        go.Scatter(
            x=ma_history["Date"],
            y=close_usd_to_indian_gold(ma_history["MA30"]),
            mode="lines",
            name="MA30",
            line=dict(color="#94a3b8", width=1.7),
            hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>",
        )
    )
ma_fig.update_layout(**dark_layout(height=430, yaxis_title="USD per Troy Ounce"))
ma_fig.update_yaxes(tickprefix="$", separatethousands=True)
st.plotly_chart(ma_fig, use_container_width=True)


# ==========================================
# HISTORICAL LOOKUP TAB
# ==========================================

with history_tab:

    st.header("Historical Gold Price Search")

    selected_date = st.date_input(
        "Select a Date",
        value=history["Date"].max().date(),
        min_value=history["Date"].min().date(),
        max_value=history["Date"].max().date(),
    )

    selected_date = pd.to_datetime(selected_date)

    nearest_idx = (
        (history["Date"] - selected_date)
        .abs()
        .idxmin()
    )

    row = history.loc[nearest_idx]

    st.success(
        f"Showing market data for **{row['Date'].strftime('%d %B %Y')}**"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Gold Close", f"${row['Close']:.2f}")

    with col2:
        st.metric("High", f"${row['High']:.2f}")

    with col3:
        st.metric("Low", f"${row['Low']:.2f}")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("USDINR", f"{row['USDINR']:.2f}")

    with col2:
        st.metric("DXY", f"{row['DXY']:.2f}")

    with col3:
        st.metric("Interest Rate", f"{row['Interest_Rate']:.2f}%")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("CPI", f"{row['CPI']:.2f}")

    with col2:
        st.metric("Trading Volume", f"{row['Volume']:,.0f}")

    window = history[
        (history["Date"] >= row["Date"] - pd.Timedelta(days=15))
        &
        (history["Date"] <= row["Date"] + pd.Timedelta(days=15))
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=window["Date"],
            y=window["Close"],
            mode="lines",
            line=dict(color="#f5c451", width=3),
            name="Gold Price",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[row["Date"]],
            y=[row["Close"]],
            mode="markers",
            marker=dict(size=14, color="red"),
            name="Selected Date",
        )
    )

    fig.update_layout(
        **dark_layout(height=430),
        yaxis_title="USD",
    )

    st.plotly_chart(fig, use_container_width=True)

    snapshot = pd.DataFrame(
        {
            "Metric": [
                "Gold Close",
                "Gold High",
                "Gold Low",
                "USDINR",
                "Dollar Index",
                "Interest Rate",
                "CPI",
                "Trading Volume",
            ],
            "Value": [
                f"${row['Close']:.2f}",
                f"${row['High']:.2f}",
                f"${row['Low']:.2f}",
                f"{row['USDINR']:.2f}",
                f"{row['DXY']:.2f}",
                f"{row['Interest_Rate']:.2f}%",
                f"{row['CPI']:.2f}",
                f"{row['Volume']:,.0f}",
            ],
        }
    )

    st.dataframe(snapshot, use_container_width=True, hide_index=True)
# ==========================================
# MODEL PIPELINE TAB
# ==========================================


with pipeline_tab:

    st.header("🏗️ Model Architecture")

    cols = st.columns(5)

    steps = [
        ("1️⃣ Data Source",
         "Yahoo Finance (GC=F)\nUSDINR\nDXY\nInterest Rate\nCPI"),

        ("2️⃣ Cleaning",
         "Datetime conversion\nForward fill\nMissing value handling"),

        ("3️⃣ Feature Engineering",
         "MA7\nMA30\nLag1\nLag7\nReturn\nVolatility"),

        ("4️⃣ Model Training",
         "Linear Regression\n1-Day\n7-Day\n30-Day"),

        ("5️⃣ Prediction",
         "Latest feature row\nForecast generation\nDashboard"),
    ]

    for col, (title, desc) in zip(cols, steps):
        with col:
            st.info(f"### {title}\n\n{desc}")

    st.divider()

    st.header("⚙ Feature Engineering Pipeline")

    raw_col, eng_col = st.columns(2)

    with raw_col:

        st.subheader("Raw Inputs")

        st.success("GC=F Gold Futures")

        st.success("USDINR Exchange Rate")

        st.success("Dollar Index (DXY)")

        st.success("Interest Rate")

        st.success("Consumer Price Index (CPI)")

    with eng_col:

        st.subheader("Engineered Features")

        st.info("MA7")

        st.info("MA30")

        st.info("Lag1")

        st.info("Lag7")

        st.info("Daily Return")

        st.info("30-Day Volatility")

    st.divider()

    st.header("Latest Model Input")

    latest_display = latest.copy()

    st.dataframe(
        latest_display[
            ["Date"] + FEATURES
            if "Date" in latest_display.columns
            else FEATURES
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    st.header("Recent Historical Dataset")

    display_cols = [
        "Date",
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

    display_cols = [
        c for c in display_cols
        if c in history.columns
    ]

    st.dataframe(
        history[display_cols].tail(20),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    st.header("📌 Complete ML Pipeline")

    st.code(
"""
Raw Data
   │
   ▼
Cleaning
   │
   ▼
Feature Engineering
   │
   ├── MA7
   ├── MA30
   ├── Lag1
   ├── Lag7
   ├── Return
   └── Volatility
   │
   ▼
Train/Test Split
   │
   ▼
Linear Regression
   │
   ├── 1-Day Model
   ├── 7-Day Model
   └── 30-Day Model
   │
   ▼
Prediction
   │
   ▼
Dashboard
""",
language="text",
)

# ==========================================
# AI PROCUREMENT ADVISOR
# ==========================================

st.divider()

st.header("🤖 AI Procurement Assistant")

col1, col2 = st.columns([8,1])

with col2:
    if st.button("🗑 Clear"):
        st.session_state.messages = []
        st.rerun()

st.caption(
    "Ask questions about the forecast, procurement strategy, or market indicators."
)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initial welcome message
if len(st.session_state.messages) == 0:
    st.session_state.messages.append({
        "role":"assistant",
        "content":f"""
# 🤖 Dabur Procurement Assistant

Hello!

I'm your AI Procurement Analyst.

I can help you:

• Explain predictions
• Recommend whether to Buy / Wait
• Analyze market indicators
• Explain why prices changed
• Summarize today's market
• Answer procurement questions

Just type your question below.

Current Gold Price : ${current_price:,.0f}

1-Day Forecast : ${pred_1:,.0f}

7-Day Forecast : ${pred_7:,.0f}

30-Day Forecast : ${pred_30:,.0f}

How can I help you today?
"""
    })

# Display messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


st.write("### 💡 Suggested Questions")

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

if col1.button("📈 Should Dabur buy now?", use_container_width=True):
    question = "Should Dabur buy gold now?"

elif col2.button("💰 Explain Forecast", use_container_width=True):
    question = "Explain today's forecast."

elif col3.button("🌍 Market Summary", use_container_width=True):
    question = "Summarize today's gold market."

elif col4.button("⚠ Procurement Risk", use_container_width=True):
    question = "What are today's procurement risks?"

else:
    question = st.chat_input("Ask Procurement AI...")




if question:

    st.session_state.messages.append(
        {
            "role":"user",
            "content":question
        }
    )

# Build conversation history
conversation = ""

for msg in st.session_state.messages[-6:]:
    conversation += f"{msg['role']}: {msg['content']}\n"

    context = f"""
    Dashboard Data

    Current Gold Price: ${current_price:,.0f}

    1-Day Forecast: ${pred_1:,.0f}

    7-Day Forecast: ${pred_7:,.0f}

    30-Day Forecast: ${pred_30:,.0f}

    USDINR: {latest["USDINR"].iloc[0]:.2f}

    DXY: {latest["DXY"].iloc[0]:.2f}

    Interest Rate: {latest["Interest_Rate"].iloc[0]:.2f}

    CPI: {latest["CPI"].iloc[0]:.2f}

    Volatility: {latest["Volatility"].iloc[0]:.2f}

    Previous Conversation

    {conversation}
    """

    # Ask only if there is a new question
if question:

    # Save user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    # Show user message
    with st.chat_message("user"):
        st.markdown(question)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Analyzing market..."):
            answer = ask_ai(question, context)

        st.markdown(answer)

    # Save assistant response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )

    st.rerun()


# ==========================================
# FOOTER
# ==========================================

st.markdown(
    """
    <div class="footer-card">
        <b>Created By:</b> Parth Kapoor<br>
        <b>Technology Stack:</b> Python | Pandas | Scikit-Learn | Streamlit | Plotly<br>
        <b>Models:</b> Linear Regression (1 Day, 7 Day, 30 Day)
    </div>
    """,
    unsafe_allow_html=True,
)