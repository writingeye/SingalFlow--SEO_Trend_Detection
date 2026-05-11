import time
import warnings
import urllib3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages

from pytrends.request import TrendReq
from scipy.stats import median_abs_deviation
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG
# =============================================================================

keyword_list   = ["Vélo Électrique"]          # change keyword here
MAX_RETRIES    = 7
WAIT_SECONDS   = 10
WEIGHTS        = {"web": 0.5, "youtube": 0.3, "news": 0.2}
FORECAST_DAYS  = 90
WINDOW_DAYS    = 90

REGIME_COLORS = {
    "Spike":             "#c8f7c5",
    "Building":          "#fef9c3",
    "Quiet":             "#f0f0f0",
    "Collapsing":        "#fdd",
    "Insufficient Data": "#ffffff"
}

REGIME_PATCHES = [
    mpatches.Patch(color="#c8f7c5", label="Spike — Act Now"),
    mpatches.Patch(color="#fef9c3", label="Building — Watch"),
    mpatches.Patch(color="#f0f0f0", label="Quiet — Wait"),
    mpatches.Patch(color="#fdd",    label="Collapsing — Hold"),
]

# =============================================================================
# 1. FETCH
# =============================================================================

def fetch_trends(keyword_list, gprop=""):
    """
    Fetches Google Trends data for a list of keywords.
    gprop: "" (web), "youtube", or "news"
    Returns a raw wide dataframe.
    """
    pytrends = TrendReq(tz=360, timeout=(10, 25), requests_args={"verify": False}, hl = "fr")
    retries  = 0

    while retries < MAX_RETRIES:
        try:
            print(f"\nFetching [{gprop or 'web'}]: {keyword_list}")
            pytrends.build_payload(keyword_list, timeframe="today 12-m", gprop=gprop)
            iot = pytrends.interest_over_time()
            if iot.empty:
                raise ValueError("Empty dataframe returned")
            print("✅ Success")
            return iot

        except Exception as e:
            retries += 1
            err = str(e)
            if "429" in err:
                wait = WAIT_SECONDS * retries * 2
                print(f"⚠️  429 Rate limit. Waiting {wait}s")
                time.sleep(wait)
            elif "500" in err:
                wait = WAIT_SECONDS * retries
                print(f"⚠️  500 Google error. Retry in {wait}s")
                time.sleep(wait)
            elif "400" in err:
                print("⚠️  400 Bad request. Skipping.")
                return pd.DataFrame()
            else:
                print(f"❌ Unknown error: {e}. Retrying in {WAIT_SECONDS}s")
                time.sleep(WAIT_SECONDS)

    print("❌ Max retries reached.")
    return pd.DataFrame()


# =============================================================================
# 2. CLEAN + COMBINE
# =============================================================================

def clean_df(df, weight):
    """
    Resets index, drops isPartial, renames to ds/y, applies weight.
    Works regardless of what the keyword column is named.
    """
    df_clean = df.reset_index()
    if "isPartial" in df_clean.columns:
        df_clean = df_clean.drop(columns=["isPartial"])
    df_clean = df_clean.rename(columns={"date": "ds"})
    keyword_col = [c for c in df_clean.columns if c != "ds"][0]
    df_clean = df_clean.rename(columns={keyword_col: "y"})
    df_clean["y"] = df_clean["y"] * weight
    return df_clean[["ds", "y"]]


def melt_and_combine(web_clean, youtube_clean, news_clean):
    web_clean     = web_clean.copy()
    youtube_clean = youtube_clean.copy()
    news_clean    = news_clean.copy()

    web_clean["source"]     = "web"
    youtube_clean["source"] = "youtube"
    news_clean["source"]    = "news"

    combined = pd.concat([web_clean, youtube_clean, news_clean], ignore_index=True)
    return combined.sort_values("ds").reset_index(drop=True)


# =============================================================================
# 3. VELOCITY + REGIME
# =============================================================================

def robust_zscore(series):
    """Z-score using median and MAD. MAD floor prevents division by zero
    for low-value or flat niche keywords."""
    series = series.astype(float)
    median = series.median()
    mad    = max(median_abs_deviation(series, nan_policy="omit"), 1e-6)
    return (series - median) / mad


def classify_regime(row, spike_threshold, building_threshold):
    """
    Regime logic uses short (15d) and long (60d) z-scores only.
    Medium (30d) is computed separately for visual use.
    Thresholds are adaptive — passed in from add_velocity based on signal std.
    """
    sz = row["velocity_short_zscore"]
    lz = row["velocity_long_zscore"]

    if pd.isna(sz):
        return "Insufficient Data"

    crossover = (sz - lz) if pd.notna(lz) else 0

    if sz > spike_threshold and crossover >= 0:
        return "Spike"
    elif sz > building_threshold and crossover >= 0:
        return "Building"
    elif sz < -1.5 or (pd.notna(lz) and crossover < -1 and lz > 0):
        return "Collapsing"
    else:
        return "Quiet"


def add_velocity(df):
    """
    Computes three velocity rolling windows on the composite signal:
      - Short  (15d): fast momentum, used in regime logic
      - Medium (30d): confirmation layer, visual only
      - Long   (60d): structural direction, used in regime logic
    Thresholds adapt to signal volatility for niche keyword support.
    """
    df = df.copy()

    std = df["y"].std()
    spike_threshold    = 1.5 if std < 10 else 2.0
    building_threshold = 0.8 if std < 10 else 1.0

    df["velocity_short"]  = df["y"].rolling(15).mean().diff()
    df["velocity_medium"] = df["y"].rolling(30).mean().diff()
    df["velocity_long"]   = df["y"].rolling(60).mean().diff()

    df["velocity_short_zscore"]  = robust_zscore(df["velocity_short"])
    df["velocity_medium_zscore"] = robust_zscore(df["velocity_medium"])
    df["velocity_long_zscore"]   = robust_zscore(df["velocity_long"])

    df["regime"] = df.apply(
        lambda row: classify_regime(row, spike_threshold, building_threshold),
        axis=1
    )

    return df


# =============================================================================
# 4. PROPHET
# =============================================================================

def prepare_prophet_input(combined_df, weights=WEIGHTS):
    """
    Collapses three-source combined_df into one weighted composite
    per date. This is the correct input format for Prophet.
    """
    df = combined_df.copy()
    df["weight"]     = df["source"].map(weights)
    df["y_weighted"] = df["y"] * df["weight"]

    prophet_df = (
        df
        .groupby("ds")["y_weighted"]
        .sum()
        .reset_index()
        .rename(columns={"y_weighted": "y"})
    )
    return prophet_df


def run_prophet(prophet_df, periods=FORECAST_DAYS):
    """
    Fits Prophet on the normalised composite signal.
    Returns full forecast and forecast-only slice.
    Inverse transform restores original scale.
    """
    df     = prophet_df.copy()
    scaler = MinMaxScaler()
    df["y"] = scaler.fit_transform(df[["y"]])

    model  = Prophet(weekly_seasonality=True)
    model.fit(df[["ds", "y"]].dropna())

    future   = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    for col in ["yhat", "yhat_lower", "yhat_upper"]:
        forecast[col] = scaler.inverse_transform(forecast[[col]])

    forecast_only = forecast[forecast["ds"] > prophet_df["ds"].max()].copy()

    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]], forecast_only


# =============================================================================
# 5. SHARED PLOT HELPERS
# =============================================================================

def get_window(df, days=WINDOW_DAYS):
    cutoff = df["ds"].max() - pd.Timedelta(days=days)
    return df[df["ds"] >= cutoff].copy()


def format_ax(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")


# =============================================================================
# 6. CHARTS
# =============================================================================

def plot_signals(combined_df, web_clean, youtube_clean, news_clean):
    """
    Chart 1 — Signal Comparison
    Web / YouTube / News as thin background lines.
    Composite (weighted sum) as the dominant thick line.
    Last 90 days.
    """
    prophet_df = prepare_prophet_input(combined_df)

    w90 = get_window(web_clean)
    y90 = get_window(youtube_clean)
    n90 = get_window(news_clean)
    c90 = get_window(prophet_df)

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(w90["ds"], w90["y"], color="steelblue", linewidth=1.2, alpha=0.6, label="Web Search")
    ax.plot(y90["ds"], y90["y"], color="tomato",    linewidth=1.2, alpha=0.6, label="YouTube")
    ax.plot(n90["ds"], n90["y"], color="goldenrod", linewidth=1.2, alpha=0.6, label="News")
    ax.plot(c90["ds"], c90["y"], color="#1a1a2e",   linewidth=2.8,            label="Composite")

    ax.set_title(f"Signal Comparison — '{keyword_list[0]}' (Last 90 Days)",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Weighted Trend Score")
    ax.set_xlabel("Date")
    ax.legend()
    format_ax(ax)
    plt.tight_layout()
    plt.show()


def plot_regime(analysis_df):
    """
    Chart 2 — Regime Classification
    Composite line with colour-coded background shading by regime.
    Last 90 days.
    """
    df90 = get_window(analysis_df)

    fig, ax = plt.subplots(figsize=(14, 5))

    for i in range(len(df90) - 1):
        regime = df90["regime"].iloc[i]
        ax.axvspan(
            df90["ds"].iloc[i],
            df90["ds"].iloc[i + 1],
            color=REGIME_COLORS.get(regime, "#ffffff"),
            alpha=0.5
        )

    ax.plot(df90["ds"], df90["y"], color="#1a1a2e", linewidth=2, label="Composite Score")
    ax.set_title(f"Regime Classification — '{keyword_list[0]}' (Last 90 Days)",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Trend Score")
    ax.legend(handles=REGIME_PATCHES, loc="upper left", fontsize=8)
    format_ax(ax)
    plt.tight_layout()
    plt.show()


def plot_zscore(analysis_df):
    """
    Chart 3 — Velocity Z-Score Panel
    Short (15d): fast, sensitive — drives regime classification
    Medium (30d): confirmation layer — visual only
    Long (60d): structural direction — drives regime classification
    Thresholds shown adapt to signal volatility.
    Last 90 days.
    """
    df90 = get_window(analysis_df)

    std          = analysis_df["y"].std()
    spike_t      = 1.5 if std < 10 else 2.0
    collapse_t   = -1.5

    fig, ax = plt.subplots(figsize=(14, 4))

    ax.plot(df90["ds"], df90["velocity_short_zscore"],
            color="tomato",    linewidth=2,   label="Short Z (15d) — regime driver")
    ax.plot(df90["ds"], df90["velocity_medium_zscore"],
            color="goldenrod", linewidth=1.5, linestyle="--", label="Medium Z (30d) — visual only")
    ax.plot(df90["ds"], df90["velocity_long_zscore"],
            color="steelblue", linewidth=2,   label="Long Z (60d) — regime driver")

    ax.axhline(y=spike_t,   color="green", linestyle="--", alpha=0.6, linewidth=1,
               label=f"Spike threshold (z={spike_t})")
    ax.axhline(y=collapse_t, color="red",  linestyle="--", alpha=0.6, linewidth=1,
               label=f"Collapse threshold (z={collapse_t})")
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.2, linewidth=1)

    ax.set_title("Velocity Z-Scores — Short / Medium / Long", fontsize=14, fontweight="bold")
    ax.set_ylabel("Z-Score")
    ax.set_xlabel("Date")
    ax.legend(fontsize=8)
    format_ax(ax)
    plt.tight_layout()
    plt.show()


def plot_forecast(analysis_df, forecast_only):
    """
    Chart 4 — Prophet Forecast
    Last 90 days of historical composite + next 90 days of forecast.
    Clear visual break at the forecast start line.
    Confidence interval shaded.
    """
    df90 = get_window(analysis_df)

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(df90["ds"], df90["y"],
            color="steelblue", linewidth=2, label="Historical (90d)")
    ax.plot(forecast_only["ds"], forecast_only["yhat"],
            color="tomato", linewidth=2, linestyle="--", label="Forecast (90d)")
    ax.fill_between(
        forecast_only["ds"],
        forecast_only["yhat_lower"],
        forecast_only["yhat_upper"],
        color="tomato", alpha=0.15, label="Confidence Interval"
    )

    cutoff = analysis_df["ds"].max()
    ax.axvline(x=cutoff, color="gray", linestyle="--", alpha=0.5, label="Forecast Start")

    ax.set_title(f"Trend Forecast — '{keyword_list[0]}' (Last 90 Days + Next 90 Days)",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Trend Score")
    ax.set_xlabel("Date")
    ax.legend()
    format_ax(ax)
    plt.tight_layout()
    plt.show()


def plot_regime_timeline(analysis_df):
    """
    Chart 5 — Regime Timeline
    Horizontal colour bar showing regime state at each date.
    Fast at-a-glance summary for non-technical readers (Imane's clients).
    Last 90 days.
    """
    df90 = get_window(analysis_df)

    fig, ax = plt.subplots(figsize=(14, 1.8))

    for i in range(len(df90) - 1):
        regime = df90["regime"].iloc[i]
        width  = (df90["ds"].iloc[i + 1] - df90["ds"].iloc[i]).days
        ax.barh(
            0,
            width=width,
            left=mdates.date2num(df90["ds"].iloc[i]),
            color=REGIME_COLORS.get(regime, "#ffffff"),
            edgecolor="none",
            height=0.5
        )

    ax.set_xlim(
        mdates.date2num(df90["ds"].min()),
        mdates.date2num(df90["ds"].max())
    )
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set_yticks([])
    ax.set_title(f"Regime Timeline — '{keyword_list[0]}' (Last 90 Days)",
                 fontsize=12, fontweight="bold")
    ax.legend(handles=REGIME_PATCHES, loc="lower right", fontsize=8, ncol=4,
              bbox_to_anchor=(1, 1.1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    plt.show()



# =============================================================================
# 7. RUN
# =============================================================================

# --- Fetch ---
web_df     = fetch_trends(keyword_list, gprop="")
youtube_df = fetch_trends(keyword_list, gprop="youtube")
news_df    = fetch_trends(keyword_list, gprop="news")

# --- Clean ---
web_clean     = clean_df(web_df,     weight=WEIGHTS["web"])
youtube_clean = clean_df(youtube_df, weight=WEIGHTS["youtube"])
news_clean    = clean_df(news_df,    weight=WEIGHTS["news"])

# --- Combine ---
combined_df = melt_and_combine(web_clean, youtube_clean, news_clean)

# --- Composite + velocity ---
prophet_df  = prepare_prophet_input(combined_df)
analysis_df = add_velocity(prophet_df)

# --- Prophet ---
forecast_df, forecast_only = run_prophet(prophet_df)

# --- Charts ---
plot_signals(combined_df, web_clean, youtube_clean, news_clean)
plot_regime(analysis_df)
plot_zscore(analysis_df)
plot_forecast(analysis_df, forecast_only)
plot_regime_timeline(analysis_df)
