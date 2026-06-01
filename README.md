Signal Flow Pro
A campaign intelligence tool that identifies emerging keyword trends before they appear in traditional SEO tools like SEMrush.
Instead of analysing keywords that already have established search volume, Signal Flow catches demand signals as they form — giving content teams a 2–4 week head start on competitors.

How it works
One seed keyword
      ↓
Auto-expand to related keywords (Google Trends)
      ↓
Fetch signals from 3 sources: Web Search · YouTube · News
      ↓
Calculate velocity + z-score + regime per keyword
      ↓
Prophet forecast (next 13 weeks)
      ↓
Verdict: Act Now · Watch · Wait · Hold Off

Key concepts
Velocity — rate of change in search interest (14-day and 60-day rolling windows)
Z-score — how unusual the current velocity is relative to historical baseline. Built using robust MAD scoring to handle outliers.
Regime classification — each keyword is labelled at every point in time:

Spike — statistically significant momentum, act now
Building — momentum forming, prepare content
Quiet — no signal, hold
Collapsing — trend reversing, do not publish

Prophet forecast — Facebook's time series model projects the trend 13 weeks forward with confidence intervals and changepoint detection.

Why this is different from SEMrush
SEMrushSignal Flow ProData sourceSearch volume historyLive trend signalsSignal typeLagging indicatorLeading indicatorSourcesGoogle Search onlySearch + YouTube + NewsOutputKeyword difficulty scoreTimed publish verdictTimingShows what rankedShows what will trend

Installation
bashgit clone https://github.com/yourname/signal-flow-pro
cd signal-flow-pro
pip install -r requirements.txt

Usage
pythonfrom signal_flow import run_pipeline

# Run full pipeline for one seed keyword
run_pipeline("voiture électrique", geo="FR")
Or run the Streamlit dashboard:
bashstreamlit run app.py

Project structure
signal-flow-pro/
│
├── app.py                  # Streamlit dashboard
├── requirements.txt
├── README.md
│
├── data/
│   └── trends_data_melted.csv
│
└── src/
    ├── fetch.py            # pytrends ingestion (web, youtube, news)
    ├── clean.py            # deduplication, weighting, melting
    ├── analysis.py         # velocity, z-score, regime classification
    ├── forecast.py         # Prophet model
    ├── plots.py            # all visualisation functions
    └── export.py           # PDF report generation

Limitations

Z-score reliability requires at least 60 data points per keyword. Niche or new keywords may return Insufficient Data.
pytrends returns relative scores (0–100), not absolute search volume. Cross-keyword comparison should be treated as directional, not precise.
Rate limiting from Google Trends is handled with exponential backoff but long keyword lists may require overnight runs.
Prophet forecast accuracy degrades beyond 8–10 weeks for volatile keywords.


What's next (v2)

Dynamic source weighting per keyword category
Competitor content detection as a fourth signal
Backtest regime predictions against historical ranking outcomes
XGBoost scoring layer once labelled outcome data is available


Author
Built by Daksh — Business & Data Analyst
Vergule Collective
