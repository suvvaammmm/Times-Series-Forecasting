# ARIMA Quant Engine

A Flask web app for time-series forecasting and lightweight quantitative trading analysis on stock/asset price data. Upload a CSV (or pull live data from Angel One) and get forecasts, anomaly detection, backtests, a trading signal, strategy simulation, and portfolio analytics — all through a browser dashboard.

## Features

- **Multiple forecasting models**
  - `ARIMA` — grid-searched (p,d,q) on log prices, with drift
  - `SARIMA` — seasonal ARIMA on log returns
  - `Ridge` — regularized linear regression on lag/rolling/momentum features
  - `ETS` — Holt-Winters exponential smoothing (additive/multiplicative, damped)
  - `AUTO` — runs ARIMA, SARIMA, and Ridge, backtests each, and picks the best by a weighted score (RMSE, MAPE, AIC, direction accuracy, Ljung-Box p-value)
- **Rolling backtesting** — walk-forward validation reporting RMSE, MAPE, and directional accuracy
- **Anomaly detection** — AR(1) residual-based outlier flagging with a configurable threshold
- **Signal engine** — BUY / SELL / HOLD signal derived from forecast direction, confidence interval width, and volatility regime (high / normal / low)
- **Strategy simulation** — simulates trading the rolling-backtest predictions with transaction costs and fixed risk-per-trade sizing, reporting total/annualized return, win rate, Sharpe ratio, max drawdown, and an equity curve vs. buy-and-hold
- **Portfolio analytics** — multi-asset CSV upload with inverse-volatility weighting, per-asset ARIMA return forecasts, and combined portfolio equity/forecast curves
- **Live market data (optional)** — pulls daily candles from Angel One (SmartAPI) for a given symbol token, with session caching and rate-limit retry/backoff
- **Web dashboard** — single-page UI (`templates/index.html`) for uploading data, choosing a model, and viewing charts/metrics

## Project Structure

```
app.py                          # Flask app and API routes
config.py                       # Loads Angel One credentials from .env
requirements.txt
Procfile / render.yaml          # Deployment config (gunicorn / Render)

model/
  forecasting/
    arima_model.py              # ARIMA (log price, grid search, drift)
    sarima_model.py             # SARIMA (log returns, seasonal)
    ridge_model.py              # Ridge regression on engineered features
    ets_model.py                # Holt-Winters exponential smoothing
    auto_selector.py            # Runs & scores all models, picks the best
  backtest/
    backtest.py                 # Static + rolling walk-forward backtesting
  anomaly/
    detect.py                   # ARIMA-residual anomaly detection
  strategy/
    strategy.py                 # Signal-driven trading simulation
    risk_engine.py               # Volatility-based position sizing, drawdown/kill-switch checks
  train.py                      # Standalone script: train/evaluate/plot an ARIMA model on sample data

portfolio/
  portfolio_engine.py           # Multi-asset portfolio construction & forecasting

services/
  angel_service.py              # Angel One (SmartAPI) live data integration

templates/
  index.html                    # Frontend dashboard

data/
  sample_data.csv               # Example single-series data (date, value)
  multi_test.csv                # Example multi-asset data (Date, TICKER1, TICKER2, ...)
test.csv                        # Example single-column ("value") data
```

## Requirements

- Python 3.10+
- See `requirements.txt` for the full list, notably: Flask, pandas, numpy, statsmodels, scikit-learn, scipy, smartapi-python, pyotp, gunicorn.

## Setup

1. **Clone and install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment variables** (only needed for the live "Angel One" data source — CSV upload works without this)

   Create a `.env` file in the project root:

   ```
   API_KEY=your_angel_one_api_key
   CLIENT_ID=your_client_id
   PASSWORD=your_password
   TOTP_SECRET=your_totp_secret
   ```

   > ⚠️ **Security note:** the repository as uploaded includes a `.env` file with sample credentials committed to it. Treat any real API key/password/TOTP secret as compromised, rotate them, and add `.env` to `.gitignore` before pushing this project anywhere public.

3. **Run the app locally**

   ```bash
   python app.py
   ```

   The dashboard will be available at `http://localhost:5000`.

## Usage

### Web dashboard

Open the app in a browser, upload a CSV (or select the Angel One live-data source), choose a model (`ARIMA`, `SARIMA`, `Ridge`, `ETS`, or `AUTO`), set an anomaly threshold, and submit to view forecasts, anomalies, signals, and strategy performance.

### API endpoints

**`POST /predict`** — single-asset forecast

Form fields:
| Field | Description |
|---|---|
| `data_source` | `"file"` or `"angel"` |
| `file` | CSV with a `value` column (required if `data_source=file`) |
| `symbol_token` | Angel One instrument token (required if `data_source=angel`) |
| `model_type` | `ARIMA`, `SARIMA`, `Ridge`, `ETS`, or `AUTO` |
| `threshold` | Anomaly detection sensitivity multiplier (default `1.5`) |

Returns JSON with `forecast`, `lower_ci`/`upper_ci`, `selected_model`, diagnostics (`aic`, `residual_mean`, `ljung_box_pvalue`), `signal`, `market_regime`, backtest metrics, strategy performance, anomaly points, and an `equity_curve`.

**`POST /predict_multi_csv`** — multi-asset portfolio analysis

Form fields:
| Field | Description |
|---|---|
| `file` | CSV with a date column followed by one column per asset |

Returns JSON with per-asset weights/returns/Sharpe/drawdown, portfolio-level metrics, and historical + forecast equity curves.

### Sample data

- `data/sample_data.csv` — `date,value` format for single-asset endpoints
- `data/multi_test.csv` — `Date,TICKER1,TICKER2,...` format for the portfolio endpoint
- `test.csv` — minimal `value`-only CSV

### Standalone training script

`model/train.py` is a self-contained script (run from inside `model/`) that fits an ARIMA(2,1,2) model on `data/sample_data.csv`, prints MAE/RMSE, plots actual vs. predicted, forecasts 7 days ahead, and saves the fitted model to `arima_model.pkl`. It requires `matplotlib` and `joblib`, which are not in `requirements.txt`.

## Status

This project currently runs locally only and has **not been deployed**. A `render.yaml` and `Procfile` are included in case you want to deploy it to [Render](https://render.com) (or another gunicorn-based host) later:

- `render.yaml` defines a Python web service that installs `requirements.txt` and starts the app with gunicorn.
- `Procfile` provides the same start command for other gunicorn-based platforms:
  ```
  web: gunicorn app:app --timeout 120 --workers 1 --bind 0.0.0.0:$PORT
  ```
- When you do deploy, set the Angel One credentials (`API_KEY`, `CLIENT_ID`, `PASSWORD`, `TOTP_SECRET`) as environment variables on the host rather than committing them in `.env`.
