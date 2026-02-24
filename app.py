from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import math

from model.anomaly.detect import detect_anomalies
from portfolio.portfolio_engine import run_portfolio_from_csv
from model.backtest.backtest import rolling_backtest
from model.forecasting.arima_model import run_arima
from model.forecasting.sarima_model import run_sarima
from model.forecasting.auto_selector import select_best_model
#from services.angel_service import get_angel_data

app = Flask(__name__)


# --------------------------------------------------
# Utility: Safe JSON Number Conversion
# --------------------------------------------------
def safe(x):
    if x is None:
        return 0.0
    if isinstance(x, (float, np.floating)):
        if math.isnan(x) or math.isinf(x):
            return 0.0
    return float(x)


@app.route("/")
def home():
    return render_template("index.html")


# ======================================================
# 🔹 MULTI ASSET (CSV)
# ======================================================
@app.route("/predict_multi_csv", methods=["POST"])
def predict_multi_csv():
    try:
        if "file" not in request.files:
            return jsonify({"error": "CSV required"}), 400

        file = request.files["file"]
        results = run_portfolio_from_csv(file)

        if not isinstance(results, dict):
            return jsonify({"error": "Invalid portfolio result format"}), 500

        return jsonify(results)

    except Exception as e:
        print("MULTI ERROR:", e)
        return jsonify({"error": str(e)}), 500


# ======================================================
# 🔹 SINGLE ASSET FORECAST
# ======================================================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data_source = request.form.get("data_source")
        model_type = request.form.get("model_type")
        threshold = request.form.get("threshold")

        if threshold is None:
            return jsonify({"error": "Threshold value required"}), 400

        threshold = float(threshold)

        # ---------------------------
        # 1️⃣ LOAD DATA
        # ---------------------------
        if data_source == "file":

            if "file" not in request.files:
                return jsonify({"error": "CSV file required"}), 400

            file = request.files["file"]
            df = pd.read_csv(file)

            if "value" not in df.columns:
                return jsonify({"error": "CSV must contain 'value' column"}), 400

            series = pd.to_numeric(df["value"], errors="coerce").dropna()
            df = pd.DataFrame({"value": series})

        #elif data_source == "angel":

            #symbol_token = request.form.get("symbol_token")

            #if not symbol_token:
                #return jsonify({"error": "Symbol token required"}), 400

            #series = get_angel_data(symbol_token)

            if series is None or len(series) == 0:
                return jsonify({"error": "No data returned from Angel API"}), 400

            series = pd.Series(series)
            df = pd.DataFrame({"value": series})

        else:
            return jsonify({"error": "Invalid data source"}), 400

        # ---------------------------
        # 2️⃣ MODEL SELECTION
        # ---------------------------
        if model_type == "ARIMA":
            forecast, lower, upper, aic, residual_mean, lb_pvalue = run_arima(series)
            selected_model = "ARIMA"

        elif model_type == "SARIMA":
            forecast, lower, upper, aic, residual_mean, lb_pvalue = run_sarima(series)
            selected_model = "SARIMA"

        elif model_type == "AUTO":

            if len(series) < 20:
                forecast, lower, upper, aic, residual_mean, lb_pvalue = run_arima(series)
                selected_model = "ARIMA (Insufficient data)"
            else:
                best_model, best_data = select_best_model(series)

                forecast = best_data["forecast"]
                lower = best_data["lower"]
                upper = best_data["upper"]
                aic = best_data["aic"]
                residual_mean = best_data["residual_mean"]
                lb_pvalue = best_data["lb_pvalue"]

                selected_model = best_model + " (Composite Score)"

        else:
            return jsonify({"error": "Invalid model type"}), 400

        # ---------------------------
        # 3️⃣ BACKTEST
        # ---------------------------
        if selected_model.startswith("ARIMA"):
            bt_rmse, bt_mape, bt_dir, rolling_preds = rolling_backtest(series, run_arima)
        else:
            bt_rmse, bt_mape, bt_dir, rolling_preds = rolling_backtest(series, run_sarima)

        # ---------------------------
        # 4️⃣ STRATEGY SIMULATION
        # ---------------------------
        returns = []
        equity = []

        split = int(len(series) * 0.8)
        transaction_cost = 0.001
        starting_capital = 100000
        capital = starting_capital
        risk_per_trade = 0.02

        equity.append(capital)

        if rolling_preds is not None:
            max_index = min(len(rolling_preds), len(series) - split - 1)

            for i in range(max_index):

                current_price = series.iloc[split + i]
                next_price = series.iloc[split + i + 1]
                predicted_price = rolling_preds[i]

                if predicted_price > current_price:
                    r = ((next_price - current_price) / current_price) - transaction_cost
                elif predicted_price < current_price:
                    r = ((current_price - next_price) / current_price) - transaction_cost
                else:
                    r = 0

                position_size = capital * risk_per_trade
                pnl = position_size * r

                capital += pnl
                equity.append(capital)
                returns.append(r)

        equity = np.array(equity)
        returns_array = np.array(returns)

        strategy_total_return = ((capital - starting_capital) / starting_capital) * 100

        # Annualized Return
        periods = len(returns_array)
        if periods > 0:
            strategy_annualized_return = (
                (capital / starting_capital) ** (252 / periods) - 1
            ) * 100
        else:
            strategy_annualized_return = 0.0

        # Sharpe
        if len(returns_array) > 1 and np.std(returns_array) != 0:
            sharpe_ratio = (np.mean(returns_array) / np.std(returns_array)) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0

        strategy_win_rate = np.mean(returns_array > 0) * 100 if len(returns_array) > 0 else 0.0

        strategy_max_drawdown = (
            np.min(equity / np.maximum.accumulate(equity) - 1) * 100
            if len(equity) > 1 else 0.0
        )

        buy_hold_return = ((series.iloc[-1] - series.iloc[split]) / series.iloc[split]) * 100

        # ---------------------------
        # 5️⃣ SIGNAL ENGINE
        # ---------------------------
        current_price = series.iloc[-1]
        forecast_mean = np.mean(forecast[:3])

        lower_first = lower.iloc[0]
        upper_first = upper.iloc[0]

        expected_return = (forecast_mean - current_price) / current_price
        volatility_regime = series.pct_change().rolling(20).std().iloc[-1]
        if volatility_regime > 0.03:
            regime = "HIGH VOLATILITY"
        elif volatility_regime < 0.01:
            regime = "LOW VOLATILITY"
        else:
            regime : "NORMAL"
        
        ci_width = upper_first - lower_first
        confidence_score = 1 - (ci_width / current_price)
        confidence_score = max(0, min(confidence_score, 1))

        signal = "HOLD"

        if confidence_score > 0.4:
            if expected_return > 0 and lower_first > current_price:
                signal = "BUY"
            elif expected_return < 0 and upper_first < current_price:
                signal = "SELL"

        # ---------------------------
        # 6️⃣ ANOMALY
        # ---------------------------
        predictions, anomalies, mae, rmse = detect_anomalies(df, threshold)

        anomaly_points = [
            df["value"].iloc[i] if anomalies[i] else None
            for i in range(len(anomalies))
        ]

        # ---------------------------
        # 7️⃣ RESPONSE
        # ---------------------------
        return jsonify({
            "actual": [safe(x) for x in df["value"]],
            "predicted": [safe(x) for x in predictions],
            "forecast": [safe(x) for x in forecast],
            "lower_ci": [safe(x) for x in lower],
            "upper_ci": [safe(x) for x in upper],
            "selected_model": selected_model,
            "aic": round(safe(aic), 3),
            "residual_mean": round(safe(residual_mean), 3),
            "ljung_box_pvalue": round(safe(lb_pvalue), 3),
            "signal": signal,
            "market regime": regime,
            "expected_return": round(safe(expected_return * 100), 3),
            "confidence_score": round(safe(confidence_score), 3),
            "anomaly_points": [safe(x) if x is not None else None for x in anomaly_points],
            "anomaly_count": int(sum(anomalies)),
            "mae": round(safe(mae), 3),
            "rmse": round(safe(rmse), 3),
            "backtest_rmse": round(safe(bt_rmse), 3) if bt_rmse else None,
            "backtest_mape": round(safe(bt_mape), 3) if bt_mape else None,
            "backtest_direction_accuracy": round(safe(bt_dir), 3) if bt_dir else None,
            "strategy_total_return": round(strategy_total_return, 3),
            "strategy_annualized_return": round(strategy_annualized_return, 3),
            "strategy_sharpe_ratio": round(sharpe_ratio, 3),
            "strategy_win_rate": round(strategy_win_rate, 3),
            "strategy_max_drawdown": round(strategy_max_drawdown, 3),
            "buy_hold_return": round(buy_hold_return, 3),
            "equity_curve": [round(float(x), 3) for x in equity]
        })

    except Exception as e:
        print("SINGLE ERROR:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)