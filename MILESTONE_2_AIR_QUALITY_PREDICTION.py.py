# train_and_save_models.py
"""
Training script for air quality forecasting.

Features:
- Trains ARIMA, Prophet (if installed), XGBoost, and LSTM (only if TensorFlow + GPU available).
- Uses the final 24 hourly rows of each city-pollutant series as evaluation period (Option 2).
- Computes 24-hour RMSE & MAE per-model and saves:
    - results_all.csv
    - best_models.csv
    - horizon_24h_accuracy.csv
    - forecast_aqi.csv
- Saves model artifacts under ./models/ where possible.
- Robust: model-level try/except so one failing model does not stop the pipeline.
"""

import os
import pickle
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

# Prophet: optional (it sometimes triggers plotly warnings). Handle gracefully.
try:
    from prophet import Prophet  # requires 'prophet' package (formerly fbprophet)
    PROPHET_AVAILABLE = True
except Exception as e:
    print("⚠️ Prophet not available or failed import — will skip Prophet. Error:", e)
    PROPHET_AVAILABLE = False

from xgboost import XGBRegressor

# ---- Optional LSTM (Option-B: Auto-disable if GPU not available) ----
TENSORFLOW_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense

    # Disable LSTM if no GPU found
    if len(tf.config.list_physical_devices("GPU")) == 0:
        print("⚠️ No GPU found — LSTM disabled for fast training.")
        TENSORFLOW_AVAILABLE = False
    else:
        print("✅ GPU found — LSTM enabled.")
        TENSORFLOW_AVAILABLE = True

except Exception as e:
    # TensorFlow not installed or some import error; LSTM disabled
    print("⚠️ TensorFlow not available — LSTM disabled. Error:", e)
    TENSORFLOW_AVAILABLE = False

POLLUTANTS = ['PM2.5', 'PM10', 'NO2', 'O3', 'SO2', 'CO']


# ---------- AQI Calculation ----------
def calculate_aqi(row):
    breakpoints = {
        'PM2.5': [(0,30,0,50),(31,60,51,100),(61,90,101,200),
                  (91,120,201,300),(121,250,301,400),(251,500,401,500)],
        'PM10':  [(0,50,0,50),(51,100,51,100),(101,250,101,200),
                  (251,350,201,300),(351,430,301,400),(431,600,401,500)],
        'NO2':   [(0,40,0,50),(41,80,51,100),(81,180,101,200),
                  (181,280,201,300),(281,400,301,400),(401,1000,401,500)],
        'O3':    [(0,50,0,50),(51,100,51,100),(101,168,101,200),
                  (169,208,201,300),(209,748,301,400),(749,1000,401,500)],
        'SO2':   [(0,40,0,50),(41,80,51,100),(81,380,101,200),
                  (381,800,201,300),(801,1600,301,400),(1601,2000,401,500)],
        'CO':    [(0,1,0,50),(1.1,2,51,100),(2.1,10,101,200),
                  (10.1,17,201,300),(17.1,34,301,400),(34.1,50,401,500)]
    }

    sub_indices = []
    for p in POLLUTANTS:
        if p in row and pd.notna(row.get(p, np.nan)):
            val = row[p]
            for bp in breakpoints.get(p, []):
                if bp[0] <= val <= bp[1]:
                    aqi = ((bp[3] - bp[2]) / (bp[1] - bp[0])) * (val - bp[0]) + bp[2]
                    sub_indices.append(aqi)
                    break
    return max(sub_indices) if sub_indices else np.nan


def aqi_category(aqi):
    if pd.isna(aqi):
        return np.nan
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 200:
        return "Unhealthy for Sensitive"
    elif aqi <= 300:
        return "Unhealthy"
    elif aqi <= 400:
        return "Very Unhealthy"
    else:
        return "Hazardous"


# ---------- Evaluation ----------
def evaluate_preds(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return rmse, mae


# ---------- Models ----------
def train_arima(ts_train, forecast_index):
    # safer ARIMA settings
    model = ARIMA(ts_train, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False).fit()
    preds = model.get_forecast(steps=len(forecast_index)).predicted_mean
    preds = pd.Series(preds.values, index=forecast_index)
    return preds, model


def train_prophet(ts_train, forecast_index):
    df_train = ts_train.reset_index()
    df_train.columns = ["ds", "y"]
    m = Prophet(daily_seasonality=True, weekly_seasonality=False, yearly_seasonality=False)
    m.fit(df_train)
    future = pd.DataFrame({"ds": forecast_index})
    forecast = m.predict(future)
    preds = pd.Series(forecast["yhat"].values, index=forecast_index)
    return preds, m


def train_xgb(ts_train, forecast_index):
    X_train = np.arange(len(ts_train)).reshape(-1, 1)
    X_test = np.arange(len(ts_train), len(ts_train) + len(forecast_index)).reshape(-1, 1)
    model = XGBRegressor(objective="reg:squarederror", verbosity=0, n_jobs=1)
    model.fit(X_train, ts_train.values)
    preds = pd.Series(model.predict(X_test), index=forecast_index)
    return preds, model


def train_lstm(ts_train, forecast_index):
    # LSTM only called when TENSORFLOW_AVAILABLE == True and GPU exists
    look_back = 24 if len(ts_train) >= 24 else max(3, len(ts_train) // 4)
    values = ts_train.values
    if len(values) <= look_back:
        raise ValueError(f"Not enough data for LSTM with look_back={look_back}")

    trainX, trainY = [], []
    for i in range(len(values) - look_back):
        trainX.append(values[i:i + look_back])
        trainY.append(values[i + look_back])

    trainX = np.array(trainX).reshape(-1, look_back, 1)
    trainY = np.array(trainY)

    model = Sequential()
    model.add(LSTM(50, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(loss="mse", optimizer="adam")

    # Smaller epochs because this will run on GPU only; adjust if needed
    model.fit(trainX, trainY, epochs=10, batch_size=8, verbose=0)

    preds = []
    last_window = values[-look_back:].copy()
    for _ in range(len(forecast_index)):
        x_input = last_window.reshape(1, look_back, 1)
        yhat = model.predict(x_input, verbose=0)[0][0]
        preds.append(yhat)
        last_window = np.append(last_window[1:], yhat)

    preds = pd.Series(preds, index=forecast_index)
    return preds, model


# -------------------------------- MAIN --------------------------------
def main():
    INPUT_FILE = "cleaned_air_quality_hourly.csv"
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Place cleaned hourly data as {INPUT_FILE} in the working directory.")

    df = pd.read_csv(INPUT_FILE)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")

    pollutants = [p for p in POLLUTANTS if p in df.columns]
    cities = df["City"].unique()

    os.makedirs("models", exist_ok=True)

    results = []              # summary RMSE/MAE (24h)
    best_models = []          # best per city-pollutant
    forecasts = []            # forecast rows for all city/pollutant (per model)
    horizon_accuracy = []     # per-model 24h RMSE/MAE (city,pollutant,model,rmse,mae)

    MIN_LEN = 50  # require at least this many hourly rows to operate (adjustable)

    for city in cities:
        print(f"\n--- Processing city: {city} ---")
        for pollutant in pollutants:
            try:
                ts_df = df[df["City"] == city][["Date", pollutant]].dropna().set_index("Date").sort_index()
                if len(ts_df) < MIN_LEN:
                    print(f"Skip {city}-{pollutant}: too few rows ({len(ts_df)})")
                    continue

                eval_h = 24
                if len(ts_df) <= eval_h:
                    print(f"Skip {city}-{pollutant}: not enough rows for 24h eval ({len(ts_df)})")
                    continue

                ts_train = ts_df.iloc[:-eval_h, 0]  # training series
                ts_eval = ts_df.iloc[-eval_h:, 0]   # final 24 hours (true values)
                forecast_index = ts_eval.index

                # Build model function map dynamically depending on availability
                model_funcs = {
                    "ARIMA": train_arima,
                    "XGBoost": train_xgb
                }
                if PROPHET_AVAILABLE:
                    model_funcs["Prophet"] = train_prophet
                else:
                    print(" - Prophet skipped (not available)")

                if TENSORFLOW_AVAILABLE:
                    model_funcs["LSTM"] = train_lstm
                else:
                    # LSTM intentionally disabled for CPU-only systems
                    pass

                model_preds = {}

                for name, func in model_funcs.items():
                    try:
                        print(f" Training {name} for {city}-{pollutant} ...", end="", flush=True)
                        preds, model_obj = func(ts_train, forecast_index)

                        # ensure length matches
                        if len(preds) != len(ts_eval):
                            preds = preds.reindex(forecast_index).fillna(method='ffill').fillna(method='bfill')

                        rmse, mae = evaluate_preds(ts_eval.values, preds.values)
                        model_preds[name] = (rmse, mae, preds, model_obj)

                        # record horizon accuracy
                        horizon_accuracy.append([city, pollutant, name, eval_h, rmse, mae])

                        # save model artifact where possible
                        model_path = f"models/{city}_{pollutant}_{name}"
                        try:
                            if name == "LSTM" and TENSORFLOW_AVAILABLE:
                                model_obj.save(model_path + ".h5")
                            elif name == "XGBoost":
                                model_obj.save_model(model_path + ".json")
                            else:
                                # For ARIMA and Prophet (and others) use pickle
                                with open(model_path + ".pkl", "wb") as f:
                                    pickle.dump(model_obj, f)
                        except Exception as e:
                            print(f"\n  ⚠️ Could not save {name} model for {city}-{pollutant}: {e}")

                        # append forecast rows for later aggregation/inspection
                        fdf = pd.DataFrame({
                            "Date": forecast_index,
                            "City": city,
                            "Pollutant": pollutant,
                            "Model": name,
                            "Forecast": preds.values
                        })
                        forecasts.append(fdf)

                        # append compact results
                        results.append([city, pollutant, name, rmse, mae])

                        print(" done. (RMSE={:.3f}, MAE={:.3f})".format(rmse, mae))

                    except Exception as me:
                        print(f"\n  ✖ {name} failed for {city}-{pollutant}: {me}")
                        # continue with next model

                # choose best model by RMSE (if any succeeded)
                if model_preds:
                    best_name, (bm_rmse, bm_mae, bm_preds, bm_model) = min(
                        model_preds.items(), key=lambda x: x[1][0]
                    )
                    best_models.append([city, pollutant, best_name, bm_rmse])
                    print(f" Selected best model for {city}-{pollutant}: {best_name} (RMSE={bm_rmse:.3f})")
                else:
                    print(f" No successful models for {city}-{pollutant}.")

            except Exception as outer_e:
                print(f"✖ Error processing {city}-{pollutant}: {outer_e}")
                # continue to next pollutant

    # ---- write outputs ----
    if results:
        pd.DataFrame(results, columns=["City", "Pollutant", "Model", "RMSE", "MAE"]).to_csv("results_all.csv", index=False)
    else:
        pd.DataFrame([], columns=["City", "Pollutant", "Model", "RMSE", "MAE"]).to_csv("results_all.csv", index=False)

    if best_models:
        pd.DataFrame(best_models, columns=["City", "Pollutant", "Best_Model", "RMSE"]).to_csv("best_models.csv", index=False)
    else:
        pd.DataFrame([], columns=["City", "Pollutant", "Best_Model", "RMSE"]).to_csv("best_models.csv", index=False)

    if horizon_accuracy:
        pd.DataFrame(horizon_accuracy, columns=["City", "Pollutant", "Model", "Horizon_hours", "RMSE", "MAE"]).to_csv("horizon_24h_accuracy.csv", index=False)
    else:
        pd.DataFrame([], columns=["City", "Pollutant", "Model", "Horizon_hours", "RMSE", "MAE"]).to_csv("horizon_24h_accuracy.csv", index=False)

    # ---- combine forecasts and compute AQI per forecast row (using best-model forecasts only) ----
    if forecasts:
        forecast_all = pd.concat(forecasts).reset_index(drop=True)

        # Keep only best model forecasts (optional). We'll keep the best per city-pollutant if available
        df_best = pd.read_csv("best_models.csv") if os.path.exists("best_models.csv") else pd.DataFrame()
        if not df_best.empty:
            merged = forecast_all.merge(df_best, left_on=["City", "Pollutant", "Model"], right_on=["City", "Pollutant", "Best_Model"])
            merged = merged[["Date", "City", "Pollutant", "Forecast"]]
        else:
            merged = forecast_all[["Date", "City", "Pollutant", "Forecast"]]

        pivot = merged.pivot_table(index=["Date", "City"], columns="Pollutant", values="Forecast").reset_index()

        # ensure pollutant columns exist
        for p in POLLUTANTS:
            if p not in pivot.columns:
                pivot[p] = np.nan

        pivot["AQI"] = pivot.apply(calculate_aqi, axis=1)
        pivot["AQI_Category"] = pivot["AQI"].apply(aqi_category)

        pivot.to_csv("forecast_aqi.csv", index=False)
    else:
        pd.DataFrame([], columns=["Date", "City"] + POLLUTANTS + ["AQI", "AQI_Category"]).to_csv("forecast_aqi.csv", index=False)

    print("\n✅ Training and evaluation complete.")
    print("Saved files (if applicable):")
    print(" - results_all.csv")
    print(" - best_models.csv")
    print(" - horizon_24h_accuracy.csv")
    print(" - forecast_aqi.csv")
    print("Model artifacts (where saved) are in ./models/")

if __name__ == "__main__":
    main()
