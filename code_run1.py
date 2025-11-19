# train_models.py
import pandas as pd
import numpy as np
import os, pickle, joblib, warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from xgboost import XGBRegressor

# --- Optional: LSTM (if TensorFlow installed) ---
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    TENSORFLOW_AVAILABLE = True
except Exception as e:
    print("⚠️ TensorFlow not available:", e)
    TENSORFLOW_AVAILABLE = False


# ---------- CONSTANTS ----------
POLLUTANTS = ['PM2.5', 'PM10', 'NO2', 'O3', 'SO2']


# ---------- AQI CALCULATION ----------
def calculate_aqi(row):
    # Simplified Indian AQI scale
    breakpoints = {
        'PM2.5': [(0,30,0,50),(31,60,51,100),(61,90,101,200),(91,120,201,300),(121,250,301,400),(251,500,401,500)],
        'PM10':  [(0,50,0,50),(51,100,51,100),(101,250,101,200),(251,350,201,300),(351,430,301,400),(431,600,401,500)],
        'NO2':   [(0,40,0,50),(41,80,51,100),(81,180,101,200),(181,280,201,300),(281,400,301,400),(401,1000,401,500)],
        'O3':    [(0,50,0,50),(51,100,51,100),(101,168,101,200),(169,208,201,300),(209,748,301,400),(749,1000,401,500)],
        'SO2':   [(0,40,0,50),(41,80,51,100),(81,380,101,200),(381,800,201,300),(801,1600,301,400),(1601,2000,401,500)]
    }

    sub_indices = []
    for p in POLLUTANTS:
        if pd.notna(row.get(p, np.nan)):
            for bp in breakpoints[p]:
                if bp[0] <= row[p] <= bp[1]:
                    aqi = ((bp[3]-bp[2])/(bp[1]-bp[0]))*(row[p]-bp[0])+bp[2]
                    sub_indices.append(aqi)
                    break
    return max(sub_indices) if sub_indices else np.nan


def aqi_category(aqi):
    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Moderate"
    elif aqi <= 200: return "Unhealthy for Sensitive"
    elif aqi <= 300: return "Unhealthy"
    elif aqi <= 400: return "Very Unhealthy"
    else: return "Hazardous"


# ---------- EVALUATION ----------
def evaluate_preds(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return rmse, mae 


# ---------- MODEL TRAINING FUNCTIONS ----------
def train_arima(ts_train, ts_test):
    model = ARIMA(ts_train, order=(1,1,1)).fit()
    preds = model.get_forecast(steps=len(ts_test)).predicted_mean
    preds.index = ts_test.index
    return preds


def train_prophet(ts_train, ts_test):
    df_train = ts_train.reset_index()
    df_train.columns = ['ds', 'y']
    m = Prophet(daily_seasonality=True)
    m.fit(df_train)
    future = pd.DataFrame({'ds': ts_test.index})
    forecast = m.predict(future)
    preds = forecast['yhat'].values
    return pd.Series(preds, index=ts_test.index)


def train_xgb(ts_train, ts_test):
    X_train = np.arange(len(ts_train)).reshape(-1,1)
    X_test = np.arange(len(ts_train), len(ts_train)+len(ts_test)).reshape(-1,1)
    model = XGBRegressor(objective="reg:squarederror")
    model.fit(X_train, ts_train)
    preds = model.predict(X_test)
    return pd.Series(preds, index=ts_test.index)


def train_lstm(ts_train, ts_test):
    look_back = 3
    trainX, trainY = [], []
    for i in range(len(ts_train)-look_back):
        trainX.append(ts_train.values[i:i+look_back])
        trainY.append(ts_train.values[i+look_back])
    trainX, trainY = np.array(trainX), np.array(trainY)
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, input_shape=(look_back,1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(trainX, trainY, epochs=15, batch_size=8, verbose=0)

    testX = []
    full_data = np.concatenate([ts_train.values, ts_test.values])
    for i in range(len(ts_train), len(full_data)-look_back):
        testX.append(full_data[i-look_back:i])
    testX = np.reshape(np.array(testX), (len(testX), look_back, 1))
    preds = model.predict(testX, verbose=0).flatten()
    preds = preds[:len(ts_test)]
    return pd.Series(preds, index=ts_test.index)


# ---------- MAIN PIPELINE ----------
def main():
    df = pd.read_csv("air_quality_data.csv")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")

    pollutants = [p for p in POLLUTANTS if p in df.columns]
    cities = df["City"].unique()

    os.makedirs("models", exist_ok=True)
    results, best_models, forecasts = [], [], []

    for city in cities:
        for pollutant in pollutants:
            ts = df[df["City"] == city][["Date", pollutant]].set_index("Date").dropna()
            if len(ts) < 30:
                continue

            ts_train, ts_test = ts.iloc[:-7, 0], ts.iloc[-7:, 0]

            model_funcs = {
                "ARIMA": train_arima,
                "Prophet": train_prophet,
                "XGBoost": train_xgb
            }
            if TENSORFLOW_AVAILABLE:
                model_funcs["LSTM"] = train_lstm

            model_preds = {}
            for name, func in model_funcs.items():
                try:
                    preds = func(ts_train, ts_test)
                    rmse, mae = evaluate_preds(ts_test, preds)
                    model_preds[name] = (rmse, mae, preds)
                    results.append([city, pollutant, name, rmse, mae])
                except Exception as e:
                    print(f"{name} failed for {city}-{pollutant}: {e}")

            if model_preds:
                best_model = min(model_preds.items(), key=lambda x: x[1][0])
                bm_name, (bm_rmse, bm_mae, bm_preds) = best_model
                best_models.append([city, pollutant, bm_name, bm_rmse])

                model_path = f"models/{city}_{pollutant}_{bm_name}.pkl"
                with open(model_path, "wb") as f:
                    pickle.dump(best_model, f)

                forecast_df = pd.DataFrame({
                    "Date": ts_test.index,
                    "City": city,
                    "Pollutant": pollutant,
                    "Forecast": bm_preds
                })
                forecasts.append(forecast_df)

    # ---------- FIXED SECTION ----------
    forecast_all = pd.concat(forecasts).reset_index(drop=True)

    # Ensure 'Date' not both index & column
    if forecast_all.index.name == 'Date' or 'Date' in forecast_all.index.names:
        forecast_all = forecast_all.reset_index()

    pivot = forecast_all.pivot_table(
        index=["Date", "City"],
        columns="Pollutant",
        values="Forecast"
    ).reset_index()

    # ---------- AQI CALCULATION ----------
    pivot["AQI"] = pivot.apply(calculate_aqi, axis=1)
    pivot["AQI_Category"] = pivot["AQI"].apply(aqi_category)

    # ---------- SAVE RESULTS ----------
    pivot.to_csv("forecast_aqi.csv", index=False)
    pd.DataFrame(results, columns=["City", "Pollutant", "Model", "RMSE", "MAE"]).to_csv("results_all.csv", index=False)
    pd.DataFrame(best_models, columns=["City", "Pollutant", "Best_Model", "RMSE"]).to_csv("best_models.csv", index=False)

    print("✅ Training complete! Forecast saved to forecast_aqi.csv")

    # Show quick preview
    print("\n🔹 Preview of forecast_aqi.csv:")
    print(pivot.head())


if __name__ == "__main__":
    main()
