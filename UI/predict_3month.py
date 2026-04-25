import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from datetime import datetime
import os

# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

# ════════════════════════════════════════════════════════
# LOAD MODEL ARTIFACTS
# ════════════════════════════════════════════════════════
def load_artifacts():
    model_path = os.path.join(MODELS_DIR, "lgbm_final.txt")
    le_target_path = os.path.join(MODELS_DIR, "le_target.pkl")
    feature_cols_path = os.path.join(MODELS_DIR, "feature_cols.csv")
    cat_mappings_path = os.path.join(MODELS_DIR, "cat_mappings.pkl")
    data_path = os.path.join(DATA_DIR, "master_dataset.csv")
    
    model        = lgb.Booster(model_file=model_path)
    le_target    = joblib.load(le_target_path)
    feature_cols = pd.read_csv(feature_cols_path)["feature"].tolist()
    cat_mappings = joblib.load(cat_mappings_path)
    df_hist      = pd.read_csv(data_path, parse_dates=["date"])
    df_hist      = df_hist.sort_values(["state", "vegetable", "date"])
    return model, le_target, feature_cols, \
           cat_mappings, df_hist


# ════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════
def get_season(month):
    if month in [6, 7, 8, 9, 10]:    return "Kharif"
    elif month in [11, 12, 1, 2, 3]: return "Rabi"
    else:                             return "Zaid"

def get_month_name(month_num):
    return [
        "January", "February", "March", "April",
        "May", "June", "July", "August", "September",
        "October", "November", "December"
    ][month_num - 1]

def estimate_price_range(modal_price, direction):
    """
    Estimate expected price range based on direction
    """
    if direction == "UP":
        low  = modal_price * 1.05
        high = modal_price * 1.20
    elif direction == "DOWN":
        low  = modal_price * 0.80
        high = modal_price * 0.95
    else:
        low  = modal_price * 0.95
        high = modal_price * 1.05

    return round(low, 1), round(high, 1)


# ════════════════════════════════════════════════════════
# FEATURE BUILDER
# ════════════════════════════════════════════════════════
def build_features(state, veg, modal_price,
                   arrival_qty, min_price, max_price,
                   weather, predict_month,
                   df_hist, feature_cols, cat_mappings):
    """
    Build all 38 features for one prediction
    """
    hist = df_hist[
        (df_hist["state"]     == state) &
        (df_hist["vegetable"] == veg)
    ].sort_values("date")

    if len(hist) < 5:
        return None

    # Lag features from history
    price_lag_1m   = hist["modal_price"].iloc[-1]
    price_lag_4m   = hist["modal_price"].iloc[-4]
    rolling_avg_3m = hist["modal_price"].iloc[-3:].mean()
    arrival_lag_1m = hist["arrival_qty"].iloc[-1]

    # Weather
    temp_max     = weather["temp_max"]
    temp_min     = weather["temp_min"]
    rainfall_mm  = weather["rainfall_mm"]
    humidity     = weather["humidity"]

    # Rainfall deviation
    hist_rain = df_hist[
        (df_hist["state"] == state) &
        (df_hist["month"] == predict_month)
    ]["rainfall_mm"].mean()
    if pd.isna(hist_rain):
        hist_rain = rainfall_mm
    rainfall_deviation = rainfall_mm - hist_rain

    # Normalize
    price_mean  = hist["modal_price"].mean()
    price_std   = hist["modal_price"].std() + 1e-8
    arr_mean    = hist["arrival_qty"].mean()
    arr_std     = hist["arrival_qty"].std()   + 1e-8

    price_norm   = (modal_price  - price_mean) / price_std
    arrival_norm = (arrival_qty  - arr_mean)   / arr_std

    # Ratios
    lag1_ratio    = modal_price  / (price_lag_1m   + 1e-8)
    lag4_ratio    = modal_price  / (price_lag_4m   + 1e-8)
    arrival_ratio = arrival_qty  / (arrival_lag_1m + 1e-8)

    # Momentum
    price_momentum   = (modal_price - price_lag_1m) / \
                       (price_lag_1m + 1e-8) * 100
    price_vs_avg     =  modal_price - rolling_avg_3m
    arrival_momentum = (arrival_qty - arrival_lag_1m) / \
                       (arrival_lag_1m + 1e-8) * 100

    # Price spread
    price_spread   = max_price - min_price
    price_position = (modal_price - min_price) / \
                     (max_price - min_price + 1e-8)

    price_volatility_3m = hist["modal_price"].iloc[
        -4:-1].std()

    price_change_abs     = abs(modal_price - price_lag_1m)
    price_change_abs_pct = price_change_abs / \
                           (modal_price + 1e-8) * 100

    month_norm_in_veg = (predict_month - 6.5) / 3.5

    # Season
    season     = get_season(predict_month)
    season_veg = f"{season}_{veg}"

    # Producer state values
    producer_map = {
        "Tomato" : "Karnataka",
        "Onion"  : "Maharashtra",
        "Potato" : "Uttar Pradesh"
    }

    prod_prices = {}
    for v, prod_state in producer_map.items():
        prod_hist = df_hist[
            (df_hist["vegetable"] == v) &
            (df_hist["state"]     == prod_state)
        ].sort_values("date")

        if len(prod_hist) >= 2:
            prod_prices[v] = {
                "price"    : prod_hist[
                    "modal_price"].iloc[-1],
                "price_lag": prod_hist[
                    "modal_price"].iloc[-2],
                "arrival"  : prod_hist[
                    "arrival_qty"].iloc[-1],
            }
        else:
            prod_prices[v] = {
                "price"    : modal_price,
                "price_lag": price_lag_1m,
                "arrival"  : arrival_qty,
            }

    # Don't encode categoricals yet - store as strings
    # We'll convert to pd.Categorical AFTER creating the DataFrame
    feat = {
        "state"               : state,
        "vegetable"           : veg,
        "season"              : season,
        "price_norm"          : price_norm,
        "arrival_norm"        : arrival_norm,
        "lag1_ratio"          : lag1_ratio,
        "lag4_ratio"          : lag4_ratio,
        "arrival_ratio"       : arrival_ratio,
        "price_momentum"      : price_momentum,
        "price_vs_avg"        : price_vs_avg,
        "arrival_momentum"    : arrival_momentum,
        "modal_price"         : modal_price,
        "price_lag_1m"        : price_lag_1m,
        "price_lag_4m"        : price_lag_4m,
        "rolling_avg_3m"      : rolling_avg_3m,
        "price_spread"        : price_spread,
        "price_position"      : price_position,
        "arrival_qty"         : arrival_qty,
        "arrival_lag_1m"      : arrival_lag_1m,
        "temp_max"            : temp_max,
        "temp_min"            : temp_min,
        "rainfall_mm"         : rainfall_mm,
        "humidity"            : humidity,
        "rainfall_deviation"  : rainfall_deviation,
        "month_sin"           : np.sin(
            2 * np.pi * predict_month / 12),
        "month_cos"           : np.cos(
            2 * np.pi * predict_month / 12),
        "month_norm_in_veg"   : month_norm_in_veg,
        "is_post_monsoon"     : int(
            predict_month in [9, 10, 11]),
        "season_veg"          : season_veg,
        "price_volatility_3m" : price_volatility_3m,
        "prod_price_Tomato"   : prod_prices[
            "Tomato"]["price"],
        "prod_price_Tomato_lag1": prod_prices[
            "Tomato"]["price_lag"],
        "prod_arrival_Tomato" : prod_prices[
            "Tomato"]["arrival"],
        "prod_price_Onion"    : prod_prices[
            "Onion"]["price"],
        "prod_price_Onion_lag1": prod_prices[
            "Onion"]["price_lag"],
        "prod_arrival_Onion"  : prod_prices[
            "Onion"]["arrival"],
        "prod_price_Potato"   : prod_prices[
            "Potato"]["price"],
        "prod_price_Potato_lag1": prod_prices[
            "Potato"]["price_lag"],
        "prod_arrival_Potato" : prod_prices[
            "Potato"]["arrival"],
    }

    if "price_change_abs" in feature_cols:
        feat["price_change_abs"]     = price_change_abs
    if "price_change_abs_pct" in feature_cols:
        feat["price_change_abs_pct"] = price_change_abs_pct

    X = pd.DataFrame([feat])

    # Convert categorical columns to Categorical dtype
    # to match model training format
    state_cats = list(cat_mappings["state"])
    if state not in state_cats:
        state_cats.append(state)
    X["state"] = pd.Categorical(X["state"], categories=state_cats)

    veg_cats = list(cat_mappings["vegetable"])
    if veg not in veg_cats:
        veg_cats.append(veg)
    X["vegetable"] = pd.Categorical(X["vegetable"], categories=veg_cats)

    season_cats = list(cat_mappings["season"])
    if season not in season_cats:
        season_cats.append(season)
    X["season"] = pd.Categorical(X["season"], categories=season_cats)

    season_veg_cats = cat_mappings.get(
        "season_veg",
        [f"{s}_{v}"
         for s in ["Kharif","Rabi","Zaid"]
         for v in ["Tomato","Onion","Potato"]]
    )
    if isinstance(season_veg_cats, list):
        season_veg_cats = list(season_veg_cats)
    else:
        season_veg_cats = list(season_veg_cats)
    if season_veg not in season_veg_cats:
        season_veg_cats.append(season_veg)
    X["season_veg"] = pd.Categorical(X["season_veg"], categories=season_veg_cats)
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0

    return X[feature_cols]


# ════════════════════════════════════════════════════════
# 3 MONTH PREDICTION ENGINE
# ════════════════════════════════════════════════════════
def predict_3_months(state, veg, current_data,
                     model, le_target, feature_cols,
                     cat_mappings, df_hist):
    """
    Predict price direction for next 3 months
    Uses recursive forecasting
    """
    today      = datetime.today()
    results    = []

    # Starting values
    current_price   = current_data["modal_price"]
    current_arrival = df_hist[
        (df_hist["state"]     == state) &
        (df_hist["vegetable"] == veg)
    ]["arrival_qty"].iloc[-1] \
        if len(df_hist[
            (df_hist["state"]     == state) &
            (df_hist["vegetable"] == veg)
        ]) > 0 else 1000

    min_price = current_data.get(
        "min_price", current_price * 0.85)
    max_price = current_data.get(
        "max_price", current_price * 1.15)

    for month_offset in range(1, 4):
        predict_month = (today.month +
                         month_offset - 1) % 12 + 1
        predict_year  = today.year + \
                        (today.month +
                         month_offset - 1) // 12

        # Get weather for this month
        weather_key = f"weather_m{month_offset}"
        weather     = current_data.get(weather_key)

        if weather is None:
            weather = {
                "temp_max"   : 32.0,
                "temp_min"   : 20.0,
                "rainfall_mm": 50.0,
                "humidity"   : 60.0
            }

        # Build features
        X = build_features(
            state, veg,
            current_price, current_arrival,
            min_price, max_price,
            weather, predict_month,
            df_hist, feature_cols, cat_mappings
        )

        if X is None:
            continue

        # Predict
        proba     = model.predict(X)[0]
        pred_idx  = int(np.argmax(proba))
        direction = le_target.inverse_transform(
                        [pred_idx])[0]
        max_prob  = float(max(proba))

        # Confidence decreases with each month
        # Month 1 = actual confidence
        # Month 2 = reduced by 15%
        # Month 3 = reduced by 30%
        confidence_penalty = [0, 0.15, 0.30]
        adjusted_prob = max_prob * \
                        (1 - confidence_penalty[
                            month_offset - 1])

        confidence = (
            "High"   if adjusted_prob > 0.60 else
            "Medium" if adjusted_prob > 0.45 else
            "Low"
        )

        price_low, price_high = estimate_price_range(
            current_price, direction)

        results.append({
            "month_num"     : predict_month,
            "month_name"    : get_month_name(
                predict_month),
            "year"          : predict_year,
            "direction"     : direction,
            "probability"   : round(max_prob,       3),
            "adj_probability": round(adjusted_prob, 3),
            "confidence"    : confidence,
            "input_price"   : round(current_price,  2),
            "price_low"     : price_low,
            "price_high"    : price_high,
            "weather"       : weather,
            "all_proba"     : {
                cls: round(float(p), 3)
                for cls, p in zip(
                    le_target.classes_, proba)
            }
        })

        # Update current price for next iteration
        # Use midpoint of predicted range
        current_price   = (price_low + price_high) / 2
        min_price       = price_low  * 0.90
        max_price       = price_high * 1.10

    return results


# ════════════════════════════════════════════════════════
# BULK PREDICT — ALL STATES FOR ONE VEGETABLE
# ════════════════════════════════════════════════════════
def predict_all_states(veg, fetched_data,
                       model, le_target, feature_cols,
                       cat_mappings, df_hist):
    """
    Run 3-month prediction for all available states
    """
    all_results = {}
    veg_data    = fetched_data.get(veg, {})

    for state, data in veg_data.items():
        try:
            pred = predict_3_months(
                state, veg, data,
                model, le_target, feature_cols,
                cat_mappings, df_hist
            )
            if pred:
                all_results[state] = pred
        except Exception as e:
            print(f"  Error for {state}: {e}")
            continue

    return all_results
