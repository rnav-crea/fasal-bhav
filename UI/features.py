"""
Shared feature engineering module for Fasal Bhav.
Contains functions to build feature vectors for prediction.
"""
import pandas as pd
import numpy as np
from datetime import datetime

def get_season(month):
    """Return season name for month (1-12)."""
    if month in [6, 7, 8, 9, 10]:
        return "Kharif"
    elif month in [11, 12, 1, 2, 3]:
        return "Rabi"
    else:
        return "Zaid"

def get_producer_latest(veg, df_hist):
    """
    Get latest price, price lag1, and arrival for the producer state of a vegetable.
    Returns tuple (price, price_lag1, arrival) or (None, None, None) if not enough history.
    """
    producer_map = {
        "Tomato": "Karnataka",
        "Onion": "Maharashtra",
        "Potato": "Uttar Pradesh"
    }
    producer = producer_map.get(veg)
    if producer is None:
        return None, None, None

    prod_hist = df_hist[
        (df_hist["vegetable"] == veg) &
        (df_hist["state"] == producer)
    ].sort_values("date")

    if len(prod_hist) < 2:
        return None, None, None

    price = prod_hist.iloc[-1]["modal_price"]
    price_lag = prod_hist.iloc[-2]["modal_price"]
    arrival = prod_hist.iloc[-1]["arrival_qty"]
    return price, price_lag, arrival

def build_features(state, veg, modal_price, arrival_qty, min_price, max_price,
                   weather, predict_month, df_hist, feature_cols, cat_mappings):
    """
    Build all features for one prediction.
    Returns a DataFrame with columns in feature_cols, or None if insufficient history.
    """
    # Filter history for this state-vegetable
    hist = df_hist[
        (df_hist["state"] == state) &
        (df_hist["vegetable"] == veg)
    ].sort_values("date")

    if len(hist) < 5:
        return None

    # Lag features from history
    price_lag_1m = hist["modal_price"].iloc[-1]
    price_lag_4m = hist["modal_price"].iloc[-4]
    rolling_avg_3m = hist["modal_price"].iloc[-3:].mean()
    arrival_lag_1m = hist["arrival_qty"].iloc[-1]

    # Use provided min/max; if missing, try to derive from history
    if min_price is None:
        min_price = hist["min_price"].iloc[-1] if "min_price" in hist.columns else modal_price * 0.85
    if max_price is None:
        max_price = hist["max_price"].iloc[-1] if "max_price" in hist.columns else modal_price * 1.15

    # Weather
    temp_max = weather["temp_max"]
    temp_min = weather["temp_min"]
    rainfall_mm = weather["rainfall_mm"]
    humidity = weather["humidity"]

    # Rainfall deviation vs historical average for this month and state
    hist_rain = df_hist[
        (df_hist["state"] == state) &
        (df_hist["month"] == predict_month)
    ]["rainfall_mm"].mean()
    if pd.isna(hist_rain):
        hist_rain = rainfall_mm
    rainfall_deviation = rainfall_mm - hist_rain

    # Normalization
    price_mean = hist["modal_price"].mean()
    price_std = hist["modal_price"].std() + 1e-8
    arr_mean = hist["arrival_qty"].mean()
    arr_std = hist["arrival_qty"].std() + 1e-8

    price_norm = (modal_price - price_mean) / price_std
    arrival_norm = (arrival_qty - arr_mean) / arr_std

    # Ratios
    lag1_ratio = modal_price / (price_lag_1m + 1e-8)
    lag4_ratio = modal_price / (price_lag_4m + 1e-8)
    arrival_ratio = arrival_qty / (arrival_lag_1m + 1e-8)

    # Momentum
    price_momentum = (modal_price - price_lag_1m) / (price_lag_1m + 1e-8) * 100
    price_vs_avg = modal_price - rolling_avg_3m
    arrival_momentum = (arrival_qty - arrival_lag_1m) / (arrival_lag_1m + 1e-8) * 100

    # Price spread and position
    price_spread = max_price - min_price
    price_position = (modal_price - min_price) / (max_price - min_price + 1e-8)

    # Volatility (std of price over last 3 months, excluding current?)
    price_volatility_3m = hist["modal_price"].iloc[-4:-1].std()

    # Absolute change
    price_change_abs = abs(modal_price - price_lag_1m)
    price_change_abs_pct = price_change_abs / (modal_price + 1e-8) * 100

    # Month normalization within vegetable
    month_norm_in_veg = (predict_month - 6.5) / 3.5

    # Producer state values for each vegetable
    producer_map = {
        "Tomato": "Karnataka",
        "Onion": "Maharashtra",
        "Potato": "Uttar Pradesh"
    }
    prod_features = {}
    for v, prod_state in producer_map.items():
        prod_hist = df_hist[
            (df_hist["vegetable"] == v) &
            (df_hist["state"] == prod_state)
        ].sort_values("date")

        if len(prod_hist) >= 2:
            prod_features[v] = {
                "price": prod_hist.iloc[-1]["modal_price"],
                "price_lag": prod_hist.iloc[-2]["modal_price"],
                "arrival": prod_hist.iloc[-1]["arrival_qty"],
            }
        else:
            # Fallback to current observation if producer history unavailable
            prod_features[v] = {
                "price": modal_price,
                "price_lag": price_lag_1m,
                "arrival": arrival_qty,
            }

    # Construct feature dictionary
    feat = {
        "state": state,
        "vegetable": veg,
        "season": get_season(predict_month),
        "price_norm": price_norm,
        "arrival_norm": arrival_norm,
        "lag1_ratio": lag1_ratio,
        "lag4_ratio": lag4_ratio,
        "arrival_ratio": arrival_ratio,
        "price_momentum": price_momentum,
        "price_vs_avg": price_vs_avg,
        "arrival_momentum": arrival_momentum,
        "modal_price": modal_price,
        "price_lag_1m": price_lag_1m,
        "price_lag_4m": price_lag_4m,
        "rolling_avg_3m": rolling_avg_3m,
        "price_spread": price_spread,
        "price_position": price_position,
        "arrival_qty": arrival_qty,
        "arrival_lag_1m": arrival_lag_1m,
        "temp_max": temp_max,
        "temp_min": temp_min,
        "rainfall_mm": rainfall_mm,
        "humidity": humidity,
        "rainfall_deviation": rainfall_deviation,
        "month_sin": np.sin(2 * np.pi * predict_month / 12),
        "month_cos": np.cos(2 * np.pi * predict_month / 12),
        "month_norm_in_veg": month_norm_in_veg,
        "is_post_monsoon": int(predict_month in [9, 10, 11]),
        "season_veg": f"{get_season(predict_month)}_{veg}",
        "price_volatility_3m": price_volatility_3m,
        f"prod_price_Tomato": prod_features["Tomato"]["price"],
        f"prod_price_Tomato_lag1": prod_features["Tomato"]["price_lag"],
        f"prod_arrival_Tomato": prod_features["Tomato"]["arrival"],
        f"prod_price_Onion": prod_features["Onion"]["price"],
        f"prod_price_Onion_lag1": prod_features["Onion"]["price_lag"],
        f"prod_arrival_Onion": prod_features["Onion"]["arrival"],
        f"prod_price_Potato": prod_features["Potato"]["price"],
        f"prod_price_Potato_lag1": prod_features["Potato"]["price_lag"],
        f"prod_arrival_Potato": prod_features["Potato"]["arrival"],
    }

    # Add optional change features if they are in the feature list
    if "price_change_abs" in feature_cols:
        feat["price_change_abs"] = price_change_abs
    if "price_change_abs_pct" in feature_cols:
        feat["price_change_abs_pct"] = price_change_abs_pct

    # Create DataFrame with a single row
    X = pd.DataFrame([feat])

    # Convert categorical columns to Categorical dtype using provided mappings
    # State
    state_cats = list(cat_mappings["state"])
    if state not in state_cats:
        state_cats.append(state)
    X["state"] = pd.Categorical(X["state"], categories=state_cats)

    # Vegetable
    veg_cats = list(cat_mappings["vegetable"])
    if veg not in veg_cats:
        veg_cats.append(veg)
    X["vegetable"] = pd.Categorical(X["vegetable"], categories=veg_cats)

    # Season
    season_cats = list(cat_mappings["season"])
    season = get_season(predict_month)
    if season not in season_cats:
        season_cats.append(season)
    X["season"] = pd.Categorical(X["season"], categories=season_cats)

    # Season_veg
    season_veg_cats = cat_mappings.get(
        "season_veg",
        [f"{s}_{v}"
         for s in ["Kharif", "Rabi", "Zaid"]
         for v in ["Tomato", "Onion", "Potato"]]
    )
    if isinstance(season_veg_cats, list):
        season_veg_cats = list(season_veg_cats)
    else:
        season_veg_cats = list(season_veg_cats)
    season_veg = f"{get_season(predict_month)}_{veg}"
    if season_veg not in season_veg_cats:
        season_veg_cats.append(season_veg)
    X["season_veg"] = pd.Categorical(X["season_veg"], categories=season_veg_cats)

    # Ensure all expected columns are present (fill missing with 0)
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0

    # Return columns in the order expected by the model
    return X[feature_cols]