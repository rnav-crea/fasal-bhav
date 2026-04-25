import os
import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from datetime import datetime, timedelta
from fetch_and_predict import fetch_all_data
from predict_3month    import (load_artifacts,
                                predict_3_months,
                                predict_all_states,
                                get_month_name)

# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

# ════════════════════════════════════════════════════════
# PAGE CONFIG & STYLING
# ════════════════════════════════════════════════════════
st.set_page_config(
    page_title = "Fasal Bhav — Price Forecast",
    page_icon  = "🌾",
    layout     = "wide",
    initial_sidebar_state="collapsed"
)

# Color Palette
GUNMETAL_GRAY = "#2F4F4F"
GHOST_WHITE = "#F8F8FF"
LIGHT_GRAY = "#475569"

# Custom CSS for styling
st.markdown(f"""
    <style>
        /* Main container */
        .main {{
            background-color: {GHOST_WHITE};
            padding-top: 20px;
            padding-bottom: 120px;
        }}
        
        /* Hide sidebar */
        [data-testid="collapsedControl"] {{
            display: none;
        }}
        
        /* Header with title and nav */
        .header-container {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: linear-gradient(135deg, {GUNMETAL_GRAY} 0%, #1a2f2f 100%);
            padding: 15px 30px;
            margin: -20px -30px 30px -30px;
            border-bottom: 3px solid #FF6B6B;
        }}
        
        .app-title {{
            font-size: 28px;
            font-weight: 700;
            color: {GHOST_WHITE};
            margin: 0;
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        
        .nav-buttons {{
            display: flex;
            gap: 10px;
            align-items: center;
        }}
        
        .nav-btn {{
            padding: 8px 16px;
            border: 2px solid {GHOST_WHITE};
            background-color: transparent;
            color: {GHOST_WHITE};
            border-radius: 6px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.3s ease;
            font-size: 14px;
        }}
        
        .nav-btn:hover {{
            background-color: {GHOST_WHITE};
            color: {GUNMETAL_GRAY};
        }}
        
        .nav-btn.active {{
            background-color: #FF6B6B;
            border-color: #FF6B6B;
            color: white;
        }}
        
        /* Fixed Footer */
        .footer-fixed {{
            position: fixed;
            bottom: 0;
            right: 0;
            left: 0;
            background-color: {GUNMETAL_GRAY};
            border-top: 2px solid #FF6B6B;
            padding: 15px 30px;
            text-align: right;
            color: {GHOST_WHITE};
            font-size: 12px;
            z-index: 999;
        }}
        
        /* Cards and sections */
        .stMetricLabel {{
            font-weight: 600;
            color: {GUNMETAL_GRAY};
        }}
        
        .warning {{
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 12px;
            border-radius: 6px;
            margin: 15px 0;
        }}
        
        .success {{
            background-color: #d4edda;
            border-left: 4px solid #28a745;
            padding: 12px;
            border-radius: 6px;
        }}
        
        /* Selectbox styling */
        [data-baseweb="select"] {{
            background-color: {GHOST_WHITE} !important;
            border: 2px solid {GUNMETAL_GRAY} !important;
        }}
        
        /* Button styling */
        .stButton > button {{
            background-color: {GUNMETAL_GRAY};
            color: {GHOST_WHITE};
            border: none;
            border-radius: 8px;
            font-weight: 600;
            padding: 10px 20px;
            transition: all 0.3s ease;
        }}
        
        .stButton > button:hover {{
            background-color: #3d6d6d;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}
        
        /* Heading styling */
        h1, h2, h3 {{
            color: {GUNMETAL_GRAY};
        }}
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 0;
        }}
    </style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# PAGE NAVIGATION (Inline - No Sidebar)
# ════════════════════════════════════════════════════════
if "page" not in st.session_state:
    st.session_state.page = "predict"

# Header with title and navigation
col_title, col_nav = st.columns([3, 2])

with col_title:
    st.markdown("""
        <div class="app-title">
            🌾 Fasal Bhav
        </div>
    """, unsafe_allow_html=True)

with col_nav:
    st.markdown("""
        <div class="nav-buttons">
    """, unsafe_allow_html=True)
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button(
            "Predict",
            key="nav_predict",
            use_container_width=True
        ):
            st.session_state.page = "predict"
    
    with col_btn2:
        if st.button(
            "Manual",
            key="nav_manual",
            use_container_width=True
        ):
            st.session_state.page = "manual"
    
    st.markdown("""
        </div>
    """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# LOAD MODEL AND ARTIFACTS
# ════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    model_path = os.path.join(MODELS_DIR, "lgbm_final.txt")
    le_target_path = os.path.join(MODELS_DIR, "le_target.pkl")
    feature_cols_path = os.path.join(MODELS_DIR, "feature_cols.csv")
    cat_mappings_path = os.path.join(MODELS_DIR, "cat_mappings.pkl")
    
    model        = lgb.Booster(model_file=model_path)
    le_target    = joblib.load(le_target_path)
    feature_cols = pd.read_csv(feature_cols_path)["feature"].tolist()
    cat_mappings = joblib.load(cat_mappings_path)
    return model, le_target, feature_cols, cat_mappings

@st.cache_data
def load_data():
    data_path = os.path.join(DATA_DIR, "master_dataset.csv")
    df = pd.read_csv(data_path, parse_dates=["date"])
    df = df.sort_values(["state", "vegetable", "date"])
    return df


@st.cache_resource
def load_prediction_artifacts():
    return load_artifacts()


@st.cache_data(ttl=60 * 60 * 6)
def get_live_snapshot(cache_version=0):
    _ = cache_version
    fetched, metadata = fetch_all_data()
    fetched_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return fetched, metadata, fetched_at

model, le_target, feature_cols, cat_mappings = load_model()
df = load_data()

# ════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════
def should_refresh_cache(last_fetched_at_str):
    """
    Determine if cache should be refreshed based on
    when market data typically updates.
    
    Market data (AGMARKNET) updates in the evening/night
    (around 8 PM IST onwards). This function checks if:
    1. Enough time has passed (> 6 hours), OR
    2. We're in evening/night hours and cache is from morning
    
    Returns: bool (True if should refresh)
    """
    try:
        last_fetch = datetime.strptime(
            last_fetched_at_str, "%Y-%m-%d %H:%M:%S")
        current_time = datetime.now()
        time_diff = current_time - last_fetch
        
        # Always refresh if more than 6 hours have passed
        if time_diff > timedelta(hours=6):
            return True
        
        # Check if we're in data update window (8 PM - 6 AM)
        current_hour = current_time.hour
        last_fetch_hour = last_fetch.hour
        
        # If last fetch was before 8 PM and current time
        # is after 8 PM, data likely updated
        if (last_fetch_hour < 20 and current_hour >= 20 and
            time_diff > timedelta(hours=2)):
            return True
        
        # If it's morning (6-10 AM) and data is from before
        # yesterday 8 PM, refresh
        if (6 <= current_hour < 10 and
            (current_time.date() > last_fetch.date())):
            return True
        
        return False
    except:
        # If timestamp parsing fails, don't refresh
        return False


def get_season(month):
    if month in [6, 7, 8, 9, 10]:    return "Kharif"
    elif month in [11, 12, 1, 2, 3]: return "Rabi"
    else:                             return "Zaid"

def get_producer_latest(veg, df):
    producer_map = {
        "Tomato" : "Karnataka",
        "Onion"  : "Maharashtra",
        "Potato" : "Uttar Pradesh"
    }
    producer = producer_map[veg]
    hist = df[
        (df["vegetable"] == veg) &
        (df["state"]     == producer)
    ].sort_values("date")

    if len(hist) < 2:
        return None, None, None

    latest     = hist.iloc[-1]["modal_price"]
    lag1       = hist.iloc[-2]["modal_price"]
    latest_arr = hist.iloc[-1]["arrival_qty"]
    return latest, lag1, latest_arr

def build_features(state, veg, modal_price,
                   arrival_qty, temp_max, temp_min,
                   rainfall_mm, humidity, predict_month,
                   df):

    hist = df[
        (df["state"]     == state) &
        (df["vegetable"] == veg)
    ].sort_values("date")

    if len(hist) < 5:
        return None

    price_lag_1m   = hist["modal_price"].iloc[-1]
    price_lag_4m   = hist["modal_price"].iloc[-4]
    rolling_avg_3m = hist["modal_price"].iloc[-3:].mean()
    arrival_lag_1m = hist["arrival_qty"].iloc[-1]
    min_price      = hist["min_price"].iloc[-1] \
                     if "min_price" in hist.columns \
                     else modal_price * 0.85
    max_price      = hist["max_price"].iloc[-1] \
                     if "max_price" in hist.columns \
                     else modal_price * 1.15

    season     = get_season(predict_month)
    season_veg = f"{season}_{veg}"

    hist_rain_mean     = df[
        (df["state"] == state) &
        (df["month"] == predict_month)
    ]["rainfall_mm"].mean()
    if pd.isna(hist_rain_mean):
        hist_rain_mean = rainfall_mm
    rainfall_deviation = rainfall_mm - hist_rain_mean

    price_mean  = hist["modal_price"].mean()
    price_std   = hist["modal_price"].std() + 1e-8
    arr_mean    = hist["arrival_qty"].mean()
    arr_std     = hist["arrival_qty"].std()   + 1e-8

    price_norm   = (modal_price  - price_mean) / price_std
    arrival_norm = (arrival_qty  - arr_mean)   / arr_std

    lag1_ratio    = modal_price  / (price_lag_1m   + 1e-8)
    lag4_ratio    = modal_price  / (price_lag_4m   + 1e-8)
    arrival_ratio = arrival_qty  / (arrival_lag_1m + 1e-8)

    price_momentum   = (modal_price - price_lag_1m) / \
                       (price_lag_1m + 1e-8) * 100
    price_vs_avg     =  modal_price - rolling_avg_3m
    arrival_momentum = (arrival_qty - arrival_lag_1m) / \
                       (arrival_lag_1m + 1e-8) * 100

    price_spread   = max_price - min_price
    price_position = (modal_price - min_price) / \
                     (max_price - min_price + 1e-8)

    price_volatility_3m = hist["modal_price"].iloc[
        -4:-1].std()

    price_change_abs = abs(modal_price - price_lag_1m)
    price_change_abs_pct = price_change_abs / \
                           (modal_price + 1e-8) * 100

    month_norm_in_veg = (predict_month - 6.5) / 3.5

    prod_price, prod_price_lag1, prod_arrival = \
        get_producer_latest(veg, df)
    if prod_price is None:
        prod_price     = modal_price
        prod_price_lag1= price_lag_1m
        prod_arrival   = arrival_qty

    # Encode categoricals
    state_cats = list(cat_mappings["state"])
    if state not in state_cats:
        state_cats.append(state)
    state_cat     = pd.Categorical(
        [state],
        categories=state_cats)
    
    veg_cats = list(cat_mappings["vegetable"])
    if veg not in veg_cats:
        veg_cats.append(veg)
    veg_cat       = pd.Categorical(
        [veg],
        categories=veg_cats)
    
    season_cats = list(cat_mappings["season"])
    if season not in season_cats:
        season_cats.append(season)
    season_cat    = pd.Categorical(
        [season],
        categories=season_cats)
    
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
    season_veg_cat = pd.Categorical(
        [season_veg],
        categories=season_veg_cats
    )

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
        f"prod_price_Tomato"  : prod_price
                                 if veg == "Tomato"
                                 else df[df["vegetable"]
                                 == "Tomato"][
                                 "modal_price"].iloc[-1],
        f"prod_price_Tomato_lag1": prod_price_lag1
                                    if veg == "Tomato"
                                    else df[
                                    df["vegetable"]
                                    == "Tomato"][
                                    "modal_price"].iloc[-2],
        f"prod_arrival_Tomato": prod_arrival
                                 if veg == "Tomato"
                                 else df[df["vegetable"]
                                 == "Tomato"][
                                 "arrival_qty"].iloc[-1],
        f"prod_price_Onion"   : prod_price
                                 if veg == "Onion"
                                 else df[df["vegetable"]
                                 == "Onion"][
                                 "modal_price"].iloc[-1],
        f"prod_price_Onion_lag1": prod_price_lag1
                                   if veg == "Onion"
                                   else df[
                                   df["vegetable"]
                                   == "Onion"][
                                   "modal_price"].iloc[-2],
        f"prod_arrival_Onion" : prod_arrival
                                 if veg == "Onion"
                                 else df[df["vegetable"]
                                 == "Onion"][
                                 "arrival_qty"].iloc[-1],
        f"prod_price_Potato"  : prod_price
                                 if veg == "Potato"
                                 else df[df["vegetable"]
                                 == "Potato"][
                                 "modal_price"].iloc[-1],
        f"prod_price_Potato_lag1": prod_price_lag1
                                    if veg == "Potato"
                                    else df[
                                    df["vegetable"]
                                    == "Potato"][
                                    "modal_price"].iloc[-2],
        f"prod_arrival_Potato": prod_arrival
                                  if veg == "Potato"
                                  else df[df["vegetable"]
                                  == "Potato"][
                                  "arrival_qty"].iloc[-1],
    }

    # Add price_change_abs if in feature_cols
    if "price_change_abs" in feature_cols:
        feat["price_change_abs"]     = price_change_abs
    if "price_change_abs_pct" in feature_cols:
        feat["price_change_abs_pct"] = price_change_abs_pct

    X = pd.DataFrame([feat])

    # Convert categorical columns to Categorical dtype
    X["state"] = pd.Categorical(X["state"], categories=state_cats)
    X["vegetable"] = pd.Categorical(X["vegetable"], categories=veg_cats)
    X["season"] = pd.Categorical(X["season"], categories=season_cats)
    X["season_veg"] = pd.Categorical(X["season_veg"], categories=season_veg_cats)

    # Keep only columns in feature_cols
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0

    return X[feature_cols]

# ════════════════════════════════════════════════════════
# PAGE LOGIC
# ════════════════════════════════════════════════════════

if st.session_state.page == "manual":
    # ════════════════════════════════════════════════════════
    # UI — HEADER
    # ════════════════════════════════════════════════════════
    st.markdown("""
        <div style='background: linear-gradient(135deg, #2F4F4F 0%, #1a2f2f 100%); 
                    padding: 30px; border-radius: 10px; margin: -20px -30px 20px -30px;'>
            <h1 style='color: #F8F8FF; text-align: center; margin: 0; font-size: 32px;'>
                Manual Price Prediction
            </h1>
            <p style='color: #B8D4D4; text-align: center; margin: 10px 0 0 0;'>
                Enter current market data and get next-month price direction forecast
            </p>
        </div>
    """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # UI — INPUT SECTION
    # ════════════════════════════════════════════════════════
    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
    st.subheader("Market Data Input")

    col1, col2, col3 = st.columns(3)

    with col1:
        vegetable = st.selectbox(
            "Commodity",
            ["Tomato", "Onion", "Potato"]
        )

        state = st.selectbox(
            "State",
            sorted(cat_mappings["state"])
        )

        predict_month = st.selectbox(
            "Predict For Month",
            options  = list(range(1, 13)),
            index    = 3,
            format_func = lambda x: [
                "January","February","March","April",
                "May","June","July","August","September",
                "October","November","December"
            ][x-1]
        )

    with col2:
        modal_price = st.number_input(
            "� Modal Price (₹/kg)",
            min_value = 1.0,
            max_value = 500.0,
            value     = 40.0,
            step      = 0.5
        )

        arrival_qty = st.number_input(
            "Arrival Quantity (MT)",
            min_value = 1.0,
            max_value = 200000.0,
            value     = 5000.0,
            step      = 100.0
        )

    with col3:
        temp_max = st.number_input(
            "Max Temperature (°C)",
            min_value = 10.0,
            max_value = 50.0,
            value     = 32.0,
            step      = 0.5
        )

        temp_min = st.number_input(
            "Min Temperature (°C)",
            min_value = 0.0,
            max_value = 40.0,
            value     = 18.0,
            step      = 0.5
        )

        rainfall_mm = st.number_input(
            "Rainfall (mm)",
            min_value = 0.0,
            max_value = 1000.0,
            value     = 10.0,
            step      = 1.0
        )

        humidity = st.number_input(
            "Humidity (%)",
            min_value = 0.0,
            max_value = 100.0,
            value     = 60.0,
            step      = 1.0
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # PREDICT BUTTON
    # ════════════════════════════════════════════════════════
    predict_clicked = st.button(
        "� Get Price Direction",
        use_container_width=True
    )

    if predict_clicked:
        X = build_features(
            state, vegetable, modal_price,
            arrival_qty, temp_max, temp_min,
            rainfall_mm, humidity,
            predict_month, df
        )

        if X is None:
            st.error(
                "Not enough historical data for this "
                "state-vegetable combination.")
        else:
            # Predict
            proba     = model.predict(X)[0]
            pred_idx  = int(np.argmax(proba))
            direction = le_target.inverse_transform(
                            [pred_idx])[0]
            max_prob  = float(max(proba))

            confidence = (
                "High"   if max_prob > 0.60 else
                "Medium" if max_prob > 0.45 else
                "Low"
            )

            # Color and arrow per direction
            color_map = {
                "UP"    : "#2D7D46",
                "DOWN"  : "#C0392B",
                "STABLE": "#2471A3"
            }
            arrow_map = {
                "UP"    : "↑ PRICE WILL GO UP",
                "DOWN"  : "↓ PRICE WILL GO DOWN",
                "STABLE": "→ PRICE WILL STAY STABLE"
            }
            emoji_map = {
                "UP"    : "↑",
                "DOWN"  : "↓",
                "STABLE": "→"
            }

            color  = color_map[direction]
            arrow  = arrow_map[direction]
            emoji  = emoji_map[direction]

            st.markdown("<hr>", unsafe_allow_html=True)
            st.subheader("Prediction Result")

            # Main result card
            st.markdown(f"""
                <div style='
                    background-color:{color};
                    padding:30px;
                    border-radius:15px;
                    text-align:center;
                    margin:10px 0;
                '>
                    <h1 style='color:white; margin:0;
                               font-size:48px;'>
                        {emoji} {arrow}
                    </h1>
                    <p style='color:white; font-size:20px;
                              margin-top:10px;'>
                        {vegetable} in {state}
                        — Next Month Forecast
                    </p>
                </div>
            """, unsafe_allow_html=True)

            # Metrics row
            m1, m2, m3, m4 = st.columns(4)

            with m1:
                st.metric("Current Price",
                          f"₹{modal_price}/kg")
            with m2:
                st.metric("Confidence",
                          confidence)
            with m3:
                st.metric("Probability",
                          f"{max_prob*100:.0f}%")
            with m4:
                st.metric("Season",
                          get_season(predict_month))

            # Probability breakdown
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("Probability Breakdown")

            prob_df = pd.DataFrame({
                "Direction"  : le_target.classes_,
                "Probability": [f"{p*100:.1f}%"
                                for p in proba],
                "Score"      : proba
            }).sort_values("Score", ascending=False)

            col_a, col_b = st.columns([1, 2])

            with col_a:
                st.dataframe(
                    prob_df[["Direction", "Probability"]],
                    hide_index=True,
                    use_container_width=True
                )

            with col_b:
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(6, 2.5))
                # Sort by probability for cleaner chart
                sorted_idx    = np.argsort(proba)
                sorted_labels = [le_target.classes_[i] for i in sorted_idx]
                sorted_proba  = proba[sorted_idx]
                sorted_colors = []
                for label in sorted_labels:
                    if label == "UP":
                        sorted_colors.append("#2D7D46")
                    elif label == "DOWN":
                        sorted_colors.append("#C0392B")
                    else:
                        sorted_colors.append("#2471A3")

                bars = ax.barh(
                    sorted_labels,
                    sorted_proba,
                    color=sorted_colors
                )
                ax.set_xlim(0, 1)
                ax.set_xlabel("Probability")
                ax.set_title("Prediction Confidence per Class")
                for bar, prob in zip(bars, sorted_proba):
                    ax.text(
                        bar.get_width() + 0.01,
                        bar.get_y() + bar.get_height() / 2,
                        f"{prob*100:.1f}%",
                        va="center", fontsize=11
                    )
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            # ── Historical Price Trend ───────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("Historical Price Trend — Last 12 Months")

            hist_data = df[
                (df["state"]     == state) &
                (df["vegetable"] == vegetable)
            ].sort_values("date").tail(12)

            if len(hist_data) > 0:
                fig2, ax2 = plt.subplots(figsize=(10, 4))

                ax2.plot(
                    hist_data["date"],
                    hist_data["modal_price"],
                    color     = "#2D7D46",
                    linewidth = 2.5,
                    marker    = "o",
                    markersize= 5,
                    label     = "Modal Price"
                )

                ax2.axhline(
                    y          = hist_data["modal_price"].mean(),
                    color      = "#888",
                    linestyle  = "--",
                    linewidth  = 1,
                    label      = f"Average ₹{hist_data['modal_price'].mean():.1f}/kg"
                )

                # Mark current input price
                ax2.axhline(
                    y         = modal_price,
                    color     = color,
                    linestyle = "-.",
                    linewidth = 1.5,
                    label     = f"Your Input ₹{modal_price}/kg"
                )

                ax2.set_xlabel("Month")
                ax2.set_ylabel("Price (₹/kg)")
                ax2.set_title(
                    f"{vegetable} Price History — {state}")
                ax2.legend()
                ax2.tick_params(axis="x", rotation=45)
                fig2.patch.set_facecolor("#0E1117")
                ax2.set_facecolor("#0E1117")
                ax2.tick_params(colors="white")
                ax2.xaxis.label.set_color("white")
                ax2.yaxis.label.set_color("white")
                ax2.title.set_color("white")
                ax2.legend(
                    facecolor="#1E2A1E",
                    labelcolor="white")
                for spine in ax2.spines.values():
                    spine.set_edgecolor("#444")

                plt.tight_layout()
                st.pyplot(fig2)
                plt.close()
            else:
                st.info("No historical data available for "
                        "this combination")

            # Key drivers
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("Key Drivers")

            drivers = []

            if abs(modal_price - df[
                (df["vegetable"] == vegetable) &
                (df["state"] == state)
            ]["modal_price"].iloc[-1]) / (
                df[(df["vegetable"] == vegetable) &
                   (df["state"] == state)
            ]["modal_price"].iloc[-1] + 1e-8) > 0.1:
                drivers.append(
                    "Price has moved significantly "
                    "from last month")

            hist_rain = df[
                (df["state"] == state) &
                (df["month"] == predict_month)
            ]["rainfall_mm"].mean()
            if not pd.isna(hist_rain):
                if rainfall_mm < hist_rain * 0.7:
                    drivers.append(
                        "Rainfall is below seasonal "
                        "normal — may reduce supply")
                elif rainfall_mm > hist_rain * 1.3:
                    drivers.append(
                        "Rainfall is above seasonal "
                        "normal — may affect harvest")

            if predict_month in [9, 10, 11]:
                drivers.append(
                    "Post-monsoon transition month — "
                    "higher price uncertainty")

            if arrival_qty < df[
                (df["vegetable"] == vegetable) &
                (df["state"] == state)
            ]["arrival_qty"].mean() * 0.7:
                drivers.append(
                    "Arrivals are below average — "
                    "supply is tight")
            elif arrival_qty > df[
                (df["vegetable"] == vegetable) &
                (df["state"] == state)
            ]["arrival_qty"].mean() * 1.3:
                drivers.append(
                    "Arrivals are above average — "
                    "supply is strong")

            if not drivers:
                drivers.append(
                    "Market conditions appear normal "
                    "— no strong signals detected")

            for driver in drivers:
                st.info(driver)

elif st.session_state.page == "predict":

    st.markdown("""
        <div style='background: linear-gradient(135deg, #2F4F4F 0%, #1a2f2f 100%); 
                    padding: 30px; border-radius: 10px; margin: -20px -30px 20px -30px;'>
            <h1 style='color: #F8F8FF; text-align: center; margin: 0; font-size: 32px;'>
                Live Price Forecast
            </h1>
            <p style='color: #B8D4D4; text-align: center; margin: 10px 0 0 0;'>
                3-month prediction based on current market data
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Initialize or refresh data intelligently
    needs_refresh = False
    
    if "available_states" not in st.session_state:
        # First load - always fetch
        needs_refresh = True
    elif "fetched_at" in st.session_state:
        # Check if we should auto-refresh based on data
        # update window
        needs_refresh = should_refresh_cache(
            st.session_state["fetched_at"])
    
    if needs_refresh:
        with st.spinner("Loading market data..."):
            if "live_cache_version" not in st.session_state:
                st.session_state["live_cache_version"] = 0
            else:
                # Increment version to bypass cache
                st.session_state["live_cache_version"] += 1
            
            fetched, metadata, fetched_at = get_live_snapshot(
                st.session_state["live_cache_version"]
            )
            st.session_state["available_states"] = metadata[
                "available_states"]
            st.session_state["failed_states"] = metadata[
                "failed_states"]
            st.session_state["fetched_at"] = fetched_at
            st.session_state["cached_data"] = fetched
    
    available_states = st.session_state[
        "available_states"]
    failed_states = st.session_state[
        "failed_states"]

    # Show unavailable states warning
    if failed_states:
        st.warning(
            f"⚠️ Data unavailable for: "
            f"{', '.join(failed_states)}. "
            f"Showing {len(available_states)} "
            f"available states.")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        selected_veg = st.selectbox(
            "Commodity",
            ["Tomato", "Onion", "Potato"]
        )

    with col2:
        selected_state = st.selectbox(
            "Market",
            available_states,
            help=("Choose from available markets. "
                  "Unavailable markets have data "
                  "fetch issues.")
        )

    with col3:
        st.write("")
        st.write("")
        predict_clicked = st.button(
            "Forecast",
            use_container_width=True,
            key="predict_btn"
        )

    if predict_clicked:
        mdl, le, feat_cols, cat_map, df_h = (
            load_prediction_artifacts())

        # Get state data from session cache
        state_data = st.session_state[
            "cached_data"].get(
            selected_veg, {}).get(selected_state)

        if state_data is None:
            st.error(
                f"Could not fetch data for "
                f"{selected_veg} in "
                f"{selected_state}. "
                f"Try another market.")
        else:
            st.success(
                f"Current price: "
                f"₹{state_data['modal_price']}/kg")

            # Run 3 month prediction
            predictions = predict_3_months(
                selected_state, selected_veg,
                state_data, mdl, le,
                feat_cols, cat_map, df_h
            )

            if not predictions:
                st.error(
                    "Insufficient historical "
                    "data for prediction")
            else:
                st.markdown("<br>",
                            unsafe_allow_html=True)
                st.subheader(
                    f"3-Month Outlook - "
                    f"{selected_veg}")

                # ── Timeline Card View ───────────────
                cols = st.columns(3)

                color_map = {
                    "UP"    : "#2D7D46",
                    "DOWN"  : "#C0392B",
                    "STABLE": "#2471A3"
                }
                arrow_map = {
                    "UP"    : "↑ UP",
                    "DOWN"  : "↓ DOWN",
                    "STABLE": "→ STABLE"
                }

                for i, pred in enumerate(predictions):
                    with cols[i]:
                        color = color_map[
                            pred["direction"]]
                        arrow = arrow_map[
                            pred["direction"]]

                        st.markdown(f"""
                            <div style='
                                background:{color};
                                padding:20px;
                                border-radius:12px;
                                text-align:center;
                            '>
                                <h3 style='
                                    color:white;
                                    margin:0;'>
                                    {pred["month_name"]}
                                </h3>
                                <h2 style='
                                    color:white;
                                    margin:10px 0;
                                    font-size:28px;'>
                                    {arrow}
                                </h2>
                                <p style='
                                    color:white;
                                    margin:5px 0;
                                    font-size:13px;'>
                                    {pred["adj_probability"]*100:.0f}%
                                    confidence
                                </p>
                                <p style='
                                    color:white;
                                    margin:5px 0 0;
                                    font-size:12px;'>
                                    ₹{pred["price_low"]} – "
                                    ₹{pred["price_high"]}/kg
                                </p>
                            </div>
                        """, unsafe_allow_html=True)

                # ── Price Trajectory Chart ────────────
                st.markdown("<br>",
                            unsafe_allow_html=True)
                st.subheader("Price Trend")

                import matplotlib.pyplot as plt
                import matplotlib.patches as mpatches

                fig, ax = plt.subplots(figsize=(10, 4))

                # Historical last 6 months
                hist_data = df_h[
                    (df_h["state"] == selected_state) &
                    (df_h["vegetable"] == selected_veg)
                ].sort_values("date").tail(6)

                hist_months = [
                    get_month_name(m)[:3]
                    for m in hist_data["month"].tolist()
                ]
                hist_prices = hist_data[
                    "modal_price"].tolist()

                # Current + predicted
                pred_months = (
                    [get_month_name(
                        hist_data["month"].iloc[-1]
                    )[:3]] +
                    [p["month_name"][:3]
                     for p in predictions]
                )
                pred_prices = (
                    [state_data["modal_price"]] +
                    [(p["price_low"] + p["price_high"])
                     / 2
                     for p in predictions]
                )
                pred_low  = (
                    [state_data["modal_price"]] +
                    [p["price_low"]
                     for p in predictions]
                )
                pred_high = (
                    [state_data["modal_price"]] +
                    [p["price_high"]
                     for p in predictions]
                )

                # Plot historical
                all_months = hist_months + \
                             pred_months[1:]
                ax.plot(
                    range(len(hist_months)),
                    hist_prices,
                    color     = "#2D7D46",
                    linewidth = 2.5,
                    marker    = "o",
                    markersize= 6,
                    label     = "Historical"
                )

                # Plot predicted
                pred_x = range(
                    len(hist_months) - 1,
                    len(hist_months) +
                    len(pred_months) - 1
                )
                ax.plot(
                    list(pred_x),
                    pred_prices,
                    color     = "#F39C12",
                    linewidth = 2.5,
                    marker    = "o",
                    markersize= 6,
                    linestyle = "--",
                    label     = "Predicted"
                )

                # Uncertainty band
                ax.fill_between(
                    list(pred_x),
                    pred_low,
                    pred_high,
                    alpha = 0.2,
                    color = "#F39C12",
                    label = "Expected Range"
                )

                ax.set_xticks(
                    range(len(all_months)))
                ax.set_xticklabels(
                    all_months, rotation=45)
                ax.set_ylabel("Price (₹/kg)")
                ax.set_title(
                    f"{selected_veg} — "
                    f"{selected_state}")
                ax.legend()
                ax.axvline(
                    x         = len(hist_months) - 1,
                    color     = "#888",
                    linestyle = ":",
                    label     = "Today"
                )

                fig.patch.set_facecolor("#0E1117")
                ax.set_facecolor("#0E1117")
                ax.tick_params(colors="white")
                ax.xaxis.label.set_color("white")
                ax.yaxis.label.set_color("white")
                ax.title.set_color("white")
                ax.legend(
                    facecolor  = "#1E2A1E",
                    labelcolor = "white")
                for spine in ax.spines.values():
                    spine.set_edgecolor("#444")

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                # ── Recommendation ────────────────────
                st.markdown("<br>",
                            unsafe_allow_html=True)
                st.subheader("Insight")

                directions = [p["direction"]
                              for p in predictions]

                if directions.count("UP") >= 2:
                    rec = (
                        f"Expect price INCREASE. "
                        f"Hold stock if possible.")
                elif directions.count("DOWN") >= 2:
                    rec = (
                        f"Expect price DECREASE. "
                        f"Consider selling soon.")
                else:
                    rec = (
                        f"Prices likely STABLE. "
                        f"Plan accordingly.")

                st.info(rec)


# FOOTER (Fixed at bottom)
# ════════════════════════════════════════════════════════
st.markdown("""
    <div class="footer-fixed">
        <div style="text-align: right; padding: 5px 0;">
            <span style="opacity: 0.8;">Fasal Bhav — AGMARKNET + Open-Meteo | LightGBM | 76% Directional Accuracy for Tomato</span>
        </div>
    </div>
""", unsafe_allow_html=True)