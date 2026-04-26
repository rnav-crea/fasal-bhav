import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import time

# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

# ════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════
API_KEY     = os.getenv(
    "DATA_GOV_API_KEY",
    "579b464db66ec23bdd000001928ace9212864ca56da90a5ae29b9aa5"
)
RESOURCE_ID = "9ef84268-d588-465a-a308-a864a43d0070"
BASE_URL    = f"https://api.data.gov.in/resource/{RESOURCE_ID}"

VEGETABLES  = ["Tomato", "Onion", "Potato"]

TARGET_STATES = [
    "Maharashtra", "Uttar Pradesh", "West Bengal",
    "Karnataka", "Andhra Pradesh", "Tamil Nadu",
    "Madhya Pradesh", "Gujarat", "Rajasthan",
    "Punjab", "Haryana", "Odisha",
    "Himachal Pradesh", "Chattisgarh", "Telangana",
    "Kerala", "Uttarakhand", "Bihar"
]

STATE_COORDS = {
    "Maharashtra"    : {"lat": 19.07, "lon": 72.87},
    "Uttar Pradesh"  : {"lat": 26.84, "lon": 80.94},
    "West Bengal"    : {"lat": 22.57, "lon": 88.36},
    "Karnataka"      : {"lat": 12.97, "lon": 77.59},
    "Andhra Pradesh" : {"lat": 15.91, "lon": 79.74},
    "Tamil Nadu"     : {"lat": 13.08, "lon": 80.27},
    "Madhya Pradesh" : {"lat": 23.25, "lon": 77.41},
    "Gujarat"        : {"lat": 23.02, "lon": 72.57},
    "Rajasthan"      : {"lat": 26.91, "lon": 75.78},
    "Punjab"         : {"lat": 30.73, "lon": 76.77},
    "Haryana"        : {"lat": 29.06, "lon": 76.08},
    "Odisha"         : {"lat": 20.29, "lon": 85.82},
    "Himachal Pradesh": {"lat": 31.10, "lon": 77.17},
    "Chattisgarh"    : {"lat": 21.27, "lon": 81.86},
    "Telangana"      : {"lat": 17.38, "lon": 78.48},
    "Kerala"         : {"lat": 8.52,  "lon": 76.93},
    "Uttarakhand"    : {"lat": 30.33, "lon": 78.03},
    "Bihar"          : {"lat": 25.59, "lon": 85.13},
}

_SEASONAL_WEATHER_LOOKUP = None


def _get_seasonal_weather_lookup():
    """Load monthly weather climatology once per process."""
    global _SEASONAL_WEATHER_LOOKUP

    if _SEASONAL_WEATHER_LOOKUP is None:
        data_path = os.path.join(DATA_DIR, "master_dataset.csv")
        df = pd.read_csv(data_path)
        df["month"] = pd.to_datetime(
            df["date"], errors="coerce").dt.month
        _SEASONAL_WEATHER_LOOKUP = (
            df.groupby(["state", "month"], dropna=True)[
                ["temp_max", "temp_min",
                 "rainfall_mm", "humidity"]
            ]
            .mean()
            .reset_index()
        )

    return _SEASONAL_WEATHER_LOOKUP

# ════════════════════════════════════════════════════════
# CACHING & ERROR HANDLING
# ════════════════════════════════════════════════════════
def get_fallback_price(vegetable, state):
    """
    Get fallback price from historical data
    when live API fails — ensures app still works
    """
    try:
        data_path = os.path.join(DATA_DIR, "master_dataset.csv")
        df = pd.read_csv(data_path)
        
        recent = df[
            (df["vegetable"].str.strip().str.title() == vegetable) &
            (df["state"].str.strip().str.title() == state)
        ].tail(5)
        
        if recent.empty:
            return None
        
        return {
            "modal_price": round(recent["modal_price"].median(), 2),
            "min_price"  : round(recent["modal_price"].min(), 2),
            "max_price"  : round(recent["modal_price"].max(), 2),
            "records"    : len(recent),
            "from_fallback": True
        }
    except Exception as e:
        print(f"  Fallback error: {e}")
        return None


# ════════════════════════════════════════════════════════
# STEP 1 — FETCH CURRENT PRICE FROM data.gov.in
# ════════════════════════════════════════════════════════
def fetch_current_prices(vegetable):
    """
    Fetch last 30 days of price data for a vegetable
    Returns state-wise median modal price
    With retry logic and fallback to historical data
    """
    print(f"Fetching prices for {vegetable}...")

    all_records = []
    offset      = 0
    limit       = 1000
    max_retries = 2

    # Date range — last 30 days
    end_date   = datetime.today()
    start_date = end_date - timedelta(days=30)

    for retry in range(max_retries):
        try:
            while True:
                params = {
                    "api-key"              : API_KEY,
                    "format"               : "json",
                    "limit"                : limit,
                    "offset"               : offset,
                    "filters[commodity]"   : vegetable,
                }

                try:
                    resp    = requests.get(
                        BASE_URL, params=params, timeout=15)
                    data    = resp.json()
                    records = data.get("records", [])

                    if not records:
                        break

                    all_records.extend(records)
                    total = int(data.get("total", 0))
                    offset += limit

                    if offset >= total or offset >= 5000:
                        break

                except (requests.Timeout, requests.ConnectionError) as e:
                    if retry < max_retries - 1:
                        print(f"  Timeout, retrying... ({retry+1}/{max_retries})")
                        time.sleep(2 ** retry)  # exponential backoff
                        continue
                    else:
                        raise
            
            # If we got here, we have data
            if all_records:
                break

        except Exception as e:
            print(f"  API error on attempt {retry+1}: {e}")
            if retry < max_retries - 1:
                time.sleep(2 ** retry)
            continue

    if not all_records:
        print(f"  No live data for {vegetable}, using fallback")
        return {}

    df = pd.DataFrame(all_records)

    # Convert price from Rs/Quintal to Rs/Kg
    df["modal_price"] = pd.to_numeric(
        df["modal_price"], errors="coerce") / 100
    df["min_price"]   = pd.to_numeric(
        df["min_price"],   errors="coerce") / 100
    df["max_price"]   = pd.to_numeric(
        df["max_price"],   errors="coerce") / 100

    # Standardize state names
    df["state"] = df["state"].str.strip().str.title()

    # Keep only latest 30-day window using arrival date
    df["arrival_date"] = pd.to_datetime(
        df.get("arrival_date"),
        format="%d/%m/%Y",
        errors="coerce"
    )
    start_ts = pd.Timestamp(start_date.date())
    end_ts   = pd.Timestamp(end_date.date())
    df = df[df["arrival_date"].between(start_ts, end_ts)]

    # Filter our 18 states
    df = df[df["state"].isin(TARGET_STATES)]

    # Remove outliers — prices below 0.5 or above 500
    df = df[
        (df["modal_price"] >= 0.5) &
        (df["modal_price"] <= 500)
    ]

    # Aggregate to state level — median price
    state_prices = (
        df.groupby("state")
        .agg(
            modal_price = ("modal_price", "median"),
            min_price   = ("min_price",   "median"),
            max_price   = ("max_price",   "median"),
            records     = ("modal_price", "count")
        )
        .reset_index()
    )

    result = {}
    for _, row in state_prices.iterrows():
        result[row["state"]] = {
            "modal_price": round(row["modal_price"], 2),
            "min_price"  : round(row["min_price"],   2),
            "max_price"  : round(row["max_price"],   2),
            "records"    : int(row["records"])
        }

    print(f"  Got prices for {len(result)} states")
    return result


# ════════════════════════════════════════════════════════
# STEP 2 — FETCH WEATHER FROM OPEN-METEO
# ════════════════════════════════════════════════════════
def fetch_weather_last_month(state):
    """
    Fetch actual weather for last month
    Returns monthly aggregated weather
    """
    coords     = STATE_COORDS.get(state)
    if not coords:
        return None

    end_date   = datetime.today()
    start_date = end_date - timedelta(days=30)

    url    = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude"  : coords["lat"],
        "longitude" : coords["lon"],
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date"  : end_date.strftime("%Y-%m-%d"),
        "daily"     : [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "relative_humidity_2m_mean"
        ],
        "timezone"  : "Asia/Kolkata"
    }

    try:
        resp = requests.get(url, params=params,
                            timeout=30)
        data = resp.json()

        if "daily" not in data:
            return None

        df = pd.DataFrame(data["daily"])

        return {
            "temp_max"   : round(
                df["temperature_2m_max"].mean(), 1),
            "temp_min"   : round(
                df["temperature_2m_min"].mean(), 1),
            "rainfall_mm": round(
                df["precipitation_sum"].sum(), 1),
            "humidity"   : round(
                df["relative_humidity_2m_mean"].mean(), 1),
        }

    except Exception as e:
        print(f"  Weather error for {state}: {e}")
        return None


def fetch_weather_forecast(state, month_offset=1):
    """
    Fetch weather forecast for future months
    Uses Open-Meteo forecast API for next 16 days
    Uses historical seasonal average beyond that
    """
    coords = STATE_COORDS.get(state)
    if not coords:
        return None

    # For month 1 (next month) use forecast API
    if month_offset == 1:
        url    = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude"  : coords["lat"],
            "longitude" : coords["lon"],
            "daily"     : [
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_sum",
                "relative_humidity_2m_mean"
            ],
            "forecast_days": 16,
            "timezone"  : "Asia/Kolkata"
        }

        try:
            resp = requests.get(
                url, params=params, timeout=30)
            data = resp.json()

            if "daily" not in data:
                return None

            df = pd.DataFrame(data["daily"])
            return {
                "temp_max"   : round(
                    df["temperature_2m_max"].mean(), 1),
                "temp_min"   : round(
                    df["temperature_2m_min"].mean(), 1),
                "rainfall_mm": round(
                    df["precipitation_sum"].sum(), 1),
                "humidity"   : round(
                    df["relative_humidity_2m_mean"]
                    .mean(), 1),
            }
        except Exception as e:
            print(f"  Forecast error: {e}")
            return None

    # For months 2 and 3 use historical seasonal average
    else:
        target_month = (datetime.today().month +
                        month_offset - 1) % 12 + 1
        return get_seasonal_weather_avg(
            state, target_month)


def get_seasonal_weather_avg(state, month):
    """
    Get historical average weather for a given
    state and month from master_dataset
    """
    try:
        lookup = _get_seasonal_weather_lookup()
        avg = lookup[
            (lookup["state"] == state) &
            (lookup["month"] == month)
        ]

        if avg.empty:
            return None

        row = avg.iloc[0]

        return {
            "temp_max"   : round(row["temp_max"],    1),
            "temp_min"   : round(row["temp_min"],    1),
            "rainfall_mm": round(row["rainfall_mm"], 1),
            "humidity"   : round(row["humidity"],    1),
        }
    except Exception:
        return None


# ════════════════════════════════════════════════════════
# STEP 3 — COMBINE ALL FETCHED DATA
# ════════════════════════════════════════════════════════
def fetch_all_data():
    """
    Master fetch function — gets everything needed
    for prediction in one call
    Returns: (data, metadata)
      - data: combined dict ready for feature engineering
      - metadata: {
          'available_states': list of states with successful data,
          'failed_states': list of states without data,
          'fetched_at': timestamp string,
          'fallback_used': True if historical data was used
        }
    """
    print("Starting data fetch...")
    print("=" * 50)

    result = {}
    available_states = set()
    failed_states = set()
    fallback_used = False
    
    # Load historical data once for fallback
    try:
        data_path = os.path.join(DATA_DIR, "master_dataset.csv")
        df_hist = pd.read_csv(data_path)
    except Exception as e:
        print(f"Warning: Could not load historical data: {e}")
        df_hist = None

    # Fetch prices for all 3 vegetables
    price_data = {}
    for veg in VEGETABLES:
        price_data[veg] = fetch_current_prices(veg)
        
        # If no live data found, use fallback for all states
        if not price_data[veg]:
            print(f"  Live API returned no data for {veg}, using fallback...")
            price_data[veg] = {}
            for state in TARGET_STATES:
                fallback = get_fallback_price(veg, state)
                if fallback:
                    price_data[veg][state] = fallback
                    fallback_used = True

    # Fetch weather for all states
    weather_data = {}
    for state in TARGET_STATES:
        print(f"Fetching weather for {state}...")

        # Last month actual weather
        last_month = fetch_weather_last_month(state)
        
        # Fallback: use historical average if current month fetch fails
        if not last_month:
            print(f"  Weather API failed for {state}, using historical average...")
            current_month = datetime.today().month
            last_month = get_seasonal_weather_avg(state, current_month)

        # Next 3 months forecast/estimate
        month1_weather = fetch_weather_forecast(
            state, month_offset=1)
        month2_weather = fetch_weather_forecast(
            state, month_offset=2)
        month3_weather = fetch_weather_forecast(
            state, month_offset=3)

        weather_data[state] = {
            "last_month": last_month,
            "month1"    : month1_weather,
            "month2"    : month2_weather,
            "month3"    : month3_weather,
        }

    # Combine into per state-vegetable records
    for veg in VEGETABLES:
        result[veg] = {}
        for state in TARGET_STATES:
            price   = price_data[veg].get(state)
            weather = weather_data.get(state, {})
            
            # Use fallback price if not in live data
            if not price:
                price = get_fallback_price(veg, state)
                if price:
                    fallback_used = True

            # Allow data if we have price (live or fallback)
            # Use fallback weather if needed
            if price:
                weather_last = weather.get("last_month")
                if not weather_last:
                    # Try fallback weather
                    current_month = datetime.today().month
                    weather_last = get_seasonal_weather_avg(state, current_month)
                
                # Include data even if weather is incomplete
                result[veg][state] = {
                    "modal_price": price["modal_price"],
                    "min_price"  : price["min_price"],
                    "max_price"  : price["max_price"],
                    "weather_last": weather_last,
                    "weather_m1"  : weather.get("month1"),
                    "weather_m2"  : weather.get("month2"),
                    "weather_m3"  : weather.get("month3"),
                }
                available_states.add(state)
            else:
                failed_states.add(state)

    # Track metadata
    metadata = {
        "available_states": sorted(list(available_states)),
        "failed_states": sorted(list(failed_states)),
        "fetched_at": datetime.now().isoformat(),
        "fallback_used": fallback_used
    }

    print("\nData fetch complete")
    print(f"Vegetables fetched : {len(result)}")
    print(f"Available states   : {len(available_states)}/18")
    if fallback_used:
        print("⚠️  Fallback to historical data used")
    if failed_states:
        print(f"Failed states      : {', '.join(failed_states)}")
    for veg in result:
        print(f"  {veg}: {len(result[veg])} states")

    return result, metadata


if __name__ == "__main__":
    data = fetch_all_data()
    print("\nSample result:")
    for veg in data:
        for state in list(data[veg].keys())[:2]:
            print(f"\n{veg} — {state}:")
            print(f"  Price  : ₹{data[veg][state]['modal_price']}/kg")
            print(f"  Weather: {data[veg][state]['weather_last']}")
