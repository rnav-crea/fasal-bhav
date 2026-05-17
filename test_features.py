"""
Unit tests for feature engineering module.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add UI directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'UI'))

from features import build_features, get_season, get_producer_latest

def test_get_season():
    """Test season mapping function."""
    assert get_season(6) == "Kharif"   # June
    assert get_season(7) == "Kharif"   # July
    assert get_season(8) == "Kharif"   # August
    assert get_season(9) == "Kharif"   # September
    assert get_season(10) == "Kharif"  # October
    assert get_season(11) == "Rabi"    # November
    assert get_season(12) == "Rabi"    # December
    assert get_season(1) == "Rabi"     # January
    assert get_season(2) == "Rabi"     # February
    assert get_season(3) == "Rabi"     # March
    assert get_season(4) == "Zaid"     # April
    assert get_season(5) == "Zaid"     # May

def test_get_producer_latest():
    """Test producer state data extraction."""
    # Create mock historical data
    df_hist = pd.DataFrame({
        'vegetable': ['Tomato', 'Tomato', 'Tomato'],
        'state': ['Karnataka', 'Karnataka', 'Karnataka'],
        'modal_price': [40.0, 42.0, 45.0],
        'arrival_qty': [1000, 1100, 1200],
        'date': pd.date_range('2023-01-01', periods=3)
    })
    
    price, price_lag, arrival = get_producer_latest('Tomato', df_hist)
    assert price == 45.0
    assert price_lag == 42.0
    assert arrival == 1200
    
    # Test with insufficient data
    df_short = pd.DataFrame({
        'vegetable': ['Tomato'],
        'state': ['Karnataka'],
        'modal_price': [45.0],
        'arrival_qty': [1200],
        'date': [pd.Timestamp('2023-01-01')]
    })
    
    price, price_lag, arrival = get_producer_latest('Tomato', df_short)
    assert price is None
    assert price_lag is None
    assert arrival is None
    
    # Test with unknown vegetable
    price, price_lag, arrival = get_producer_latest('Unknown', df_hist)
    assert price is None
    assert price_lag is None
    assert arrival is None

def test_build_features_insufficient_data():
    """Test feature building with insufficient historical data."""
    # Create minimal historical data (less than 5 rows)
    df_hist = pd.DataFrame({
        'state': ['Maharashtra'] * 3,
        'vegetable': ['Tomato'] * 3,
        'modal_price': [40.0, 42.0, 45.0],
        'min_price': [35.0, 38.0, 40.0],
        'max_price': [45.0, 48.0, 50.0],
        'arrival_qty': [1000, 1100, 1200],
        'date': pd.date_range('2023-01-01', periods=3)
    })
    
    # Mock categorical mappings
    cat_mappings = {
        'state': ['Maharashtra'],
        'vegetable': ['Tomato'],
        'season': ['Kharif', 'Rabi', 'Zaid'],
        'season_veg': ['Kharif_Tomato', 'Rabi_Tomato', 'Zaid_Tomato']
    }
    
    # Feature columns (subset for testing)
    feature_cols = [
        'state', 'vegetable', 'season', 'price_norm', 'arrival_norm',
        'lag1_ratio', 'lag4_ratio', 'arrival_ratio', 'price_momentum',
        'price_vs_avg', 'arrival_momentum', 'modal_price', 'price_lag_1m',
        'price_lag_4m', 'rolling_avg_3m', 'price_spread', 'price_position',
        'arrival_qty', 'arrival_lag_1m', 'temp_max', 'temp_min',
        'rainfall_mm', 'humidity', 'rainfall_deviation', 'month_sin',
        'month_cos', 'month_norm_in_veg', 'is_post_monsoon', 'season_veg',
        'price_volatility_3m', 'price_change_abs', 'price_change_abs_pct',
        'prod_price_Tomato', 'prod_price_Tomato_lag1', 'prod_arrival_Tomato',
        'prod_price_Onion', 'prod_price_Onion_lag1', 'prod_arrival_Onion',
        'prod_price_Potato', 'prod_price_Potato_lag1', 'prod_arrival_Potato'
    ]
    
    weather = {
        'temp_max': 32.0,
        'temp_min': 20.0,
        'rainfall_mm': 50.0,
        'humidity': 60.0
    }
    
    # Should return None due to insufficient data (<5 rows)
    result = build_features(
        state='Maharashtra',
        veg='Tomato',
        modal_price=45.0,
        arrival_qty=1200,
        min_price=None,  # Will be derived from history
        max_price=None,  # Will be derived from history
        weather=weather,
        predict_month=6,
        df_hist=df_hist,
        feature_cols=feature_cols,
        cat_mappings=cat_mappings
    )
    
    assert result is None

if __name__ == "__main__":
    test_get_season()
    test_get_producer_latest()
    test_build_features_insufficient_data()
    print("All tests passed!")