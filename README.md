# Fasal Bhav — Climate-Driven Vegetable Price Prediction

> Predicting next-month wholesale price direction for Tomato, Onion and Potato
> across 18 Indian states using weather data and mandi market signals.

---

## Problem Statement

Agricultural commodity prices in India fluctuate **30–50% within a single season**,
driven by unpredictable rainfall, supply shocks, and market dynamics. Farmers lack
reliable tools to decide **when to sell** their produce, leading to distress selling
and significant income loss.

This project builds a **data-driven price forecasting system** using real government
mandi data combined with climate signals — giving farmers and traders a monthly
price direction signal backed by machine learning.

---

## Results

| Vegetable | Accuracy | Directional Accuracy | Test Rows |
|-----------|----------|----------------------|-----------|
| Tomato    | 59%      | 76% ✅               | 237       |
| Onion     | 56%      | 49% ⚠️               | 246       |
| Potato    | 66%      | 46% ⚠️               | 264       |
| **Overall**   | **61%**  | **60%**              | **747**   |

> **Directional Accuracy** measures whether the model correctly predicts
> UP or DOWN price movement, excluding STABLE predictions.
> Random baseline = 33%. Our model = 61% — **85% above random.**

---

## Sample Forecast Output

| Vegetable | State         | Current Price | Prediction  | Confidence |
|-----------|---------------|---------------|-------------|------------|
| Tomato    | Karnataka     | ₹45/kg        | ↑ UP        | High 77%   |
| Tomato    | Maharashtra   | ₹38/kg        | ↓ DOWN      | High 89%   |
| Onion     | Maharashtra   | ₹25/kg        | ↓ DOWN      | High 88%   |
| Onion     | Rajasthan     | ₹28/kg        | ↓ DOWN      | High 90%   |
| Potato    | Uttar Pradesh | ₹16/kg        | ↑ UP        | High 84%   |
| Potato    | West Bengal   | ₹18/kg        | ↑ UP        | High 91%   |

---

## Data Sources

| Source | What We Collected | Coverage |
|--------|-------------------|----------|
| [AGMARKNET](https://agmarknet.gov.in) | Wholesale modal price + arrival quantity | 18 states, Jan 2022 – Dec 2025 |
| [Open-Meteo API](https://open-meteo.com) | Temperature max/min, rainfall, humidity | Daily historical, free API |

### States Covered (18)
Maharashtra, Uttar Pradesh, West Bengal, Karnataka, Andhra Pradesh,
Tamil Nadu, Madhya Pradesh, Gujarat, Rajasthan, Punjab, Haryana,
Odisha, Himachal Pradesh, Chhattisgarh, Telangana, Kerala,
Uttarakhand, Bihar

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10 | Core language |
| LightGBM | Main ML model |
| Pandas / NumPy | Data processing |
| Scikit-learn | Evaluation metrics |
| Matplotlib / Seaborn | Visualizations |
| Open-Meteo API | Weather data collection |
| Google Colab | Training environment |

---

## Pipeline

```
Raw Data Collection          Cleaning & Merging         Feature Engineering
─────────────────            ──────────────────         ───────────────────
AGMARKNET price data    →    Filter 18 states      →    Price lag ratios
Open-Meteo weather           3 vegetables                Rainfall deviation
18 states coverage           Forward fill gaps           Producer state signal
Jan 2022 – Dec 2025          Merge on state + date       Cyclical time encoding
                             2216 rows, 0 missing        38 total features

Model Training               Evaluation                 Prediction Output
──────────────               ──────────────             ─────────────────
LightGBM multiclass →        Time-based split      →    UP / DOWN / STABLE
Single model                 Train 2022-2024             + Confidence %
38 input features            Test 2025                   Per state per vegetable
Balanced class weights       61% overall accuracy        Monthly forecast
Custom STABLE threshold      76% Tomato directional
```

---

## Key Features Engineered

| Feature | Description | Why It Matters |
|---------|-------------|----------------|
| `price_lag_1m` | Price 1 month ago | Captures recent momentum |
| `price_lag_4m` | Price 4 months ago | Captures seasonal cycle |
| `lag1_ratio` | Current / lag1 price | Direction signal normalized |
| `rainfall_deviation` | Actual vs normal rainfall | Weather shock detection |
| `prod_price_Tomato_lag1` | Karnataka Tomato price last month | Cross-state spillover signal |
| `price_volatility_3m` | Rolling 3-month price std | Uncertainty signal |
| `price_change_abs` | Absolute price change | STABLE vs moving detection |
| `month_sin / month_cos` | Cyclical month encoding | Seasonal patterns |
| `is_post_monsoon` | Sep-Oct-Nov flag | Transition period signal |
| `price_norm` | Price normalized within vegetable | Removes scale differences |

---

## Model Details

```
Algorithm          : LightGBM Multiclass Classifier
Classes            : UP / DOWN / STABLE (±8% threshold)
Features           : 38 input features
Training period    : May 2022 – December 2024
Test period        : January 2025 – November 2025
Train rows         : 2347
Test rows          : 747
Best iteration     : 350
Class weights      : Balanced + STABLE boosted manually
Prediction rule    : Custom STABLE threshold (not plain argmax)
```

---

## Repository Structure

```
fasal-bhav/
│
├── README.md                          
├── requirements.txt                   ← Root dependencies
├── .gitignore
│
├── notebook/
│   └── LightGBM.ipynb                ← Full pipeline notebook
│
├── UI/                                ← Streamlit web application
│   ├── app.py                         ← Main Streamlit dashboard
│   ├── fetch_and_predict.py           ← Data fetching & predictions
│   ├── predict_3month.py              ← 3-month forecast logic
│   ├── requirements.txt               ← UI dependencies
│   ├── data/
│   │   └── master_dataset.csv         ← Historical data for UI
│   └── models/
│       ├── lgbm_final.txt             ← Trained LightGBM model
│       ├── le_target.pkl              ← Label encoder
│       ├── cat_mappings.pkl           ← Category mappings
│       └── feature_cols.csv           ← Feature list
│
├── data/                              ⚠️ Excluded from git
│   └── master_dataset.csv            
│
├── models/                            ⚠️ Excluded from git
│   └── *.pkl, *.txt                   
│
└── outputs/                           ⚠️ Excluded from git
    └── Results and visualizations
```

---

## How to Run

### Option 1: Interactive Dashboard (Recommended for Users)

**1. Clone and setup**
```bash
git clone https://github.com/YOUR_USERNAME/fasal-bhav.git
cd fasal-bhav/UI
pip install -r requirements.txt
```

**2. Run the Streamlit app**
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` — browse vegetable prices,
view 3-month forecasts, and check confidence scores interactively.

---

### Option 2: Jupyter Notebook (For Model Exploration)

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/fasal-bhav.git
cd fasal-bhav
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Open the notebook**
```bash
jupyter notebook notebook/LightGBM.ipynb
```

**4. Run all cells**

The notebook runs end to end — data loading, feature engineering,
model training, and evaluation.

---

## Dashboard Features

The **Streamlit UI** (`UI/app.py`) provides an interactive interface for farmers and traders:

- 📊 **Real-time Price Browse** — Current prices across 18 states for Tomato, Onion, Potato
- 🔮 **3-Month Forecast** — View predicted price movements (UP/DOWN/STABLE) with confidence scores
- 🗺️ **State-wise Comparison** — Compare prices and predictions across regions
- 📈 **Price Trends** — Historical price charts with seasonal patterns
- 🎯 **Confidence Indicators** — Model confidence for each prediction
- ⚡ **Fresh Data Fetch** — Option to fetch latest AGMARKNET and weather data

**Tech Stack:** Streamlit + LightGBM + Open-Meteo API + Pandas

---

## Deploy on Streamlit Cloud

### Step 1: Push to GitHub
```bash
git push origin main
```

### Step 2: Connect to Streamlit Cloud
1. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
2. Click **"New app"**
3. Select your repository: `vegetable-price-prediction-india`
4. Choose branch: `main`
5. Set main file path to: `UI/app.py`
6. Click **"Deploy"**

### Step 3: (Optional) Set Environment Variables
If you need custom AGMARKNET API keys:
1. In Streamlit Cloud dashboard, go to **"Manage app"**
2. Click **"Secrets"**
3. Add:
   ```
   DATA_GOV_API_KEY = "your_api_key_here"
   ```

**Your app will be live in ~2-3 minutes!** 🚀

---

## Key Findings

**Tomato is most predictable (76% directional accuracy)**
Price follows clear weather patterns — Karnataka production shocks
ripple across all states within 4–6 weeks. Monsoon and summer
seasons show 76–78% accuracy.

**Onion is hardest to predict (49% directional accuracy)**
Government policy interventions — export bans, buffer stock releases,
import decisions — override all weather and supply signals. This is
a genuine real-world limitation, not a model failure.

**Post-monsoon months (Sep–Nov) are hardest across all vegetables**
Supply transitions from kharif to rabi season create extreme price
volatility that is difficult to predict even with weather signals.

**Weather features confirmed necessary for long-term prediction**
Rainfall deviation and temperature signals contribute meaningfully
to accuracy, especially for Tomato. Removing weather features
drops Tomato accuracy by 4–6%.

---

## What Makes This Project Different

| Standard Student Project | This Project                                   |
|--------------------------|------------------------------------------------|
| Kaggle dataset           | Real government AGMARKNET data                 |
| Single commodity         | 3 vegetables, 18 states                        |
| No domain knowledge      | Producer state spillover signal                |
| Random train-test split  | Time-based split (no data leakage)             |
| Generic features         | 38 engineered domain-specific features         |
| Accuracy only            | Directional accuracy + per-vegetable breakdown |
| No leakage check         | Explicit leakage verification at each step     |

---

## Future Scope

- Add 5+ vegetables: Brinjal, Okra, Cauliflower, Green Chilli, Cabbage
- Expand to all 28 states with sufficient mandi data
- ✅ ~~Build Streamlit dashboard~~ (Completed — see UI/ folder)
- Add SHAP explainability for feature interpretation
- Integrate government policy event flags (export ban, MSP changes)
- Deploy as REST API for integration with agricultural advisory systems
- Add real-time price notifications via SMS/email for price threshold alerts
- Multi-language UI for regional farmer accessibility

