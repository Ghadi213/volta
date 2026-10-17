# ⚡ VOLTA — Predicting Tomorrow's Electricity Price

VOLTA is a time-series forecasting project that predicts Spain's actual electricity price for each of the next 24 hours.

We trained and compared LSTM and GRU models on historical prices, tested them against simple baselines, and built an interactive Streamlit dashboard that shows the 24-hour forecast and suggests the cheapest time to run a device.

## Results

The final model is an **LSTM with an `is_weekend` feature**, trained on 2015–2017 and tested on 2018, a year it never saw.

| Model | Test MAE (€/MWh) |
|---|---|
| **Final LSTM** | **4.03** |
| Lag-24 (same hour yesterday) | 5.21 |
| Lag-168 (same hour last week) | 6.39 |
| Last value | 7.44 |

On average, the forecast is about 4 € away from the real price, which is **22.5% more accurate than the best baseline (Lag-24)**.

## Data

- **Source:** [Kaggle — Energy consumption, generation, prices and weather in Spain](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather)
- **Period:** 2015–2018, hourly (35,064 hours)
- **Target:** `price actual` (€/MWh)

## Approach

- **Input:** the previous 168 hours (7 days) of prices, plus `is_weekend`
- **Output:** 24 hourly prices, forecast once a day at 00:00 UTC
- **Split by time :** train 2015–2016, validation 2017, test 2018
- **Model selection:** LSTM vs GRU, feature engineering, then hyperparameter tuning.

## Project Structure

```
volta/
├─ notebooks/
│  ├─ data_quality.ipynb                        Data quality checks on the raw data
│  ├─ data_preparation.ipynb                    Cleaning; creates data/processed/volta_clean_hourly.csv
│  ├─ eda.ipynb                                 Exploratory data analysis
│  ├─ feature_experiments.ipynb                 LSTM/GRU feature experiments
│  └─ final_model.ipynb                         Final modeling: baselines, LSTM/GRU, tuning, test 2018
├─ streamlit_app/                               Interactive dashboard
├─ README.md
└─ requirements.txt
```

## How to Run

```bash
pip install -r requirements.txt

# Dashboard
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

The dataset is not in the repo. Download it from Kaggle into `data/raw/`.

## Team

Rahaf Alrehaili · Ayah Qalo · Mohammed Alyami · Ghadi Alrashdi
