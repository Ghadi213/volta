# VOLTA – Streamlit App

Predicting tomorrow's electricity price in Spain. The app runs the final model from
`notebooks/04_models.ipynb` (LSTM + `is_weekend`, trained on 2015–2017, tested on 2018).
Every number in the app is computed from the model's predictions.

## Folder structure

```
streamlit_app/
├─ app.py                      The app: forecast & device optimizer, forecast insights, about
├─ requirements.txt
├─ .streamlit/config.toml      Theme (presentation colours)
├─ data/volta_clean_hourly.csv Hourly dataset
├─ models/
│  ├─ volta_lstm_final.keras   Final LSTM
│  └─ volta_scaler_final.pkl   MinMaxScaler fitted on 2015–2017
└─ volta/
   ├─ forecast.py              Model inputs, predictions, metrics, find_cheapest_hours (no Streamlit)
   └─ cache.py                 Loads data/model once and caches the predictions
```

## Run locally

```
pip install -r requirements.txt
streamlit run app.py
```

Opens at http://localhost:8501. The first load takes a few seconds while TensorFlow
loads and the model predicts every day.

## Deploy (Streamlit Community Cloud)

1. Push to GitHub.
2. On share.streamlit.io: **Create app** → repository `Ghadi213/volta`, branch `main`,
   main file `streamlit_app/app.py`.
3. In **Advanced settings** choose Python 3.12, then **Deploy**.

## Updating the model

Save the new model and scaler from the notebook into `streamlit_app/models/` with the
same file names. If the inputs change (window length, extra features), update
`volta/forecast.py` so it builds the inputs exactly like the notebook.
