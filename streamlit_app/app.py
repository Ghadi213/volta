import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from volta import cache, forecast as fc

# ---------------------------------------------------------
# Page Configuration & High-End Dark Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="VOLTA · Energy Price Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .stApp {
        background-color: #0b0f19;
        color: #f3f4f6;
    }
    
    /* Metrics Styling */
    div[data-testid="stMetricValue"] {
        font-size: 26px !important;
        font-weight: 700 !important;
        color: #10b981 !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 13px !important;
        color: #9ca3af !important;
    }
    
    /* Custom Card for Device Optimizer */
    .device-card {
        background-color: #111827;
        border: 1px solid #1f2937;
        padding: 18px;
        border-radius: 12px;
        margin-bottom: 15px;
    }
    
    /* Recommendation Output Box */
    .recommendation-box {
        background-color: #064e3b;
        border: 1px solid #10b981;
        padding: 16px;
        border-radius: 10px;
        margin-top: 15px;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Data & Model: the final LSTM from notebooks/final_model.ipynb
# (same data, model and scaler as app.py, through volta/cache.py)
# ---------------------------------------------------------
MODEL_COLUMNS = {
    "LSTM (Best)": "predicted",
    "Lag-24 baseline (yesterday)": "lag_24",
    "Lag-168 baseline (last week)": "lag_168",
}

all_predictions = cache.predictions()
FIRST_DAY = all_predictions["day"].dt.date.min()
LAST_DAY = all_predictions["day"].dt.date.max()


def get_24h_predictions(selected_date, model_choice="LSTM (Best)"):
  """The 24 forecast hours (00:00-23:00 UTC) of one day, plus the real price."""
  day = cache.day_forecast(selected_date)
  return pd.DataFrame({
      "Hour_Label": day["timestamp"].dt.strftime("%H:%M"),
      "Hour": day["timestamp"].dt.hour,
      "Price_EUR": day[MODEL_COLUMNS[model_choice]].round(2),
      "Actual_EUR": day["actual"].round(2),
  })


# One date shared by every page: picking a day on one page keeps it on the others
st.session_state.setdefault("selected_day", datetime.date(2018, 7, 15))


def date_picker(label, container=st):
  key = f"date_{label}"

  def remember_day():
    st.session_state["selected_day"] = st.session_state[key]

  return container.date_input(
      label,
      value=st.session_state["selected_day"],
      min_value=FIRST_DAY,
      max_value=LAST_DAY,
      key=key,
      on_change=remember_day,
  )


# Algorithm to calculate the cheapest consecutive duration for device running
def find_cheapest_window(df, power_kw, duration_hours):
  prices = df["Price_EUR"].values
  n = len(prices)
  duration_hours = min(int(duration_hours), n)

  min_cost = float("inf")
  best_start_idx = 0

  for i in range(n - duration_hours + 1):
    window_prices = prices[i : i + duration_hours]
    # Total cost = sum of (price in EUR/MWh / 1000) * power_kw
    window_cost = np.sum((window_prices / 1000.0) * power_kw)
    if window_cost < min_cost:
      min_cost = window_cost
      best_start_idx = i

  best_end_idx = best_start_idx + duration_hours
  start_hour = df.loc[best_start_idx, "Hour_Label"]
  end_hour = f"{best_end_idx:02d}:00" if best_end_idx < 24 else "24:00"

  avg_window_price = np.mean(prices[best_start_idx:best_end_idx])

  return {
      "time_window": f"{start_hour} – {end_hour}",
      "estimated_cost": min_cost,
      "avg_price": avg_window_price,
  }


# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------
# Purple bolt from assets/logo.svg (the old icon was an external link that did not load)
BOLT_ICON = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 28 40">'
    '<polygon points="15,2 3,23 12,23 9,38 24,15 15,15 19,2" fill="#8B5CF6"/></svg>'
)
st.sidebar.image(BOLT_ICON, width=40)
st.sidebar.title("VOLTA System")

st.sidebar.success("🟢 Final LSTM model loaded")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "1. ⚡ Forecast & Device Optimizer",
        "2. 📊 Forecast Insights",
        "3. ℹ️ About VOLTA",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption("© 2026 VOLTA · Smart Electricity Price Forecasting")

# =========================================================
# PAGE 1: FORECAST & DEVICE OPTIMIZER
# =========================================================
if page == "1. ⚡ Forecast & Device Optimizer":
  st.title("⚡ Predict Energy. Plan Smarter.")
  st.markdown(
      "Forecast Spain's electricity prices, and find the best time to run your"
      " devices."
  )
  st.markdown("---")

  # Header Inputs Row
  col_date, col_model = st.columns(2)
  with col_date:
    selected_date = date_picker("Select date:")
  with col_model:
    model_choice = st.selectbox("Model:", list(MODEL_COLUMNS))

  df_day = get_24h_predictions(selected_date, model_choice)

  min_p_row = df_day.loc[df_day["Price_EUR"].idxmin()]
  max_p_row = df_day.loc[df_day["Price_EUR"].idxmax()]
  avg_p = df_day["Price_EUR"].mean()

  # Metric Summary Row
  m1, m2, m3 = st.columns(3)
  m1.metric(
      "Lowest Price",
      f"€{min_p_row['Price_EUR']:.1f} /MWh",
      f"at {min_p_row['Hour_Label']}",
  )
  m2.metric(
      "Highest Price",
      f"€{max_p_row['Price_EUR']:.1f} /MWh",
      f"at {max_p_row['Hour_Label']}",
      delta_color="inverse",
  )
  m3.metric("Average Price", f"€{avg_p:.1f} /MWh", "Next 24 hours", delta_color="off")

  st.markdown("<br>", unsafe_allow_html=True)

  # Graph & Device Optimizer Section
  col_chart, col_device = st.columns([2.2, 1])

  with col_chart:
    st.subheader("📈 Predicted Electricity Price (Next 24 Hours)")
    # product view: show the forecast only (the real price is not known in advance)
    chart_df = df_day.rename(columns={"Price_EUR": model_choice})
    fig_line = px.line(
        chart_df,
        x="Hour_Label",
        y=[model_choice],
        markers=True,
        color_discrete_sequence=["#10b981"],
        labels={"value": "Price (€/MWh)", "Hour_Label": "Hour (UTC)", "variable": ""},
    )
    fig_line.update_traces(line_width=3, selector={"name": model_choice})
    fig_line.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=380,
        margin=dict(l=10, r=10, t=20, b=20),
        legend=dict(orientation="h", y=1.1, x=0),
    )
    st.plotly_chart(fig_line, width="stretch")

  with col_device:
    st.markdown(
        """
        <div class="device-card">
            <h4 style="margin:0; color:#10b981;">💡 Find the Best Time to Run Your Device</h4>
            <p style="font-size: 12px; color: #9ca3af; margin-top: 5px;">Enter your device details and VOLTA will suggest the cheapest time to run it.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # bottom alignment keeps both boxes level even when one label wraps
    pow_col, dur_col = st.columns(2, vertical_alignment="bottom")
    with pow_col:
      power_kw = st.number_input(
          "Device power (kW)", min_value=0.1, max_value=50.0, value=2.0, step=0.5
      )
    with dur_col:
      duration_hours = st.number_input(
          "Duration (hours)", min_value=1, max_value=24, value=3, step=1
      )

    if st.button("🔍 Find Best Time", width="stretch"):
      res = find_cheapest_window(df_day, power_kw, duration_hours)

      st.markdown(
          f"""
            <div class="recommendation-box">
                <span style="background-color: #10b981; color: black; padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 11px;">Lowest cost</span>
                <p style="margin-top: 6px; margin-bottom: 2px; color: #d1d5db; font-size: 12px;">⏰ Recommended Time</p>
                <h3 style="color: #ffffff; margin: 0;">{res['time_window']}</h3>
                <hr style="border-color: #059669; margin: 8px 0;">
                <div style="display: flex; justify-content: space-between;">
                    <div>
                        <p style="margin:0; font-size: 11px; color: #a7f3d0;">Estimated Cost</p>
                        <h4 style="margin:0; color: #ffffff;">€{res['estimated_cost']:.2f}</h4>
                    </div>
                    <div>
                        <p style="margin:0; font-size: 11px; color: #a7f3d0;">Average Price</p>
                        <h4 style="margin:0; color: #ffffff;">€{res['avg_price']:.1f} /MWh</h4>
                    </div>
                </div>
            </div>
            """,
          unsafe_allow_html=True,
      )

# =========================================================
# PAGE 2: FORECAST INSIGHTS
# =========================================================
elif page == "2. 📊 Forecast Insights":
  st.title("📊 Forecast Insights & Breakdown")
  st.markdown(
      "The cheapest and most expensive forecast hours of the selected day."
      " Hours are grouped into low, moderate and high prices compared with"
      " that day's average."
  )
  st.markdown("---")

  selected_date = date_picker("Filter Date:", st.sidebar)
  df_day = get_24h_predictions(selected_date)

  p_mean = df_day["Price_EUR"].mean()
  p_std = df_day["Price_EUR"].std()

  def cat(p):
    if p > p_mean + 0.6 * p_std:
      return "High Price"
    elif p < p_mean - 0.6 * p_std:
      return "Low Price"
    else:
      return "Moderate"

  df_day["Category"] = df_day["Price_EUR"].apply(cat)

  c1, c2 = st.columns(2)
  with c1:
    st.markdown("### 🟢 Top 5 Cheapest Hours")
    st.dataframe(
        df_day.sort_values(by="Price_EUR").head(5)[
            ["Hour_Label", "Price_EUR", "Category"]
        ],
        width="stretch",
        hide_index=True,
    )
  with c2:
    st.markdown("### 🔴 Top 5 Peak Hours")
    st.dataframe(
        df_day.sort_values(by="Price_EUR", ascending=False).head(5)[
            ["Hour_Label", "Price_EUR", "Category"]
        ],
        width="stretch",
        hide_index=True,
    )

  st.markdown("<br>", unsafe_allow_html=True)
  fig_bar = px.bar(
      df_day,
      x="Hour_Label",
      y="Price_EUR",
      color="Category",
      color_discrete_map={
          "Low Price": "#10b981",
          "Moderate": "#f59e0b",
          "High Price": "#ef4444",
      },
      labels={"Price_EUR": "Price (€/MWh)", "Hour_Label": "Hour (UTC)"},
      # keep the hours in time order (00:00 -> 23:00), not grouped by colour
      category_orders={"Hour_Label": [f"{h:02d}:00" for h in range(24)]},
  )
  fig_bar.update_layout(
      template="plotly_dark",
      paper_bgcolor="rgba(0,0,0,0)",
      plot_bgcolor="rgba(0,0,0,0)",
      height=380,
  )
  st.plotly_chart(fig_bar, width="stretch")

# =========================================================
# PAGE 3: ABOUT VOLTA
# =========================================================
elif page == "3. ℹ️ About VOLTA":
  st.title("ℹ️ About VOLTA Project")

  test = all_predictions[all_predictions["split"] == "Test (2018)"]
  scores = fc.score(test)
  lstm_mae = scores.loc["Final LSTM", "MAE"]
  lag24_mae = scores.loc["Lag-24 (same hour yesterday)", "MAE"]
  gain = scores.loc["Lag-24 (same hour yesterday)", "LSTM improvement (%)"]

  st.markdown(f"""
    ### ⚡ Executive Summary
    **VOLTA** forecasts Spain's actual electricity price (`price actual`) for each of the next 24 hours, so people can run their devices in the cheapest hours.

    ---

    ### 🧠 Machine Learning Model
    * **Model:** Keras LSTM + `is_weekend` feature (168h input window $\\rightarrow$ 24h forecast).
    * **Training:** 2015–2017 · **Test:** 2018 (never seen during training).
    * **Performance (Test 2018):** **MAE = {lstm_mae:.2f} EUR/MWh** vs Lag-24 baseline MAE = {lag24_mae:.2f} EUR/MWh — **{gain:.1f}% more accurate**.
    """)

  st.markdown("### 📊 Test 2018: Model vs Baselines")
  # The LSTM is not compared with itself, so show "—" instead of an empty value
  table = scores.round(2)
  table["LSTM improvement (%)"] = table["LSTM improvement (%)"].map(
      lambda v: "—" if pd.isna(v) else f"{v:.1f}%"
  )
  st.dataframe(
      table,
      width="stretch",
      column_config={
          "MAE": st.column_config.NumberColumn("MAE (EUR/MWh)", format="%.2f"),
          "RMSE": st.column_config.NumberColumn("RMSE (EUR/MWh)", format="%.2f"),
          "LSTM improvement (%)": st.column_config.TextColumn(),
      },
  )
