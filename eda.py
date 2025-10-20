# eda.py
import os
import pandas as pd
import plotly.express as px
import plotly.io as pio
from etl import load_data, clean_data

# Ensure charts render in notebook (optional)
pio.renderers.default = "browser"

# --------------------------
# Load and clean data
# --------------------------
df = load_data()
df = clean_data(df)

# Standardize column names
df.columns = df.columns.str.strip().str.lower()

# Output folder for dashboards
output_dir = "dashboards"
os.makedirs(output_dir, exist_ok=True)

# --------------------------
# 1. Total distance by carrier
# --------------------------
fig1 = px.bar(
    df.groupby('carrier')['distance'].sum().reset_index(),
    x='carrier',
    y='distance',
    color='carrier',
    title='Total Distance Flown by Carrier'
)
fig1.write_html(os.path.join(output_dir, "distance_by_carrier.html"))
fig1.write_image(os.path.join(output_dir, "distance_by_carrier.png"))

# --------------------------
# 2. Monthly distance trend
# --------------------------
if 'month' in df.columns:
    fig2 = px.line(
        df.groupby(['month', 'carrier'])['distance'].sum().reset_index(),
        x='month',
        y='distance',
        color='carrier',
        title='Monthly Distance Flown per Carrier'
    )
    fig2.write_html(os.path.join(output_dir, "monthly_distance_trend.html"))
    fig2.write_image(os.path.join(output_dir, "monthly_distance_trend.png"))

# --------------------------
# 3. Delay vs Scheduled Time
# --------------------------
fig3 = px.scatter(
    df,
    x='sched_arr_time',
    y='arr_delay',
    color='carrier',
    size='distance',
    hover_data=['carrier', 'flight'],
    title='Flight Delay vs Scheduled Arrival Time'
)
fig3.write_html(os.path.join(output_dir, "delay_vs_sched_time.html"))
fig3.write_image(os.path.join(output_dir, "delay_vs_sched_time.png"))

print(f"✅ Dashboards saved in HTML and PNG format in: {os.path.abspath(output_dir)}")
