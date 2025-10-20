import pandas as pd
import plotly.express as px
import plotly.io as pio
from etl import load_data, clean_data

#'browser' if running as script, 'notebook' if inside Jupyter
pio.renderers.default = "browser"

# Load and clean data
df = load_data()
df = clean_data(df)

# Standardize column names
df.columns = df.columns.str.strip().str.lower()

# Determine which column to use as airline equivalent
airline_col = None
for col in ['airline', 'carrier', 'flight']:
    if col in df.columns:
        airline_col = col
        break

if airline_col is None:
    raise ValueError("❌ No suitable airline column found (expected 'Airline', 'Carrier', or 'Flight').")

# --------------------------
# 1. Distance by Airline/Carrier (Bar Chart)
# --------------------------
fig1 = px.bar(
    df.groupby(airline_col)['distance'].sum().reset_index(),
    x=airline_col,
    y='distance',
    color=airline_col,
    title=f'Total Distance Flown by {airline_col.capitalize()}'
)
fig1.show()

# --------------------------
# 2. Monthly Distance Trend (if Month column exists)
# --------------------------
if 'month' in df.columns:
    fig2 = px.line(
        df.groupby(['month', airline_col])['distance'].sum().reset_index(),
        x='month',
        y='distance',
        color=airline_col,
        title=f'Monthly Distance Flown per {airline_col.capitalize()}'
    )
    fig2.show()

# --------------------------
# 3. Delay vs Scheduled Time (Scatter Plot)
# --------------------------
if 'sched_arr_time' in df.columns and 'arr_delay' in df.columns:
    fig3 = px.scatter(
        df,
        x='sched_arr_time',
        y='arr_delay',
        color=airline_col,
        size='distance' if 'distance' in df.columns else None,
        hover_data=[airline_col],
        title='Flight Delay vs Scheduled Time'
    )
    fig3.show()

else:
    print("⚠️ Missing 'sched_arr_time' or 'arr_delay' columns, skipping scatter plot.")


# --------------------------
# Save dashboards as HTML and PNG
# --------------------------
import os

# Create a folder to store your dashboards
output_dir = "dashboards"
os.makedirs(output_dir, exist_ok=True)

# Save interactive dashboards as HTML
fig1.write_html(os.path.join(output_dir, "distance_by_airline.html"))
if 'month' in df.columns:
    fig2.write_html(os.path.join(output_dir, "monthly_trend.html"))
if 'scheduledtime' in df.columns and 'delayed' in df.columns:
    fig3.write_html(os.path.join(output_dir, "delay_vs_scheduledtime.html"))

# Optionally save as static PNG images (requires kaleido)
try:
    import kaleido  # noqa
    fig1.write_image(os.path.join(output_dir, "distance_by_airline.png"))
    if 'month' in df.columns:
        fig2.write_image(os.path.join(output_dir, "monthly_trend.png"))
    if 'scheduledtime' in df.columns and 'delayed' in df.columns:
        fig3.write_image(os.path.join(output_dir, "delay_vs_scheduledtime.png"))
except Exception as e:
    print(f"⚠️ Skipping PNG export: {e}")

print(f"✅ Dashboards saved in: {os.path.abspath(output_dir)}")
