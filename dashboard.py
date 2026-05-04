import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report

st.set_page_config(page_title="Flight Delay Prediction", layout="wide")

# ------------------------
# 1. Load data and artifacts
# ------------------------
st.title("✈️ Flight Delay Prediction")

df = pd.read_csv("ny-flights.csv")
df.columns = df.columns.str.strip().str.lower()

required_cols = ['arr_delay', 'distance', 'sched_arr_time', 'month', 'day', 'carrier']
df = df.dropna(subset=required_cols).copy()

model = joblib.load("artifacts/flight_delay_model.pkl")
scaler = joblib.load("artifacts/scaler.pkl")
trained_columns = joblib.load("artifacts/trained_columns.pkl")
delay_threshold = joblib.load("artifacts/delay_threshold.pkl")
model_name = joblib.load("artifacts/model_name.pkl")

carrier_choices = sorted(
    col.replace("carrier_", "") for col in trained_columns if col.startswith("carrier_")
)

st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a view", ["Overview", "Predict a Flight", "Visual Dashboards"])

st.info(f"Delay definition: flights with arrival delay greater than {delay_threshold} minutes are labeled delayed.")


def build_feature_frame(input_df: pd.DataFrame) -> pd.DataFrame:
    feature_df = input_df.copy()
    feature_df['hour'] = (feature_df['sched_arr_time'] // 100).astype(int)
    feature_df['minute'] = (feature_df['sched_arr_time'] % 100).astype(int)
    feature_df['is_weekend'] = feature_df['day'].isin([6, 7]).astype(int)

    feature_df['time_of_day'] = pd.cut(
        feature_df['hour'],
        bins=[0, 6, 12, 18, 24],
        labels=['Night', 'Morning', 'Afternoon', 'Evening'],
        include_lowest=True,
    )
    feature_df['season'] = pd.cut(
        feature_df['month'],
        bins=[0, 3, 6, 9, 12],
        labels=['Winter', 'Spring', 'Summer', 'Fall'],
    )
    feature_df['distance_category'] = pd.cut(
        feature_df['distance'],
        bins=[0, 500, 1000, 2000, 5000],
        labels=['Short', 'Medium', 'Long', 'VeryLong'],
    )

    feature_df = pd.get_dummies(
        feature_df[['distance', 'hour', 'minute', 'month', 'day', 'is_weekend', 'carrier', 'time_of_day', 'season', 'distance_category']],
        columns=['carrier', 'time_of_day', 'season', 'distance_category'],
        drop_first=True,
    )

    for col in trained_columns:
        if col not in feature_df.columns:
            feature_df[col] = 0

    return feature_df[trained_columns]


def predict_delays(input_df: pd.DataFrame):
    features = build_feature_frame(input_df)

    if model_name == 'Random Forest':
        predictions = model.predict(features)
        probabilities = model.predict_proba(features)[:, 1]
    else:
        scaled_features = scaler.transform(features)
        predictions = model.predict(scaled_features)
        probabilities = model.predict_proba(scaled_features)[:, 1]

    return pd.Series(predictions, index=input_df.index), pd.Series(probabilities, index=input_df.index)


def render_dashboard_file(title: str, file_name: str) -> None:
    dashboard_path = Path("dashboards") / file_name
    if dashboard_path.exists():
        st.subheader(title)
        components.html(dashboard_path.read_text(encoding="utf-8"), height=700, scrolling=True)
    else:
        st.warning(f"{file_name} not found in dashboards/")


if page == "Overview":
    st.subheader("📊 Sample of Cleaned Data")
    st.dataframe(df.head(10))

    df_overview = df.copy()
    df_overview['is_delayed'] = (df_overview['arr_delay'] > delay_threshold).astype(int)
    y_true = df_overview['is_delayed']
    y_pred, y_prob = predict_delays(df_overview)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("On-Time Flights", f"{(df_overview['is_delayed'] == 0).sum()}")
    with col2:
        st.metric("Delayed Flights", f"{(df_overview['is_delayed'] == 1).sum()}")

    accuracy = accuracy_score(y_true, y_pred)
    st.subheader("🎯 Model Performance")
    st.metric("Model Accuracy", f"{accuracy * 100:.2f}%")

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['On-Time', 'Delayed'])
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues')
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

    st.subheader("📈 Classification Report")
    report = classification_report(y_true, y_pred, target_names=['On-Time', 'Delayed'], output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    df_preds = df_overview.copy()
    df_preds['predicted_delay'] = y_pred
    df_preds['prediction_label'] = df_preds['predicted_delay'].map({0: 'On-Time', 1: 'Delayed'})
    df_preds['actual_label'] = df_preds['is_delayed'].map({0: 'On-Time', 1: 'Delayed'})
    df_preds['correct'] = df_preds['predicted_delay'] == df_preds['is_delayed']

    st.subheader("🔍 Sample Predictions")
    display_cols = ['carrier', 'flight', 'sched_arr_time', 'arr_delay', 'actual_label', 'prediction_label', 'correct']
    st.dataframe(df_preds[display_cols].head(20))

    correct_preds = int(df_preds['correct'].sum())
    total_preds = int(len(df_preds))
    st.success(f"✅ Correct Predictions: {correct_preds}/{total_preds} ({correct_preds / total_preds * 100:.2f}%)")

elif page == "Predict a Flight":
    st.subheader("🔮 Predict a Single Flight")
    st.write("Enter flight details below to get a delay prediction.")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            distance = st.number_input("Distance (miles)", min_value=1, value=500, step=1)
            month = st.selectbox("Month", list(range(1, 13)), index=0)
        with col2:
            sched_arr_time = st.number_input("Scheduled arrival time (HHMM)", min_value=0, max_value=2359, value=1500, step=1)
            day = st.selectbox("Day of month", list(range(1, 32)), index=0)
        with col3:
            carrier = st.selectbox("Carrier", carrier_choices)
            submit = st.form_submit_button("Predict delay")

    if submit:
        input_row = pd.DataFrame([{
            'distance': distance,
            'sched_arr_time': sched_arr_time,
            'month': month,
            'day': day,
            'carrier': carrier,
        }])

        prediction, probability = predict_delays(input_row)
        delayed = int(prediction.iloc[0])
        label = 'Delayed' if delayed == 1 else 'On-Time'

        result_col1, result_col2 = st.columns(2)
        with result_col1:
            if delayed == 1:
                st.error(f"Predicted outcome: {label}")
            else:
                st.success(f"Predicted outcome: {label}")
        with result_col2:
            st.metric("Delay probability", f"{probability.iloc[0] * 100:.2f}%")

        st.caption("Derived model features")
        st.dataframe(build_feature_frame(input_row))

elif page == "Visual Dashboards":
    st.subheader("📊 Exploratory Dashboards")
    st.write("These are the saved Plotly charts generated from the EDA workflow.")

    render_dashboard_file("Top 10 airline / carrier views", "distance_by_airline.html")
    render_dashboard_file("Total distance flown by carrier", "distance_by_carrier.html")
    render_dashboard_file("Monthly trend", "monthly_trend.html")
    render_dashboard_file("Monthly distance trend", "monthly_distance_trend.html")
    render_dashboard_file("Delay vs scheduled arrival time", "delay_vs_sched_time.html")