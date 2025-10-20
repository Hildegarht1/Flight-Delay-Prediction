# dashboard.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report

# ------------------------
# 1. Load data
# ------------------------
st.title("✈️ Flight Delay Prediction Dashboard")

df = pd.read_csv("ny-flights.csv")

# Drop rows with missing values in features or target
required_cols = ['arr_delay', 'distance', 'sched_arr_time', 'month', 'day', 'carrier']
df = df.dropna(subset=required_cols)

# Standardize column names
df.columns = df.columns.str.strip().str.lower()

st.subheader("📊 Sample of Cleaned Data")
st.dataframe(df.head(10))

# ------------------------
# 2. Load trained model, scaler, and metadata
# ------------------------
model = joblib.load("artifacts/flight_delay_model.pkl")
scaler = joblib.load("artifacts/scaler.pkl")
trained_columns = joblib.load("artifacts/trained_columns.pkl")
delay_threshold = joblib.load("artifacts/delay_threshold.pkl")

st.info(f"**Delay Definition:** Flights with arrival delay > {delay_threshold} minutes are considered delayed")

# ------------------------
# 3. Prepare features
# ------------------------
# Create binary target
df['is_delayed'] = (df['arr_delay'] > delay_threshold).astype(int)

# Display class distribution
col1, col2 = st.columns(2)
with col1:
    st.metric("On-Time Flights", f"{(df['is_delayed']==0).sum()}")
with col2:
    st.metric("Delayed Flights", f"{(df['is_delayed']==1).sum()}")

# Select features
X = df[['distance', 'sched_arr_time', 'month', 'day', 'carrier']]

# One-hot encode categorical columns
X = pd.get_dummies(X, columns=['carrier'], drop_first=True)

# Add missing columns with 0s
for col in trained_columns:
    if col not in X.columns:
        X[col] = 0

# Ensure same order as during training
X = X[trained_columns]

# Scale features
X_scaled = scaler.transform(X)

# ------------------------
# 4. Make predictions
# ------------------------
y_true = df['is_delayed']
y_pred = model.predict(X_scaled)

accuracy = accuracy_score(y_true, y_pred)
st.subheader("🎯 Model Performance")
st.metric("Model Accuracy", f"{accuracy*100:.2f}%")

# ------------------------
# 5. Confusion Matrix
# ------------------------
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=['On-Time', 'Delayed'])
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap='Blues')
ax.set_title("Confusion Matrix")
st.pyplot(fig)

# Classification report
st.subheader("📈 Classification Report")
report = classification_report(y_true, y_pred, target_names=['On-Time', 'Delayed'], output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())

# ------------------------
# 6. Sample Predictions
# ------------------------
df_preds = df.copy()
df_preds['predicted_delay'] = y_pred
df_preds['prediction_label'] = df_preds['predicted_delay'].map({0: 'On-Time', 1: 'Delayed'})
df_preds['actual_label'] = df_preds['is_delayed'].map({0: 'On-Time', 1: 'Delayed'})
df_preds['correct'] = (df_preds['predicted_delay'] == df_preds['is_delayed'])

st.subheader("🔍 Sample Predictions")
display_cols = ['carrier', 'flight', 'sched_arr_time', 'arr_delay', 'actual_label', 'prediction_label', 'correct']
st.dataframe(df_preds[display_cols].head(20))

# Show prediction statistics
correct_preds = df_preds['correct'].sum()
total_preds = len(df_preds)
st.success(f"✅ Correct Predictions: {correct_preds}/{total_preds} ({correct_preds/total_preds*100:.2f}%)")