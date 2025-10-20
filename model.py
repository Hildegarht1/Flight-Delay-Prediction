# model.py
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from etl import load_data, clean_data

# --------------------------
# 1. Load and preprocess data
# --------------------------
print("📦 Loading data...")
df = load_data()
df = clean_data(df)

# Standardize column names
df.columns = df.columns.str.strip().str.lower()

# --------------------------
# 2. Feature selection
# --------------------------
target = 'arr_delay'

# Keep only relevant features
features = ['distance', 'sched_arr_time', 'month', 'day', 'carrier']  # drop flight numbers
df = df[[*features, target]]

# --------------------------
# 3. Encode categorical features
# --------------------------
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
if cat_cols:
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    print(f"✅ Encoded categorical columns: {cat_cols}")

X = df.drop(columns=[target])
y = df[target]

# --------------------------
# 4. Split and scale data
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------
# 5. Train logistic regression model
# --------------------------
print("🚀 Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# --------------------------
# 6. Evaluate the model
# --------------------------
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

print("\n✅ Model Evaluation Results:")
print(f"Accuracy: {acc:.3f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --------------------------
# 7. Save model and scaler
# --------------------------
output_dir = "artifacts"
os.makedirs(output_dir, exist_ok=True)

joblib.dump(model, os.path.join(output_dir, "flight_delay_model.pkl"))
joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))

print(f"\n💾 Model and scaler saved in: {os.path.abspath(output_dir)}")
