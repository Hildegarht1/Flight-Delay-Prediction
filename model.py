# model.py (Enhanced Version)
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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
# 2. Feature Engineering
# --------------------------
print("\n🔧 Engineering features...")

# Time-based features
df['hour'] = (df['sched_arr_time'] // 100).astype(int)
df['minute'] = (df['sched_arr_time'] % 100).astype(int)

# Create time of day categories
df['time_of_day'] = pd.cut(df['hour'],
                            bins=[0, 6, 12, 18, 24],
                            labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                            include_lowest=True)

# Weekend indicator
df['is_weekend'] = df['day'].isin([6, 7]).astype(int)

# Season
df['season'] = pd.cut(df['month'],
                       bins=[0, 3, 6, 9, 12],
                       labels=['Winter', 'Spring', 'Summer', 'Fall'])

# Distance categories
df['distance_category'] = pd.cut(df['distance'],
                                  bins=[0, 500, 1000, 2000, 5000],
                                  labels=['Short', 'Medium', 'Long', 'VeryLong'])

# --------------------------
# 3. Define target variable
# --------------------------
DELAY_THRESHOLD = 15
df['is_delayed'] = (df['arr_delay'] > DELAY_THRESHOLD).astype(int)

print(f"\n📊 Class distribution:")
print(df['is_delayed'].value_counts())
print(f"Delayed flights: {df['is_delayed'].sum()} ({df['is_delayed'].mean()*100:.1f}%)")

# --------------------------
# 4. Select features
# --------------------------
# Original + engineered features
features = [
    'distance', 'hour', 'minute', 'month', 'day',
    'carrier', 'time_of_day', 'is_weekend', 'season', 'distance_category'
]

df_model = df[features + ['is_delayed']].copy()

# Encode categorical features
cat_cols = df_model.select_dtypes(include=['object', 'category']).columns.tolist()
if cat_cols:
    df_model = pd.get_dummies(df_model, columns=cat_cols, drop_first=True)
    print(f"✅ Encoded categorical columns: {cat_cols}")

X = df_model.drop(columns=['is_delayed'])
y = df_model['is_delayed']

print(f"\n📐 Feature shape: {X.shape}")
print(f"Number of features: {X.shape[1]}")

# --------------------------
# 5. Split and scale data
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------
# 6. Train multiple models and compare
# --------------------------
print("\n🚀 Training models...\n")

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', max_depth=10)
}

best_model = None
best_score = 0
best_name = ""

for name, model in models.items():
    print(f"Training {name}...")

    if name == 'Random Forest':
        # Random Forest doesn't need scaling
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    else:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)

    # Cross-validation score
    cv_scores = cross_val_score(model, X_train_scaled if name != 'Random Forest' else X_train,
                                  y_train, cv=5, scoring='accuracy')

    print(f"  Test Accuracy: {acc:.3f}")
    print(f"  CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    print()

    if acc > best_score:
        best_score = acc
        best_model = model
        best_name = name

print(f"✅ Best model: {best_name} with accuracy {best_score:.3f}\n")

# --------------------------
# 7. Evaluate best model
# --------------------------
if best_name == 'Random Forest':
    y_pred = best_model.predict(X_test)
else:
    y_pred = best_model.predict(X_test_scaled)

print("="*50)
print("FINAL MODEL EVALUATION")
print("="*50)
print(f"\nAccuracy: {best_score:.3f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['On-Time', 'Delayed']))

# Feature importance (if Random Forest)
if best_name == 'Random Forest':
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n📊 Top 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))

# --------------------------
# 8. Save artifacts
# --------------------------
output_dir = "artifacts"
os.makedirs(output_dir, exist_ok=True)

trained_columns = X.columns.tolist()

joblib.dump(best_model, os.path.join(output_dir, "flight_delay_model.pkl"))
joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))
joblib.dump(trained_columns, os.path.join(output_dir, "trained_columns.pkl"))
joblib.dump(DELAY_THRESHOLD, os.path.join(output_dir, "delay_threshold.pkl"))
joblib.dump(best_name, os.path.join(output_dir, "model_name.pkl"))

print(f"\n💾 Model artifacts saved in: {os.path.abspath(output_dir)}")
print(f"   - Model type: {best_name}")
print(f"   - Delay threshold: {DELAY_THRESHOLD} minutes")
print(f"   - Number of features: {len(trained_columns)}")