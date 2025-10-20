# ✈️ Flight Delay Prediction

Machine learning model that predicts flight delays using historical data from New York airports. Achieves **59.33% accuracy** using logistic regression.

## 🚀 Quick Start

### Installation
```bash
pip install pandas numpy scikit-learn streamlit plotly matplotlib joblib
```

### Usage
```bash
# Generate visualizations
python eda.py

# Train the model
python model.py

# Launch dashboard
streamlit run dashboard.py
```

## 📁 Project Structure

```
├── etl.py              # Data loading and cleaning
├── model.py            # Model training
├── dashboard.py        # Streamlit dashboard
├── eda.py              # Data visualizations
├── artifacts/          # Saved models
└── dashboards/         # Generated charts
```

## 🎯 Features

- Binary classification (delayed >15 min vs on-time)
- Interactive Streamlit dashboard with confusion matrix
- Plotly visualizations for exploratory analysis
- Logistic regression with balanced classes

**Model Features:**
- Distance, scheduled arrival time, month, day, carrier

## 📊 Performance

- **Accuracy**: 59.33%
- **Model**: Logistic Regression
- **Threshold**: 15 minutes

## 🔮 Future Improvements

- Feature engineering (time of day, seasons, weekends)
- Try Random Forest or Gradient Boosting
- Add weather data
- Hyperparameter tuning

## 👤 Author

Hilda Amadu - [@Hildegarht1](https://github.com/hildegarht1)
