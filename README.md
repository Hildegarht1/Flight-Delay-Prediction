# ✈️ Flight Delay Prediction

Machine learning project that predicts flight delays using historical data from New York airports. The current model uses a random forest classifier and an interactive Streamlit dashboard.

## 📘 Full Documentation

For a detailed explanation of the project, design choices, KPI rationale, tech stack, and interview Q&A, see [docs/PROJECT_DOCUMENTATION.md](docs/PROJECT_DOCUMENTATION.md).

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

### Optional PySpark Pipeline
```bash
# Install Spark dependency
pip install -r requirements-pyspark.txt

# Build a curated feature dataset with PySpark
spark-submit pyspark_pipeline.py --input ny-flights.csv --output data/gold/flights_features
```

The PySpark script mirrors the pandas ETL workflow and prepares the project for
larger-scale Spark or Databricks-style processing. See
[docs/PYSPARK_DATABRICKS_NOTES.md](docs/PYSPARK_DATABRICKS_NOTES.md).

## 📁 Project Structure

```
├── etl.py              # Data loading and cleaning
├── pyspark_pipeline.py # Spark-based feature preparation
├── model.py            # Model training
├── dashboard.py        # Streamlit dashboard
├── eda.py              # Data visualizations
├── artifacts/          # Saved models
└── dashboards/         # Generated charts
```

## 🎯 Features

- Binary classification (delayed >15 min vs on-time)
- Interactive Streamlit dashboard with model metrics, visual dashboards, and single-flight prediction
- Plotly visualizations for exploratory analysis
- Random forest classifier with engineered time, route, and airport features

**Model Features:**
- Distance, departure delay, scheduled arrival time, month, day, carrier, origin, destination, and engineered time features

## 📊 Performance

- **Accuracy**: 88.8% (training evaluation)
- **Model**: Random Forest
- **Threshold**: 15 minutes

## 🔮 Future Improvements

- Feature engineering (time of day, seasons, weekends)
- Try Random Forest or Gradient Boosting
- Add weather data
- Hyperparameter tuning

## 📎 Downloadable Docs

- [Project documentation (Markdown)](docs/PROJECT_DOCUMENTATION.md)
- [Project documentation (PDF)](docs/Flight_Delay_Project_Documentation.pdf)

## 👤 Author

Hilda Amadu - [@Hildegarht1](https://github.com/hildegarht1)
