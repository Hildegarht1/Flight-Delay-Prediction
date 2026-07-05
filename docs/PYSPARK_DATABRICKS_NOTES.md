# PySpark and Databricks Notes

This project originally uses pandas and scikit-learn for local development. The
`pyspark_pipeline.py` file adds a Spark version of the data preparation layer so
the same workflow can be discussed in a big-data or Databricks context.

## What the PySpark pipeline does

- Reads the raw flight CSV with Spark.
- Standardizes column names.
- Checks that required columns exist.
- Removes rows with missing critical values.
- Casts numeric columns to stable types.
- Creates model-ready features such as hour, minute, time of day, season,
  distance category, and the binary delayed label.
- Writes a curated output dataset in Parquet by default.

## How this maps to Databricks

In Databricks, the same pipeline could run as a notebook or scheduled job. The
input path could be DBFS, S3, ADLS, or a Unity Catalog volume. The output could
be written as a Delta table instead of Parquet:

```bash
spark-submit pyspark_pipeline.py \
  --input /Volumes/raw/flights/ny-flights.csv \
  --output /Volumes/gold/flights/features \
  --format delta
```

A production version would add orchestration, data quality checks, monitoring,
alerts, and tests around the transformations.

## Interview wording

The honest way to present this is:

> The main project was first built with pandas, scikit-learn, and Streamlit. I
> then added a PySpark version of the ETL and feature-engineering layer to show
> how I would move the same data workflow toward a Spark or Databricks
> environment.

