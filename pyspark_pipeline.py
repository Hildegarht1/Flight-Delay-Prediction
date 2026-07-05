"""PySpark version of the flight delay data preparation workflow.

This script mirrors the pandas ETL/model-preparation logic in the project and is
intended as a small Spark/Databricks-ready pipeline example. It reads raw flight
CSV data, applies basic quality checks and feature engineering, then writes a
curated analytics/modeling dataset.

Local example:
    spark-submit pyspark_pipeline.py --input ny-flights.csv --output data/gold/flights_features

Databricks example:
    Use this file as a notebook/script job and pass DBFS, S3, or volume paths.
"""

from __future__ import annotations

import argparse
from typing import Iterable

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T


REQUIRED_COLUMNS = [
    "arr_delay",
    "dep_delay",
    "distance",
    "sched_arr_time",
    "month",
    "day",
    "carrier",
    "origin",
    "dest",
]

NUMERIC_COLUMNS = {
    "arr_delay": T.DoubleType(),
    "dep_delay": T.DoubleType(),
    "distance": T.DoubleType(),
    "sched_arr_time": T.IntegerType(),
    "month": T.IntegerType(),
    "day": T.IntegerType(),
}

DELAY_THRESHOLD_MINUTES = 15


def build_spark(app_name: str = "flight-delay-pyspark-pipeline") -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )


def standardize_columns(df: DataFrame) -> DataFrame:
    """Normalize source column names so downstream transformations are stable."""
    for column in df.columns:
        normalized = column.strip().lower().replace(" ", "_")
        if normalized != column:
            df = df.withColumnRenamed(column, normalized)
    return df


def require_columns(df: DataFrame, required_columns: Iterable[str]) -> None:
    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def load_raw_flights(spark: SparkSession, input_path: str) -> DataFrame:
    return spark.read.option("header", True).option("inferSchema", True).csv(input_path)


def clean_flights(df: DataFrame) -> DataFrame:
    df = standardize_columns(df)
    require_columns(df, REQUIRED_COLUMNS)

    cleaned = df.dropna(subset=REQUIRED_COLUMNS)
    for column, data_type in NUMERIC_COLUMNS.items():
        cleaned = cleaned.withColumn(column, F.col(column).cast(data_type))

    return cleaned.dropna(subset=list(NUMERIC_COLUMNS.keys()))


def add_model_features(df: DataFrame) -> DataFrame:
    featured = (
        df.withColumn("hour", F.floor(F.col("sched_arr_time") / F.lit(100)).cast("int"))
        .withColumn("minute", (F.col("sched_arr_time") % F.lit(100)).cast("int"))
        .withColumn(
            "is_delayed",
            (F.col("arr_delay") > F.lit(DELAY_THRESHOLD_MINUTES)).cast("int"),
        )
        .withColumn(
            "time_of_day",
            F.when(F.col("hour") < 6, F.lit("Night"))
            .when(F.col("hour") < 12, F.lit("Morning"))
            .when(F.col("hour") < 18, F.lit("Afternoon"))
            .otherwise(F.lit("Evening")),
        )
        .withColumn(
            "season",
            F.when(F.col("month").between(1, 3), F.lit("Winter"))
            .when(F.col("month").between(4, 6), F.lit("Spring"))
            .when(F.col("month").between(7, 9), F.lit("Summer"))
            .otherwise(F.lit("Fall")),
        )
        .withColumn(
            "distance_category",
            F.when(F.col("distance") <= 500, F.lit("Short"))
            .when(F.col("distance") <= 1000, F.lit("Medium"))
            .when(F.col("distance") <= 2000, F.lit("Long"))
            .otherwise(F.lit("VeryLong")),
        )
    )

    return featured.select(
        "carrier",
        "origin",
        "dest",
        "distance",
        "dep_delay",
        "arr_delay",
        "sched_arr_time",
        "hour",
        "minute",
        "month",
        "day",
        "time_of_day",
        "season",
        "distance_category",
        "is_delayed",
    )


def run_pipeline(
    spark: SparkSession, input_path: str, output_path: str, output_format: str
) -> None:
    raw_df = load_raw_flights(spark, input_path)
    gold_df = add_model_features(clean_flights(raw_df))

    delayed_summary = gold_df.groupBy("is_delayed").count().orderBy("is_delayed")
    delayed_summary.show(truncate=False)

    gold_df.write.mode("overwrite").format(output_format).save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare flight delay features with PySpark.")
    parser.add_argument("--input", default="ny-flights.csv", help="Raw flight CSV path")
    parser.add_argument(
        "--output", default="data/gold/flights_features", help="Output dataset path"
    )
    parser.add_argument(
        "--format",
        default="parquet",
        choices=["parquet", "delta"],
        help="Output format. Use delta in Databricks or with delta-spark installed.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    spark_session = build_spark()
    try:
        run_pipeline(spark_session, args.input, args.output, args.format)
    finally:
        spark_session.stop()
