from collections.abc import Iterator
from time import time

import polars as pl

from rtmdf.constant import FEATURES, INDEX, RESPONDERS


def serve_data(df: pl.LazyFrame) -> Iterator[tuple[pl.DataFrame, pl.DataFrame | None]]:
    """Serve test and lags data like the submission API does."""
    unique_dates = df.select("date_id").unique().sort(by=["date_id"]).collect().to_series(0)
    for date_id in unique_dates:
        time_per_date = time()
        subset = df.filter(pl.col("date_id") == date_id).collect()
        unique_times = subset.select("time_id").unique().sort(by=["time_id"]).to_series(0)
        for time_id in unique_times:
            lags = None
            if time_id == 0:
                lags = (
                    df.filter(pl.col("date_id") == date_id - 1)
                    .select(INDEX + RESPONDERS)
                    .rename({f"responder_{idx}": f"responder_{idx}_lag_1" for idx in range(9)})
                    .collect()
                )
                if lags.is_empty():
                    lags = None
                else:
                    lags = lags.with_columns(pl.lit(date_id).alias("date_id"))
            test = (
                subset.filter(pl.col("time_id") == time_id)
                .select(INDEX + ["weight", pl.lit(True).alias("is_scored")] + FEATURES)
                .with_row_index(name="row_id")
            )
            yield test, lags
        print(f"Processing date {date_id} took {time() - time_per_date:.2f}s.")


if __name__ == "__main__":
    from pathlib import Path

    INPUT_PATH = Path("")  # TODO: Input your own path to the data sources.
    train_df = pl.scan_parquet(INPUT_PATH / "train.parquet")
    sample_test = pl.read_parquet(INPUT_PATH / "test.parquet")
    sample_lags = pl.read_parquet(INPUT_PATH / "lags.parquet")
    for test, lags in serve_data(train_df.filter(pl.col("date_id") >= 1696)):
        assert test.columns == sample_test.columns
        if lags is not None:
            assert lags.columns == sample_lags.columns
        # TODO: Call your own `predict()` function, the one you're passing to
        #  `inference_server = kaggle_evaluation.jane_street_inference_server.JSInferenceServer(predict)`.
