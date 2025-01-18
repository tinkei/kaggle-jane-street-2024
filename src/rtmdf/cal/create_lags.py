from pathlib import Path

import polars as pl

from rtmdf.constant import INDEX, RESPONDERS


def create_and_cache_lag_responders(df: pl.LazyFrame, parquet_path: Path) -> pl.LazyFrame:
    """
    Create lag-one-date responders somewhat like how the `lags` DataFrame are provided during test time.

    "time_id" goes from 0 to T-1. The corresponding "lag_time_id" goes from T to 1.
    Lagged responders are suffixed with "_shift" to avoid naming conflict with "_lag".
    """
    if not parquet_path.is_dir():
        lag_date = 1
        lag_responders_df = (
            df.select(INDEX + RESPONDERS)
            .with_columns(
                pl.col("time_id")
                .sub(pl.col("time_id").max().over(partition_by=["date_id", "symbol_id"], order_by=["time_id"]).add(1))
                .mul(-1)
                .alias("lag_time_id")
            )
            .with_columns(
                pl.col(f"responder_{idx:d}")
                .shift(lag_date)
                .over(partition_by=["symbol_id", "lag_time_id"], order_by=["date_id"])
                .alias(f"responder_{idx:d}_shift_{lag_date:d}")
                for idx in range(9)
            )
            .select(
                ["date_id", "time_id", "lag_time_id", "symbol_id"]
                + [f"responder_{idx:d}_shift_{lag_date:d}" for idx in range(9)]
            )
            .collect()
        )
        lag_responders_df.write_parquet(
            parquet_path, partition_by=["date_id"], compression="zstd", compression_level=11
        )
    lag_responders_df = pl.scan_parquet(parquet_path)
    return lag_responders_df


if __name__ == "__main__":
    from time import time

    INPUT_PATH = Path("")  # TODO: Input your own path to the data sources.
    OUTPUT_PATH = Path("")  # TODO: Input your own path to the data sources.
    train_df = pl.scan_parquet(INPUT_PATH / "train.parquet")
    parquet_path = OUTPUT_PATH / "lag_responders.parquet"
    time_lag_cache = time()
    lag_responders_df = create_and_cache_lag_responders(df=train_df, parquet_path=parquet_path)
    file_size = sum(f.stat().st_size for f in parquet_path.glob("**/*") if f.is_file()) / 1024 / 1024
    # 37s without write_parquet, 48s with compression level 1, 50s with level 7, 55s with level 1, and 152s(!) with level 22!
    print(f"Creating lag responders took {time() - time_lag_cache:.2f}s and {file_size:.2f}MB.")
