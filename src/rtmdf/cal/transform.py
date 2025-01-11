import polars as pl


def append_mean_features(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    """Append features averaged over Date x Time cross sections (i.e. over Symbols)."""
    df = df.with_columns(
        pl.col(f"feature_{idx:02d}")
        .fill_nan(None)
        .mean()
        .over(["date_id", "time_id"])
        .alias(f"mean_feature_{idx:02d}")
        .cast(pl.Float32)
        for idx in range(79)
    )
    return df


def append_lagged_features(df: pl.DataFrame | pl.LazyFrame, lag_time: int) -> pl.DataFrame | pl.LazyFrame:
    """Append features lagged over Date x Symbol cross sections (i.e. over Time)."""
    df = df.with_columns(
        pl.col(f"feature_{idx:02d}")
        .shift(lag_time)
        .over(partition_by=["date_id", "symbol_id"], order_by=["time_id"])
        .alias(f"feature_{idx:02d}_lag_{lag_time:d}")
        for idx in range(79)
    )
    return df
