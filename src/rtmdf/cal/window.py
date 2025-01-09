from collections.abc import Iterable

import polars as pl
from polars._typing import IntoExpr


def expanding_window(
    df: pl.DataFrame,
    period: int,
    index_column: str,
    group_by: IntoExpr | Iterable[IntoExpr],
    aggs: IntoExpr | Iterable[IntoExpr] | None = None,
    named_aggs: dict[str, IntoExpr] | None = None,
) -> pl.DataFrame:
    """
    Create an expanding+rolling window over "time_id", after grouping by "symbol_id" and "date_id".

    The windowed column (e.g. "time_id") must be sorted.
    This is a complicated substitute for the simple `rolling()` function, which unfortunately doesn't support group-by.
    """
    if aggs is None and named_aggs is None:
        raise ValueError("Please provide at least one aggregation operation to either `aggs` or `named_aggs`.")

    if named_aggs is None:
        named_aggs = dict()

    smoothed_df = (
        df.with_columns(
            pl.col(index_column).cast(pl.Int32),  # `group_by_dynamic()` supports only Int32 or above.
        )
        .group_by_dynamic(
            index_column=index_column,
            every="1i",
            period=f"{period}i",
            offset=f"-{period}i",
            closed="right",
            label="right",
            group_by=group_by,
            start_by="window",
        )
        .agg(aggs, **named_aggs)
        .with_columns(
            pl.col(index_column).cast(pl.Int16),
        )
        .filter(
            pl.col(index_column).is_in(df.select(index_column).unique())
        )  # Remove excess rows created by rolling window.
    )
    return smoothed_df


if __name__ == "__main__":
    # Compare implementations of `rolling()` vs `group_by_dynamic()`.
    from pathlib import Path

    INPUT_PATH = Path("")  # TODO: Input your own path to the data sources.
    train_df = pl.scan_parquet(INPUT_PATH / "train.parquet")
    subset = train_df.filter((pl.col("date_id") == 1134) & (pl.col("symbol_id") == 23)).collect()

    left = (
        subset.with_columns(
            pl.col("time_id").cast(pl.Int32),
        )
        .with_columns(
            rolling_mean=pl.col("responder_6")
            .mean()
            .rolling(
                index_column="time_id",
                period="5i",
            )
        )
        .select("time_id", "responder_6", "rolling_mean")
        .with_columns(
            pl.col("time_id").cast(pl.Int16),
        )
    )

    right = expanding_window(
        df=subset,
        period=5,
        index_column="time_id",
        group_by=["symbol_id", "date_id"],
        aggs=pl.col("responder_6").mean().alias("rolling_mean2"),
        named_aggs={"rolling_mean3": pl.col("responder_6").mean()},
    ).with_columns(
        pl.col("time_id").cast(pl.Int32),
    )

    comparison = left.join(right, how="outer", on="time_id").filter(
        (pl.col("time_id") < 5) | (pl.col("time_id").is_null())
    )
    display(comparison)
