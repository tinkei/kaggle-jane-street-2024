from typing import Optional

import altair as alt
import polars as pl
from IPython.display import display
from tqdm import tqdm


def plot_column_given_date_symbol(df: pl.LazyFrame, col: str, symbol_id: int, date_id: int) -> None:
    subset = (
        df.filter(pl.col("symbol_id") == symbol_id, pl.col("date_id") == date_id)
        .select("date_id", "time_id", "symbol_id", col)
        .collect()
    )
    subset = subset.with_columns(opacity=0.5)
    if "responder" in col:
        y_data = alt.Y(f"{col}:Q").scale(domain=(-5, 5), clamp=True)
    else:
        y_data = alt.Y(f"{col}:Q")
    chart = (
        subset.plot.line(x="time_id", y=y_data, opacity="opacity")
        .properties(width=1024)
        .properties(title=f'"{col}" for symbol {symbol_id} on date {date_id} over time')
    )
    chart.encoding.x.title = "Time"
    chart.encoding.y.title = col
    chart.display()


def plot_column_given_time_symbol(df: pl.LazyFrame, col: str, symbol_id: int, time_id: int) -> None:
    subset = (
        df.filter(pl.col("symbol_id") == symbol_id, pl.col("time_id") == time_id)
        .select("date_id", "time_id", "symbol_id", col)
        .collect()
    )
    subset = subset.with_columns(opacity=0.5)
    if "responder" in col:
        y_data = alt.Y(f"{col}:Q").scale(domain=(-5, 5), clamp=True)
    else:
        y_data = alt.Y(f"{col}:Q")
    chart = (
        subset.plot.line(x="date_id", y=y_data, opacity="opacity")
        .properties(width=1024)
        .properties(title=f'"{col}" for symbol {symbol_id} at time {time_id} over different dates')
    )
    chart.encoding.x.title = "Date"
    chart.encoding.y.title = col
    chart.display()


def plot_column_given_symbol_averaged_over_date(
    df: pl.LazyFrame,
    col: str,
    symbol_id: int,
) -> None:
    subset = df.filter(pl.col("symbol_id") == symbol_id).select("date_id", "time_id", "symbol_id", col).collect()
    subset = subset.group_by("time_id").agg(
        pl.col(col).mean().name.suffix("_mean"), pl.col(col).std().name.suffix("_std")
    )
    subset = subset.with_columns(opacity=0.5).sort("time_id")
    display(subset)
    scale = subset.select(pl.col(f"{col}_mean").abs()).max().item()
    chart = (
        subset.plot.line(
            x="time_id", y=alt.Y(f"{col}_mean:Q").scale(domain=(-scale, scale), clamp=True), opacity="opacity"
        )
        .properties(width=1024)
        .properties(title=f'"{col}" averaged over date for symbol {symbol_id}')
    )
    chart.encoding.x.title = "Time"
    chart.encoding.y.title = col
    chart.display()
    chart = (
        subset.plot.line(x="time_id", y=alt.Y(f"{col}_std:Q").scale(domainMin=0.0), opacity="opacity")
        .properties(width=1024)
        .properties(title=f'"{col}" volatility over dates for symbol {symbol_id}')
    )
    chart.encoding.x.title = "Time"
    chart.encoding.y.title = col
    chart.display()


def plot_column_given_symbol_averaged_over_time(
    df: pl.LazyFrame,
    col: str,
    symbol_id: int,
) -> None:
    subset = df.filter(pl.col("symbol_id") == symbol_id).select("date_id", "time_id", "symbol_id", col).collect()
    subset = subset.group_by("date_id").agg(
        pl.col(col).mean().name.suffix("_mean"), pl.col(col).std().name.suffix("_std")
    )
    subset = subset.with_columns(opacity=0.5).sort("date_id")
    display(subset)
    scale = subset.select(pl.col(f"{col}_mean").abs()).max().item()
    chart = (
        subset.plot.line(
            x="date_id", y=alt.Y(f"{col}_mean:Q").scale(domain=(-scale, scale), clamp=True), opacity="opacity"
        )
        .properties(width=1024)
        .properties(title=f'"{col}" averaged over time for symbol {symbol_id}')
    )
    chart.encoding.x.title = "Date"
    chart.encoding.y.title = col
    chart.display()
    chart = (
        subset.plot.line(x="date_id", y=alt.Y(f"{col}_std:Q").scale(domainMin=0.0), opacity="opacity")
        .properties(width=1024)
        .properties(title=f'"{col}" daily volatility for symbol {symbol_id}')
    )
    chart.encoding.x.title = "Date"
    chart.encoding.y.title = col
    chart.display()


def plot_lagged_correlations_given_symbol(
    df: pl.LazyFrame,
    col: str,
    symbol_id: int,
    lag: int = 3,
    from_date: Optional[int] = None,
    to_date: Optional[int] = None,
) -> pl.DataFrame:

    lag_n_corrs = []
    for i in range(lag):
        lag_n_corrs.append([])

    dates = df.select("date_id").collect().unique().sort(by="date_id").to_series(0).to_list()
    if from_date is not None and to_date is not None:
        dates = dates[from_date : to_date + 1]
    elif from_date is not None:
        dates = dates[from_date:]
    elif to_date is not None:
        dates = dates[: to_date + 1]

    for date_id in tqdm(dates[lag + 1 :]):
        for i in range(lag + 1):
            if i == 0:
                subset = (
                    df.filter(pl.col("symbol_id") == symbol_id, pl.col("date_id") == date_id - i)
                    .select("time_id", "symbol_id", pl.col(col).alias(f"{col}_D-{i}"))
                    .collect()
                )
            else:
                subset_i = (
                    df.filter(pl.col("symbol_id") == symbol_id, pl.col("date_id") == date_id - i)
                    .select("time_id", "symbol_id", pl.col(col).alias(f"{col}_D-{i}"))
                    .collect()
                )
                subset = subset.join(subset_i, on=["time_id", "symbol_id"], how="inner")
        for i in range(lag):
            lag_n_corrs[i].append(subset.select(pl.corr(f"{col}_D-0", f"{col}_D-{i + 1}")).item())

    subset = pl.DataFrame(
        [
            dates[lag + 1 :],
            *lag_n_corrs,
        ],
        schema=["date_id"] + [f"lag_{i + 1}" for i in range(lag)],
    )
    display(subset)

    subset_unpivot = subset.unpivot(index="date_id").with_columns(opacity=0.5)
    try:
        alt.data_transformers.disable_max_rows()
        chart = (
            subset_unpivot.plot.line(x="date_id", y="value:Q", color="variable", opacity="opacity")
            .properties(width=1024)
            .properties(title=f'Correlation of lagged "{col}" for symbol {symbol_id}')
        )
        chart.encoding.x.title = "Date"
        chart.encoding.y.title = col
        chart.encoding.color.title = "Lags"
        chart.display()
    except:
        print("Plot failed.")
    return subset
