import altair as alt
import polars as pl
from IPython.display import display


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
    chart.encoding.x.title = "Date"
    chart.encoding.y.title = col
    chart.display()
    chart = (
        subset.plot.line(x="time_id", y=alt.Y(f"{col}_std:Q").scale(domainMin=0.0), opacity="opacity")
        .properties(width=1024)
        .properties(title=f'"{col}" volatility over dates for symbol {symbol_id}')
    )
    chart.encoding.x.title = "Date"
    chart.encoding.y.title = col
    chart.display()
