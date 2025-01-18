from pathlib import Path

import numpy as np
import polars as pl
import torch

from rtmdf.constant import INDEX
from rtmdf.model.dataset import FullDataset
from rtmdf.model.spec import BaseModelSpec


def train_loop(
    dataset: FullDataset, model_spec: BaseModelSpec, batch_size: int, optimizer: torch.optim.Optimizer, print_every: int
):
    # Set the model to training mode - important for batch normalization and dropout layers.
    model_spec.model.train()
    batch_loss = 0.0
    batch_named_losses = dict()

    for batch in range(0, len(dataset), batch_size):
        # Fetch a training batch from dataset.
        (X, y, w) = dataset[batch : batch + batch_size]

        # Compute prediction and loss.
        loss, named_losses = model_spec.eval_loss_train(X, y, w)
        batch_loss, batch_named_losses = loss.item(), {k: v.item() for k, v in named_losses.items()}

        # Backpropagation.
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (batch + batch_size) % (print_every * batch_size) == 0:
            current = batch + len(X)
            log_str = model_spec.log_loss_train(batch_loss, batch_named_losses, sum_batch_sizes=1)
            print(f"{log_str}  [{current:>5d}/{len(dataset):>5d}]")


def test_loop(dataset: FullDataset, model_spec: BaseModelSpec, batch_size: int):
    # Set the model to evaluation mode - important for batch normalization and dropout layers.
    model_spec.model.eval()
    cum_loss = 0.0
    cum_named_losses = dict()

    # Evaluating the model with `torch.no_grad()` ensures that no gradients are computed during test mode.
    # Also serves to reduce unnecessary gradient computations and memory usage for tensors with `requires_grad=True`.
    with torch.no_grad():
        for batch in range(0, len(dataset), batch_size):
            (X, y, w) = dataset[batch : batch + batch_size]
            loss, named_losses = model_spec.eval_loss_test(X, y, w)
            cum_loss, cum_named_losses = model_spec.accumulate_losses(
                loss, named_losses, cum_loss, cum_named_losses, real_batch_size=len(X)
            )

    model_spec.log_loss_test(cum_loss, cum_named_losses, sum_batch_sizes=len(dataset))


def nn_predict(
    dataset: FullDataset, model_spec: BaseModelSpec, batch_size: int = 100000, return_all: bool = False
) -> pl.DataFrame:
    model_spec.model.eval()
    batch_predicts = []

    # Evaluating the model with `torch.no_grad()` ensures that no gradients are computed during test mode.
    # Also serves to reduce unnecessary gradient computations and memory usage for tensors with `requires_grad=True`.
    with torch.no_grad():
        for batch in range(0, len(dataset), batch_size):
            (X, y, w) = dataset[batch : batch + batch_size]
            if model_spec.version in {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}:
                pred = model_spec.predict(X)
            elif model_spec.version in {11, 12, 13}:
                if return_all:
                    pred = model_spec.predict_custom(X)
                else:
                    pred = model_spec.predict(X)
            else:
                raise NotImplementedError()
            batch_predicts.append(pred)

    return pl.concat(batch_predicts, how="vertical_relaxed")


def sample_data_for_online_learning(
    model_spec: BaseModelSpec,
    train_df: pl.LazyFrame,
    test_df: pl.LazyFrame | None,
    latest_df: pl.LazyFrame | None,
    num_train: int,
    num_test: int,
    sort_output: bool = False,
    epoch: int = 0,
    lag_responders_df: pl.LazyFrame | None = None,
) -> pl.LazyFrame:
    """Sample from each of three input data sources for model online training."""
    samples = []
    model_version = model_spec.version

    USE_CACHE = False  # Always disabled due to performance issues.
    # cache_path = OUTPUT_PATH / "cache" / f"model_version={model_version:02d},sort_output={sort_output},num_train={num_train:04d}"
    # cache_path.mkdir(parents=True, exist_ok=True)
    # epoch_cache_path = cache_path / f"train_transform_cache_epoch={epoch:04d}.parquet"
    epoch_cache_path = Path("")
    if epoch_cache_path.is_file() and USE_CACHE:
        train_subset.scan_parquet(epoch_cache_path)
    else:
        train_dates = train_df.select("date_id").unique().sort(by="date_id").collect()
        train_dates = np.random.choice(train_dates.to_series(0), num_train, replace=False)
        # train_dates_select = np.concatenate([train_dates, train_dates - 1])  # To fetch date-lagged features. No need because we prepared a `lag_responders_df`.
        train_subset = train_df.filter(pl.col("date_id").is_in(train_dates)).select(
            pl.exclude("row_id", "is_scored", "partition_id")
        )
        if model_version in {6, 7, 8, 9, 10, 11, 12, 13}:
            train_subset = model_spec.transform_source(train_subset)
        if model_version in {12}:  # Note: Not else-if here!
            train_subset = model_spec.create_lag_responders_v12(train_subset, lag_responders_df)
            train_subset = model_spec.create_lead_responder_targets_v12(train_subset)
        train_subset = train_subset.filter(pl.col("date_id").is_in(train_dates))
        if USE_CACHE:
            train_subset.collect().write_parquet(epoch_cache_path, compression="zstd", compression_level=7)
    samples.append(train_subset)
    # print(train_subset.collect_schema().names())

    if latest_df is not None:
        latest_subset = latest_df.select(pl.exclude("row_id", "is_scored", "partition_id"))
        if model_version in {6, 7, 8, 9, 10, 11, 12, 13}:
            latest_subset = model_spec.transform_source(latest_subset)
        if model_version in {12}:  # Note: Not else-if here!
            latest_subset = model_spec.create_lag_responders_v12(latest_subset, lag_responders_df)
            latest_subset = model_spec.create_lead_responder_targets_v12(latest_subset)
        samples.append(latest_subset)

    if test_df is not None:
        test_dates = test_df.select("date_id").unique().sort(by="date_id").collect()
        test_dates = np.random.choice(test_dates.to_series(0), num_test, replace=False)
        test_subset = test_df.filter(pl.col("date_id").is_in(test_dates)).select(
            pl.exclude("row_id", "is_scored", "partition_id")
        )
        if model_version in {6, 7, 8, 9, 10, 11, 12, 13}:
            test_subset = model_spec.transform_source(test_subset)
        if model_version in {12}:  # Note: Not else-if here!
            test_subset = model_spec.create_lag_responders_v12(test_subset, lag_responders_df)
            test_subset = model_spec.create_lead_responder_targets_v12(test_subset)
        samples.append(test_subset)

    if latest_df is not None:
        samples.append(latest_subset)

    online_sample = pl.concat(samples, how="vertical_relaxed", rechunk=True)
    if sort_output:
        online_sample = online_sample.sort(by=INDEX)
    # display(online_sample)
    return online_sample
