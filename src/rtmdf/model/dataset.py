import polars as pl
import torch
from torch.utils.data import Dataset

from rtmdf.model.spec import BaseModelSpec


class FullDataset(Dataset):
    """Load everything to memory."""

    _source_data: pl.DataFrame

    def __init__(self, df: pl.LazyFrame, model_spec: BaseModelSpec):
        self._source_data = df.select(pl.exclude("partition_id")).collect()
        self.cols_x = model_spec.cols_x
        self.cols_y = model_spec.cols_y
        self.cols_w = model_spec.cols_w

        # Nope, cannot pre-transform training data. Will OOM.
        #  Instead, we use the online-training code `sample_data_for_online_learning()`
        #  to fetch a subset of the training data as each "epoch".
        # if MODEL_VERSION in {6}:
        #     self._source_data = self._source_data.with_columns(
        #         pl.col(f"feature_{i:02d}")
        #         .fill_nan(None)
        #         .mean()
        #         .over(self.index)
        #         .alias(f"mean_feature_{i:02d}")
        #         .cast(pl.Float32)
        #         for i in range(79)
        #     )

    def __len__(self) -> int:
        return len(self._source_data)

    def __getitem__(self, idx: int | slice) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        # if isinstance(idx, int):
        #     return pl.DataFrame(
        #         [self._source_data.row(idx)], schema=list(zip(self._source_data.columns, self._source_data.dtypes))
        #     )
        # if isinstance(idx, slice):
        #     start, stop, step = idx.start, idx.stop, idx.step
        subset = self._source_data[idx].fill_nan(0).fill_null(strategy="zero")
        if "responder_6" in subset.columns:
            return (
                subset.select(self.cols_x).to_torch(dtype=pl.Float32),
                subset.select(self.cols_y).to_torch(dtype=pl.Float32),
                subset.select(self.cols_w).to_torch(dtype=pl.Float32),
            )
        else:
            return (
                subset.select(self.cols_x).to_torch(dtype=pl.Float32),
                None,
                subset.select(self.cols_w).to_torch(dtype=pl.Float32),
            )

    @property
    def columns(self) -> list[str]:
        """Data source's columns."""
        return self._source_data.columns
