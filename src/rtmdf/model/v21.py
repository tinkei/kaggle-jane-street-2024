import polars as pl
import torch
from darts.models import TSMixerModel
from darts.utils.callbacks import TFMProgressBar
from darts.utils.likelihood_models import QuantileRegression
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.preprocessing import FunctionTransformer
from torch import nn

# from rtmdf.constant import SEED
from rtmdf.constant import SEED
from rtmdf.model.spec import BaseModelSpec


class ModelSpecV21(BaseModelSpec):

    def __init__(self):
        super().__init__()

        # Config inputs, if they are different from default.
        self._cols_x = self._cols_x

        # Config targets, if they are different from default.
        self._cols_y = self._responders

        # PyTorch model.
        self._model: TSMixerModel = TSMixerModel(
            **self.create_params(
                input_chunk_length=120,
                output_chunk_length=120,
                full_training=True,
            ),
            use_static_covariates=True,
            model_name="tsm_v21",
        )

    @property
    def model(self) -> nn.Module:
        """PyTorch neural network for prediction."""
        return self._model.model

    @property
    def device(self) -> str:
        """Get PyTorch device."""
        # return self._device
        raise NotImplementedError("Darts' model is just a wrapper.")

    @device.setter
    def device(self, device: str):
        """Set PyTorch device."""
        # self._model = self._model.to(device)
        # self._device = device
        raise NotImplementedError("Darts' model is just a wrapper.")

    def create_params(
        self,
        input_chunk_length: int,
        output_chunk_length: int,
        full_training=True,
    ) -> dict:
        # Early stopping: this setting stops training once the the validation loss has not decreased by more than 1e-5 for 10 epochs.
        # early_stopper = EarlyStopping(
        #     monitor="val_loss",
        #     patience=10,
        #     min_delta=1e-5,
        #     mode="min",
        # )

        # PyTorch Lightning Trainer arguments (you can add any custom callback).
        if full_training:
            limit_train_batches = None
            limit_val_batches = None
            max_epochs = 1000  # 200
            batch_size = 256
        else:
            limit_train_batches = 20
            limit_val_batches = 10
            max_epochs = 5
            batch_size = 64

        # Only show the training and prediction progress bars.
        progress_bar = TFMProgressBar(enable_sanity_check_bar=False, enable_validation_bar=False)
        # logger = TensorBoardLogger(save_dir=os.getcwd(), version=1, name="lightning_logs")
        pl_trainer_kwargs = {
            "gradient_clip_val": 1,
            "max_epochs": max_epochs,
            "limit_train_batches": limit_train_batches,
            "limit_val_batches": limit_val_batches,
            "accelerator": "auto",
            # "callbacks": [early_stopper, progress_bar],
            "callbacks": [progress_bar],
            # "logger": logger,
        }

        # Optimizer setup, uses Adam by default.
        # optimizer_cls = torch.optim.Adam
        optimizer_kwargs = {
            "lr": 1e-4,
        }

        # Learning rate scheduler.
        lr_scheduler_cls = torch.optim.lr_scheduler.ExponentialLR
        lr_scheduler_kwargs = {"gamma": 0.999}

        # For probabilistic models, we use quantile regression, and set `loss_fn` to `None`.
        likelihood = QuantileRegression()
        loss_fn = None
        # likelihood = None
        # loss_fn = r_square_loss

        return {
            "input_chunk_length": input_chunk_length,  # Lookback window.
            "output_chunk_length": output_chunk_length,  # Forecast/lookahead window.
            "ff_size": 384,
            "hidden_size": 256,
            "num_blocks": 2,
            "activation": "GELU",
            "dropout": 0.1,
            "use_reversible_instance_norm": True,
            "optimizer_kwargs": optimizer_kwargs,
            "pl_trainer_kwargs": pl_trainer_kwargs,
            "lr_scheduler_cls": lr_scheduler_cls,
            "lr_scheduler_kwargs": lr_scheduler_kwargs,
            "likelihood": likelihood,  # Use a `likelihood` for probabilistic forecasts.
            "loss_fn": loss_fn,  # Use a `loss_fn` for determinsitic model.
            "save_checkpoints": True,  # Checkpoint to retrieve the best performing model state.
            "force_reset": True,  # If set to True, any previously-existing model with the same name will be reset (all checkpoints will be discarded). Default: False.
            "batch_size": batch_size,
            "random_state": SEED,
            "add_encoders": {
                # Add cyclic time axis encodings as future covariates.
                # https://unit8co.github.io/darts/generated_api/darts.dataprocessing.encoders.encoders.html
                "cyclic": {
                    "future": ["microsecond"],
                },
            },
        }

    def eval_loss_train(
        self, X: torch.Tensor, y: torch.Tensor, w: torch.Tensor, to_device: bool = True
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Evaluate prediction loss for training."""
        raise NotImplementedError("V20 and above uses PyTorch Lightning Trainer instead of our custom implementation.")

    def eval_loss_test(
        self, X: torch.Tensor, y: torch.Tensor, w: torch.Tensor, to_device: bool = True
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Evaluate prediction loss for test set."""
        raise NotImplementedError("V20 and above uses PyTorch Lightning Trainer instead of our custom implementation.")

    def log_loss_train(self, cum_loss: float, cum_named_losses: dict[str, float], sum_batch_sizes: int = 1) -> str:
        """Return formatted training loss to be printed."""
        raise NotImplementedError("V20 and above uses PyTorch Lightning Trainer instead of our custom implementation.")

    def log_loss_test(self, cum_loss: float, cum_named_losses: dict[str, float], sum_batch_sizes: int = 1) -> None:
        """Print test loss."""
        raise NotImplementedError("V20 and above uses PyTorch Lightning Trainer instead of our custom implementation.")

    def predict(self, X: torch.Tensor, to_device: bool = True) -> pl.DataFrame:
        """Predict "responder_6" given input. Returns a single-columned DataFrame "predict"."""
        if to_device:
            X = X.to(self.device)
        y_pred = self._model(X)

        pred = pl.DataFrame(y_pred.detach().cpu().numpy(), schema=[f"responder_{i:01d}" for i in range(9)])
        pred = pred.select(pl.col("responder_6").alias("predict"))
        return pred  # Always a single column.

    def transform_source(self, df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
        """Transform data source before splitting into inputs and targets."""
        # Flatten
        return df
