import torch

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
