# Jane Street Real-Time Market Data Forecasting

My code for ["Jane Street Real-Time Market Data Forecasting" Kaggle competition](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/).

## Set up (Windows PowerShell)

- `git clone https://github.com/tinkei/kaggle-jane-street-2024.git` and `cd .\kaggle-jane-street-2024` to the root of this repository.
- [Download a current version of Python](https://www.python.org/downloads/). In this project we use Python 3.12 (pinned in `pyproject.toml`).
- Create virtual environment once: `& "$env:HomeDrive$env:HomePath\AppData\Local\Programs\Python\Python312\python.exe" -m venv $env:HomeDrive$env:HomePath\venvs\rtmdf`
- Activate virtual environment in PowerShell: `& "$env:HomeDrive$env:HomePath\venvs\rtmdf\Scripts\Activate.ps1"`
- Install this repository and all its dependencies to the active virtual environment: `pip install -e .`

## Set up (Kaggle)

- Attach the Kaggle Datasets [RTMDF | Code](https://www.kaggle.com/datasets/tinkei/rtmdf-code) and [RTMDF | Model](https://www.kaggle.com/datasets/tinkei/rtmdf-model) to your Kaggle Notebook.
- `sys.path.append("/kaggle/input/rtmdf-code/src/")`
- `MODEL_PATH = Path(f"/kaggle/input/rtmdf-model/v{VERSION:02d}/model_version={MODEL_VERSION:02d}")`

## Overview

To no one's surprise, simple 3-layered models again dominates the leaderboard in a time series competition.
Here is a list of ideas that I deemed intellectually more worthwhile to explore:
[What are your brightest ideas (that didn't pan out)?](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/discussion/556553)
And these are what you will find in this repository.

Most models in this repository explore the engineering of elaborate loss functions and auxiliary training targets to assist a barebone ResNet to hone in on [physical relationships](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/discussion/555562) among various noisy signals. (Spoiler: <s>None of that panned out.</s>)

## Model architecture

All NN models implement the `BaseModelSpec` abstract class, which defines the model parameters, data transforms, and loss calculations.
All models &lt;v20 are pure PyTorch implementations for easy experimentation.
LGBM models failed so miserably I didn't bother to organize them here.

- LGBM models (not listed): Worked well during last year's Optiver competition. Not this year.
  - Ensemble partitioned over ranges of `date_id`.
  - Apply a single gradient boosting model to all responders (which normally can only predict a single output): Add a `responder_num` as input feature, duplicate the other features, then train the model with a single flattened responders target.
  - Train a per-feature-tag ensemble. Not the worst idea.
- Model versions 1-4: 3-4 layer MLP, regressing the responders directly.
  - Regressing all responders performs significantly better than regressing only responder 6.
  - Version 2 is already my second best submission. Everything that follows is futile.
- Model versions 5-13: 10-20 layer ResNet.
  - Version 8 is my best submission, a marginal 0.0002 better than MLP.
- Model versions 20+: TSMixer.
  - Scored 0.0000. Likely timed out due to predict's 1-minute limit.
