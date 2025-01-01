# Jane Street Real-Time Market Data Forecasting

My code for ["Jane Street Real-Time Market Data Forecasting" Kaggle competition](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/).

## Set up

- `git clone https://github.com/tinkei/kaggle-jane-street-2024.git` and `cd .\kaggle-jane-street-2024` to the root of this repository.
- [Download a current version of Python](https://www.python.org/downloads/). In this project we use Python 3.12 (pinned in `pyproject.toml`).
- Create virtual environment once: `& "$env:HomeDrive$env:HomePath\AppData\Local\Programs\Python\Python312\python.exe" -m venv $env:HomeDrive$env:HomePath\venvs\rtmdf`
- Activate virtual environment in PowerShell: `& "$env:HomeDrive$env:HomePath\venvs\rtmdf\Scripts\Activate.ps1"`
- Install this repository and all its dependencies to the active virtual environment: `pip install -e .`
