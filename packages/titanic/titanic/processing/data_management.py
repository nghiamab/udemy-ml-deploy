import pandas as pd
import joblib
from sklearn.pipeline import Pipeline

from packages.titanic.titanic.config import config
from titanic import __version__ as _version

import logging


def load_dataset(*, file_name: str) -> pd.DataFrame:
    _data = pd.read_csv(f"{config.TRAINING_DATA_FILE}/{file_name}")
    return _data
