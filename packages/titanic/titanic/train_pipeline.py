import pandas as pd

from sklearn.model_selection import train_test_split
import joblib

from packages.titanic.titanic import pipeline as pl
from packages.titanic.titanic.config import config


def run_training():
    """Train the model."""

    # read training data
    data = pd.read_csv(config.TRAINING_DATA_FILE)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(config.TARGET, axis=1),
        data[config.TARGET],
        test_size=0.2,
        random_state=0
    )

    # fit pipeline
    pl.titanic_pipe.fit(X_train, y_train)

    # save pipeline
    joblib.dump(pl.titanic_pipe, config.PIPELINE_NAME)

if __name__ == '__main__':
    run_training()
