import pathlib
import titanic
import pandas as pd


# ====   PATHS ===================

PACKAGE_ROOT = pathlib.Path(titanic.__file__).resolve().parent
TRAINING_DATA_FILE = "../dataset/titanic.csv"
PIPELINE_NAME = 'logistic_regression.pkl'


# ======= FEATURE GROUPS =============

TARGET = 'survived'

CATEGORICAL_VARS = ['sex', 'cabin', 'embarked', 'title']

NUMERICAL_VARS = ['pclass', 'age', 'sibsp', 'parch', 'fare']

VARS_WITH_NA = ['age', 'fare', 'sex', 'cabin', 'embarked', 'title']

CATEGORICAL_VARS_WITH_NA = ['sex', 'cabin', 'embarked', 'title']

NUMERICAL_VARS_WITH_NA = ['age', 'fare']

CABIN = 'cabin'