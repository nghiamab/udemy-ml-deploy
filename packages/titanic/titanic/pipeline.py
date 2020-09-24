from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from packages.titanic.titanic.processing import preprocessors as pp
from packages.titanic.titanic.config import config

titanic_pipe = Pipeline(
    # complete with the list of steps from the preprocessors file
    # and the list of variables from the config
    [
        ('indicate_missing',
         pp.MissingIndicator(variables=config.VARS_WITH_NA)),

        ('categorical_imputer',
         pp.CategoricalImputer(variables=config.CATEGORICAL_VARS_WITH_NA)),

        ('numerical_imputer',
         pp.NumericalImputer(variables=config.NUMERICAL_VARS_WITH_NA)),

        ('extract_first_letter',
         pp.ExtractFirstLetter(variables=config.CABIN)),

        ('rare_cate_encoder',
         pp.RareLabelCategoricalEncoder(variables=config.CATEGORICAL_VARS)),

        ('cate_encoder',
         pp.CategoricalEncoder(variables=config.CATEGORICAL_VARS)),

        ('scaler', StandardScaler()),

        ('logistic_regression', LogisticRegression(C=0.0005, random_state=0))
    ]
)
