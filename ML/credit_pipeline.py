from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

from ml.features.credit_features import CreditFeatureEngineer

NUM_COLS = [
    'age', 'decline_app_cnt', 'score_bki',
    'bki_request_cnt', 'income', 'first_time',
    'region_rating', 'mean_income_region',
    'mean_income_age', 'mean_bki_age'
]

CAT_COLS = [
    'education', 'sex', 'car', 'car_type',
    'good_work', 'home_address', 'work_address',
    'foreign_passport', 'sna', 'month'
]

def build_credit_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), CAT_COLS),
            ('num', 'passthrough', NUM_COLS)
        ]
    )

    model = LogisticRegression(
        class_weight='balanced',
        C=500.5,
        max_iter=400,
        penalty='l2',
        solver='lbfgs'
    )

    pipeline = Pipeline(steps=[
        ('features', CreditFeatureEngineer()),
        ('preprocess', preprocessor),
        ('model', model)
    ])

    return pipeline
