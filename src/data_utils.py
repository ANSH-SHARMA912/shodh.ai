import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

def parse_int_rate(x):
    """Convert interest rate strings like '13.56%' to numeric."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, str) and x.endswith('%'):
        try:
            return float(x.strip().strip('%')) / 100.0
        except:
            return np.nan
    try:
        return float(x)
    except:
        return np.nan

def build_preprocessor(num_features, cat_features):
    """Create ColumnTransformer for preprocessing."""
    try:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

    num_pipeline = Pipeline([("scaler", StandardScaler())])
    cat_pipeline = Pipeline([("onehot", encoder)])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_features),
            ("cat", cat_pipeline, cat_features),
        ]
    )
    return preprocessor

def save_preprocessor(preprocessor, path):
    joblib.dump(preprocessor, path)
