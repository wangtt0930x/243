import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
import urllib.request
import tempfile
import requests
import io
import joblib

DATA_URL = "https://www.dropbox.com/scl/fi/lwqa9zeesnn7wfdqi4jq4/Module2_final_data-1.csv?rlkey=andu9myoza0oa7u22pwz3v2n1&dl=1"
MODEL_URL = "https://www.dropbox.com/scl/fi/ndil67djnbu4r6zhf7i43/final_rf_model.pkl?rlkey=5wdq26hkfaq5cqfh6bcxxc9o5&dl=1"

from io import StringIO

def load_data():
    url = "https://www.dropbox.com/scl/fi/lwqa9zeesnn7wfdqi4jq4/Module2_final_data-1.csv?rlkey=andu9myoza0oa7u22pwz3v2n1&dl=1"
    
    # Fetch content safely using requests
    response = requests.get(url)
    response.raise_for_status()  # Raise error if download failed
    
    # Convert content to a pandas DataFrame
    csv_data = StringIO(response.text)
    df = pd.read_csv(csv_data)
    return df

def load_model():
    # Download model to a temporary file and load it
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        urllib.request.urlretrieve(MODEL_URL, tmp.name)
        model = joblib.load(tmp.name)
    return model

def apply_filters(df, filters):
    for key, value in filters.items():
        if key in df.columns and value is not None:
            if isinstance(value, tuple) and len(value) == 2:
                df = df[(df[key] >= value[0]) & (df[key] <= value[1])]
            else:
                df = df[df[key] == value]
    return df

def recommend_top_10(filtered_df, model):
    if filtered_df.empty:
        return pd.DataFrame(), []

    X_filtered = filtered_df.drop(columns=['RENT_PRICE'])
    y_actual = filtered_df['RENT_PRICE']

    # Clean: remove inf and NaN values
    X_filtered = X_filtered.replace([np.inf, -np.inf], np.nan)
    X_filtered = X_filtered.dropna()
    y_actual = y_actual.loc[X_filtered.index]

    if X_filtered.empty:
        return pd.DataFrame(), []

    y_pred = model.predict(X_filtered)
    ratio = y_pred / y_actual
    top_idx = np.argsort(ratio)[-10:][::-1]
    top_df = filtered_df.iloc[top_idx].copy()
    top_df["PREDICTED_RENT"] = y_pred[top_idx]
    top_df["PREDICTED/ACTUAL_RATIO"] = ratio[top_idx]

    return top_df.reset_index(drop=True)

# Main function to call from backend
def get_recommendations(filters):
    df = load_data()
    model = load_model()
    filtered_df = apply_filters(df, filters)
    return recommend_top_10(filtered_df, model)
