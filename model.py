import streamlit as st  # ← needed for caching
import pandas as pd
import numpy as np
import requests
import joblib
import tempfile
from io import StringIO
import urllib.request

DATA_URL = "https://www.dropbox.com/scl/fi/lwqa9zeesnn7wfdqi4jq4/Module2_final_data-1.csv?rlkey=andu9myoza0oa7u22pwz3v2n1&dl=1"
MODEL_URL = "https://www.dropbox.com/scl/fi/r1l48ib8ckzzrfg91irmx/df1000.csv?rlkey=a9m2362vtnu9f566a0y0dzuy1&st=yffpk167&dl=0"

@st.cache_data
def load_data():
    response = requests.get(DATA_URL)
    response.raise_for_status()
    df = pd.read_csv(StringIO(response.text))
    return df

@st.cache_resource
def load_model():
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
    X_filtered = X_filtered.replace([np.inf, -np.inf], np.nan).dropna()
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

def get_recommendations(filters):
    df = load_data()
    model = load_model()
    filtered_df = apply_filters(df, filters)
    return recommend_top_10(filtered_df, model)
