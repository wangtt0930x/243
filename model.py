import streamlit as st  # ← needed for caching
import pandas as pd
import numpy as np
import requests
import joblib
import tempfile
from io import StringIO
import urllib.request

DATA_URL = "https://www.dropbox.com/scl/fi/r1l48ib8ckzzrfg91irmx/df1000.csv?rlkey=a9m2362vtnu9f566a0y0dzuy1&st=yffpk167&dl=0"
MODEL_URL = "https://www.dropbox.com/scl/fi/ndil67djnbu4r6zhf7i43/final_rf_model.pkl?rlkey=5wdq26hkfaq5cqfh6bcxxc9o5&dl=1"

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

    try:
        # Separate features and target
        X_filtered = filtered_df.drop(columns=['RENT_PRICE'])
        y_actual = filtered_df['RENT_PRICE']

        # Clean data: replace infs and drop NaNs
        X_filtered = X_filtered.replace([np.inf, -np.inf], np.nan).dropna()
        y_actual = y_actual.loc[X_filtered.index]

        if X_filtered.empty:
            return pd.DataFrame(), []

        # Print debug information
        st.write("Available features in data:", X_filtered.columns.tolist())
        st.write("Model expected features:", model.feature_names_in_)

        # Ensure all required features are present
        missing_features = set(model.feature_names_in_) - set(X_filtered.columns)
        if missing_features:
            st.write("Adding missing features:", missing_features)
            for feature in missing_features:
                X_filtered[feature] = 0  # Add missing features with default value 0

        # Align input columns to model's training feature set
        X_filtered = X_filtered.reindex(columns=model.feature_names_in_, fill_value=0)

        # Print shape information
        st.write("Final feature matrix shape:", X_filtered.shape)

        # Predict and compute ratio
        y_pred = model.predict(X_filtered)
        ratio = y_pred / y_actual

        # Get top 10 highest predicted/actual ratios
        top_idx = np.argsort(ratio)[-10:][::-1]

        # Assemble result
        top_df = X_filtered.iloc[top_idx].copy()
        top_df["RENT_PRICE"] = y_actual.iloc[top_idx].values
        top_df["PREDICTED_RENT"] = y_pred[top_idx]
        top_df["PREDICTED/ACTUAL_RATIO"] = ratio.iloc[top_idx].values

        return top_df.reset_index(drop=True), ratio.iloc[top_idx]
    except Exception as e:
        st.error(f"An error occurred while processing recommendations: {str(e)}")
        st.error(f"Error type: {type(e).__name__}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return pd.DataFrame(), []

def get_recommendations(filters):
    df = load_data()
    model = load_model()
    filtered_df = apply_filters(df, filters)
    return recommend_top_10(filtered_df, model)
