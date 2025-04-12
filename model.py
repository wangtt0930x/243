import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
import urllib.request
import tempfile

DATA_URL = "https://www.dropbox.com/scl/fi/lwqa9zeesnn7wfdqi4jq4/Module2_final_data-1.csv?rlkey=andu9myoza0oa7u22pwz3v2n1&dl=1"
MODEL_URL = "https://www.dropbox.com/scl/fi/ndil67djnbu4r6zhf7i43/final_rf_model.pkl?rlkey=5wdq26hkfaq5cqfh6bcxxc9o5&dl=1"

def load_data():
    df = pd.read_csv(DATA_URL)
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
    y_pred = model.predict(X_filtered)

    ratio = y_pred / y_actual
    top_idx = np.argsort(ratio)[-10:][::-1]  # Top 10 in descending order
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
