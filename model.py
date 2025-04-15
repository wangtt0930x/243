import streamlit as st  # ← needed for caching
import pandas as pd
import numpy as np
import requests
import joblib
import tempfile
from io import StringIO
import urllib.request

DATA_URL = "https://www.dropbox.com/scl/fi/r1l48ib8ckzzrfg91irmx/df1000.csv?rlkey=a9m2362vtnu9f566a0y0dzuy1&st=yffpk167&dl=1"
MODEL_URL = "https://www.dropbox.com/scl/fi/ndil67djnbu4r6zhf7i43/final_rf_model.pkl?rlkey=5wdq26hkfaq5cqfh6bcxxc9o5&dl=1"

@st.cache_data
def load_data():
    """Load and preprocess the dataset from Dropbox"""
    try:
        response = requests.get(DATA_URL)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))
        
        # Debug info about loaded data
        print(f"Loaded dataframe shape: {df.shape}")
        
        # Ensure required columns exist or create them
        if 'lat' not in df.columns or 'lng' not in df.columns:
            print("Warning: Latitude and longitude data missing")
            
            # Create dummy lat/lng data for California properties
            # These are rough approximations based on zip code ranges
            if 'zip' in df.columns:
                # Create basic mapping of zip codes to lat/lng for California
                # This is a very simplified approach and would be improved with geocoding
                def zip_to_coords(zip_code, add_jitter=True):
                    # Default to center of California
                    lat, lng = 36.7783, -119.4179
                    
                    if pd.isna(zip_code):
                        return lat, lng
                    
                    try:
                        zip_str = str(int(zip_code))
                        
                        # LA area (roughly)
                        if zip_str.startswith('9'):
                            lat = 34.0
                            lng = -118.2
                        # SF area (roughly)
                        elif zip_str.startswith('94'):
                            lat = 37.7
                            lng = -122.4
                        # San Diego area (roughly)
                        elif zip_str.startswith('92'):
                            lat = 32.7
                            lng = -117.1
                    except (ValueError, TypeError):
                        pass
                    
                    # Add jitter to spread out properties with the same ZIP code
                    if add_jitter:
                        # Generate a unique jitter for each property
                        # This will spread properties out in roughly a 500m radius
                        lat += np.random.uniform(-0.005, 0.005)
                        lng += np.random.uniform(-0.005, 0.005)
                        
                    return lat, lng
                
                # Apply the function to create lat/lng columns
                coords = df['zip'].apply(lambda x: zip_to_coords(x, add_jitter=True))
                df['lat'] = coords.apply(lambda x: x[0])
                df['lng'] = coords.apply(lambda x: x[1])
                print("Created approximate lat/lng data based on ZIP codes")
            else:
                # If no zip codes, create random coordinates in California
                df['lat'] = np.random.uniform(32.5, 42.0, size=len(df))  # CA latitudes
                df['lng'] = np.random.uniform(-124.4, -114.1, size=len(df))  # CA longitudes
                print("Created random lat/lng data for California")
        
        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        # Return minimal dataframe to prevent app crash
        return pd.DataFrame(columns=['zip', 'BEDS', 'BATHS', 'SQFT', 'RENT_PRICE', 'lat', 'lng'])

@st.cache_resource
def load_model():
    """Load the rental price prediction model"""
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            urllib.request.urlretrieve(MODEL_URL, tmp.name)
            model = joblib.load(tmp.name)
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        # Return a simple model to prevent app crash
        from sklearn.ensemble import RandomForestRegressor
        dummy_model = RandomForestRegressor()
        dummy_model.feature_names_in_ = ['BEDS', 'BATHS', 'SQFT', 'zip']
        dummy_model.predict = lambda X: np.ones(len(X)) * 2000  # predict $2000 for everything
        return dummy_model

def apply_filters(df, filters):
    """Apply user filters to the dataframe"""
    filtered_df = df.copy()
    print(f"Starting with {len(filtered_df)} properties")
    
    for key, value in filters.items():
        if key in filtered_df.columns and value is not None:
            if isinstance(value, tuple) and len(value) == 2:
                filtered_df = filtered_df[(filtered_df[key] >= value[0]) & (filtered_df[key] <= value[1])]
                print(f"After filter {key} [{value[0]}-{value[1]}]: {len(filtered_df)} properties")
            else:
                filtered_df = filtered_df[filtered_df[key] == value]
                print(f"After filter {key} = {value}: {len(filtered_df)} properties")
    
    # Log how many items remain after filtering
    if len(filtered_df) < 10:
        return filtered_df, f"Only {len(filtered_df)} properties match your filters. Consider broadening your search."
    else:
        return filtered_df, None

def recommend_top_10(filtered_df, model):
    """Find top 10 underpriced properties based on model predictions"""
    print(f"Starting recommendation with {len(filtered_df)} filtered properties")
    
    if filtered_df.empty:
        return pd.DataFrame(), [], "No properties match your search criteria. Try adjusting your filters."

    try:
        # Ensure lat/lng columns exist for map display
        if 'lat' not in filtered_df.columns or 'lng' not in filtered_df.columns:
            print("Warning: lat/lng columns missing from dataframe")
            # Add basic lat/lng for California (this would be improved with geocoding)
            if 'zip' in filtered_df.columns:
                # Same approach as in load_data
                def zip_to_coords(zip_code):
                    lat, lng = 36.7783, -119.4179  # Default to center of CA
                    
                    if pd.isna(zip_code):
                        return lat, lng
                    
                    try:
                        zip_str = str(int(zip_code))
                        
                        # Very simplified mapping
                        if zip_str.startswith('9'):  # LA area (roughly)
                            lat = 34.0 + np.random.uniform(-0.2, 0.2)
                            lng = -118.2 + np.random.uniform(-0.2, 0.2)
                        elif zip_str.startswith('94'):  # SF area (roughly)
                            lat = 37.7 + np.random.uniform(-0.2, 0.2)
                            lng = -122.4 + np.random.uniform(-0.2, 0.2)
                        elif zip_str.startswith('92'):  # San Diego area (roughly)
                            lat = 32.7 + np.random.uniform(-0.2, 0.2)
                            lng = -117.1 + np.random.uniform(-0.2, 0.2)
                    except (ValueError, TypeError):
                        pass
                        
                    return lat, lng
                
                # Apply the function to create lat/lng columns
                coords = filtered_df['zip'].apply(lambda x: zip_to_coords(x, add_jitter=True))
                filtered_df['lat'] = coords.apply(lambda x: x[0])
                filtered_df['lng'] = coords.apply(lambda x: x[1])
                print("Created approximate lat/lng data based on ZIP codes")
            else:
                # Random coordinates in California
                filtered_df['lat'] = np.random.uniform(32.5, 42.0, size=len(filtered_df))
                filtered_df['lng'] = np.random.uniform(-124.4, -114.1, size=len(filtered_df))
                print("Created random lat/lng data for California")
        
        # Separate features and target
        X_filtered = filtered_df.drop(columns=['RENT_PRICE'])
        y_actual = filtered_df['RENT_PRICE']

        # Clean data: replace infs and drop NaNs
        X_filtered = X_filtered.replace([np.inf, -np.inf], np.nan)
        
        # Find rows with NaN values in features used for prediction
        model_features = set(model.feature_names_in_)
        relevant_columns = [col for col in X_filtered.columns if col in model_features]
        rows_with_nan = X_filtered[relevant_columns].isna().any(axis=1)
        
        print(f"Found {rows_with_nan.sum()} rows with NaN values in prediction columns")
        
        # Keep track of original indices and filter out rows with NaN in feature columns
        valid_indices = X_filtered.index[~rows_with_nan]
        X_filtered = X_filtered.loc[valid_indices]
        y_actual = y_actual.loc[valid_indices]

        if X_filtered.empty:
            return pd.DataFrame(), [], "No valid properties remain after cleaning data. Try different criteria."

        # Ensure all required features are present
        missing_features = set(model.feature_names_in_) - set(X_filtered.columns)
        if missing_features:
            print(f"Adding missing features: {missing_features}")
            for feature in missing_features:
                X_filtered[feature] = 0  # Add missing features with default value 0

        # Align input columns to model's training feature set
        X_for_prediction = X_filtered.reindex(columns=model.feature_names_in_, fill_value=0)
        print(f"Prepared {len(X_for_prediction)} properties for prediction")

        # Predict and compute ratio
        y_pred = model.predict(X_for_prediction)
        ratio = pd.Series(y_pred / y_actual.values, index=valid_indices)
        
        print(f"Predictions completed. Min ratio: {ratio.min():.2f}, Max ratio: {ratio.max():.2f}")

        # Get top 10 highest predicted/actual ratios (or fewer if less available)
        num_results = min(10, len(ratio))
        if num_results == 0:
            return pd.DataFrame(), [], "Found no valid results to rank. Try adjusting your filters."
            
        # Sort by ratio and get top indices
        sorted_indices = ratio.sort_values(ascending=False).index[:num_results]
        
        # Assemble result
        top_df = filtered_df.loc[sorted_indices].copy()
        
        # Map predictions back to original indices
        prediction_map = dict(zip(valid_indices, y_pred))
        top_df["PREDICTED_RENT"] = [prediction_map.get(idx) for idx in sorted_indices]
        top_df["PREDICTED/ACTUAL_RATIO"] = ratio.loc[sorted_indices].values

        print(f"Returning {len(top_df)} recommended properties")
        print(f"Sample lat/lng values: {top_df[['lat', 'lng']].head(3).values}")
        
        return top_df.reset_index(drop=True), ratio.loc[sorted_indices].values, None
    except Exception as e:
        import traceback
        error_message = f"An error occurred: {str(e)}"
        error_details = traceback.format_exc()
        print(error_details)
        return pd.DataFrame(), [], error_message

def get_recommendations(filters):
    """Main function to get property recommendations based on filters"""
    print(f"Getting recommendations with filters: {filters}")
    df = load_data()
    model = load_model()
    
    # Apply filters and get any warning message
    filtered_df, filter_warning = apply_filters(df, filters)
    
    # Get recommendations and any error message
    results_df, ratios, error_message = recommend_top_10(filtered_df, model)
    
    # Return results and any messages
    return results_df, ratios, filter_warning, error_message
