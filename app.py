import streamlit as st
import pandas as pd
from model import get_recommendations  # Your function from model.py

# Set page title and style
st.set_page_config(page_title="CA Rent Recommender", layout="centered")

st.title("🏡 California Housing Recommendation System")
st.markdown("Filter to find the top 10 underpriced rental listings!")

# --- Collect Filter Inputs from User ---
filters = {}

filters["ZIPCODE"] = st.selectbox("Zip Code", options=[None, 90001, 90002, 90003, 94110, 94111])  # Replace with real zipcodes
filters["SQFT"] = st.slider("Square Footage Range", min_value=300, max_value=3000, value=(500, 1500))
filters["BEDROOMS"] = st.selectbox("Number of Bedrooms", options=[None, 1, 2, 3, 4])
filters["BATHROOMS"] = st.selectbox("Number of Bathrooms", options=[None, 1, 2, 3])

filters["GARAGE"] = st.radio("Garage", options=[None, 1, 0], format_func=lambda x: "Any" if x is None else "Yes" if x else "No")
filters["FURNISHED"] = st.radio("Furnished", options=[None, 1, 0], format_func=lambda x: "Any" if x is None else "Yes" if x else "No")
filters["GYM"] = st.radio("Gym", options=[None, 1, 0], format_func=lambda x: "Any" if x is None else "Yes" if x else "No")
filters["DOORMAN"] = st.radio("Doorman", options=[None, 1, 0], format_func=lambda x: "Any" if x is None else "Yes" if x else "No")
filters["LAUNDRY"] = st.radio("Laundry", options=[None, 1, 0], format_func=lambda x: "Any" if x is None else "Yes" if x else "No")
filters["POOL"] = st.radio("Pool", options=[None, 1, 0], format_func=lambda x: "Any" if x is None else "Yes" if x else "No")
filters["RENT_PRICE"] = st.slider("Rent Price Range", min_value=500, max_value=8000, value=(1000, 4000))

# --- Show results on button click ---
if st.button("🔍 Show Top 10 Listings"):
    with st.spinner("Finding the best deals..."):
        results_df = get_recommendations(filters)[0]  # just get DataFrame
        if results_df.empty:
            st.warning("No listings match your criteria. Try adjusting filters.")
        else:
            st.success(f"Top {len(results_df)} recommended listings:")
            st.dataframe(results_df.style.format({"PREDICTED/ACTUAL_RATIO": "{:.2f}", "PREDICTED_RENT": "${:.0f}"}))
