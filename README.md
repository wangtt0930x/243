# 🏡 California Housing Recommendation System

This project is a **Streamlit-based web application** that helps users discover **top underpriced rental listings in California**. By combining user filters and a pre-trained machine learning model, the app highlights listings offering the best rental value compared to predicted market prices.

## 🔍 Features

- Filter by ZIP code, square footage, bedrooms, bathrooms, and amenities
- Predict rental price using a Random Forest model
- Show top 10 "best value" listings based on predicted-to-actual rent ratio
- Visualize recommendations on an interactive map (with custom markers)
- Download top listings as CSV

## 📁 File Overview

| File | Description |
|------|-------------|
| `app.py` | Main Streamlit app with UI, map, and result rendering |
| `model.py` | Model loading, filtering logic, and top-10 recommendation function |
| `requirements.txt` | Required Python packages for deployment |

## 🔧 Installation

1. **Clone the repository**  
```bash
git clone https://github.com/yourusername/ca-housing-recommender.git
cd ca-housing-recommender
```

2. **Install dependencies**  
It's recommended to use a virtual environment.
```bash
pip install -r requirements.txt
```

3. **Run the app**
```bash
streamlit run app.py
```

## 📊 Dataset and Model

- **Dataset**: `df1000.csv` hosted on Dropbox (automatically downloaded)
- **Model**: `final_rf_model.pkl`, a pre-trained Random Forest regression model (also hosted on Dropbox)

Both files are fetched at runtime from:
- `https://www.dropbox.com/scl/fi/r1l48ib8ckzzrfg91irmx/df1000.csv?dl=1`
- `https://www.dropbox.com/scl/fi/ndil67djnbu4r6zhf7i43/final_rf_model.pkl?dl=1`

## 🧠 ML Model

The model is trained to predict rent price using features like:
- ZIP code
- Number of beds, baths
- Square footage
- Amenities (Garage, Furnished, Pool, etc.)

The top recommendations are computed based on the ratio:
```python
ratio = predicted_rent / actual_rent
```
Higher values suggest better deals.

## 📦 Dependencies

```text
streamlit
pandas
numpy
scikit-learn
joblib
requests
folium
streamlit-folium
```

## 🖼 UI Preview

The app features a clean UI inspired by Airbnb:
- Filter sidebar for user inputs
- Map view with color-coded deal scores
- Top 10 listings with rental price predictions

## 📌 Credits

Built with ❤️ using Python and Streamlit.  
Model and data processing by the California Housing Team.
