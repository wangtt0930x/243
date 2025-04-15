import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
from model import get_recommendations  # Your function from model.py

# Set page config with wide layout and custom title
st.set_page_config(
    page_title="CA Rent Recommender", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Add custom CSS for Airbnb-like styling
st.markdown("""
<style>
    /* Remove all padding and margins at the top */
    .main .block-container {
        padding-top: 1rem !important;
        margin-top: 0 !important;
    }
    
    /* Hide header decoration */
    header {
        visibility: hidden;
    }
    
    /* Hide footer */
    footer {
        visibility: hidden;
    }
    
    /* Main theme colors */
    :root {
        --airbnb-red: #FF5A5F;
        --airbnb-dark: #484848;
        --airbnb-light: #767676;
    }
    
    /* Title styling */
    h1 {
        color: var(--airbnb-red);
        font-weight: bold !important;
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    
    /* Cards for filter options */
    .filter-card {
        background-color: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: var(--airbnb-red);
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 20px;
        border: none;
        width: 100%;
    }
    
    /* Property card styling */
    .property-card {
        border: 1px solid #DDDDDD;
        border-radius: 12px;
        overflow: hidden;
        transition: transform 0.2s, box-shadow 0.2s;
        margin-bottom: 15px;
    }
    
    .property-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    /* Make the dataframe pretty */
    .dataframe-container {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 12px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# App header with logo-like styling
st.markdown("""
<div style="display: flex; align-items: center; margin-bottom: 0px;">  <!-- Reduced margin -->
    <h1>🏡 California Rental Explorer</h1>
</div>
<p style="color: #767676; margin-bottom: 15px;">Find underpriced rentals in California with our AI-powered recommendation engine</p>
""", unsafe_allow_html=True)

# Create container for filters
with st.container():
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Where")
        # 1. Change zipcode to text input
        filters = {}
        filters["zip"] = st.text_input("Enter ZIP code", placeholder="e.g. 92225, 96007")
        try:
            if filters["zip"] and filters["zip"].strip():
                filters["zip"] = int(filters["zip"])
            else:
                filters["zip"] = None
        except ValueError:
            st.error("Please enter a valid ZIP code")
            filters["zip"] = None
    
    with col2:
        st.markdown("### Price Range")
        filters["RENT_PRICE"] = st.slider("", min_value=500, max_value=8000, value=(1000, 4000), format="$%d")

    st.markdown("### Property Features")
    
    # Create three columns for property features
    feat_col1, feat_col2, feat_col3 = st.columns(3)
    
    with feat_col1:
        filters["SQFT"] = st.slider("Square Footage", min_value=300, max_value=3000, value=(500, 1500))
        filters["BEDS"] = st.selectbox("Bedrooms", options=[None, 1, 2, 3, 4], format_func=lambda x: "Any" if x is None else f"{x} Beds")
        filters["BATHS"] = st.selectbox("Bathrooms", options=[None, 1, 2, 3], format_func=lambda x: "Any" if x is None else f"{x} Baths")
    
    with feat_col2:
        filters["GARAGE"] = st.toggle("Garage", value=False)
        filters["FURNISHED"] = st.toggle("Furnished", value=False)
        filters["GYM"] = st.toggle("Gym", value=False)
    
    with feat_col3:
        filters["DOORMAN"] = st.toggle("Doorman", value=False)
        filters["LAUNDRY"] = st.toggle("Laundry", value=False)
        filters["POOL"] = st.toggle("Pool", value=False)
    
    # Convert toggle values (True/False) to 1/0
    toggle_fields = ["GARAGE", "FURNISHED", "GYM", "DOORMAN", "LAUNDRY", "POOL"]
    for field in toggle_fields:
        filters[field] = 1 if filters[field] else None
        
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Search button
    search_col1, search_col2, search_col3 = st.columns([1, 2, 1])
    with search_col2:
        search_clicked = st.button("🔍 Find Top Deals")

# Initialize session state to store results
if 'results_df' not in st.session_state:
    st.session_state.results_df = pd.DataFrame()
if 'ratios' not in st.session_state:
    st.session_state.ratios = []
if 'filter_warning' not in st.session_state:
    st.session_state.filter_warning = None
if 'error_message' not in st.session_state:
    st.session_state.error_message = None

# Display results section
if search_clicked:
    with st.spinner("Finding the best deals for you..."):
        try:
            # Get recommendations with the updated function signature
            results_df, ratios, filter_warning, error_message = get_recommendations(filters)
            
            # Store results in session state
            st.session_state.results_df = results_df
            st.session_state.ratios = ratios
            st.session_state.filter_warning = filter_warning
            st.session_state.error_message = error_message
        except Exception as e:
            import traceback
            st.error(f"An error occurred: {str(e)}")
            st.error(traceback.format_exc())

# Display any warnings
if st.session_state.filter_warning:
    st.warning(st.session_state.filter_warning)
if st.session_state.error_message:
    st.error(st.session_state.error_message)

# Display results if available
if not st.session_state.results_df.empty:
    st.markdown(f"<h3>Top {len(st.session_state.results_df)} Recommended Listings</h3>", unsafe_allow_html=True)
    
    # Create two columns: map and results table
    map_col, table_col = st.columns([1, 1])
    
    with map_col:
        # Create the map centered at the mean lat/lng of results or default to California center if no data
        if len(st.session_state.results_df) > 0 and 'lat' in st.session_state.results_df.columns and 'lng' in st.session_state.results_df.columns:
            # Filter out rows with NaN lat/lng
            valid_locations = st.session_state.results_df.dropna(subset=['lat', 'lng'])
            
            if not valid_locations.empty:
                avg_lat = valid_locations['lat'].mean()
                avg_lng = valid_locations['lng'].mean()
                zoom_start = 11  # Closer zoom when we have specific properties
            else:
                # Default to California center if no valid locations
                avg_lat = 36.7783
                avg_lng = -119.4179
                zoom_start = 6  # Wider zoom to show all of California
        else:
            # Default to California center if no locations
            avg_lat = 36.7783
            avg_lng = -119.4179
            zoom_start = 6  # Wider zoom to show all of California
        
        m = folium.Map(location=[avg_lat, avg_lng], zoom_start=zoom_start, tiles="OpenStreetMap")
        
        # Check if we have locations to display
        if len(st.session_state.results_df) > 0 and 'lat' in st.session_state.results_df.columns and 'lng' in st.session_state.results_df.columns:
            # Group properties by lat/lng coordinates to handle multiple properties at the same location
            coords_df = st.session_state.results_df.copy()
            coords_df['coord'] = list(zip(coords_df['lat'], coords_df['lng']))
            grouped_by_coords = coords_df.groupby('coord')
            
            # For each location, create a marker
            for coords, props in grouped_by_coords:
                lat, lng = coords
                
                if pd.notna(lat) and pd.notna(lng):
                    # Find the lowest price property for this location to display on marker
                    min_price_row = props.loc[props['RENT_PRICE'].idxmin()]
                    min_price = int(min_price_row['RENT_PRICE']) if pd.notna(min_price_row['RENT_PRICE']) else '?'
                    
                    # Calculate a score-based color (greener = better deal)
                    # Use the best deal score for marker color
                    max_ratio = props['PREDICTED/ACTUAL_RATIO'].max()
                    if max_ratio > 1.5:
                        color = '#00AF87'  # Green for great deals
                    elif max_ratio > 1.2:
                        color = '#FF9800'  # Orange for good deals
                    else:
                        color = '#FF5A5F'  # Airbnb red for standard deals
                    
                    # Create popup HTML that shows ALL properties at this location
                    popup_html = f"""
                    <div style="width: 300px; font-family: 'Arial', sans-serif; max-height: 400px; overflow-y: auto;">
                        <h3 style="color: #484848; margin-bottom: 8px;">Properties at this location</h3>
                    """
                    
                    # Add each property to the popup
                    for idx, row in props.iterrows():
                        # Format property details (with error checking)
                        beds = int(row['BEDS']) if 'BEDS' in row and pd.notna(row['BEDS']) else '?'
                        baths = int(row['BATHS']) if 'BATHS' in row and pd.notna(row['BATHS']) else '?'
                        sqft = int(row['SQFT']) if 'SQFT' in row and pd.notna(row['SQFT']) else '?'
                        price = int(row['RENT_PRICE']) if 'RENT_PRICE' in row and pd.notna(row['RENT_PRICE']) else '?'
                        deal_score = f"{row.get('PREDICTED/ACTUAL_RATIO', 1.0):.2f}x" if pd.notna(row.get('PREDICTED/ACTUAL_RATIO', None)) else 'N/A'
                        
                        # Determine color based on individual property's deal score
                        prop_ratio = row.get('PREDICTED/ACTUAL_RATIO', 1.0)
                        if prop_ratio > 1.5:
                            prop_color = '#00AF87'  # Green for great deals
                        elif prop_ratio > 1.2:
                            prop_color = '#FF9800'  # Orange for good deals
                        else:
                            prop_color = '#FF5A5F'  # Airbnb red for standard deals
                        
                        popup_html += f"""
                        <div style="border-bottom: 1px solid #E8E8E8; padding: 10px 0; margin-bottom: 10px;">
                            <h4 style="color: #484848; margin-bottom: 5px;">${price:,}/month</h4>
                            <p style="color: #767676; margin-bottom: 5px;">
                                {beds} bed · {baths} bath · {sqft:,} sqft
                            </p>
                            <p style="font-weight: bold; color: {prop_color};">
                                Deal Score: {deal_score} value ratio
                            </p>
                            <div style="display: flex; flex-wrap: wrap; gap: 5px;">
                        """
                        
                        # Add amenity badges if available
                        amenities = []
                        if row.get('GARAGE') == 1:
                            amenities.append('Garage')
                        if row.get('FURNISHED') == 1:
                            amenities.append('Furnished')
                        if row.get('GYM') == 1:
                            amenities.append('Gym')
                        if row.get('DOORMAN') == 1:
                            amenities.append('Doorman')
                        if row.get('LAUNDRY') == 1:
                            amenities.append('Laundry')
                        if row.get('POOL') == 1:
                            amenities.append('Pool')
                        
                        for amenity in amenities:
                            popup_html += f"""
                            <span style="background-color: #F7F7F7; 
                                        border-radius: 4px; 
                                        padding: 3px 8px; 
                                        font-size: 11px; 
                                        color: #484848;">
                                {amenity}
                            </span>
                            """
                        
                        popup_html += """
                            </div>
                        </div>
                        """
                    
                    popup_html += """
                    </div>
                    """
                    
                    # Create popup with custom width
                    popup = folium.Popup(popup_html, max_width=320)
                    
                    # Calculate a price display value for the LOWEST price (K for thousands)
                    if isinstance(min_price, int) or isinstance(min_price, float):
                        price_display = f"${min_price//1000}k" if min_price >= 1000 else f"${min_price}"
                    else:
                        price_display = "$?"
                    
                    # Create marker with custom icon - using the lowest price
                    folium.Marker(
                        location=[lat, lng],
                        popup=popup,
                        icon=folium.DivIcon(
                            icon_size=(40, 40),
                            icon_anchor=(20, 20),
                            html=f"""
                            <div style="
                                background-color: {color};
                                color: white;
                                border-radius: 50%;
                                width: 40px;
                                height: 40px;
                                display: flex;
                                align-items: center;
                                justify-content: center;
                                font-weight: bold;
                                font-size: 14px;
                                box-shadow: 0 2px 5px rgba(0,0,0,0.3);
                                border: 2px solid white;
                            ">
                                {price_display}
                            </div>
                            """
                        )
                    ).add_to(m)
            
            # Add a circle to highlight the ZIP code area if provided
            if filters.get('zip') is not None:
                # Find a property with that ZIP to center the circle
                zip_props = st.session_state.results_df[st.session_state.results_df['zip'] == filters['zip']]
                valid_zip_props = zip_props.dropna(subset=['lat', 'lng'])
                if not valid_zip_props.empty:
                    zip_lat = valid_zip_props['lat'].mean()
                    zip_lng = valid_zip_props['lng'].mean()
                    folium.Circle(
                        location=[zip_lat, zip_lng],
                        radius=2000,  # 2km radius (approximate for ZIP code)
                        color='#FF5A5F',
                        fill=True,
                        fill_color='#FF5A5F',
                        fill_opacity=0.1,
                        weight=2,
                        popup=f"ZIP Code: {filters['zip']}"
                    ).add_to(m)
        
        # Display the map
        folium_static(m, width=600, height=500)
    
    with table_col:
        if not st.session_state.results_df.empty:
            try:
                # Get only the columns we need
                columns_to_show = ["zip", "BEDS", "BATHS", "SQFT", "RENT_PRICE", "PREDICTED_RENT", "PREDICTED/ACTUAL_RATIO"]
                # Make sure all columns exist
                existing_columns = [col for col in columns_to_show if col in st.session_state.results_df.columns]
                
                # Format the results dataframe for display
                # Ensure we only show the top 10 results
                display_df = st.session_state.results_df[existing_columns].head(10).copy()
                
                # Format BEDS and BATHS as integers (safely)
                if "BEDS" in display_df.columns:
                    display_df["BEDS"] = display_df["BEDS"].fillna(0).astype(int)
                if "BATHS" in display_df.columns:
                    display_df["BATHS"] = display_df["BATHS"].fillna(0).astype(int)
                
                # Rename columns for better display
                column_mapping = {
                    "zip": "ZIP", 
                    "BEDS": "Beds", 
                    "BATHS": "Baths", 
                    "SQFT": "Sq.Ft", 
                    "RENT_PRICE": "Price", 
                    "PREDICTED_RENT": "Est. Value", 
                    "PREDICTED/ACTUAL_RATIO": "Deal Score"
                }
                
                # Only rename columns that exist
                rename_dict = {k: v for k, v in column_mapping.items() if k in display_df.columns}
                display_df = display_df.rename(columns=rename_dict)
                
                # Display the table with formatting
                st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
                
                # Define format based on available columns
                format_dict = {}
                if "Price" in display_df.columns:
                    format_dict["Price"] = "${:.0f}"
                if "Est. Value" in display_df.columns:
                    format_dict["Est. Value"] = "${:.0f}"
                if "Deal Score" in display_df.columns:
                    format_dict["Deal Score"] = "{:.2f}x"
                
                st.dataframe(
                    display_df.style.format(format_dict),
                    height=500,
                    use_container_width=True,
                    hide_index=True  # Hide the index
                )
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Add a download button for the results
                csv = display_df.to_csv(index=False)
                st.download_button(
                    label="📋 Download Results",
                    data=csv,
                    file_name="rental_recommendations.csv",
                    mime="text/csv",
                )
            except Exception as e:
                import traceback
                st.error(f"Error displaying results: {str(e)}")
                st.error(traceback.format_exc())
