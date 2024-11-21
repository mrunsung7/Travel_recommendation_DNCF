import numpy as np
import pandas as pd
import mysql.connector
from keras.models import load_model
import json
import sys
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# Suppress TensorFlow logging
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow logging

# Load the model once at the beginning
ncf_model = load_model('/Applications/XAMPP/xamppfiles/htdocs/idbms/ncf_model.h5')

# Database connection and data loading
def fetch_data_from_db():
    db_config = {
        'host': 'localhost',
        'user': 'root',
        'password': '',  # Add your MySQL password
        'database': 'travelrecommendation'
    }
    
    try:
        connection = mysql.connector.connect(**db_config)
        query = """
        SELECT City, Place, Rating, Review
        FROM travel_data
        """
        df = pd.read_sql(query, connection)
        connection.close()
    except Exception as e:
        print(f"Database connection error: {e}", file=sys.stderr)
        return pd.DataFrame()  # Return an empty DataFrame if there's an error

    return df

# Function to recommend places based on city
# Function to recommend places based on city
def recommend_places_by_city(city):
    df_cleaned = fetch_data_from_db()

    if df_cleaned.empty:
        return {"error": "No data fetched from the database."}

    # Convert 'Place' and 'City' to categorical codes (IDs)
    df_cleaned['place_id_code'] = df_cleaned['Place'].astype('category').cat.codes
    df_cleaned['city_id_code'] = df_cleaned['City'].astype('category').cat.codes

    # Filter places based on the selected city
    city_places = df_cleaned[df_cleaned['City'] == city]
    
    if city_places.empty:
        return {"error": f"No data found for the city: {city}"}

    # Prepare data for the model
    place_ids = city_places['place_id_code'].values.reshape(-1, 1)
    city_ids = city_places['city_id_code'].values.reshape(-1, 1)

    # Predict ratings without verbose output
    predicted_ratings = ncf_model.predict([place_ids, city_ids], verbose=0).flatten()

    # Add predictions to the DataFrame
    city_places['Predicted_Rating'] = predicted_ratings

    # Drop duplicates to ensure unique places
    city_places = city_places.drop_duplicates(subset=['Place'])

    # Get the lowest-rated place with review
    lowest_rated_place = city_places.nsmallest(1, 'Predicted_Rating')[['Place', 'Predicted_Rating', 'Review']]

    # Exclude the lowest-rated place from the top places
    if not lowest_rated_place.empty:
        lowest_place_name = lowest_rated_place['Place'].iloc[0]
        top_rated_places = city_places[city_places['Place'] != lowest_place_name].nlargest(5, 'Predicted_Rating')[['Place', 'Predicted_Rating']]
    else:
        top_rated_places = city_places.nlargest(5, 'Predicted_Rating')[['Place', 'Predicted_Rating']]

    recommendations = {
        "top": top_rated_places.to_dict(orient='records'),
        "lowest": lowest_rated_place.to_dict(orient='records') if not lowest_rated_place.empty else []
    }

    return recommendations




# Check for user input from the command line
if __name__ == "__main__":
    selected_city = sys.argv[1] if len(sys.argv) > 1 else "Default City"
    recommendations = recommend_places_by_city(selected_city)
    
    # Print as JSON
    print(json.dumps(recommendations, ensure_ascii=False))  # Ensure JSON is valid
