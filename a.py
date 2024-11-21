import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Embedding, Flatten, Input, Dense, Concatenate, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from textblob import TextBlob

# Load the dataset
df = pd.read_csv('final_travel_data.csv')
# Preprocess Data: Keep necessary columns and convert 'Review_Date' to datetime
df['Review_Date'] = pd.to_datetime(df['Review_Date'], format='%d-%m-%Y', errors='coerce')

# Filter relevant columns: 'User_ID', 'Place', 'City', 'Rating', 'Review'
df_cleaned = df[['User_ID', 'Place', 'City', 'Rating', 'Review']]

# Convert 'User_ID' and 'Place' to categorical codes (IDs)
df_cleaned['user_id_code'] = df_cleaned['User_ID'].astype('category').cat.codes
df_cleaned['place_id_code'] = df_cleaned['Place'].astype('category').cat.codes

# Split the data into training and testing sets (90% training, 10% testing)
train, test = train_test_split(df_cleaned, test_size=0.1, random_state=42)

user_ids_train = train['user_id_code'].values
place_ids_train = train['place_id_code'].values
ratings_train = train['Rating'].values

user_ids_test = test['user_id_code'].values
place_ids_test = test['place_id_code'].values
ratings_test = test['Rating'].values

n_users = df_cleaned['user_id_code'].nunique()
n_places = df_cleaned['place_id_code'].nunique()

# Neural Collaborative Filtering Model (Deep)
def create_deep_ncf_model(n_users, n_places, embedding_dim=50):
    # Input layers
    user_input = Input(shape=[1], name='User')
    place_input = Input(shape=[1], name='Place')

    # Embedding layers
    user_embedding = Embedding(n_users, embedding_dim, name='User_Embedding')(user_input)
    place_embedding = Embedding(n_places, embedding_dim, name='Place_Embedding')(place_input)

    # Flatten the embeddings
    user_vec = Flatten()(user_embedding)
    place_vec = Flatten()(place_embedding)

    # Concatenate user and place embeddings
    concat = Concatenate()([user_vec, place_vec])

    # Deep layers
    dense = Dense(256, activation='relu')(concat)
    dense = Dropout(0.5)(dense)
    dense = Dense(128, activation='relu')(dense)
    dense = Dropout(0.5)(dense)
    dense = Dense(64, activation='relu')(dense)
    output = Dense(1, activation='linear')(dense)  # Output predicted rating

    # Create and compile the model
    model = Model([user_input, place_input], output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mean_absolute_error', 'mean_squared_error'])

    return model

# Create the model
ncf_model = create_deep_ncf_model(n_users, n_places)

# Train the model (you need to add epochs and batch_size)
history = ncf_model.fit([user_ids_train, place_ids_train], ratings_train, 
                        epochs=100, batch_size=20, 
                        validation_split=0.1, verbose=1)

# Evaluate the model on the test set
test_loss, test_mae, test_mse = ncf_model.evaluate([user_ids_test, place_ids_test], ratings_test, verbose=1)

print(f"Test Loss (MSE): {test_loss:.4f}")
print(f"Test MAE: {test_mae:.4f}")
print(f"Test MSE: {test_mse:.4f}")

# Predictions for performance metrics
predicted_ratings = ncf_model.predict([user_ids_test, place_ids_test])

# Calculate MSE, RMSE, and MAE
mse = mean_squared_error(ratings_test, predicted_ratings)
rmse = np.sqrt(mse)
mae = mean_absolute_error(ratings_test, predicted_ratings)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")

# Save the model


# Sentiment Analysis on Reviews
def analyze_sentiment(review_text):
    analysis = TextBlob(review_text)
    return analysis.sentiment.polarity

# Apply sentiment analysis to the 'Review' column
df_cleaned['Sentiment'] = df_cleaned['Review'].apply(lambda x: analyze_sentiment(str(x)))

# Correlation between sentiment and ratings
correlation = df_cleaned['Sentiment'].corr(df_cleaned['Rating'])
print(f"\nCorrelation between Sentiment and Rating: {correlation:.4f}")

# Function to recommend top N places based on city
def recommend_places_by_city(city, n_recommendations=5):
    # Filter places based on the selected city
    city_places = df_cleaned[df_cleaned['City'] == city]
    
    if city_places.empty:
        print(f"No data found for the city: {city}")
        return None

    # Get top N highest-rated places in the selected city
    top_rated_places = city_places.sort_values(by='Rating', ascending=False).head(n_recommendations)

    # Get the lowest-rated place with review
    lowest_rated_place = city_places.sort_values(by='Rating', ascending=True).iloc[0]

    print(f"\nTop {n_recommendations} Highest-Rated Places in {city}:")
    for idx, row in top_rated_places.iterrows():
        print(f"{row['Place']} - Rating: {row['Rating']}")
    
    print(f"\nLowest-Rated Place in {city}:")
    print(f"{lowest_rated_place['Place']} - Rating: {lowest_rated_place['Rating']}")
    print(f"Reason (Review): {lowest_rated_place['Review']}")

# Allow user to choose a city for recommendations
def get_user_city():
    # Display unique cities for selection
    unique_cities = df_cleaned['City'].unique()
    print("\nAvailable Cities:")
    for i, city in enumerate(unique_cities):
        print(f"{i + 1}. {city}")
    
    # Get user's choice
    city_choice = int(input("\nSelect a city by number: ")) - 1
    selected_city = unique_cities[city_choice]
    
    return selected_city

# Function to interact with the user and provide recommendations
def personalized_travel_recommendation():
    print("\nWelcome to the Personalized Travel Recommendation System!")
    selected_city = get_user_city()  # Get user-selected city
    n_recommendations = 5  # Customize this as needed
    recommend_places_by_city(selected_city, n_recommendations)

# Run the personalized recommendation system
ncf_model.save('ncf_model.h5')
print("Model saved as 'ncf_model.h5'.")
personalized_travel_recommendation()

