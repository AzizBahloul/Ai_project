import streamlit as st
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load the trained machine learning model
def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Function to make predictionscd
def predict_price(model, input_features):
    # Convert input features to DataFrame
    input_df = pd.DataFrame([input_features])
    
    # Make predictions
    prediction = model.predict(input_df)
    return prediction

# Main function to run the Streamlit app
def main():
    st.title('House Price Prediction App')

    # Load the model
    model_path = 'model_train.pkl'  # Path to your trained model
    model = load_model(model_path)

    # User input features
    st.sidebar.header('User Input')
    longitude = st.sidebar.number_input('Longitude', value=0.0)
    latitude = st.sidebar.number_input('Latitude', value=0.0)
    housing_median_age = st.sidebar.number_input('Housing Median Age', value=0.0)
    total_rooms = st.sidebar.number_input('Total Rooms', value=0.0)
    population = st.sidebar.number_input('Population', value=0.0)
    households = st.sidebar.number_input('Households', value=0.0)
    median_income = st.sidebar.number_input('Median Income', value=0.0)
    rooms_per_household = st.sidebar.number_input('Rooms per Household', value=0.0)
    bedrooms_per_room = st.sidebar.number_input('Bedrooms per Room', value=0.0)
    population_per_household = st.sidebar.number_input('Population per Household', value=0.0)
    ocean_1 = st.sidebar.selectbox('NEAR BAY', [0, 1])
    ocean_2 = st.sidebar.selectbox('<1H OCEAN', [0, 1])
    ocean_3 = st.sidebar.selectbox('INLAND', [0, 1])
    ocean_4 = st.sidebar.selectbox('NEAR OCEAN', [0, 1])
    ocean_5 = st.sidebar.selectbox('ISLAND', [0, 1])
    
    # Collect user input
    input_features = {
        'longitude': longitude,
        'latitude': latitude,
        'housing_median_age': housing_median_age,
        'total_rooms': total_rooms,
        'population': population,
        'households': households,
        'median_income': median_income,
        'rooms_per_household': rooms_per_household,
        'bedrooms_per_room': bedrooms_per_room,
        'population_per_household': population_per_household,
        'ocean_1': ocean_1,
        'ocean_2': ocean_2,
        'ocean_3': ocean_3,
        'ocean_4': ocean_4,
        'ocean_5': ocean_5
    }

    # Make predictions
    if st.sidebar.button('Predict'):
        prediction = predict_price(model, input_features)
        st.subheader('Prediction')
        st.write(f'The predicted house price is ${prediction[0]:,.2f}')

if __name__ == '__main__':
    main()
