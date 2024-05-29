import streamlit as st
import pickle
import pandas as pd
import joblib

# Load the trained machine learning model for medical cost prediction
def load_medical_model(model_path):
    model = joblib.load(model_path)
    return model

# Load the trained machine learning model for house price prediction
def load_housing_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Function to make medical cost predictions
def predict_medical_cost(model, age, bmi, children, sex, smoker, region):
    input_data = pd.DataFrame({
        'age': [age],
        'bmi': [bmi],
        'children': [children],
        'sex_male': [1 if sex == 'male' else 0],
        'smoker_yes': [1 if smoker == 'yes' else 0],
        'region_northwest': [1 if region == 'northwest' else 0],
        'region_southeast': [1 if region == 'southeast' else 0],
        'region_southwest': [1 if region == 'southwest' else 0]
    })
    prediction = model.predict(input_data)[0]
    return prediction

# Function to make house price predictions
def predict_house_price(model, input_features):
    input_df = pd.DataFrame([input_features])
    prediction = model.predict(input_df)[0]
    return prediction

# Function to load and inject custom CSS
def load_css():
    css = """
    <style>
    body {
        font-family: 'Arial', sans-serif;
        background-color: #f8f9fa;
        color: #212529;
    }
    .main .block-container {
        padding-top: 2rem;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        cursor: pointer;
        font-size: 1rem;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    .stSelectbox, .stNumberInput, .stTextInput {
        margin-bottom: 1rem;
    }
    .stTextInput input {
        padding: 0.5rem;
        border: 1px solid #ced4da;
        border-radius: 4px;
        font-size: 1rem;
    }
    .stNumberInput input {
        padding: 0.5rem;
        border: 1px solid #ced4da;
        border-radius: 4px;
        font-size: 1rem;
    }
    .stSelectbox>div>div>div {
        padding: 0.5rem;
        border: 1px solid #ced4da;
        border-radius: 4px;
        font-size: 1rem;
    }
    .stAlert {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 4px;
        border: 1px solid #c3e6cb;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Main function to run the Streamlit app
def main():
    load_css()

    st.title('Prediction App')

    # Load the models
    medical_model_path = 'medical_model.pkl'  # Path to your medical cost prediction model
    medical_model = load_medical_model(medical_model_path)

    housing_model_path = 'housing_model_train.pkl'  # Path to your house price prediction model
    housing_model = load_housing_model(housing_model_path)

    # User selects prediction type
    prediction_type = st.selectbox('Select Prediction Type', ['Medical Cost', 'House Price'])

    if prediction_type == 'Medical Cost':
        st.subheader('Medical Cost Prediction')
        age = st.number_input('Age', min_value=0, max_value=100, value=25)
        bmi = st.number_input('BMI', min_value=0.0, max_value=100.0, value=25.0)
        children = st.number_input('Number of Children', min_value=0, max_value=10, value=0)
        sex = st.selectbox('Sex', ['male', 'female'])
        smoker = st.selectbox('Smoker', ['yes', 'no'])
        region = st.selectbox('Region', ['northwest', 'southeast', 'southwest'])

        if st.button('Predict Medical Cost'):
            medical_prediction = predict_medical_cost(medical_model, age, bmi, children, sex, smoker, region)
            st.success(f'Predicted Medical Cost: ${medical_prediction:.2f}')

    elif prediction_type == 'House Price':
        st.subheader('House Price Prediction')
        longitude = st.number_input('Longitude', value=0.0)
        latitude = st.number_input('Latitude', value=0.0)
        housing_median_age = st.number_input('Housing Median Age', value=0.0)
        total_rooms = st.number_input('Total Rooms', value=0.0)
        population = st.number_input('Population', value=0.0)
        households = st.number_input('Households', value=0.0)
        median_income = st.number_input('Median Income', value=0.0)
        rooms_per_household = st.number_input('Rooms per Household', value=0.0)
        bedrooms_per_room = st.number_input('Bedrooms per Room', value=0.0)
        population_per_household = st.number_input('Population per Household', value=0.0)
        ocean_proximity = st.selectbox('Ocean Proximity', ['NEAR BAY', '<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'ISLAND'])

        ocean_mapping = {'NEAR BAY': 1, '<1H OCEAN': 2, 'INLAND': 3, 'NEAR OCEAN': 4, 'ISLAND': 5}
        ocean_1 = ocean_mapping['NEAR BAY'] == ocean_mapping[ocean_proximity]
        ocean_2 = ocean_mapping['<1H OCEAN'] == ocean_mapping[ocean_proximity]
        ocean_3 = ocean_mapping['INLAND'] == ocean_mapping[ocean_proximity]
        ocean_4 = ocean_mapping['NEAR OCEAN'] == ocean_mapping[ocean_proximity]
        ocean_5 = ocean_mapping['ISLAND'] == ocean_mapping[ocean_proximity]

        if st.button('Predict House Price'):
            house_prediction = predict_house_price(housing_model, {
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
            })
            st.success(f'Predicted House Price: ${house_prediction:,.2f}')

if __name__ == '__main__':
    main()
