
import streamlit as st
import joblib
import pandas as pd

# Load the AdaBoost model
model = joblib.load('ada_boost_model.pkl')

# Define function to get user input
def user_input_features():
    fulfilment = st.selectbox('Fulfilment', ['Merchant', 'Amazon'])
    shipservicelevel = st.selectbox('Ship Service Level', ['Standard', 'Expedited'])
    category = st.selectbox('Category', ['Western Dress', 'kurta', 'Set', 'Top', 'Blouse', 'Bottom', 'Ethnic Dress', 'Saree'])
    size = st.selectbox('Size', ['3XL', 'L', 'XXL', 'S', 'XS', 'M', '5XL', 'XL', 'Free', '6XL', '4XL'])
    amount = st.number_input('Amount', min_value=0.0, value=500.0)  # Set a default value
    region = st.selectbox('Region', ['westindia', 'eastindia', 'centralindia', 'northindia', 'southindia', 'northeastindia'])

    # Create DataFrame
    data = {
        'fulfilment': fulfilment,
        'shipservicelevel': shipservicelevel,
        'category': category,
        'size': size,
        'amount': amount,
        'region': region
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Title
st.write("""
# AdaBoost Model Prediction App
This app predicts if a product will be rejected or not!
""")

# User input features
input_df = user_input_features()

# Display the user input features
st.subheader('User Input Features')
st.write(input_df)

# Predict and display the output
if st.button('Predict'):
    prediction = model.predict(input_df)
    st.subheader('Prediction')
    predicted_label = 'Rejected' if prediction[0] == 1 else 'Accepted/Not rejected'
    st.write(predicted_label)
