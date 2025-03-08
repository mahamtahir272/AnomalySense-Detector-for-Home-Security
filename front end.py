import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the saved model and encoders
rf_model = joblib.load('random_forest_model.pkl')  # Load your trained Random Forest model
label_encoders = {
    'noise_level': joblib.load('label_encoder_noise_level.pkl'),  # Load encoder for noise_level
}
target_encoder = joblib.load('target_encoder.pkl')  # Load target encoder

# Set a title for the app
st.title('Anomaly Sense Detector App')

# Add background image via custom CSS (either local or URL image)
image_url = "https://wallpapercave.com/wp/wp12670889.jpg"  # Replace with your image URL or path to local file
background_css = f"""
    <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-position: center center;
            background-attachment: fixed;
        }}
        
        /* Custom form styles */
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: "Poppins", sans-serif;
        }}
        body {{
            height: 100vh;
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: center;
            background: #22232e;
            gap: 20px;
            margin: 10px;
        }}
    </style>
"""

# Inject the CSS into the page
st.markdown(background_css, unsafe_allow_html=True)

# Add the form with fields
temp = st.number_input('Enter Temperature:')
humidity = st.number_input('Enter Humidity:')
# Add a dropdown for noise_level input
noise_level = st.selectbox('Select Noise Level:', ['Normal', 'Loud'])
hour = st.number_input('Enter Hour:', min_value=1, max_value=24)
minute = st.number_input('Enter Minute:', min_value=1, max_value=59)

# Handling the form submission when 'Submit' button is clicked (the default Streamlit button)
if st.button('Submit'):
    # Create a DataFrame from the user input for prediction
    unseen_data = pd.DataFrame({
        'temp': [float(temp)],  # Make sure to convert to correct type (float)
        'humidity': [float(humidity)],
        'noise_level': [noise_level],
        'hour': [hour],
        'minute': [minute]
    })
    
    # Step 1: Apply Label Encoding to the 'noise_level' column
    unseen_data['noise_level'] = label_encoders['noise_level'].transform(unseen_data['noise_level'])
    
    unseen_data_scaled = unseen_data  # If no scaling was used
    
    # Step 3: Make prediction using the trained model
    prediction = rf_model.predict(unseen_data_scaled)
    
    # Step 4: Decode the predicted label (if target variable was encoded)
    predicted_label = target_encoder.inverse_transform([prediction[0]])[0]
    
    # Display the prediction result
    st.write(f"### Prediction Result: The model predicts the anomaly as: **{predicted_label}**")
