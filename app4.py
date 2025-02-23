import hashlib
import pandas as pd
import streamlit as st
import time
import os
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
import streamlit.components.v1 as components

# Hash password function using hashlib (SHA-256)
def hash_password(password):
    """Hashes a password using SHA-256."""
    hashed_password = hashlib.sha256(password.encode()).hexdigest()  # Hash the password
    return hashed_password

# Verify hashed password during login
def verify_password(stored_hashed_password, entered_password):
    """Verify that the entered password matches the stored hashed password."""
    hashed_entered_password = hashlib.sha256(entered_password.encode()).hexdigest()
    return stored_hashed_password == hashed_entered_password

# Load customer data from CSV
def load_data():
    """Load the customer data from CSV."""
    if not os.path.exists('customer.csv'):
        st.error("Customer data file is missing!")
        return pd.DataFrame()  # Return an empty DataFrame to avoid errors
    df = pd.read_csv('customer.csv')  # Read from CSV file
    df['Customer_ID'] = df['Customer_ID'].astype(str).str.strip()  # Ensure Customer_ID is treated as a string and stripped of spaces
    return df

# Load pre-trained deep learning model for customer behavior prediction
@st.cache_resource
def load_behavior_model():
    """Load the pre-trained AI model."""
    if not os.path.exists('customer_behavior_model.h5'):
        st.error("Model file is missing!")
        return None
    model = load_model('customer_behavior_model.h5')  # Load pre-trained AI model
    return model

# Standardize data for model prediction using loaded scaler
def scale_data(data, scaler):
    return scaler.transform(data)




# Authenticate user login
def authenticate_user(customer_id, password, df):
    """Authenticate the user based on Customer ID and password."""
    customer_id = customer_id.strip()  # Strip spaces from entered Customer ID
    
    # Check if customer ID exists in the DataFrame (stripping any spaces)
    if customer_id in df['Customer_ID'].values:
        stored_hashed_password = df[df['Customer_ID'] == customer_id]['Password'].values[0]
        
        # Verify the entered password by comparing with the stored hashed password
        if verify_password(stored_hashed_password, password):
            return True  # Login successful
        else:
            st.error("Incorrect password entered!")  # Debugging message
    else:
        st.error("Customer ID not found!")  # Debugging message
    
    return False  # Login failed

# Load model & scaler
model = load_model('customer_behavior_model.h5')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

import numpy as np
import pandas as pd
import pickle

def recommend_policy(customer_data, model, scaler):
    customer_df = pd.DataFrame([customer_data])

    # One-hot encode as in training
    customer_df = pd.get_dummies(customer_df, columns=['Current_Plan', 'Subscription_Status'])

    # Ensure all required columns exist
    required_features = scaler.feature_names_in_
    for col in required_features:
        if col not in customer_df.columns:
            customer_df[col] = 0  # Add missing columns

    # Reorder columns to match training data
    customer_df = customer_df[required_features]

    # Standardize the data
    scaled_features = scaler.transform(customer_df)

    # Predict the insurance plan
    prediction = model.predict(scaled_features)[0]  # Get the first row's prediction

    prediction = np.argmax(prediction)  # ✅ Get index of the highest probability

    # Map prediction index to plan names
    plan_mapping = {0: "Basic Plan", 1: "Premium Plan", 2: "Standard Plan"}
    return plan_mapping.get(prediction, "Unknown Plan")  # Return mapped plan






def main():
    st.title("Customer Insurance Management App")

    # Load customer data and AI model
    df = load_data()
    model = load_behavior_model()

    # Load the scaler used during training using pickle
    scaler = None
    if os.path.exists('scaler.pkl'):
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)  # Load pre-fitted scaler
            st.write("Scaler loaded successfully.")
    else:
        st.error("Scaler file is missing. Please ensure the model training includes saving the scaler.")

    if scaler is None:
        return  # Prevent further processing if scaler is not available

    # Home Page
    st.markdown(
        """
        <div class="home-container">
            <h2 class="title">Welcome to the Insurance Portal</h2>
            <p class="plan-description">Your insurance, our responsibility. Choose the best plan tailored to you!</p>
            <button class="button" onclick="window.location.href='#login'">Get Started</button>
        </div>
        """, unsafe_allow_html=True)

    # Display Insurance Plans with Sliding Cards
    st.markdown(
        """
        <h2 class="title">Our Insurance Plans</h2>
        <div class="slider-container">
            <div class="slider-card">
                <h3 class="card-header">Basic Plan</h3>
                <p class="plan-description">Affordable coverage for young individuals and small families.</p>
            </div>
            <div class="slider-card">
                <h3 class="card-header">Premium Plan</h3>
                <p class="plan-description">Comprehensive plan for families with extensive coverage.</p>
            </div>
            <div class="slider-card">
                <h3 class="card-header">Standard Plan</h3>
                <p class="plan-description">Balanced coverage ideal for individuals seeking good benefits.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Login or Sign Up page
    login = st.sidebar.selectbox("Select an option", ["Login", "Sign Up"])

    if login == "Login":
        st.subheader("Login to Your Account")
        customer_id = st.text_input("Customer ID")
        password = st.text_input("Password", type='password')

        if st.button("Login"):
            if authenticate_user(customer_id, password, df):
                st.success(f"Logged in successfully as Customer ID: {customer_id}")

                # Fetch customer details after login
                customer_data = df[df['Customer_ID'] == customer_id].iloc[0]

                st.subheader("Customer Profile")
                st.write(f"Name: {customer_data['Name']}")
                st.write(f"Age: {customer_data['Age']}")
                st.write(f"Tenure: {customer_data['Tenure']} years")
                st.write(f"Selected Plan: {customer_data['Current_Plan']}")  # ✅ Use the actual plan from the data
                st.subheader("Chat with Our Virtual Insurance Assistant 🤖")
                chatbot_html = """
                <iframe 
                src="https://www.chatbase.co/chatbot-iframe/nB1Du0wNzteKYTGiOUkJD" 
                width="100%" 
                height="700px"
                style="border:none;">
                </iframe>
                """
                st.components.v1.html(chatbot_html, height=700)



                # AI-driven Recommendation
                recommended_plan = recommend_policy(customer_data, model, scaler)
                st.write(f"Recommended Insurance Plan: {recommended_plan}")

                # Provide personalized feedback page
                feedback = st.text_area("Provide your feedback here")
                if st.button("Submit Feedback"):
                    st.write("Feedback submitted successfully!")

            else:
                st.error("Invalid Customer ID or Password")

    elif login == "Sign Up":
        st.subheader("Sign Up to Create a New Account")
        new_customer_id = st.text_input("Enter a new Customer ID")
        new_password = st.text_input("Create a Password", type='password')
        new_name = st.text_input("Enter your Full Name")
        new_age = st.number_input("Enter your Age", min_value=18)
        new_tenure = st.number_input("Enter your Tenure (in years)", min_value=0)
        new_plan = st.selectbox("Select Your Insurance Plan", ["Basic Plan", "Standard Plan", "Premium Plan"])
        new_income = st.number_input("Enter your Income")
        new_claims = st.number_input("Enter your Claims History")
        new_email = st.text_input("Enter Your Email")
        new_no = st.number_input("Enter your mobile_no", min_value=10)
        if new_email and ("@" not in new_email or "." not in new_email):
            st.error("Please enter a valid email address.")
    elif login == "Chatbot":
        chatbot_page()
    
    

        if st.button("Sign Up"):
            if new_customer_id and new_password and new_name:
                # Hash the password before saving
                hashed_password = hash_password(new_password)
                
                # Append the new user's data to the DataFrame
                new_data = pd.DataFrame([{
                    'Customer_ID': new_customer_id,
                    'Password': hashed_password,  # Store the hashed password
                    'Name': new_name,
                    'Age': new_age,
                    'Tenure': new_tenure,
                    'Income': new_income,
                    'Claims': new_claims,
                    'Current_Plan': new_plan,  # Default value, can be updated later
                    'Email': new_email,
                    'Phone Number': new_no,
                    'Last_Login': '',
                    'Account_Created': time.strftime("%Y-%m-%d %H:%M:%S"),  # Timestamp when account is created
                    'Plan_Expiry_Date': '',
                    'Feedback': '',
                    'Subscription_Status': 'Active'
                }])
                df = pd.concat([df, new_data], ignore_index=True)

                # Save the updated DataFrame back to CSV
                df.to_csv('customer.csv', index=False)

                # Reload the data after saving to ensure the login works immediately
                df = load_data()

                st.success(f"Account created successfully for {new_name}!")
                st.success(f"Selected Plan: {new_plan}")

# Run the app
if __name__ == "__main__":
    main()
