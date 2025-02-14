import streamlit as st
import pandas as pd
import hashlib
import time

# Load customer data from the CSV file
@st.cache_data
def load_data():
    df = pd.read_csv('customer.csv')  # Read from CSV file
    return df

# Save updated customer data to the CSV file
def save_data(df):
    df.to_csv('customer.csv', index=False)  # Save as CSV

# Hash password function for security
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Authenticate user login
def authenticate_user(customer_id, password, df):
    if customer_id in df['Customer_ID'].values:
        stored_password = df[df['Customer_ID'] == customer_id]['Password'].values[0]
        if stored_password == hash_password(password):
            return True
    return False

# Add a new customer (Sign Up)
def add_new_customer(customer_id, password, name, age, tenure, current_plan, df):
    new_data = {
        'Customer_ID': customer_id,
        'Password': hash_password(password),
        'Name': name,
        'Age': age,
        'Tenure': tenure,
        'Current_Plan': current_plan
    }
    df = df.append(new_data, ignore_index=True)
    save_data(df)

# Recommendation based on Age
def recommend_plan(age):
    if age < 30:
        return "Youth Plan"
    elif 30 <= age < 50:
        return "Family Plan"
    else:
        return "Senior Plan"

# Add custom CSS styling to Streamlit app
st.markdown(
    """
    <style>
        body {
            background-color: #6A1B9A;
            color: #FF0000;
            font-family: 'Roboto', sans-serif;
        }
        .title {
            font-size: 50px;
            font-weight: bold;
            color: #FF0000;
            text-align: center;
            font-family: 'Montserrat', sans-serif;
            margin-top: 20px;
        }
        .plan-card {
            background-color: #4A148C;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.3);
            margin: 20px;
            transition: transform 0.3s ease;
            cursor: pointer;
        }
        .plan-card:hover {
            transform: scale(1.05);
        }
        .card-header {
            font-size: 22px;
            font-weight: bold;
            color: #FFEB3B;
        }
        .plan-description {
            font-size: 16px;
            color: #FFEB3B;
            margin-top: 10px;
        }
        .button {
            background-color: #FF4081;
            color: white;
            border-radius: 5px;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
            border: none;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease-in-out;
        }
        .button:hover {
            background-color: #F50057;
            transform: scale(1.05);
        }
        .slider-container {
            display: flex;
            overflow-x: scroll;
            margin-top: 20px;
            padding: 10px;
        }
        .slider-card {
            background-color: #6A1B9A;
            border-radius: 10px;
            padding: 20px;
            margin-right: 20px;
            min-width: 300px;
            color: #FFEB3B;
            text-align: center;
            transition: transform 0.3s ease;
        }
        .slider-card:hover {
            transform: scale(1.1);
        }
        .advertisement-banner {
            background-color: #FF4081;
            color: white;
            padding: 10px;
            text-align: center;
            font-weight: bold;
            font-size: 20px;
        }
    </style>
    """, unsafe_allow_html=True
)

# Main function to render the Streamlit app
def main():
    st.title("Customer Insurance Management App")

    # Load customer data from CSV
    df = load_data()

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

    # Advertisement Section
    st.markdown(
        """
        <div class="advertisement-banner">
            Get the Best Insurance Plan for Your Needs! Contact us for more details.
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
                st.write(f"Current Insurance Plan: {customer_data['Current_Plan']}")

                # Insurance Plan Recommendation
                recommended_plan = recommend_plan(customer_data['Age'])
                st.write(f"Recommended Insurance Plan: {recommended_plan}")

                # Feedback Page
                feedback = st.text_area("Provide your feedback here")
                if st.button("Submit Feedback"):
                    st.write("Feedback submitted successfully!")

            else:
                st.error("Invalid Customer ID or Password")

    elif login == "Sign Up":
        st.subheader("Sign Up to Create a New Account")

        customer_id = st.text_input("Enter a new Customer ID")
        if customer_id in df['Customer_ID'].values:
            st.error("Customer ID already exists. Please choose a different ID.")
        else:
            name = st.text_input("Name")
            password = st.text_input("Choose a Password", type='password')
            confirm_password = st.text_input("Confirm Password", type='password')
            age = st.number_input("Age", min_value=18)
            tenure = st.number_input("Tenure (in years)", min_value=0)
            current_plan = st.selectbox("Select Insurance Plan", ["Basic", "Premium", "Standard"])

            if password == confirm_password:
                if st.button("Sign Up"):
                    # Add the new customer to the database
                    add_new_customer(customer_id, password, name, age, tenure, current_plan, df)
                    st.success("Account created successfully! You can now log in.")
            else:
                st.error("Passwords do not match.")

# Run the app
if __name__ == "__main__":
    main()
