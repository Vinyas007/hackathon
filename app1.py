import streamlit as st
import pandas as pd
import hashlib
import time
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

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

# Machine Learning Model for Recommendation
def train_model(df):
    # Encoding categorical variables
    label_encoder = LabelEncoder()
    df['Current_Plan'] = label_encoder.fit_transform(df['Current_Plan'])

    # Features and target variable
    features = ['Age', 'Tenure']
    target = 'Current_Plan'

    # Handle missing values (if any)
    df = df.dropna(subset=features + [target])

    # Split the data into training and testing sets
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a random forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Stratified Cross-validation to ensure balance
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cross_val_score_model = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
    model_accuracy = cross_val_score_model.mean()

    # Evaluate the model accuracy on the test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.write(f"Model Accuracy (Cross-validation): {model_accuracy*100:.2f}%")
    st.write(f"Model Accuracy (Test Set): {accuracy*100:.2f}%")
    return model, label_encoder

# Recommend insurance plan using the trained model
def recommend_plan_using_ml(age, tenure, model, label_encoder):
    prediction = model.predict([[age, tenure]])[0]
    predicted_plan = label_encoder.inverse_transform([prediction])[0]
    return predicted_plan

# Initialize Chatbot model
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

def chatbot_response(user_input):
    # Tokenize and generate the model's response
    inputs = tokenizer.encode("chat: " + user_input, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2, top_p=0.95)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

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

    # Train the machine learning model once and store it in memory
    if 'model' not in st.session_state:
        model, label_encoder = train_model(df)
        st.session_state.model = model
        st.session_state.label_encoder = label_encoder

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
                st.write(f"Tenure: {customer_data['Tenure']}")
                st.write(f"Current Plan: {customer_data['Current_Plan']}")

                # Plan recommendation using ML model
                st.subheader("Plan Recommendation (Using ML Model)")
                recommended_plan = recommend_plan_using_ml(customer_data['Age'], customer_data['Tenure'],
                                                           st.session_state.model, st.session_state.label_encoder)
                st.write(f"Recommended Plan: {recommended_plan}")
            else:
                st.error("Invalid Customer ID or Password")

    elif login == "Sign Up":
        st.subheader("Create a New Account")
        new_customer_id = st.text_input("New Customer ID")
        new_password = st.text_input("New Password", type='password')
        new_name = st.text_input("Full Name")
        new_age = st.number_input("Age", min_value=18, max_value=100)
        new_tenure = st.number_input("Tenure (in years)", min_value=1, max_value=30)
        new_plan = st.selectbox("Current Plan", ["Basic", "Premium", "Standard"])

        if st.button("Sign Up"):
            add_new_customer(new_customer_id, new_password, new_name, new_age, new_tenure, new_plan, df)
            st.success("Account created successfully! Please login now.")

    # Chatbot Section
    st.subheader("Chatbot")
    user_input = st.text_input("Ask anything about insurance:")
    if user_input:
        response = chatbot_response(user_input)
        st.write(f"Bot: {response}")

if __name__ == "__main__":
    main()
