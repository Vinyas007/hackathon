import streamlit as st
import pandas as pd
import hashlib
import time
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model  # Add the import for Keras model loading
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the customer data from the CSV file with error handling
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('customer.csv')
    except FileNotFoundError:
        st.error("The customer data file is missing.")
        df = pd.DataFrame()  # Return an empty dataframe if file is missing
    return df

# Save updated customer data to the CSV file with exception handling
def save_data(df):
    try:
        df.to_csv('customer.csv', index=False)
    except Exception as e:
        st.error(f"Error saving data: {e}")

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
def add_new_customer(customer_id, password, name, age, tenure, current_plan, email, phone_number, df):
    new_data = pd.DataFrame({
        'Customer_ID': [customer_id],
        'Password': [hash_password(password)],  # Hash the password
        'Name': [name],
        'Age': [age],
        'Tenure': [tenure],
        'Current_Plan': [current_plan],
        'Email': [email],
        'Phone Number': [phone_number],
        'Last Login': [''],
        'Account Created': [time.strftime("%m/%d/%Y %H:%M")],
        'Plan Expiry Date': [time.strftime("%m/%d/%Y %H:%M", time.localtime(time.time() + 365*24*60*60))],  # 1 year expiry
        'Feedback': [''],
        'Subscription_Status': ['Active']
    })
    
    df = pd.concat([df, new_data], ignore_index=True)
    save_data(df)

# Train Random Forest and ANN models for customer churn prediction
def train_models(df):
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

    # Train a RandomForestClassifier model
    churn_rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    churn_rf_model.fit(X_train, y_train)

    # Train a Neural Network (ANN) model
    churn_ann_model = Sequential()
    churn_ann_model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    churn_ann_model.add(Dense(32, activation='relu'))
    churn_ann_model.add(Dense(1, activation='sigmoid'))  # Binary output for classification
    churn_ann_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    churn_ann_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Save models after training
    save_models(churn_rf_model, churn_ann_model, label_encoder)

    return churn_rf_model, churn_ann_model, label_encoder

# Save Random Forest, ANN, and label encoder models to disk
def save_models(churn_rf_model, churn_ann_model, label_encoder):
    try:
        joblib.dump(churn_rf_model, 'churn_rf_model.pkl')
        churn_ann_model.save('churn_ann_model.h5')  # Keras model save method
        joblib.dump(label_encoder, 'label_encoder.pkl')
    except Exception as e:
        st.error(f"Error saving models: {e}")

# Load pre-trained models from disk
def load_models():
    try:
        # Loading the Random Forest model and Label Encoder using joblib
        churn_rf_model = joblib.load('churn_rf_model.pkl')
        label_encoder = joblib.load('label_encoder.pkl')
        
        # Loading the Keras ANN model using load_model
        churn_ann_model = load_model('churn_ann_model.h5')
        
    except FileNotFoundError:
        churn_rf_model = churn_ann_model = label_encoder = None
        # Handle model loading errors, such as if the models haven't been saved yet.
    return churn_rf_model, churn_ann_model, label_encoder

# Recommendation for the insurance plan using the ML models
def recommend_plan_using_ml(age, tenure, churn_rf_model, churn_ann_model, label_encoder):
    # Get prediction from both models (RF and ANN)
    rf_prediction = churn_rf_model.predict([[age, tenure]])[0]
    ann_prediction = churn_ann_model.predict([[age, tenure]])[0][0]

    # Get the most common prediction (majority vote)
    final_prediction = rf_prediction if rf_prediction == ann_prediction else max([rf_prediction, ann_prediction], key=lambda x: [rf_prediction, ann_prediction].count(x))

    # Decode the predicted plan
    predicted_plan = label_encoder.inverse_transform([final_prediction])[0]
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
    </style>
    """, unsafe_allow_html=True
)

# Main function to render the Streamlit app
def main():
    st.title("Customer Insurance Management App")

    # Load customer data from CSV
    df = load_data()

    # Load pre-trained models or train new ones if not available
    if 'churn_rf_model' not in st.session_state:
        churn_rf_model, churn_ann_model, label_encoder = load_models()
        if churn_rf_model is None or churn_ann_model is None or label_encoder is None:
            churn_rf_model, churn_ann_model, label_encoder = train_models(df)
            st.session_state.churn_rf_model = churn_rf_model
            st.session_state.churn_ann_model = churn_ann_model
            st.session_state.label_encoder = label_encoder
        else:
            st.session_state.churn_rf_model = churn_rf_model
            st.session_state.churn_ann_model = churn_ann_model
            st.session_state.label_encoder = label_encoder

    # User authentication
    if 'authenticated' not in st.session_state or not st.session_state.authenticated:
        # Show login form
        st.subheader("Login")
        customer_id = st.text_input("Customer ID")
        password = st.text_input("Password", type="password")
        login_button = st.button("Login")

        if login_button:
            if authenticate_user(customer_id, password, df):
                st.session_state.authenticated = True
                st.session_state.customer_id = customer_id
                st.success("Logged in successfully!")
            else:
                st.error("Invalid login credentials. Please try again.")

        # Sign Up section
        st.subheader("Sign Up")
        new_customer_id = st.text_input("New Customer ID")
        new_password = st.text_input("New Password", type="password")
        name = st.text_input("Name")
        age = st.number_input("Age", min_value=18, max_value=100)
        tenure = st.number_input("Tenure (in months)", min_value=1)
        current_plan = st.selectbox("Current Plan", ["Basic", "Premium", "Gold"])
        email = st.text_input("Email")
        phone_number = st.text_input("Phone Number")
        signup_button = st.button("Sign Up")

        if signup_button:
            if new_customer_id and new_password and name:
                add_new_customer(new_customer_id, new_password, name, age, tenure, current_plan, email, phone_number, df)
                st.success(f"Account for {new_customer_id} created successfully!")
            else:
                st.error("Please fill out all the fields.")

    if st.session_state.authenticated:
        st.subheader("Welcome, " + st.session_state.customer_id)
        
        # Display Plan Recommendations
        age = df.loc[df['Customer_ID'] == st.session_state.customer_id, 'Age'].values[0]
        tenure = df.loc[df['Customer_ID'] == st.session_state.customer_id, 'Tenure'].values[0]
        
        predicted_plan = recommend_plan_using_ml(age, tenure, st.session_state.churn_rf_model, st.session_state.churn_ann_model, st.session_state.label_encoder)
        
        st.write(f"Based on your profile (Age: {age}, Tenure: {tenure} months), we recommend the '{predicted_plan}' plan for you.")

        # Chatbot interaction
        st.subheader("Chatbot")
        user_input = st.text_input("Ask me anything related to your plan.")
        if user_input:
            response = chatbot_response(user_input)
            st.write("Chatbot:", response)

if __name__ == "__main__":
    main()
