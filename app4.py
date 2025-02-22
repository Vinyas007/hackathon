import hashlib
import pandas as pd
import streamlit as st
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
import pickle
pip install matplotlib

def generate_synthetic_data(n_samples=1000):
    np.random.seed(42)
    customer_ids = [f'C{i:03d}' for i in range(n_samples)]
    ages = np.random.randint(18, 70, size=n_samples)
    incomes = np.random.randint(20000, 150000, size=n_samples)
    tenures = np.random.randint(0, 30, size=n_samples)
    claims = np.random.randint(0, 10, size=n_samples)
    plans = np.random.choice(['Basic Plan', 'Standard Plan', 'Premium Plan'], size=n_samples)
    emails = [f'customer{i}@example.com' for i in range(n_samples)]
    phone_numbers = [f'{np.random.randint(6000000000, 9999999999)}' for _ in range(n_samples)]
    
    df = pd.DataFrame({
        'Customer_ID': customer_ids,
        'Age': ages,
        'Income': incomes,
        'Tenure': tenures,
        'Claims': claims,
        'Current_Plan': plans,
        'Email': emails,
        'Phone Number': phone_numbers
    })
    df.to_csv('customer.csv', index=False)
    return df

def train_kmeans(df, n_clusters=3):
    features = ['Age', 'Income', 'Tenure', 'Claims']
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[features])
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['Segment'] = kmeans.fit_predict(df_scaled)
    
    segment_mapping = {
        0: "Stable & Responsible: You have a strong financial profile and a low claim history.",
        1: "Moderate Risk: Your profile is balanced, but some claims or financial factors impact your category.",
        2: "Needs Attention: Your claim history or financial status suggests a higher risk profile."
    }
    df['Segment'] = df['Segment'].map(segment_mapping)
    
    df.to_csv('customer.csv', index=False)
    
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('kmeans.pkl', 'wb') as f:
        pickle.dump(kmeans, f)
    
    return df

def visualize_clusters(df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df['Age'], y=df['Income'], hue=df['Segment'], palette='coolwarm')
    plt.title("Customer Clusters by Age and Income")
    plt.xlabel("Age")
    plt.ylabel("Income")
    st.pyplot(plt)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(stored_hashed_password, entered_password):
    return stored_hashed_password == hashlib.sha256(entered_password.encode()).hexdigest()

def load_data():
    if not os.path.exists('customer.csv'):
        generate_synthetic_data()
    df = pd.read_csv('customer.csv')
    df['Customer_ID'] = df['Customer_ID'].astype(str).str.strip()
    return df

@st.cache_resource
def load_behavior_model():
    if not os.path.exists('customer_behavior_model.h5'):
        st.error("Model file is missing!")
        return None
    return load_model('customer_behavior_model.h5')

@st.cache_resource
def load_scaler():
    if os.path.exists('scaler.pkl'):
        with open('scaler.pkl', 'rb') as f:
            return pickle.load(f)
    return None

@st.cache_resource
def load_kmeans():
    if os.path.exists('kmeans.pkl'):
        with open('kmeans.pkl', 'rb') as f:
            return pickle.load(f)
    return None

def recommend_policy(customer_data, model, scaler, kmeans):
    customer_df = pd.DataFrame([customer_data])
    features = ['Age', 'Income', 'Tenure', 'Claims']
    customer_df = customer_df[features]  
    
    scaled_features = scaler.transform(customer_df)
    segment = kmeans.predict(scaled_features)[0]
    segment_recommendations = {0: "Basic Plan", 1: "Standard Plan", 2: "Premium Plan"}
    return segment_recommendations.get(segment, "Standard Plan")

def authenticate_user(customer_id, password, df):
    customer_id = customer_id.strip()
    user_row = df.loc[df['Customer_ID'] == customer_id]
    if not user_row.empty:
        stored_hashed_password = user_row['Password'].iloc[0]
        return verify_password(stored_hashed_password, password)
    return False

def main():
    st.title("Customer Insurance Management App")
    df = load_data()
    model = load_behavior_model()
    scaler = load_scaler()
    kmeans = load_kmeans()
    
    if scaler is None or kmeans is None:
        df = train_kmeans(df)
        kmeans, scaler = load_kmeans(), load_scaler()
    
    login = st.sidebar.selectbox("Select an option", ["Login", "Sign Up"])
    
    if login == "Login":
        st.subheader("Login to Your Account")
        customer_id = st.text_input("Customer ID")
        password = st.text_input("Password", type='password')
        
        if st.button("Login"):
            if authenticate_user(customer_id, password, df):
                st.success(f"Logged in successfully as Customer ID: {customer_id}")
                customer_data = df[df['Customer_ID'] == customer_id].iloc[0]
                st.subheader("Customer Profile")
                st.write(f"Name: {customer_data['Name']}")
                st.write(f"Segment: {customer_data['Segment']}")
                st.write(f"Recommended Plan: {recommend_policy(customer_data, model, scaler, kmeans)}")
                
                st.subheader("Cluster Visualization")
                visualize_clusters(df)
    
if __name__ == "__main__":
    main()
