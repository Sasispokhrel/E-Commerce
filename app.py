import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

# Streamlit interface
st.title("E-Commerce")
st.subheader("Help the company maximize customer spending by identifying which platform (app or website) is more valuable.")


# Add image
st.image("https://daxg39y63pxwu.cloudfront.net/images/blog/feature-store-in-machine-learning/ML_Feature_Store.webp")

st.subheader("Enter customer attributes")

# Taking input from the user
avg_session_length = st.number_input("Average Session Length", min_value=0.0, max_value=37.0, value=30.0)
time_on_app = st.number_input("Time on App (minutes)", min_value=0.0, max_value=16.0, value=12.0)
time_on_website = st.number_input("Time on Website (minutes)", min_value=0.0, max_value=41.0, value=35.0)
length_of_membership = st.number_input("Length of Membership (in years)", min_value=0.0, max_value=10.0, value=4.0)

# Example training data (for illustration)
# Ideally, this should be a well-prepared dataset.
data = {
    "avg_session_length": [10, 20, 30, 40, 50],
    "time_on_app": [5, 10, 15, 20, 25],
    "time_on_website": [10, 20, 30, 40, 50],
    "length_of_membership": [1, 2, 3, 4, 5],
    "purchase_value": [100, 200, 300, 400, 500],  # This could be your target variable (y_train)
}

df = pd.DataFrame(data)

# X_train and y_train preparation
X_train = df[["avg_session_length", "time_on_app", "time_on_website", "length_of_membership"]]
y_train = df["purchase_value"]

# Model training and saving
def train_and_save_model():
    lr = LinearRegression()
    lr_model = lr.fit(X_train, y_train)

    # Save the trained model to a .pkl file
    with open('model.pkl', 'wb') as f:
        pickle.dump(lr_model, f)

    st.write("Model trained and saved as model.pkl!")

# Check if the model exists and train if necessary
# if not 'model.pkl' in locals():
#     if st.button("Train Model"):
#         train_and_save_model()

# Making predictions
def make_prediction():
    # Load the trained model from the pickle file
    with open('model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)

    # Create the user input as a pandas DataFrame
    user_input = pd.DataFrame({
        "avg_session_length": [avg_session_length],
        "time_on_app": [time_on_app],
        "time_on_website": [time_on_website],
        "length_of_membership": [length_of_membership]
    })

    # Make prediction
    prediction = loaded_model.predict(user_input)
    st.write(f"Predicted Purchase Value: ${prediction[0]:.2f}")

    # Get feature names and coefficients
    features = ["avg_session_length", "time_on_app", "time_on_website", "length_of_membership"]
    coefs = loaded_model.coef_

    # Calculate contributions
    app_contribution = coefs[features.index("time_on_app")] * time_on_app
    website_contribution = coefs[features.index("time_on_website")] * time_on_website

    # Compare and display
    if app_contribution > website_contribution:
        st.success("📱 The App contributes more to the predicted purchase value.")
    elif website_contribution > app_contribution:
        st.success("💻 The Website contributes more to the predicted purchase value.")
    else:
        st.info("📊 Both platforms contribute equally.")
# Prediction trigger
if st.button("Make a Prediction"):
    make_prediction()

