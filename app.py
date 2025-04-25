import mlflow
import mlflow.sklearn
import streamlit as st
import pandas as pd
import joblib

# Set the MLflow tracking URI (Ensure it's pointing to your local MLflow server)
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Load the saved vectorizer (ensure this matches the vectorizer used in the training)
vectorizer = joblib.load(r'C:\Users\hp\vectorizer.pkl')

# Model paths (with versioning)
model_paths = {
    "Logistic Regression": "models:/Logistic Regression/latest",  # Ensure the correct version
    "Naive Bayes": "models:/Naive Bayes/latest",
    "Support Vector Machine": "models:/Support Vector Machine/latest",
}

# Load models from MLflow
models = {name: mlflow.sklearn.load_model(path) for name, path in model_paths.items()}

# Streamlit UI
st.set_page_config(page_title="Spam or Ham Prediction", page_icon="📩", layout="centered")

# Custom Styling
st.markdown("""
    <style>
        /* Body Styling */
        body {
            background-color: #f0f8ff;
            font-family: 'Arial', sans-serif;
            color: #3c3c3c;
            margin: 0;
            padding: 0;
        }

        /* Title Styling */
        .title {
            font-size: 48px;
            font-weight: 700;
            color: #0a74da;
            text-align: center;
            margin-top: 30px;
            margin-bottom: 20px;
        }

        /* Container Styling */
        .container {
            width: 80%;
            margin: auto;
            padding: 40px 0;
            background-color: #ffffff;
            border-radius: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }

        /* Input Box */
        .stTextInput input {
            font-size: 18px;
            padding: 15px;
            border-radius: 12px;
            border: 2px solid #0a74da;
            margin-bottom: 20px;
            width: 100%;
            box-sizing: border-box;
            transition: all 0.3s ease;
        }

        .stTextInput input:focus {
            border-color: #4d9fbb;
            box-shadow: 0 0 10px rgba(0, 120, 200, 0.4);
        }

        /* Dropdown Styling */
        .stSelectbox select {
            font-size: 18px;
            padding: 15px;
            border-radius: 12px;
            border: 2px solid #0a74da;
            margin-bottom: 20px;
            width: 100%;
            box-sizing: border-box;
            transition: all 0.3s ease;
        }

        .stSelectbox select:focus {
            border-color: #4d9fbb;
            box-shadow: 0 0 10px rgba(0, 120, 200, 0.4);
        }

        /* Button Styling */
        .stButton button {
            background-color: #0a74da;
            color: white;
            padding: 15px 30px;
            font-size: 20px;
            border-radius: 12px;
            border: none;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .stButton button:hover {
            background-color: #4d9fbb;
            transform: translateY(-3px);
        }

        /* Prediction Result Styling */
        .stWrite {
            font-size: 24px;
            font-weight: 600;
            color: #0a74da;
            text-align: center;
            margin-top: 20px;
            margin-bottom: 30px;
        }

        /* Footer Styling */
        footer {
            background-color: #0a74da;
            color: white;
            padding: 20px;
            text-align: center;
            font-size: 16px;
            border-radius: 10px;
            margin-top: 40px;
        }

        /* Image */
        .image {
            width: 50%;
            margin-top: 30px;
            margin-bottom: 30px;
            display: block;
            margin-left: auto;
            margin-right: auto;
            border-radius: 20px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        }

        /* Result Card */
        .result-card {
            padding: 30px;
            background-color: #e3f2fd;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            font-size: 20px;
            font-weight: 600;
            color: #1565c0;
            margin-top: 20px;
        }
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown('<p class="title">Spam or Ham Prediction</p>', unsafe_allow_html=True)


# Container to hold the input and prediction button
with st.container():
    # Text input for the user message
    text_input = st.text_input("Enter your message:", placeholder="Type a message here...", help="Type any message you want to check for spam or ham.")

    # Dropdown to select the model
    selected_model_name = st.selectbox(
        "Select the model:",
        options=["Logistic Regression", "Naive Bayes", "Support Vector Machine"],
        help="Choose the model you want to use for prediction."
    )

    # If the user clicks the "Predict" button
    if st.button("Predict"):
        if text_input:  # Ensure the input is not empty
            # Vectorize the input text
            input_vectorized = vectorizer.transform([text_input])

            # Predict using the selected model
            selected_model = models[selected_model_name]
            prediction = selected_model.predict(input_vectorized)[0]

            # Convert prediction to label
            prediction_label = "Spam" if prediction == 1 else "Ham"
            
            # Display the prediction result with a stylish result card
            st.markdown(f"""
                <div class="result-card">
                    The message is: <strong>{prediction_label}</strong>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.write("### Please enter a message to predict.")

# Footer with additional information
st.markdown("""
    <footer>
        <p>Made with 💙 by <strong>Zizoune Badr</strong></p>
    </footer>
    """, unsafe_allow_html=True)
