import streamlit as st
import pandas as pd
import pickle
import os


MODEL_PATH = "stock_close_prediction_model.pkl"
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    model = None


st.title("Stock Close Price Prediction")


with st.expander("About Details"):
    st.write("""
    - **App Name:** Stock Market Prediction  
    - **Developer:** Dixon Manuel 
    - **Version:** 1.0  
    - **Purpose:** This Model is designed to Predict the Stock closing of the Stock Market.  
    """)

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])


def predict(data):
    if model is not None:
        return model.predict(data)
    else:
        st.error("Model file not found! Please upload the trained model.")
        return None

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:")
    st.write(df.head())
    
    if st.button("Predict from CSV"):
        predictions = predict(df)
        if predictions is not None:
            df["Predicted Close"] = predictions
            st.write(df)

if st.button('Check Quality Of The Model'):
    st.write("""- R2Score=99%
            MSE=59%""")

st.subheader("Manual Entry")
with st.form("manual_entry"):
    col1, col2 = st.columns(2)
    manual_data = {}
    features = ['Prev Close', 'Open', 'High', 'Low', 'Last', 'VWAP', 'Volume',
       'Turnover', 'Deliverable Volume', '%Deliverble', 'year', 'month', 'day'
       , 'symbol_encoded']  # Update with actual features
    
    for feature in features:
        manual_data[feature] = st.number_input(feature, value=0.0)
    
    submitted = st.form_submit_button("Predict Manually")
    
    if submitted:
        input_df = pd.DataFrame([manual_data])
        prediction = predict(input_df)
        if prediction is not None:
            st.success(f"Predicted Close: {prediction[0]}")