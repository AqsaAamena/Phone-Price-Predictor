import streamlit as st
import pandas as pd
import joblib

# Load dataset to extract options
df = pd.read_csv("Mobile phone price.csv")
df.columns = df.columns.str.strip()  # remove trailing spaces

# Load saved model, scaler, and encoded columns
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
encoded_columns = joblib.load("columns.pkl")

# Configure Streamlit page
st.set_page_config(page_title="📱 Mobile Phone Price Predictor", layout="centered")
st.title("📱 Mobile Phone Price Predictor")
st.write("Select the phone details below and get the predicted **selling price** instantly.")

# Extract unique options from dataset
brand_options = sorted(df["Brand"].dropna().unique())
model_options = sorted(df["Model"].dropna().unique())
storage_options = sorted(df["Storage"].dropna().unique())
ram_options = sorted(df["RAM"].dropna().unique())
camera_options = sorted(df["Camera (MP)"].dropna().unique())

# User inputs via selectbox
brand = st.selectbox("Brand", options=brand_options)
model_name = st.selectbox("Model", options=model_options)
storage = st.selectbox("Storage", options=storage_options)
ram = st.selectbox("RAM", options=ram_options)
screen_size = st.number_input("Screen Size (inches)", min_value=4.0, max_value=7.5, value=6.1)
camera = st.selectbox("Camera (MP)", options=camera_options)
battery = st.number_input("Battery Capacity (mAh)", min_value=1000, max_value=7000, value=4000)
currency = st.selectbox(
        "Select currency for prediction", 
        options=["USD", "INR", "EUR", "GBP"],
        index=1
        )
# Correct: numeric exchange rates
exchange_rates = {
    "USD": 1.0,
    "EUR": 0.92,
    "INR": 83.5,
    "GBP": 0.78
}

currency_symbols = {
    "USD": "$",
    "EUR": "€",
    "INR": "₹",
    "GBP": "£"
}

# Predict button
if st.button("Click to Predict Price"):
    # Build input DataFrame
    input_data = pd.DataFrame({
        "Brand": [brand],
        "Model": [model_name],
        "Storage": [storage],
        "RAM": [ram],
        "Screen Size (inches)": [screen_size],
        "Camera (MP)": [camera],
        "Battery Capacity (mAh)": [battery]
    })

    # One-hot encode
    X_encoded = pd.get_dummies(input_data)

    # Align with training columns
    X_encoded = X_encoded.reindex(columns=encoded_columns, fill_value=0)

    # Scale numeric column
    if "Battery Capacity (mAh)" in X_encoded.columns:
        X_encoded[["Battery Capacity (mAh)"]] = scaler.transform(X_encoded[["Battery Capacity (mAh)"]])

    # Predict
    prediction = model.predict(X_encoded)

    # Show processed input for debugging
    st.write("Processed Input DataFrame:", X_encoded)

    # Display result
    pred_value = float(prediction[0])  # ensure numeric
    st.success(
    f"Predicted Selling Price: {currency_symbols[currency]}{pred_value * exchange_rates[currency]:.2f}"
    )

# Dark background image with readable text
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1518770660439-4636190af475?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=100");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
    color: white;
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0); /* transparent header */
}
[data-testid="stSidebar"] {
    background-color: rgba(25,25,25,0.9); /* dark sidebar */
    color: white;
}
h1, h2, h3, h4, h5, h6, p, label {
    color: #f0f0f0 !important; /* light text for readability */
}
</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)