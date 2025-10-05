import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Page Config
st.set_page_config(page_title="Amazon Delivery Time Prediction", page_icon="📦", layout="wide")

# Load trained model
loaded_model = joblib.load("xgboost_pipeline.pkl")

# ---- Top Navigation ----
st.markdown("""
    <style>
        .nav-container {
            display: flex;
            justify-content: center;
            gap: 2rem;
            background-color: #f0f2f6;
            padding: 12px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .nav-item {
            font-size: 18px;
            font-weight: bold;
            color: #333;
            text-decoration: none;
            padding: 6px 12px;
        }
        .nav-item:hover {
            background-color: #ddd;
            border-radius: 6px;
        }
    </style>
""", unsafe_allow_html=True)

# Navigation buttons
nav = st.radio("", ["🏠 Home", "🔮 Prediction", "📊 Insights", "📬 Contact"], horizontal=True)

# ----------------------------- HOME PAGE -----------------------------
if nav == "🏠 Home":
    st.title("📦 Amazon Delivery Time Prediction")
    st.markdown("""
    Welcome to the **Amazon Delivery Time Prediction App** 🚀  
    This project predicts **delivery times** based on:
    - 👤 Agent details  
    - 📍 Store & drop locations  
    - 🛒 Order info (category, weather, traffic)  
    - ⏱ Time & delay factors  

    ---
    🔮 Go to the **Prediction Page** to try it out!
    """)

# ----------------------------- PREDICTION PAGE -----------------------------
elif nav == "🔮 Prediction":
    st.title("🔮 Delivery Time Prediction Form")
    st.write("Please fill in all details below. The **Predict button** will appear at the end.")

    with st.form("prediction_form"):
        # Layout in sections
        st.subheader("👤 Agent Details")
        agent_age = st.number_input("Agent Age", min_value=18, max_value=70, value=30)
        agent_rating = st.number_input("Agent Rating", min_value=0.0, max_value=5.0, value=4.5, step=0.1)

        st.subheader("📍 Location Details")
        store_lat = st.number_input("Store Latitude", value=28.70)
        store_long = st.number_input("Store Longitude", value=77.10)
        drop_lat = st.number_input("Drop Latitude", value=28.54)
        drop_long = st.number_input("Drop Longitude", value=77.39)
        distance_km = st.number_input("Distance (km)", min_value=0.0, value=5.0)

        st.subheader("🛒 Order Details")
        weather = st.selectbox("Weather", ["Sunny", "Rainy", "Cloudy", "Foggy"])
        traffic = st.selectbox("Traffic", ["Low", "Medium", "High"])
        vehicle = st.selectbox("Vehicle", ["Bike", "Car", "Scooter"])
        area = st.selectbox("Area", ["Urban", "Semi-Urban", "Rural"])
        category = st.selectbox("Category", ["Grocery", "Food", "Electronics"])

        st.subheader("⏱ Time Details")
        order_year = st.number_input("Order Year", min_value=2020, max_value=2030, value=2023)
        order_month = st.number_input("Order Month", min_value=1, max_value=12, value=5)
        order_day = st.number_input("Order Day", min_value=1, max_value=31, value=15)
        order_dayofweek = st.number_input("Order Day of Week (0=Mon)", min_value=0, max_value=6, value=2)
        order_hour = st.number_input("Order Hour", min_value=0, max_value=23, value=14)
        order_minute = st.number_input("Order Minute", min_value=0, max_value=59, value=30)
        pickup_hour = st.number_input("Pickup Hour", min_value=0, max_value=23, value=15)
        pickup_minute = st.number_input("Pickup Minute", min_value=0, max_value=59, value=0)
        pickup_delay = st.number_input("Pickup Delay (minutes)", min_value=0, max_value=120, value=10)

        # Submit button at the end
        submitted = st.form_submit_button("🚀 Predict Delivery Time")

    if submitted:
        new_data = pd.DataFrame([{
            "Agent_Age": agent_age,
            "Agent_Rating": agent_rating,
            "Store_Latitude": store_lat,
            "Store_Longitude": store_long,
            "Drop_Latitude": drop_lat,
            "Drop_Longitude": drop_long,
            "Weather": weather,
            "Traffic": traffic,
            "Vehicle": vehicle,
            "Area": area,
            "Category": category,
            "Distance_km": distance_km,
            "Order_Year": order_year,
            "Order_Month": order_month,
            "Order_Day": order_day,
            "Order_DayOfWeek": order_dayofweek,
            "Order_Hour": order_hour,
            "Order_Minute": order_minute,
            "Pickup_Hour": pickup_hour,
            "Pickup_Minute": pickup_minute,
            "Pickup_Delay_Minutes": pickup_delay
        }])

        prediction = loaded_model.predict(new_data)[0]
        st.success(f"⏱ Predicted Delivery Time: **{prediction:.2f} minutes**")

# ----------------------------- INSIGHTS PAGE -----------------------------
elif nav == "📊 Insights":
    st.title("📊 Insights Dashboard")

    df = pd.DataFrame({
        "Traffic": ["Low", "Medium", "High", "Low", "High"],
        "Delivery_Time": [20, 35, 50, 22, 55],
        "Weather": ["Sunny", "Rainy", "Cloudy", "Sunny", "Rainy"]
    })

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🚦 Avg Delivery Time by Traffic")
        traffic_mean = df.groupby("Traffic")["Delivery_Time"].mean()
        st.bar_chart(traffic_mean)

    with col2:
        st.subheader("🌤 Avg Delivery Time by Weather")
        weather_mean = df.groupby("Weather")["Delivery_Time"].mean()
        st.bar_chart(weather_mean)

    st.subheader("📈 Delivery Time Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["Delivery_Time"], kde=True, ax=ax)
    st.pyplot(fig)

# ----------------------------- CONTACT PAGE -----------------------------
elif nav == "📬 Contact":
    st.title("📬 Contact")
    st.markdown("""
    If you’d like to connect:  
    - 📧 Email: **gunturridhi@gmail.com**  
    - 💼 LinkedIn: [Ridhi Guntur ](https://www.linkedin.com/in/ridhi-guntur/)
    - 🌐 GitHub: [Ridhi-215](https://github.com/Ridhi-215/amazon-delivery-time-prediction)  
    """)
