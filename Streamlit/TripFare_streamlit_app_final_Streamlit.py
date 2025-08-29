# TripFare_streamlit_app_finall.py
# ---------------------------------------------------------
# Streamlit app for Taxi Fare & Total Amount Prediction
# ---------------------------------------------------------

# ========================
# Import Required Libraries
# ========================
import streamlit as st
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ========================
# Load Models & Scaler
# ========================
@st.cache_resource
def load_artifacts():
    with open("C:/Users/Admin/Downloads/FareAmount_prediction_model_lasso.pkl", "rb") as f1:
        fare_model = pickle.load(f1)
    with open("C:/Users/Admin/Downloads/TotalAmount_prediction_model_lasso.pkl", "rb") as f2:
        total_model = pickle.load(f2)
    with open("scaler.pkl", "rb") as f3:
        scaler = pickle.load(f3)
    return fare_model, total_model, scaler

fare_model, total_model, scaler = load_artifacts()

# ✅ Set page config
st.set_page_config(page_title="🚖 Taxi Fare & Total Amount Prediction", layout="wide")

# Sidebar
st.sidebar.image(
    "https://tripoventure.com/wp-content/uploads/2024/04/its-time-travel-travel-world-vacation-road-trip-tourism-travel-banner_508931-29.webp",
    width=100,
)
st.sidebar.title("TripFare Dashboard")

# Page Navigation
page = st.sidebar.radio("Navigate", ["Home", "Predict Fare", "Data Analytics", "About"])

# ========================
# Load Data
# ========================
@st.cache_data
def load_data():
    df = pd.read_csv("C:/Users/Admin/Downloads/TripFare_Cleaned_Data.csv")   # Ensure file is in same folder
    return df

df = load_data()

# ========================
# Home Page
# ========================
if page == "Home":
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("🚖 **Welcome to the Taxi Fare & Trip Dashboard**")
        st.markdown(
            """
            This app lets you:

            🔍 Predict **Fare Amount** and **Total Amount** using trained ML models.  
            📊 Explore relationships in Taxi trip data with interactive graphs.  

            Use the sidebar to get started.
            """
        )
    with col2:
        st.image("https://miro.medium.com/v2/resize:fit:1400/1*-MmCUj_WyrxMj8wMQZHtAg.png")

# ========================
# Predict Fare Page
# ========================
elif page == "Predict Fare":
    st.header("Step 1: Predict Fare Amount")

    duration = st.number_input("Trip Duration (minutes)", min_value=1.0, value=10.0)
    trip_distance = st.number_input("Trip Distance (miles)", min_value=0.1, value=2.5)
    passenger_count = st.number_input("Passenger Count", min_value=1, max_value=6, value=1)
    store_and_fwd_flag = st.selectbox("Store and Forward Flag", ["No", "Yes"], index=0)
    payment_type = st.selectbox("Payment Type", ["Cash", "Card"], index=0)

    # Encode categorical
    store_and_fwd_flag = 1 if store_and_fwd_flag == "Yes" else 0
    payment_type = 1 if payment_type == "Card" else 0

    fare_pred = None

    if st.button("Predict Fare Amount"):
        fare_per_mile = 0.0  # can improve if you compute later
        input_data = pd.DataFrame(
            [[passenger_count, store_and_fwd_flag, payment_type, trip_distance, duration, fare_per_mile]],
            columns=["passenger_count", "store_and_fwd_flag", "payment_type", "trip_distance_miles", "Duration", "fare_per_mile"],
        )
        input_data[["trip_distance_miles", "Duration", "fare_per_mile"]] = scaler.transform(
            input_data[["trip_distance_miles", "Duration", "fare_per_mile"]]
        )
        fare_pred = fare_model.predict(input_data)[0]
        st.success(f"Predicted Fare Amount: ${fare_pred:.2f}")

    # Step 2: Predict Total Amount
    st.header("Step 2: Predict Total Amount")
    use_total = st.radio("Do you want to predict Total Amount?", ["No", "Yes"])

    if use_total == "Yes":
        st.subheader("Enter charges for Total Amount prediction")
        if fare_pred is not None:
            fare_amount = st.number_input("Fare Amount", value=float(fare_pred))
        else:
            fare_amount = st.number_input("Fare Amount", value=0.0)

        extra = st.number_input("Extra Charges", value=0.0)
        mta_tax = st.number_input("MTA Tax", value=0.5)
        tip_amount = st.number_input("Tip Amount", value=0.0)
        tolls_amount = st.number_input("Tolls Amount", value=0.0)
        improvement_surcharge = st.number_input("Improvement Surcharge", value=0.3)

        if st.button("Predict Total Amount"):
            input_data2 = pd.DataFrame(
                [[fare_amount, extra, mta_tax, tip_amount, tolls_amount, improvement_surcharge]],
                columns=["fare_amount", "extra", "mta_tax", "tip_amount", "tolls_amount", "improvement_surcharge"],
            )
            total_pred = total_model.predict(input_data2)[0]
            st.success(f"Predicted Total Amount: ${total_pred:.2f}")

# ========================
# Data Analytics Page
# ========================
elif page == "Data Analytics":
    st.title("🚖 Taxi Fare & Trip Data - Relationship Analysis")

    # Two-column layout (controls left, plots right)
    col_controls, col_plots = st.columns([1, 3])

    with col_controls:
        st.header("⚙️ Plot Options")

        # Predefined Relations Dropdown
        relation_plot = st.selectbox(
            "Choose a relation to visualize:",
            [
                "Fare Amount vs Trip Distance (Fare Data)",
                "Fare per Mile vs Duration Bin (Fare Data)",
                "Fare Amount by Payment Type (Fare Data)",
                "Number of Trips by Passenger Count (Fare Data)",
                "Number of Trips by Payment Type (Fare Data)",
                "Total Amount vs Fare Amount (Total Data)",
                "Tip Amount vs Fare Amount (Total Data)",
                "Tip Amount vs Payment Type (Both Files)",
                "Tolls Amount vs Total Amount (Total Data)",
                "Total Amount vs Passenger Count (Both Files)"
            ]
        )

        st.markdown("---")

    # -----------------------------
    # RIGHT SIDE: DISPLAY PLOTS
    # -----------------------------
    with col_plots:
        # ----- Predefined Relations -----
        if relation_plot == "Fare Amount vs Trip Distance (Fare Data)":
            st.subheader("Fare Amount vs Trip Distance")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(x="trip_distance_miles", y="fare_amount", data=df, alpha=0.3, ax=ax)
            st.pyplot(fig)

        elif relation_plot == "Fare per Mile vs Duration Bin (Fare Data)":
            st.subheader("Fare per Mile vs Duration Bin")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.violinplot(x="Duration", y="fare_amount", data=df, ax=ax)  # better than boxplot for spread
            st.pyplot(fig)

        elif relation_plot == "Fare Amount by Payment Type (Fare Data)":
            st.subheader("Fare Amount by Payment Type")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(x="payment_type", y="fare_amount", data=df, ax=ax)
            st.pyplot(fig)

        elif relation_plot == "Number of Trips by Passenger Count (Fare Data)":
            st.subheader("Trips by Passenger Count")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.barplot(x="passenger_count", y="fare_amount", data=df, estimator=len, ax=ax)  # bar instead of countplot
            st.pyplot(fig)

        elif relation_plot == "Number of Trips by Payment Type (Fare Data)":
            st.subheader("Trips by Payment Type")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.countplot(x="payment_type", data=df, ax=ax)
            st.pyplot(fig)

        elif relation_plot == "Total Amount vs Fare Amount (Total Data)":
            st.subheader("Total Amount vs Fare Amount")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(x="fare_amount", y="total_amount", data=df, alpha=0.3, ax=ax)
            st.pyplot(fig)

        elif relation_plot == "Tip Amount vs Fare Amount (Total Data)":
            st.subheader("Tip Amount vs Fare Amount")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(x="fare_amount", y="tip_amount", data=df, alpha=0.3, ax=ax)
            st.pyplot(fig)

        elif relation_plot == "Tip Amount vs Payment Type (Both Files)":
            st.subheader("Tip Amount vs Payment Type")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.violinplot(x="payment_type", y="tip_amount", data=df, ax=ax)
            st.pyplot(fig)

        elif relation_plot == "Tolls Amount vs Total Amount (Total Data)":
            st.subheader("Tolls Amount vs Total Amount")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(x="total_amount", y="tolls_amount", data=df, alpha=0.3, ax=ax)
            st.pyplot(fig)

        elif relation_plot == "Total Amount vs Passenger Count (Both Files)":
            st.subheader("Total Amount vs Passenger Count")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(x="passenger_count", y="total_amount", data=df, ax=ax)
            st.pyplot(fig)

        elif relation_plot == "Correlation Heatmap (All Data)":
            st.subheader("Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)


        st.sidebar.success("✅ Select any relationship graph to view insights.")

# ========================
# About Page
# ========================
elif page == "About":
    st.header("📚 About This App")
    st.write("This dashboard is built using **Streamlit**.")
    st.write("**Data Source**: TripFare_Database")
    st.markdown("---")
    st.write("Developed by **Arun Kumar**")
    st.caption("Thank you for visiting!")

# ========================
# Footer
# ========================
st.markdown("---")
st.caption("Developed by Arun Kumar | Powered by GUVI")