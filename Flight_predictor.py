import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib as plt
from sklearn.metrics import mean_squared_error, r2_score

# ---------- Load Model ----------
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# ---------- Load City Options ----------
@st.cache_data
def get_city_list():
    df = pd.read_csv("domestic.csv")
    city1_list = sorted(df['city1'].dropna().unique().tolist())
    city2_list = sorted(df['city2'].dropna().unique().tolist())
    return city1_list, city2_list

city1_list, city2_list = get_city_list()

# ---------- Streamlit UI ----------
st.title("Flight Fare Predictor ✈️")
st.write("Estimate the average fare for a U.S. domestic flight route.")

# ---------- City Selection ----------
st.subheader("Select Cities")
origin = st.selectbox("Origin City", city1_list)
destination = st.selectbox("Destination City", city2_list)

# ---------- User Input ----------
st.subheader("Enter Route Features")

nsmiles = st.number_input("Distance between airports (miles)", min_value=0, value=500)
passengers = st.number_input("Number of passengers", min_value=0, value=200)
quarter = st.selectbox("Quarter", [1, 2, 3, 4])
large_ms = st.slider("Market Share of Largest Carrier", 0.0, 1.0, 0.5)
lf_ms = st.slider("Market Share of Lowest Fare Carrier", 0.0, 1.0, 0.2)
fare_low = st.number_input("Lowest Fare Offered ($)", min_value=0.0, value=100.0)
fare_lg = st.number_input("Average Fare of Largest Carrier ($)", min_value=0.0, value=150.0)

# ---------- Predict Fare ----------
if st.button("Predict Fare"):
    input_arr = np.array([[nsmiles, passengers, quarter, large_ms, lf_ms, fare_low, fare_lg]])
    prediction = model.predict(input_arr)[0]
    st.success(f"Estimated Average Fare 🎯: ${prediction:.2f}")

# ---------- Feature Importance ----------
if st.checkbox("Show Feature Importance 📊"):
    importances = model.feature_importances_
    features = ['nsmiles', 'passengers', 'quarter', 'large_ms', 'lf_ms', 'fare_low', 'fare_lg']

    fig, ax = plt.subplots()
    ax.barh(features, importances)
    ax.set_title("Feature Importance")
    st.pyplot(fig)

# ---------- Predicted vs. Actual ---------- 
if st.checkbox("Show Predicted vs. Actual 📈"):
    @st.cache_data
    def load_sample_data():
        df = pd.read_csv("domestic.csv")
        df = df.dropna(subset=['nsmiles', 'passengers', 'quarter', 'large_ms', 'lf_ms', 'fare_low', 'fare_lg'])
        y = df['fare']
        return X, y
    
    try:
        X, y = load_sample_data()
        y_pred = model.predict(X)

        fig, ax = plt.subplots()
        ax.scatter(y, y_pred, alpha=0.3)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
        ax.set_xlabel("Actual Fare")
        ax.set_ylabel("Predicted Fare")
        ax.set_title("Predicted vs. Actual Fare")
        st.pyplot(fig)

        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        st.markdown(f"**MSE:** {mse:.2f} &nbsp;&nbsp;&nbsp; **R²:** {r2:.2f}")

    except Exception as e:
        st.warning(f"Couldn't load sample data: {e}")
