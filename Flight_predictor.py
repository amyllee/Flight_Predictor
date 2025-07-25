import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gdown
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import os

# ---------- Load Data ----------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("domestic.csv")
        return df
    except FileNotFoundError:
        st.error(f"domestic.csv file no found. Please ensure the file is in the same directory.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# ---------- Load Full Model ----------
@st.cache_resource
def load_model():
    file_id = "1yK9Xn30O_EotvDpHDgBSnO-cSaI4Hdyq"
    download_url = f"https://drive.google.com/uc?id={file_id}"
    output = "model.pkl"

    # Only download if file doesn't exist
    if not os.path.exists(output):
        try:
            gdown.download(download_url, output, quiet=False)
        except Exception as e:
            st.error(f"Failed to download model: {e}")
            return None
    
    try:
        with open(output, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# ---------- Load Minimal Model ----------
@st.cache_resource
def load_minimal_model():
    df = load_data()
    if df is None:
        return None
    
    df_clean = df.dropna(subset=['nsmiles', 'quarter', 'fare'])
    if df_clean.empty:
        st.error("No valid data found for minimal model training")
        return None
    
    X_min = df_clean[['nsmiles', 'quarter']]
    y_min = df_clean['fare']
    minimal_model = RandomForestRegressor(random_state=42, n_estimators=50)
    minimal_model.fit(X_min, y_min)
    return minimal_model

# ---------- Load City Options ----------
@st.cache_data
def get_city_list():
    df = load_data()
    if df is None:
        return [], []
    
    city1_list = sorted(df['city1'].dropna().unique().tolist())
    city2_list = sorted(df['city2'].dropna().unique().tolist())
    return city1_list, city2_list

# ---------- Distance Lookup Function ----------
@st.cache_data
def lookup_distance():
    df = load_data()
    if df is None:
        return pd.DataFrame()
    
    distance_df = df[['city1', 'city2', 'nsmiles']].dropna().drop_duplicates()
    return distance_df

# Initialize everything
df_main = load_data()
if df_main is not None:
    model = load_model()
    minimal_model = load_minimal_model()
    city1_list, city2_list = get_city_list()
    distance_df = lookup_distance()
else:
    st.stop()

# ---------- Streamlit UI ----------
st.title("Flight Fare Predictor ✈️")
st.write("Estimate the average fare for a U.S. domestic flight route")

# ---------- Model Selection ----------
st.subheader("Choose Prediction Model")
model_type = st.radio(
    "Select Model Type",
    ("Full Model (more accurate)", "Minimal Model (simplified)")
)

# Check if models loaded successfully
if model_type == "Full Model (more accurate)" and model is None:
    st.error("Full model failed to load. Please try the minimal model.")
    st.stop()
elif model_type == "Minimal Model (simplified)" and minimal_model is None:
    st.error("Minimal model failed to load. Please check your data file.")
    st.stop()

# ---------- City Selection ----------
st.subheader("Select Cities")
if city1_list and city2_list:
    origin = st.selectbox("Origin City", city1_list)
    destination = st.selectbox("Destination City", city2_list)

    # Find distance between cities
    matched_row = distance_df[
        ((distance_df['city1'] == origin) & (distance_df['city2'] == destination)) |
        ((distance_df['city1'] == destination) & (distance_df['city2'] == origin))
    ]

    if not matched_row.empty:
        nsmiles = matched_row.iloc[0]['nsmiles']
        st.write(f"Distance between {origin} and {destination}: {nsmiles:.0f} miles")
    else:
        st.warning("Distance not found for this route. Please enter manually.")
        nsmiles = st.number_input("Distance between airports (miles)", min_value=0, value=500)
else:
    st.error("No city data available")
    st.stop()

# ---------- User Input ----------
st.subheader("Enter Route Features")
quarter = st.selectbox("Quarter", [1, 2, 3, 4])

# Full model inputs
if model_type == "Full Model (more accurate)":
    passengers = st.number_input("Number of passengers", min_value=0, value=200)
    large_ms = st.slider("Market Share of Largest Carrier", 0.0, 1.0, 0.5)
    lf_ms = st.slider("Market Share of Lowest Fare Carrier", 0.0, 1.0, 0.2)
    fare_low = st.number_input("Lowest Fare Offered ($)", min_value=0.0, value=100.0)
    fare_lg = st.number_input("Average Fare of Largest Carrier ($)", min_value=0.0, value=150.0)

# ---------- Predict Fare ----------
if st.button("Predict Fare"):
    try:
        if model_type == "Full Model (more accurate)":
            input_arr = np.array([[nsmiles, passengers, quarter, large_ms, lf_ms, fare_low, fare_lg]])
            prediction = model.predict(input_arr)[0]
        # Use minimal model
        else:
            input_arr = np.array([[nsmiles, quarter]])
            prediction = minimal_model.predict(input_arr)[0]

        st.success(f"Estimated Average Fare 🎯: ${prediction:.2f}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ---------- Feature Importance ----------
if st.checkbox("Show Feature Importance 📊"):
    try:
        if model_type == "Full Model (more accurate)": 
            importances = model.feature_importances_
            features = ['nsmiles', 'passengers', 'quarter', 'large_ms', 'lf_ms', 'fare_low', 'fare_lg']
        else:
            importances = minimal_model.feature_importances_
            features = ['nsmiles', 'quarter']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.barh(features, importances)
        ax.set_title("Feature Importance")
        ax.set_xlabel("Importance")

        # Label values on bars
        for bar, importance in zip(bars, importances):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{importance:.3f}', va='center')
        
        plt.tight_layout()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Feature importance plot failed: {e}")

# ---------- Predicted vs. Actual ---------- 
if st.checkbox("Show Predicted vs. Actual 📈"):
    @st.cache_data
    def load_sample_data(model_type_param):
        df = load_data()
        if df is None:
            return None, None
    
        try:
            if model_type_param == "Full Model (more accurate)":
                # Check that all required columns exist
                required_cols = ['nsmiles', 'passengers', 'quarter', 'large_ms', 'lf_ms', 'fare_low', 'fare_lg', 'fare']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    st.warning(f"Missing columns for full model: {missing_cols}")
                    return None, None
            
                df_clean = df.dropna(subset=required_cols)
                X = df_clean[['nsmiles', 'passengers', 'quarter', 'large_ms', 'lf_ms', 'fare_low', 'fare_lg']]
            # Use minimal model
            else:
                df_clean = df.dropna(subset=['nsmiles', 'quarter', 'fare'])
                X = df_clean[['nsmiles', 'quarter']]
        
            y = df_clean['fare']

            # Limit sample size for performance
            if len(X) > 1000:
                sample_idx = np.random.choice(len(X), 1000, replace=False)
                X = X.iloc[sample_idx]
                y = y.iloc[sample_idx]

            return X, y
        except Exception as e:
            st.error(f"Error preparing sample data: {e}")
            return None, None

    try:
        X, y = load_sample_data(model_type)

        if X is not None and y is not None:
            # Use the user-chosen model
            if model_type == "Full Model (more accurate)" and model is not None:
                y_pred = model.predict(X)
            else:
                y_pred = minimal_model.predict(X)
                
            fig, ax = plt.subplots(figsize=(8,6))
            ax.scatter(y, y_pred, alpha=0.5)

            # Create perfect prediction line
            min_val, max_val = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')

            ax.set_xlabel("Actual Fare ($)")
            ax.set_ylabel("Predicted Fare ($)")
            ax.set_title("Predicted vs. Actual Fare")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)

            # Calculate metrics
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mse)
                
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("R² Score", f"{r2:.3f}")
            with col2:
                st.metric("RMSE", f"{rmse:.2f}")
            with col3:
                st.metric("MSE", f"{mse:.2f}")
        else:
            st.warning(f"Couldn't load sample data for evaluation")

    except Exception as e:
        st.error(f"Evaluation plot failed: {e}")

# ---------- Display Data Info ----------
if st.checkbox("Show Data Info 📋"):
    if df_main is not None:
        st.write("### Dataset Overview")
        st.write(f"**Total Rows:** {len(df_main):,}")
        st.write(f"**Columns:** {', '.join(df_main.columns)}")

        # Show sample data
        st.write("### Sample Data")
        st.dataframe(df_main.head())

        # Show missing values
        st.write("### Missing Values")
        missing_data = df_main.isnull().sum()
        if missing_data.sum() > 0:
            st.dataframe(missing_data[missing_data > 0])
        else:
            st.write("No missing values found!")