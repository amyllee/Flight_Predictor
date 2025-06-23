import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load data
df = pd.read_csv("domestic.csv")
df.info()

# Select Features and Target
X = df[['nsmiles',
        'passengers',
        'quarter',
        'large_ms',
        'lf_ms',
        'fare_low',
        'fare_lg' ]]
y = df['fare']

# Data Cleanup - Handle Missing Values
X = X.dropna()
y = y.loc[X.index]

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error: ", mse)
print("R² Score: ", r2)

# Save model for streamlit
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)