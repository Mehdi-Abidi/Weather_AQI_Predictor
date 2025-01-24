#Model #1
import os
from math import sqrt
import hopsworks
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error,root_mean_squared_error
import numpy as np

# Fetch the data
def fetch_historical_data():
    project = hopsworks.login(api_key_value=os.getenv("HW_API_KEY"))
    fs = project.get_feature_store()
    feature_group_name = "air_quality_features_f_cleaned"
    feature_group_version = 1

    try:
        feature_group = fs.get_feature_group(name=feature_group_name, version=feature_group_version)
        historical_data = feature_group.read()
        print("Fetched historical data successfully!")
        return historical_data
    except Exception as e:
        print(f"Failed to fetch historical data: {e}")
        return None

historical_data = fetch_historical_data()
from scipy.stats import zscore
historical_data = historical_data[(np.abs(zscore(historical_data[['pm25', 'pm10']])) < 3).all(axis=1)]
if historical_data is not None:
    # Preprocessing
    X = historical_data.drop(["future_aqi", "id"], axis=1, errors="ignore")  # Drop target and ID
    y = historical_data["future_aqi"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)

    alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

# Store results
results = []

# Iterate through each alpha and fit the Ridge model
for alpha in alphas:
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)

    # Predict on test data
    test_predictions = model.predict(X_test)
    train_predictions = model.predict(X_train)
    # Calculate MSE and R²
    mae_train = mean_absolute_error(y_train, train_predictions)
    mae_test = mean_absolute_error(y_test, test_predictions)
    mse_train = mean_squared_error(y_train, train_predictions)
    mse_test = mean_squared_error(y_test, test_predictions)
    rmse_train = root_mean_squared_error(y_train, train_predictions)
    rmse_test = root_mean_squared_error(y_test, test_predictions)
    r2_test = model.score(X_test, y_test)  # R² score

    # Save results
    results.append((alpha,mse_train,mse_test,mae_train,mae_test,rmse_train,rmse_test,r2_test))

# Display the results
print("Ridge Regression Results:")
for alpha,mse_train,mse,maet,mae,rmset,rmse,r2 in results:
    #rootmse = sqrt(mse)
    print(f"Alpha: {alpha}, Train MSE: {mse_train:.6f}, Test MSE: {mse:.6f}, Train MAE: {maet:.6f}, Test MAE: {mae:.6f}, Train RMSE: {rmset:.6f}, Test RMSE:{rmse:.6f},  Test R²: {r2:.7f}")


# Model 2

import hopsworks
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error,root_mean_squared_error
from math import sqrt
import numpy as np

# Fetch the data
def fetch_historical_data():
    project = hopsworks.login(api_key_value=os.getenv("HW_API_KEY"))
    fs = project.get_feature_store()
    feature_group_name = "air_quality_features_f_cleaned"
    feature_group_version = 1

    try:
        feature_group = fs.get_feature_group(name=feature_group_name, version=feature_group_version)
        historical_data = feature_group.read()
        print("Fetched historical data successfully!")
        return historical_data
    except Exception as e:
        print(f"Failed to fetch historical data: {e}")
        return None

historical_data = fetch_historical_data()

if historical_data is not None:
    historical_data = historical_data.replace(-9999, np.nan).dropna()
    from scipy.stats import zscore
    historical_data = historical_data[(np.abs(zscore(historical_data[['pm25', 'pm10']])) < 3).all(axis=1)]

    # Preprocessing
    X = historical_data.drop(["future_aqi", "id"], axis=1, errors="ignore")  # Drop target and ID
    y = historical_data["future_aqi"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)

    # Initialize the Random Forest model
    rf_model = RandomForestRegressor(
        n_estimators=200,  # Number of trees in the forest
        max_depth=None,    # Maximum depth of each tree
        random_state=42    # Seed for reproducibility
    )

    # Train the model
    rf_model.fit(X_train, y_train)

    # Predict on test data
    test_predictions = rf_model.predict(X_test)
    train_predictions = rf_model.predict(X_train)

    # Evaluate the model
    mae_train = mean_absolute_error(y_train, train_predictions)
    mae_test = mean_absolute_error(y_test, test_predictions)
    mse_train = mean_squared_error(y_train, train_predictions)
    mse_test = mean_squared_error(y_test, test_predictions)
    rmse_train = sqrt(mse_train)
    rmse_test = sqrt(mse_test)
    r2_test = rf_model.score(X_test, y_test)  # R² score

    # Display the results
    print("Random Forest Results:")
    print(f"Train MSE: {mse_train:.4f}, Test MSE: {mse_test:.4f}")
    print(f"Train MAE: {mae_train:.4f}, Test MAE: {mae_test:.4f}")
    print(f"Train RMSE: {rmse_train:.4f}, Test RMSE: {rmse_test:.4f}")
    print(f"Test R²: {r2_test:.4f}")
else:
    print("No data fetched from the feature store.")



# Storing Models into the Model Registry

import os
import joblib

# Specify the model directory
model_dir = "models"
# Create directories if they don't exist
os.makedirs(model_dir, exist_ok=True)

import hopsworks
project = hopsworks.login(api_key_value=os.getenv("HW_API_KEY"))
mr = project.get_model_registry()

#exporting ridge regression model to model dir
model_path = os.path.join(model_dir, "rr_model.json")
joblib.dump(model, model_path)

#exporting random forest regression model to model dir
model_path = os.path.join(model_dir, "rf_model.json")
joblib.dump(rf_model, model_path)

model_dir = "/content/models"
model_files = [f for f in os.listdir(model_dir) if f.endswith(".json")]

for model_file in model_files:
    model_path = os.path.join(model_dir, model_file)
    model_name = os.path.splitext(model_file)[0]

    # Createing and register the model in Hopsworks

    model = mr.python.create_model(
        name=model_name,
        description=f"Model {model_name} for AQI prediction"
    )

    # Upload the model file to the Model Registry

    model.save(model_path)
    print(f"Registered and uploaded model: {model_name}")
