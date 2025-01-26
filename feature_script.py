import requests
import datetime
import time
import pandas as pd
import numpy as np
import hopsworks
from requests.exceptions import RequestException
from scipy.stats import zscore
import os
 
# API = 'OW_API_KEY' # OpenWeather API key
API = os.getenv("OW_API_KEY")
URL = "http://api.openweathermap.org/data/2.5/air_pollution/history?lat=24.8607&lon=67.0011&start={start}&end={end}&appid={API}"  # Karachi city coordinates

def fetch_data(api, s, e):
    url = URL.format(start=s, end=e, API=api)  # Format URL with correct start and end times
    resp = requests.get(url)
    if resp.status_code == 200:
        return resp.json()
    else:
        print(f"Error: {resp.status_code} - {resp.text}")  # Print error code and message
        return None

def time_from_date(date):
    return int(date.timestamp())


def store_data(df):
    if df.empty:
        print("No data to upload to the Feature Store.")
        return

    try:
        # Login to Hopsworks
        project = hopsworks.login(api_key_value=os.getenv("HW_API_KEY")) 
        fs = project.get_feature_store()

        # Feature group details
        feature_group_name = "air_quality_features_f_cleaned"
        feature_group_version = 1
        description = "Air quality features"
        primary_key = ['id']
        online_enabled = False

        try:
            # Attempt to retrieve the feature group
            feature_group = fs.get_feature_group(name=feature_group_name, version=feature_group_version)
            feature_group.delete()
            print(f"Existing feature group '{feature_group_name}' deleted.")
        except AttributeError:
            print("Error: The FeatureStore object does not have the required methods.")
            return

        # Create and insert data into the feature group
        feature_group = fs.create_feature_group(
            name=feature_group_name,
            version=feature_group_version,
            description=description,
            primary_key=primary_key,
            online_enabled=online_enabled
        )
        feature_group.insert(df)
        print(f"Feature group '{feature_group_name}' created and data inserted successfully!")

    except Exception as e:
        print(f"Failed to update the Feature Store: {e}")


def fetch_and_process_data(api, s, days):
    response = []  #List to store collected data
    for i in range(days):
        start = s - datetime.timedelta(days=i + 1)  #Subtract i + 1 days to move backward in time
        end = s - datetime.timedelta(days=i)  #One day backward range
        start_time = time_from_date(start)
        end_time = time_from_date(end)
        print(f"Fetching data for: {end.strftime('%Y-%m-%d')} to {start.strftime('%Y-%m-%d')}")
        data = fetch_data(api, start_time, end_time)
        if data and "list" in data:  # Check if the data contains 'list' key
            for j in data["list"]:  # Loop through each entry in 'list'
                res = {
                    "date": datetime.datetime.fromtimestamp(j["dt"]).strftime("%Y-%m-%d %H:%M:%S"),
                    "aqi": j.get("main", {}).get("aqi"),
                    "pm25": j.get("components", {}).get("pm2_5"),
                    "pm10": j.get("components", {}).get("pm10"),
                    "no2": j.get("components", {}).get("no2"),
                    "o3": j.get("components", {}).get("o3"),
                }
                print(res)  # Print result for each entry
                response.append(res)  # Append to the response list
        time.sleep(1)  # Sleep to avoid rate limits

    # Process the data
    data = pd.DataFrame(response)
    data["date"] = pd.to_datetime(data["date"])  # Ensure 'date' column is in datetime format
    data.sort_values(by="date", inplace=True)
    data.set_index("date", inplace=True)

    # Define Target
    data["future_aqi"] = data["aqi"].shift(-72)  # Shift AQI to create a future prediction target
    data.replace(-9999, np.nan, inplace=True)
    features_to_fill = [col for col in data.columns if col != "future_aqi"]
    data[features_to_fill] = data[features_to_fill].fillna(data[features_to_fill].median())

    # Log-transform skewed data
    data["pm25"] = np.log1p(data["pm25"])
    data["pm10"] = np.log1p(data["pm10"])

    # Detect and remove outliers using z-score
    data = data[(np.abs(zscore(data[["pm25", "pm10"]])) < 3).all(axis=1)]

    # Basic Features
    data["hour"] = data.index.hour
    data["day"] = data.index.day
    data["month"] = data.index.month
    data["day_of_week"] = data.index.dayofweek
    data["is_weekend"] = data["day_of_week"].isin([5, 6]).astype(int)

    # Derived Features
    data["rolling_pm25_mean"] = data["pm25"].rolling(window=3).mean()
    data["rolling_pm10_mean"] = data["pm10"].rolling(window=3).mean()
    data["rolling_pm25_std"] = data["pm25"].rolling(window=3).std()
    data["rolling_pm10_std"] = data["pm10"].rolling(window=3).std()

    data["pm25_change"] = data["pm25"].diff()  # Change rate for PM2.5
    data["pm10_change"] = data["pm10"].diff()  # Change rate for PM10

    # Lag Features
    data["lag_1_pm25"] = data["pm25"].shift(1)
    data["lag_1_pm10"] = data["pm10"].shift(1)
    data["lag_1_aqi"] = data["aqi"].shift(1)

    # Interaction Features
    data["pm25_pm10_ratio"] = data["pm25"] / (data["pm10"] + 1e-9)
    # Drop rows with NaN in features but retain NaN in 'future_aqi'
    feature_columns = [col for col in data.columns if col != "future_aqi"]
    data = data.dropna(subset=feature_columns)
    data.drop_duplicates(inplace=True)

    data.reset_index(drop=True, inplace=True)
    data['id'] = data.index + 1

    try:
        print("Features to be stored")
        print(data.head())
        store_data(data)
    except Exception as e:
        print(f"Error saving file: {e}")

if __name__ == "__main__":
    start_date = datetime.datetime.today()
    days = 5
    fetch_and_process_data(API, start_date, days)
