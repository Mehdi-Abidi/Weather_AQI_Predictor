import streamlit as st
import pandas as pd
import plotly.express as px
import datetime
import random
import requests
import json
import hopsworks
import os
import joblib




# Fetch Current Weather Data
def get_current_weather(latitude, longitude , api_key):
    base_url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": latitude, "lon": longitude , "appid": api_key, "units": "metric"}  # Units: Metric for °C
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return None

# Your OpenWeather API key
API_KEY = "ced66015f7438be60809c67c3c2876d4"

# Get Karachi Weather Data
weather_data = get_current_weather(24.8607, 67.0011, API_KEY)

# Page Configurations
st.set_page_config(
    page_title="Karachi AQI Predictor",
    page_icon="🌥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Enhanced UI
st.markdown("""
<style>
    .stApp {
        background-image: radial-gradient( circle farthest-corner at 50% 50%,  rgba(100,43,115,1) 10%, rgba(4,0,4,1) 100% );
        font-family: 'Segoe UI', sans-serif;
        padding: 20px;
    }
    .title-container {
        text-align: center;
        margin-bottom: 20px;
    }
    .main-container {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }

    .weather-box {
        background: rgba(0, 0, 0, 0.5);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .weather-box img {
        width: 100px;
        height: 100px;
    }
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 10px;
        max-width: 400px;
    }
    .st-emotion-cache-ocqkz7{
            background:white;}
    .st-emotion-cache-1cvow4s{
            all:unset;}
    .stMarkdown{
            all:unset;}
    .weather-box{
            all:unset;
            width:50%;}
    .st-emotion-cache-o1a98x{
            all:unset;}  
    .weather-box img {
        width: 50%;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.8);
        border-radius: 8px;
        padding: 10px;
        transition: transform 0.2s ease;
        text-align: center;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
    }
    [data-testid="stMetricValue"] {
        color: #333333 !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        margin-bottom: 2px !important;
    }
    [data-testid="stMetricLabel"] {
        color: #666666 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    .weather-icon {
        display: block;
        margin: 0 auto;
        width: 250px; /* Adjust the size as needed */
        height: 250px; /* Adjust the size as needed */
    }
    .st-emotion-cache-6qob1r{
        background: #230129;
    }
    .st-emotion-cache-h4xjwg{
        background: #230129;
    }
</style>
""", unsafe_allow_html=True)

# Add custom CSS for alerts
st.markdown("""
<style>
.alert-hazard {
    background: rgba(255, 0, 4, 0.2);
    backdrop-filter: blur(10px);
    color: white;
    border: 1px solid rgba(255, 23, 4, 0.6);
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    font-size: 16px;
    text-align: center;
    margin-bottom: 25px 0;
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.2);
    margin-top: 50px;
    font-family: 'trebuchet MS', 'Lucida Sans', Arial;
}

.alert-safe {
    background: rgba(77, 255, 77, 0.1);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(77, 255, 77, 0.5);
    border-radius: 12px;
    padding: 20px;
    margin: 25px 0;
    margin-top: 50px;
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.2);
    transition: all 0.3s ease;
    font-family: 'trebuchet MS', 'Lucida Sans', Arial;

}

.alert-title {
    font-size: 1.4rem;
    font-weight: 600;
    margin-bottom: 12px;
    text-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.alert-content {
    font-size: 1.1rem;
    line-height: 1.5;
    opacity: 0.9;
}
</style>
""", unsafe_allow_html=True)

# Add custom CSS with width constraints
st.markdown("""
<style>
    div[data-testid="stLottieContainer"] {
        display: flex !important;
        justify-content: center !important;
        max-width: 400px !important;
        margin: 0 auto !important;
        background: white !important;
        margin-top: 20px !important;
        margin-bottom: 20px !important;
    }
    
    .element-container:has(.stLottie) {
        display: flex !important;
        justify-content: center !important;
        max-width: 300px !important;
        margin: 0 auto !important;
        padding: 20px !important;
        margin-top: 20px !important;
        margin-bottom: 20px !important;
    }
    
    iframe {
        max-width: 300px !important;
        margin-top: 20px !important;
        margin-bottom: 20px !important;
    }
</style>
""", unsafe_allow_html=True)

# Add custom CSS for metrics
st.markdown("""
<style>
    /* Reduce space between metrics */
    [data-testid="metric-container"] {
        padding: 0px !important;
        margin: 0px !important;
    }
    
    /* Compact metric layout */
    [data-testid="stMetricValue"] {
        font-size: 1.2rem !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
    }
    
    /* Adjust column spacing */
    [data-testid="column"] {
        padding: 0 !important;
        margin: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# Add custom CSS for metrics layout
st.markdown("""
<style>
    /* Metrics grid container */
    .metrics-container {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 5px;
        padding: 10px;
        max-width: 400px;
        margin: 0 auto;
    }
    
    /* Individual metric styling */
    [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.05);
        padding: 8px !important;
        margin: 0 !important;
        border-radius: 8px;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 1.1rem !important;
        line-height: 1.2 !important;
        margin: 0 !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.8rem !important;
        margin: 0 !important;
    }
    
    [data-testid="stMetricDelta"] {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# Add custom CSS for metrics layout
st.markdown("""
<style>
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 10px;
        max-width: 400px;
        margin: 10px auto;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 15px;
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
    }
    
    [data-testid="metric-container"] {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 1.4rem !important;
        font-weight: 600 !important;
        margin-bottom: 5px !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: rgba(255, 255, 255, 0.8) !important;
        font-size: 0.9rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
</style>
""", unsafe_allow_html=True)

# Title and Description
st.markdown('<div class="title-container"><h1>Karachi AQI Predictor</h1></div>', unsafe_allow_html=True)

st.markdown("""
<div class="main-container">
    <h2>Welcome to the Karachi Air Quality Predictor App! 🚀</h2>
    <p>This app provides AQI predictions for the next 3 days using machine learning models and OpenWeather API data. Stay informed about the air quality in the city of Karachi and plan accordingly!</p>
</div>
""", unsafe_allow_html=True)

# Weather display section
st.subheader("🌧️ Current Weather in Karachi")
if weather_data:
    weather_description = weather_data["weather"][0]["main"]
    weather_icon = weather_data["weather"][0]["icon"]
    temperature = weather_data["main"]["temp"]
    feels_like = weather_data["main"]["feels_like"]
    humidity = weather_data["main"]["humidity"]
    wind_speed = weather_data["wind"]["speed"]
    pressure = weather_data["main"]["pressure"]
    current_time = weather_data["dt"]

    # Weather icon URL
    icon_url = f"https://openweathermap.org/img/wn/{weather_icon}@4x.png"

    st.markdown("""
    <style>
        .weather-box {
            display: flex;
            justify-content: center;
            align-items: center;
            width: 100%;
            background-color: rgba(A19, 255, 255, 1); /* Change this to your desired background color */
            padding: 20px;
            border-radius: 10px;
        }
        .weather-box img {
            display: block;
            margin: auto;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="weather-box">', unsafe_allow_html=True)
    left_co, cent_co1, cent_co2, last_co = st.columns([1, 0.70, 1, 1.5])
    with cent_co2:
        st.image(icon_url, use_container_width=False)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display weather metrics in grid
    st.markdown('<div class="metric-grid">', unsafe_allow_html=True)

    metrics = [
        {"label": "Temperature", "value": f"{temperature}°C"},
        {"label": "Feels Like", "value": f"{feels_like}°C"},
        {"label": "Weather", "value": weather_description},
        {"label": "Humidity", "value": f"{humidity}%"},
        {"label": "Wind Speed", "value": f"{wind_speed} m/s"},
        {"label": "Pressure", "value": f"{pressure} hPa"}
    ]

    for metric in metrics:
        st.markdown(f"""
            <div class="metric-card">
                <div data-testid="metric-container">
                    <div data-testid="stMetricValue">{metric['value']}</div>
                    <div data-testid="stMetricLabel">{metric['label']}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------------------------

def fetch_historical_data():
    project = hopsworks.login(api_key_value="CfEguTsh4ZmL50zX.XacymCLSjFjetGdkZ5kGRr5RMu4cMe0NVT7zFQKLE2E8pjsilqVhLbFZOj3YX3CE")
    fs = project.get_feature_store()
    feature_group_name = "air_quality_nan_rows"
    feature_group_version = 1

    try:
        # Retrieve data from the feature group
        feature_group = fs.get_feature_group(name=feature_group_name, version=feature_group_version)
        historical_data = feature_group.read()
        print("Fetched historical data successfully!")
        return historical_data
    except Exception as e:
        print(f"Failed to fetch historical data: {e}")
        return None
historical_data = fetch_historical_data()
if historical_data is not None:
    print(historical_data.head())  # Display the first few rows
    
# ---------------------------------------------------------------------------------------------------------------------------------

project = hopsworks.login(api_key_value="CfEguTsh4ZmL50zX.XacymCLSjFjetGdkZ5kGRr5RMu4cMe0NVT7zFQKLE2E8pjsilqVhLbFZOj3YX3CE")
mr = project.get_model_registry()

retrieved_model_dir = "retrieved_models"
os.makedirs(retrieved_model_dir, exist_ok=True)

# List of model names to retrieve
model_names = ["rf_model"]  # Versioned names
model_version = 14  # Specify the version of the models to retrieve

# Loop through model names and download them
for model_name in model_names:
    print(f"Retrieving model: {model_name} (version {model_version})")

    # Get the model from Hopsworks
    model = mr.get_model(model_name, version=model_version)

    # Download the model to the local directory
    model_dir = model.download(retrieved_model_dir)
    print(f"Model '{model_name}' downloaded to: {model_dir}")

    # Load the model using joblib
    model_file_path = os.path.join(retrieved_model_dir, f"{model_name}.json")
    if os.path.exists(model_file_path):
        rf_model = joblib.load(model_file_path)
        print(f"Random Forest model '{model_name}' loaded successfully!")
    else:
        print(f"Model file '{model_file_path}' not found!")


df = historical_data

print(df.head())

ids = df['id']
old_df = df
df.drop(columns=['id', 'future_aqi'], inplace=True)
X_test_final = df
preds = rf_model.predict(X_test_final)

if 'year' not in old_df.columns:
    old_df['year'] = datetime.datetime.now().year

# Create the 'datetime' column with date and hour
old_df['datetime'] = pd.to_datetime(old_df[['year', 'month', 'day', 'hour']])

print("Updated DataFrame with 'datetime' column:")
print(old_df.head())

print(old_df.columns.tolist())

old_df['datetime'] = old_df['datetime'] + pd.Timedelta(days=3)

result_df = pd.DataFrame({
    'id': ids,
    'current_aqi': df['aqi'],
    '3_day_future_aqi': preds,
    'datetime': old_df['datetime']  # Add the new datetime column
})

result_df = result_df.sort_values(by=['id'])
selected_data = result_df.iloc[::8]
table_data = result_df.iloc[[23, 47, 71]]

print("These are the predictions in sorted order")
print(result_df)

print("These are the selected predictions to be shown on the chart (every 8th value)")
print(selected_data)

selected_data['datetime'] = pd.to_datetime(selected_data['datetime'])

# --------------------------------------------------------------------

# Mapping AQI Levels to Descriptions
aqi_descriptions = {
    1: "Good",
    2: "Fair",
    3: "Moderate",
    4: "Poor",
    5: "Very Poor",
}

# Generate AQI Predictions
st.subheader("🔁 Predicted AQI for Karachi (Next 3 Days)")
# Generate the prediction data
aqi_data = table_data.copy()
aqi_data["Description"] = aqi_data["3_day_future_aqi"].map(aqi_descriptions)
aqi_data["Date"] = aqi_data["datetime"].dt.strftime('%Y-%m-%d')  # Show only the date
aqi_data["3_day_future_aqi"] = aqi_data["3_day_future_aqi"].astype(int)
aqi_data["Predicted AQI"] = aqi_data["3_day_future_aqi"]

# Set the index to start from 1 to 3
aqi_data.index = range(1, len(aqi_data) + 1)

# Select only the necessary columns
aqi_data = aqi_data[["Date", "Predicted AQI", "Description"]]

# Centered Table for Predictions
st.markdown('<div class="prediction-table-container">', unsafe_allow_html=True)
st.table(aqi_data)
st.markdown("</div>", unsafe_allow_html=True)

# Visualization

# Sample data for AQI levels for the next 3 days with every 8-hour interval
data = {
    "datetime": pd.date_range(start=pd.Timestamp.now(), periods=9, freq='8h'),
    "AQI": selected_data["3_day_future_aqi"].values,
}

df = pd.DataFrame(data)

# Define AQI levels and corresponding colors
aqi_levels = [
    {"range": (1, 1), "color": "rgb(75,209,84,0.3)", "label": "Good"},
    {"range": (2, 2), "color": "#rgb(224,227,70,0.3)", "label": "Fair"},
    {"range": (3, 3), "color": "rgb(242,102,56,0.3)", "label": "Moderate"},
    {"range": (4, 4), "color": "rgb(232,42,58,0.3)", "label": "Poor"},
    {"range": (5, 5), "color": "rgb(87,5,21,0.3)", "label": "Very Poor"}
]

# Function to get color based on AQI value
def get_aqi_color(aqi):
    for level in aqi_levels:
        if level["range"][0] <= aqi <= level["range"][1]:
            return level["color"]
    return "black"

# Add color column to the dataframe
df["color"] = df["AQI"].apply(get_aqi_color)

# Create the plotly figure
fig = px.line(df, x='datetime', y='AQI', title='📈 AQI Trend for the Next 3 Days',
              labels={'datetime': 'Date and Time', 'AQI': 'Air Quality Index'},
              markers=True)

# Update the trace to use the colors based on AQI levels
fig.update_traces(marker=dict(color=df["color"], size=12,line=dict(color='rgba(0, 0, 0, 0.8)',width=2)), line=dict(color="#a0b0a4", width=2))

# Update layout for better styling
fig.update_layout(
    title_font_size=20,
    xaxis_title_font_size=16,
    yaxis_title_font_size=16,
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(255,255,255,0.1)",
    font=dict(color="white"),
    xaxis=dict(
        showgrid=True,
        gridcolor='rgba(255, 255, 255, 0.2)',  # Change this to your desired gridline color
        tickmode='array',
        tickvals=df['datetime'],
        ticktext=df['datetime'].dt.strftime('%H:00 %d-%m-%Y'),  # Show date and time
        tickfont=dict(size=12),  # Smaller tick label font size
        tickson="labels"  # Align ticks with labels
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor='rgba(255, 255, 255, 0.2)',  # Change this to your desired gridline color
        range=[0.5, 5.5],
        tickmode='linear',
        dtick=1
    ),
    autosize = True,
    margin=dict(l=60, r=60, t=80, b=100),  # Adjusted margins
    height=500,  # Set chart height
    hovermode='x unified'  # Enhanced hover interaction
)

# Display the chart in Streamlit
st.plotly_chart(fig, use_container_width=True)

# Sidebar Alert
hazardous_detected = any(aqi_data["Predicted AQI"].isin([4, 5]))
# Update sidebar alert
if hazardous_detected:
    with st.sidebar:
        st.markdown("""
        <div class="alert-hazard">
            <div class="alert-title">
                🚨 Hazardous Air Quality Alert!
            </div>
            <div class="alert-content">
                Air quality levels are hazardous!<br><br>
                • Avoid outdoor activities<br>
                • Use air purifiers indoors<br>
                • Wear N95 masks if outside<br>
                • Keep windows closed
            </div>
        </div>
        """, unsafe_allow_html=True)
else:
    with st.sidebar:
        st.markdown("""
        <div class="alert-safe">
            <div class="alert-title">
                ✅ Air Quality is Safe
            </div>
            <div class="alert-content">
                Current air quality levels are acceptable.<br><br>
                • Enjoy outdoor activities<br>
                • Air quality is within safe limits<br>
                • Regular monitoring is ongoing
            </div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.write("---")
st.markdown("""
<div class="footer">
    <p>📘 <i>This app is powered by machine learning and OpenWeather API.</i></p>
    <p>Made with ❤️ by Muhammad Mehdi</p>
</div>
""", unsafe_allow_html=True)

def main():
    # Initialize weather and AQI data first
    current_aqi = aqi_data["current_aqi"].iloc[0]  # Get current AQI

    # Sidebar with conditional alerts
    with st.sidebar:
        st.title("Air Quality Monitor")
        
        if current_aqi >= 4:
            st.markdown("""
                <div style="
                    background-color: #ffe5e5;
                    color: #800000;
                    border: 2px solid #ff4d4d;
                    border-radius: 10px;
                    padding: 15px;
                    text-align: center;
                    margin: 10px 0;">
                    <h3 style="color: #cc0000;">🚨 Hazardous Air Quality Alert!</h3>
                    <p style="color: #333333;">
                        Air quality levels are hazardous!<br>
                        • Avoid outdoor activities<br>
                        • Use air purifiers indoors<br>
                        • Wear N95 masks if outside
                    </p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style="
                    background-color: #f0fff0;
                    color: #006400;
                    border: 2px solid #4dff4d;
                    border-radius: 10px;
                    padding: 15px;
                    text-align: center;
                    margin: 10px 0;">
                    <h3 style="color: #006400;">✅ Air Quality is Safe</h3>
                    <p style="color: #333333;">
                        Current air quality levels are acceptable.<br>
                        Enjoy outdoor activities!
                    </p>
                </div>
            """, unsafe_allow_html=True)

    # Continue with rest of the dashboard...
