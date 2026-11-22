# Weather_AQI_Predictor
Applying machine learning and data science to real-world scenarios. This project predicts Karachi's weather and air quality for the next three days using processed data from external APIs.
You can checkout the app visiting at https://weatheraqipredictor-gkehryzvj7munc4rwqsumy.streamlit.app/

## Live dashboard

https://mehdi-abidi.github.io/Weather_AQI_Predictor/ shows the forecast for the next three days, the last 60 days of PM2.5, and the model's error against a persistence baseline. A GitHub Actions workflow fetches fresh data, retrains and redeploys it every morning. The code is in [site/](site/).

## Evaluation (November 2026)

The forecast was measured again on a held-out year with a time-based split. A Random Forest predicts the daily mean PM2.5 three days ahead with a mean absolute error of 5.61 µg/m³ (R² 0.41), which is 23.8 percent below the persistence baseline. The method, the full tables and the script are in [evaluation/](evaluation/README.md).
