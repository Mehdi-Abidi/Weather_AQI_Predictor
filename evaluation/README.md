# Evaluation: 3-day PM2.5 forecast for Karachi

Added in November 2026. The original 2024 project trained three classifiers on the OpenWeather AQI category (1 to 5) with a random train/test split. In that test set 177 of 225 rows were category 5, so a model that always answers 5 scores 78.7 percent, and all three classifiers scored 79.1 percent. The accuracy figure said little about the forecast.

This folder measures the forecast again with a setup that can fail:

- The target is the daily mean PM2.5 concentration (µg/m³), 1 and 3 days ahead. PM2.5 is the pollutant that sets Karachi's AQI category on most days.
- The split is by time. Models train on 5 August 2022 to late October 2025 and are tested on 1 November 2025 to 19 November 2026, a year they never saw.
- Every model is compared with three baselines: persistence (the forecast equals today's value), the mean of the last 7 days, and the monthly average from the training years.

## Data

Hourly pollutant concentrations and daily weather for Karachi (24.8607 N, 67.0011 E) from the free Open-Meteo air quality and archive APIs. No API key is needed. The script downloads about 2 MB on the first run and caches it in `evaluation/data/`.

## Results on the held-out year

3 days ahead, 382 test days, mean PM2.5 28.7 µg/m³:

| Method | MAE | RMSE | R² | MAE below persistence |
| --- | --- | --- | --- | --- |
| Persistence | 7.36 | 9.88 | -0.03 | |
| Mean of last 7 days | 6.96 | 8.93 | 0.16 | 5.4% |
| Monthly climatology | 7.22 | 9.44 | 0.06 | 1.9% |
| Ridge | 5.78 | 7.55 | 0.40 | 21.5% |
| Random Forest | 5.61 | 7.50 | 0.41 | 23.8% |
| Histogram gradient boosting | 5.90 | 7.90 | 0.34 | 19.8% |

1 day ahead, 384 test days:

| Method | MAE | RMSE | R² | MAE below persistence |
| --- | --- | --- | --- | --- |
| Persistence | 5.01 | 7.00 | 0.49 | |
| Ridge | 4.47 | 6.17 | 0.61 | 10.8% |
| Random Forest | 4.57 | 6.30 | 0.59 | 8.8% |
| Histogram gradient boosting | 4.70 | 6.57 | 0.55 | 6.2% |

When the 3-day Random Forest forecast is mapped back to the OpenWeather PM2.5 bands, it lands in the right band on 72.3 percent of test days, against 65.7 percent for persistence.

## What did not work

Predicting the AQI category directly as a 5-class problem never beat persistence on macro-F1 under the time split, hourly or daily. A random split on the same data gave a Random Forest 84 percent accuracy, because neighbouring hours leak into both sides of the split. That number is not a forecast score and is not reported.

## Limits

- Open-Meteo concentrations come from the CAMS global model, about 40 km resolution, and are estimates for the grid cell around Karachi. They are not readings from a ground station.
- Weather features use only the days up to the forecast date. A version that also used the weather of the target day would need a weather forecast at prediction time.
- One city and one test year.

## Reproduce

```bash
pip install -r evaluation/requirements.txt
python evaluation/pm25_forecast.py
```

The script prints the tables above and writes `evaluation/results.json`.
