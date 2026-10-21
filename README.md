# KisanMitra: ML-Based District-Level Crop Recommendation and Profit Estimation

A farmer selects **State, District and Season**. The system ranks crops by predicted
performance (Low / Medium / High), effort level, weather suitability and estimated
profit or loss per hectare.

## Features
- Yield prediction with LightGBM (time-based train/test split, no leakage)
- Low / Medium / High performance, computed per crop from historical yield percentiles
- Effort level from labour days and irrigation count
- Weather fit using Open-Meteo data (2017-2019 state-season averages)
- Estimated profit/loss = `Yield x 10 x Price - Cost` (price in Rs/quintal, yield in t/ha)
- Low-confidence flag for crops with weak model accuracy
- Two UIs: Streamlit (`app.py`) and HTML/CSS/JS + FastAPI (`api.py`, `index.html`)

## Data sources
| Data | Source |
|---|---|
| Crop area, production, yield | APY (data.gov.in / Kaggle), 2008-2019 |
| Prices and MSP | Agmarknet Market-wise Price & Arrival report |
| Weather | Open-Meteo Historical API |
| Cost, labour, irrigation | Hand-made approximate table (verify with CACP) |

## Results
- R2 0.95, MAE about 0.97 t/ha
- Low/Medium/High label accuracy: **70.9%** (random baseline 33%)
- Tried and rejected (no gain): state-level weather, neighbouring-district yield

## Project structure
```
crop recommendation system/
|-- APY.csv                  raw crop data
|-- crop_clean.csv           cleaned data
|-- final_data.csv           data with Prev_Yield, Prev3_Avg
|-- final_model.pkl          trained LightGBM model
|-- crop_reference.csv       price, cost, effort per crop
|-- state_weather.csv        state-season weather
|-- app.py                   Streamlit app
|-- api.py                   FastAPI backend
|-- index.html               HTML/CSS/JS frontend
```

## Setup
```
pip install pandas numpy lightgbm scikit-learn joblib streamlit fastapi uvicorn
```



## Limitations
- Weather is a state's top 5 producter city estimated , so all districts in a state share it.
- Prices come from a few days of mandi data, not a full-season average.
- treat output as guidance, not a guarantee.

## Future work
crop yield prediction
