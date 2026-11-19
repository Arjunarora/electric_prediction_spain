# Forecasting Spain's Hourly Electricity Price

Predicting the **actual hourly electricity price in Spain (€/MWh)** from generation by source, total load, the TSO's day-ahead price and weather features. Tree-based models are compared on a strict **time-based split**: trained on 2015–2017, tested on the unseen year 2018.

## Data

[Hourly energy demand, generation and weather (Kaggle)](https://www.kaggle.com/nicholasjhana/energy-consumption-generation-prices-and-weather): four years (2015–2018) of hourly data from ENTSO-E and Red Eléctrica de España, plus OpenWeather data for five Spanish cities. This notebook joins the energy data with **Madrid** weather.

## Approach

1. Merge energy and weather data on the timestamp. Add calendar features (year, month, day).
2. Impute missing values, scale numeric features, one-hot encode categorical weather descriptions.
3. Split by time: **train 2015–2017, test 2018**. Prices rose between the periods (mean 56 → 63 €/MWh), which makes the test realistic and hard.
4. Compare Decision Tree, Random Forest, XGBoost and LightGBM, and tune depth, leaves and estimators.

## Results (test year 2018)

| Model | Test RMSE (€/MWh) | Test R² |
|---|---|---|
| Decision tree (unconstrained) | – | 0.21 |
| Decision tree (max depth 3) | – | 0.41 |
| Random forest (default) | – | 0.43 |
| XGBoost (20 trees, depth 5) | 9.39 | – |
| LightGBM (default) | 9.30 | – |
| **LightGBM (max depth 8)** | **9.14** | – |

Every model fits the training years far better than 2018 (e.g. random-forest train R² 0.99 vs. test 0.43). That reflects real distribution shift in the market, and it's why the time-based split matters: a random split would have hidden it.

## Run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/spain_electricity_price_prediction.ipynb
```

The notebook downloads the dataset with `opendatasets`, which needs a Kaggle API key.

## Ideas for improvement

- Lag features (price 24 h / 168 h earlier) and rolling statistics: the strongest signals for price forecasting
- Exclude `price day ahead` to measure how much the model adds beyond the TSO forecast
- Weather from all five cities, and walk-forward validation

*Built as a course project (2021).*

## License

MIT
