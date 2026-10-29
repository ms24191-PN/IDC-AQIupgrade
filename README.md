# Air Quality Prediction

## 1. Project Overview

Air pollution is a major environmental concern in India, affecting public health and environmental quality. The Air Quality Index (AQI) provides a numerical representation of air pollution levels based on measured pollutant concentrations.

This project looks at daily air quality data from Indian cities and builds regression models to predict the Air Quality Index (AQI). It has two parts:

1. **AQI prediction using pollutant concentrations:** Regression models are trained to predict AQI using twelve pollutant-related features.
2. **City-specific next-day AQI prediction:** Separate Random Forest regression models are trained for six Indian cities using historical AQI values and lag-based features.

The project compares regression performance, examines feature importance, investigates temporal patterns in AQI, and evaluates the use of city-specific models for next-day forecasting.

## 2. Objectives

- Analyze air-quality measurements from different Indian cities.
- Investigate monthly and yearly AQI trends.
- Examine relationships between air pollutants using correlation analysis.
- Predict numerical AQI values using Linear Regression, Decision Tree, and Random Forest regression.
- Optimize model hyperparameters using RandomizedSearchCV.
- Compare model performance using MAE, RMSE, and R².
- Analyze feature importance for pollutant-based AQI prediction.
- Develop separate next-day AQI prediction models for Bengaluru, Chennai, Hyderabad, Delhi, Mumbai, and Chandigarh.
- Compare city-specific forecasting performance.

## 3. Dataset

The dataset used in this project is **India Air Quality Data (2015–2020)** from Kaggle.

**Dataset source:** [India Air Quality Data — Kaggle](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)

The dataset contains air-quality measurements collected from monitoring locations across Indian cities.

### Important variables

| Feature | Description |
|---|---|
| City | Name of the city |
| Date | Date of observation |
| PM2.5 | Fine particulate matter |
| PM10 | Particulate matter |
| NO | Nitric oxide |
| NO2 | Nitrogen dioxide |
| NOx | Oxides of nitrogen |
| NH3 | Ammonia |
| CO | Carbon monoxide |
| SO2 | Sulphur dioxide |
| O3 | Ozone |
| Benzene | Benzene concentration |
| Toluene | Toluene concentration |
| Xylene | Xylene concentration |
| AQI | Numerical Air Quality Index |
| AQI_Bucket | Air-quality category |

The pollutant measurements are used as input features for the first part of the project, while AQI is the target variable.

## 4. What the notebook does

**4.1. Loading and cleaning**
- Downloads the dataset and reads `city_day.csv`.
- Drops rows where `AQI` is missing.
- Fills missing pollutant values with the median of each pollutant column.
- Checks null counts, duplicate rows, city names, record count per city and the type of the `Date` column.

**4.2. Feature creation**
- Converts `Date` to datetime and extracts `Year`, `Month`, `Day` and `Weekday`.
- Adds `Season` using standard Northern Hemisphere months (Winter: Dec-Feb, Spring: Mar-May, Summer: Jun-Aug, Monsoon: Sep-Nov).
- Adds a `Winter` flag (1 for Dec, Jan, Feb, otherwise 0).
- Keeps a copy of the processed data as `AQI_data`. A commented-out cell exports it to CSV.

**4.3. Exploratory analysis**
- Average yearly AQI for every city (line plot).
- The 10 cities with the most AQI records, their yearly averages, a ranking table by average AQI, and a summary of the highest and lowest cities.
- Daily and monthly AQI for Delhi, Mumbai, Chandigarh, Bengaluru, Chennai and Hyderabad between Sep 2019 and Jul 2020.
- Box plot of the pollutant values and a Pearson correlation heatmap of the pollutants.
- For each city, daily and monthly AQI curves with one line per year, to show seasonal patterns.

**4.4. Same-day AQI models**
- Features are the 12 pollutants and the target is `AQI`. The data is split 80/20 with `random_state=42`.
- Linear Regression (baseline), Decision Tree (`max_depth=10`, `min_samples_leaf=20`) and Random Forest (`n_estimators=120`, `max_depth=20`, `max_features=0.5`, `min_samples_split=10`).
- Each model is scored with MAE, RMSE and R2, and feature importance is shown for all three. For Linear Regression, the absolute coefficients are used.
- A correlation filter (threshold 0.90) is computed to find redundant pollutants, but the models above are trained on all 12 features.
- The Random Forest is also tested on two hand-entered pollutant samples whose actual AQI values were 80 and 127.
- A commented-out block shows how to save the Linear Regression model with `joblib` and use it on a new CSV.

**4.5. Next-day AQI model**
- Features are built from each city's own history: AQI of the previous 1, 2, 3 and 7 days, and the mean of the previous 3 and 7 days.
- A Random Forest is tuned separately for each city, and the best parameters and results are listed below.
- The final models are trained on all available rows of each city. The notebook asks for a city name and predicts the AQI for the day after the last date recorded in the dataset.

### Key observation: Monthly AQI variation

The monthly AQI visualization showed a seasonal pattern in which AQI generally decreased during the summer months, while AQI tended to remain elevated during other parts of the year.

This observation suggests that air quality varies seasonally. However, the visualization alone does not establish the causes of this pattern.

The observed trend may be investigated further using meteorological variables, pollutant concentrations, and seasonal differences between cities.

## 5. Part I: AQI Prediction Using Pollutant Concentration Values

### 5.1 Problem formulation

This is a supervised regression task in which the model learns the relationship between pollutant concentrations and numerical AQI values.

**Input features:**

```python
[
    'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3',
    'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene'
]
```

**Target variable:**

```python
'AQI'
```

The objective is to estimate AQI from the measured pollutant concentrations.

### 5.2 Model optimization

Three regression models were evaluated:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor

RandomizedSearchCV was used to optimize the hyperparameters of the models. The selected configurations were subsequently evaluated using the model evaluation procedure used in the experiments.

### 5.3 Regression evaluation metrics

The models were evaluated using the following metrics:

- **MAE (Mean Absolute Error):** Measures the average absolute difference between actual and predicted AQI.
- **RMSE (Root Mean Squared Error):** Penalizes larger prediction errors more strongly than MAE.
- **R² Score:** Measures the proportion of variance in AQI explained by the regression model.

### 5.4 Regression results

| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| Linear Regression | 31.170 | 59.442 | 0.807 |
| Decision Tree Regressor | 24.455 | 45.205 | 0.888 |
| Random Forest Regressor | 20.537 | 40.311 | 0.911 |

The Random Forest Regressor achieved the lowest MAE and RMSE and the highest R² score among the three evaluated models.

This indicates that Random Forest provided the strongest predictive performance among the tested models under the reported evaluation setup.

### 5.5 Linear Regression

Linear Regression was used as a baseline model to examine the relationship between pollutant concentrations and AQI.

**Model evaluation:**

| Metric | Value |
|---|---:|
| MAE | 31.1705 |
| RMSE | 59.4417 |
| R² Score | 0.8070 |

### 5.6 Decision Tree Regressor

The Decision Tree Regressor captures nonlinear relationships between pollutant concentrations and AQI.

**Optimized hyperparameters:**

```python
max_depth = 10
min_samples_leaf = 20
min_samples_split = 2
```

**Model evaluation:**

| Metric | Value |
|---|---:|
| MAE | 24.4546 |
| RMSE | 45.2055 |
| R² Score | 0.8884 |

### 5.7 Random Forest Regressor

Random Forest combines multiple decision trees to improve predictive performance and model nonlinear relationships in the data.

**Optimized hyperparameters:**

```python
n_estimators = 120
min_samples_split = 10
min_samples_leaf = 1
max_features = 0.5
max_depth = 20
bootstrap = True
```

**Model evaluation:**

| Metric | Value |
|---|---:|
| MAE | 20.5366 |
| RMSE | 40.3106 |
| R² Score | 0.9113 |

Random Forest achieved the best overall performance among the three models evaluated in Part I.

## 6. Feature Importance Analysis

Feature importance was examined to understand the relative contribution of pollutant variables to the predictions(was done to see the relationship with the target variable and no feature was removed).

### 6.1 Linear Regression

For Linear Regression, the reported values are the absolute magnitudes of the fitted coefficients, if calculated using the absolute values of the coefficients. These magnitudes are not directly comparable as measures of predictive importance when features have different scales.

| Feature | Reported importance |
|---|---:|
| CO | 11.750274 |
| PM2.5 | 1.150264 |
| SO2 | 0.742530 |
| PM10 | 0.274891 |
| NO2 | 0.209746 |
| O3 | 0.193859 |
| NO | 0.138584 |
| NOx | 0.088042 |
| NH3 | 0.085267 |
| Xylene | 0.053917 |
| Toluene | 0.037001 |
| Benzene | 0.025156 |

CO had the largest reported coefficient magnitude, followed by PM2.5 and SO2.

### 6.2 Decision Tree Regressor

| Feature | Importance |
|---|---:|
| PM2.5 | 0.530857 |
| CO | 0.390730 |
| NO | 0.038088 |
| PM10 | 0.028473 |
| O3 | 0.003022 |
| NOx | 0.002765 |
| Toluene | 0.002076 |
| SO2 | 0.001465 |
| NO2 | 0.001333 |
| Xylene | 0.000573 |
| NH3 | 0.000561 |
| Benzene | 0.000059 |

PM2.5 and CO accounted for most of the Decision Tree's reported feature importance.

### 6.3 Random Forest Regressor

| Feature | Importance |
|---|---:|
| PM2.5 | 0.401273 |
| CO | 0.319405 |
| PM10 | 0.088770 |
| SO2 | 0.046723 |
| NO | 0.045557 |
| NO2 | 0.039696 |
| NOx | 0.017093 |
| Toluene | 0.012673 |
| O3 | 0.011228 |
| Xylene | 0.009069 |
| Benzene | 0.004984 |
| NH3 | 0.003529 |

PM2.5 and CO were the two most important features in the Random Forest model, followed by PM10.

### 6.4 Interpretation

The Decision Tree and Random Forest results consistently identify PM2.5 and CO as the dominant features in their respective models.

The relative importance of the remaining pollutants differs between models because the algorithms capture relationships and interactions differently.

Feature importance describes how a fitted model uses its inputs. It does not establish that a pollutant causes a particular AQI change.

### 6.5 Correlation-based feature selection

A correlation heatmap was used to inspect relationships between pollutant variables. A separate correlation-filtering procedure was not applied because the analysis did not provide sufficient justification for removing the selected features.

All twelve pollutant features were therefore retained for the reported regression experiments.

## 7. Part II: City-Specific Next-Day AQI Prediction

### 7.1 Motivation

AQI patterns can differ between cities because of differences in emission sources, local conditions, seasonal behavior, and historical pollution trends.

A single model trained on combined data from multiple cities may not capture the temporal characteristics of each city equally well.

Therefore, this part of the project trains **six separate Random Forest regression models**, one for each selected city.

Each model uses the historical AQI pattern of its corresponding city to predict the next day's AQI.

### 7.2 Cities included

- Bengaluru
- Chennai
- Hyderabad
- Delhi
- Mumbai
- Chandigarh

### 7.3 Input features

The following historical AQI features(because they help in time - series pattern learning to the model) were used:

```python
[
    "AQI_lag_1",
    "AQI_lag_2",
    "AQI_lag_3",
    "AQI_lag_7",
    "AQI_rolling_mean_3",
    "AQI_rolling_mean_7"
]
```

**Feature descriptions:**

| Feature | Description |
|---|---|
| AQI_lag_1 | AQI from the previous day |
| AQI_lag_2 | AQI from two days earlier |
| AQI_lag_3 | AQI from three days earlier |
| AQI_lag_7 | AQI from seven days earlier |
| AQI_rolling_mean_3 | Rolling mean of AQI over three days |
| AQI_rolling_mean_7 | Rolling mean of AQI over seven days |

The target variable is the AQI for the next day.

### 7.4 Modeling workflow

1. The user enters a city name.
2. The data is filtered to retain observations for that city.
3. Daily AQI observations are prepared and ordered chronologically.
4. Lag features and rolling means are calculated using historical observations.
5. The city-specific dataset is split into training and testing sets while preserving temporal order.
6. RandomizedSearchCV is used to select the Random Forest hyperparameters.
7. The selected model is evaluated on the test period.
8. The model uses the latest available historical AQI features to predict the next day's AQI.

Each city has its own optimized hyperparameters and evaluation results.

## 8. Next-Day AQI Prediction Results

### 8.1 Next-day AQI by city (Random Forest, test set)

| City | CV MAE | Test MAE | Test RMSE | Test R2 |
|---|---|---|---|---|
| Bengaluru | 20.56 | 9.19 | 12.40 | 0.600 |
| Delhi | 45.57 | 32.75 | 44.66 | 0.846 |
| Hyderabad | 18.26 | 11.35 | 15.35 | 0.814 |
| Mumbai | 20.86 | 11.80 | 18.08 | 0.825 |
| Chandigarh | 31.06 | 12.44 | 15.69 | 0.223 |
| Chennai | 26.98 | 18.13 | 26.04 | 0.475 |

*CV MAE refers to the best cross-validation MAE reported during hyperparameter optimization. Test metrics refer to evaluation on the held-out test period.*

### 8.2 Optimized hyperparameters
All the hyperparameters are found with the help of RandomizedSearchCV for Random Forest regression model(as this was the best model for the predication)
#### Bengaluru

```python
{
    "n_estimators": 50,
    "min_samples_split": 8,
    "min_samples_leaf": 10,
    "max_features": 0.7,
    "max_depth": 5,
    "bootstrap": True
}
```

#### Delhi

```python
{
    "n_estimators": 75,
    "min_samples_split": 10,
    "min_samples_leaf": 8,
    "max_features": 1.0,
    "max_depth": None,
    "bootstrap": True
}
```

#### Hyderabad

```python
{
    "n_estimators": 150,
    "min_samples_split": 15,
    "min_samples_leaf": 8,
    "max_features": 1.0,
    "max_depth": None,
    "bootstrap": True
}
```

#### Mumbai

```python
{
    "n_estimators": 125,
    "min_samples_split": 10,
    "min_samples_leaf": 8,
    "max_features": 0.7,
    "max_depth": 15,
    "bootstrap": False
}
```

#### Chandigarh

```python
{
    "n_estimators": 150,
    "min_samples_split": 10,
    "min_samples_leaf": 2,
    "max_features": 1.0,
    "max_depth": 30,
    "bootstrap": True
}
```

#### Chennai

```python
{
    "n_estimators": 50,
    "min_samples_split": 8,
    "min_samples_leaf": 10,
    "max_features": 0.7,
    "max_depth": 5,
    "bootstrap": True
}
```

### 8.3 Interpretation of city-specific results

The results show differences in forecasting performance across cities.

- **Delhi:** Achieved the highest test R² of 0.846, indicating that the model explained a large proportion of AQI variation during the test period. Its absolute prediction errors were also higher than those of several other cities.
- **Hyderabad:** Achieved a test R² of 0.814 with a test MAE of 11.352.
- **Mumbai:** Achieved a test R² of 0.825 with a test MAE of 11.803.
- **Bengaluru:** Achieved a relatively low test MAE of 9.191, although its R² was 0.600.
- **Chennai:** Achieved a test MAE of 18.132 and a test R² of 0.475.
- **Chandigarh:** Achieved a test MAE of 12.442 but a comparatively low R² of 0.223.

The difference between MAE and R² across cities highlights why multiple evaluation metrics are important. A model may have a reasonably low average absolute error but explain relatively little of the variation in a city's AQI.

The results support evaluating each city separately rather than assuming that one model will perform equally well across all cities. However, they do not by themselves establish that city-specific models will always outperform a pooled model; that requires a direct comparison using the same temporal evaluation procedure.

## 9. Repository Structure

The repository can be organized as follows:

```text
Indian-Air-Quality-Prediction/
│
├── README.md
├── requirements.txt
├── notebooks/
│   ├── AQI predication and Next day AQI predication - CSV pipeline.ipynb
│   └── EDA_notebook.ipynb
└── data/
    └── AQI_data.csv
```

Adjust the file names and folders to match the actual repository. 
## 10. Installation and Setup

### Prerequisites

Python 3 with:

```
kagglehub
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
```

Install with `pip install kagglehub pandas numpy matplotlib seaborn scikit-learn joblib`. Downloading the dataset may need a Kaggle account.

## How to run

1. Open the notebook in Jupyter or Google Colab.
2. Run all cells from top to bottom. Later cells depend on variables from earlier ones.
3. In the last cell, type one of Delhi, Bengaluru, Chennai, Hyderabad, Mumbai or Chandigarh when asked to get the next-day forecast.
4. To save the cleaned data, uncomment the export cell. Outside Colab, use only `data.to_csv('AQI_data.csv', index=False)`.

## 11. Limitations

- Prediction quality depends on the quality and completeness of the historical data.
- Pollutant-based regression estimates AQI from pollutant measurements and does not independently forecast future pollutant concentrations.
- Historical AQI lag features may not capture sudden changes caused by weather, emissions, or other external factors.
- The city-specific models may perform differently because AQI variability and temporal patterns differ across cities.
- Correlation and feature importance do not establish causal relationships.
- Predictions should not be treated as official AQI measurements or public-health advisories.

## 12. Future Scope

Potential extensions include:

- Incorporating meteorological variables such as temperature, humidity, wind speed, and rainfall.
- Comparing city-specific models directly against a pooled model.
- Testing additional time-series forecasting approaches.
- Investigating feature selection using model-based importance and other methods.
- Extending the analysis to more Indian cities and newer observations.
- Developing an interactive application for city-wise AQI prediction.

## 13. Conclusion

This project investigates Indian air-quality patterns and applies machine learning to AQI prediction.

In the first part, Linear Regression, Decision Tree, and Random Forest models were trained using twelve pollutant features. Random Forest achieved the best reported performance, with an MAE of 20.537, RMSE of 40.311, and R² of 0.911.

In the second part, six separate Random Forest models were developed for next-day AQI prediction in Bengaluru, Chennai, Hyderabad, Delhi, Mumbai, and Chandigarh. Each model used historical AQI lag features and rolling means, with hyperparameters optimized separately for each city.

The project demonstrates two complementary approaches to AQI modeling: estimating AQI from pollutant concentrations and forecasting AQI from historical temporal patterns.

## 14. Dataset Attribution

The dataset was obtained from Kaggle:

Rohan Rao, *Air Quality Data in India*.

Dataset link: https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india

Please follow the dataset provider's terms and applicable licensing conditions when redistributing the data.

## 15. Author

**Name:** Achalesh Nimbiwal(MS24255),Peehu(MS24191), and Vaibhav(MS24090)

**Institution:** IISER Mohali

**Project:** Air Quality Prediction
