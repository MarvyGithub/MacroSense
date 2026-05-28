# MacroSense -- US Economic Forecasting System

A machine learning system that forecasts GDP growth, inflation, 
and unemployment 6 months ahead using Federal Reserve data.

---

## The Problem

Most organisations make economic decisions reactively. They 
look at what the economy did last quarter and assume the near 
future looks similar. By the time official data confirms a 
slowdown or a surge the window to act has often already closed.

MacroSense addresses this by providing a forward-looking 
data-driven view of US economic conditions -- built entirely 
from public Federal Reserve data and standard machine learning 
tools.

---

## What MacroSense Forecasts

| Indicator | Description | Horizon |
|---|---|---|
| GDP Growth Rate | How fast the economy is expanding or contracting | 6 months |
| Inflation Rate | Year over year consumer price changes | 6 months |
| Unemployment Rate | Percentage of workforce without jobs | 6 months |

---

## Results

Walk forward validation simulated MacroSense running live 
from the late 1990s onward -- retraining every month on all 
available history and forecasting one step ahead.

| Indicator | Direction Accuracy | Recession Accuracy | Normal Period |
|---|---|---|---|
| GDP Growth | 88.9% | 70.0% | 90.5% |
| Inflation Rate | 96.2% | 70.0% | 98.3% |
| Unemployment Rate | 100.0% | 100.0% | 100.0% |

Direction accuracy measures how often the model correctly 
predicted whether each indicator would rise or fall over 
the following 6 months. A random baseline would achieve 50%.

Note on unemployment accuracy: The 100% direction accuracy 
for unemployment reflects the high persistence of this series 
rather than extraordinary predictive power. Unemployment 
trends in one direction for extended periods making 
directional prediction more straightforward than for GDP 
or inflation.

---

## Data Sources

All data sourced from FRED -- Federal Reserve Bank of 
St. Louis. Free, authoritative, and used by central banks, 
governments, and academic institutions worldwide.

| Variable | FRED Code | Type | What It Measures |
|---|---|---|---|
| Real GDP | GDPC1 | Target | Total economic output |
| CPI | CPIAUCSL | Target | Consumer price inflation |
| Unemployment | UNRATE | Target | Labour market health |
| Fed Funds Rate | FEDFUNDS | Input | Monetary policy stance |
| Money Supply M2 | M2SL | Input | Money in circulation |
| Yield Curve | T10Y2Y | Input | Recession early warning |
| Industrial Production | INDPRO | Input | Factory output |
| Retail Sales | RSAFS | Input | Consumer spending |
| Consumer Sentiment | UMCSENT | Input | Economic confidence |
| Jobless Claims | ICSA | Input | Labour market stress |

---

## Methodology

**Feature Engineering**

Raw FRED data cannot go directly into a machine learning 
model. Level variables like GDP trend upward indefinitely 
creating spurious correlations that teach models 
mathematical coincidences rather than genuine economic 
relationships. Three transformations address this:

1. Growth rates -- level variables converted to percentage 
   changes achieving stationarity
2. Lag features -- past values at 1, 3, 6, and 12 month 
   horizons give the model memory of economic history
3. Target shift -- target variables shifted 6 months 
   forward so the model learns to predict future outcomes 
   from current conditions

**Models**

| Model | Role |
|---|---|
| Linear Regression | Baseline |
| Ridge Regression | Regularised baseline |
| Random Forest | Non-linear tree ensemble |
| XGBoost | Primary model |
| Ensemble | Average of all four |

A separate model is trained for each target giving 
12 trained models in total.

**Validation**

Walk forward validation was used rather than a single 
static train-test split. Starting with 10 years of 
initial training data the model retrained each month 
on all available history and forecasted one step ahead. 
This simulates real deployment conditions and generates 
hundreds of evaluation scores across different economic 
regimes rather than one aggregate score.

---

## Project Structure

MacroSense/

notebooks/
- 01_data_collection.ipynb -- connects to FRED API and downloads all economic data
- 02_EDA.ipynb -- exploratory data analysis with recession visualisation
- 03_Feature_Engineering.ipynb -- growth rate transformations, lag features, target creation
- 04_modelling.ipynb -- trains four ML models across three targets
- 05_evaluation.ipynb -- static evaluation, SHAP analysis, performance metrics
- 06_walk_forward.ipynb -- walk forward validation across 25 years of economic history

app.py -- Streamlit interactive dashboard
requirements.txt -- all required Python libraries
SETUP.md -- setup instructions including how to get FRED API key
README.md -- this file
---

## Key Findings

The yield curve spread is the single most powerful 
predictor in the dataset. It inverted before the 2001 
Dot-com recession, the 2008 Financial Crisis, and again 
in 2023. Every major US recession since 1976 was preceded 
by yield curve inversion.

Consumer sentiment leads economic downturns. It declined 
before official recession dates in 2001 and 2008 -- 
consumers sensed trouble before economists confirmed it.

Unemployment is a lagging indicator. It only rises after 
a recession has already begun making it less useful for 
early warning than jobless claims which react within weeks.

The 2020 COVID recession produced errors significantly 
larger than other periods across all models. This is 
expected. A pandemic-driven economic shutdown has no 
historical precedent that any model trained on past data 
could anticipate.

---

## Honest Limitations

1. Novel shocks -- no model trained on historical 
   patterns can predict unprecedented events like 
   pandemics or novel financial crises

2. US-centric -- the system uses US data only and 
   does not generalise to other economies without 
   retraining

3. Manual retraining -- the system requires manual 
   retraining when new FRED data is released rather 
   than updating automatically

4. Spurious accuracy -- unemployment direction 
   accuracy reflects series persistence rather than 
   genuine predictive power

5. Structural breaks -- if the economy behaves 
   fundamentally differently from historical patterns 
   model performance will degrade

---

## How To Run This Project

See SETUP.md for complete setup instructions including 
how to obtain your free FRED API key.

Quick start:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Live Dashboard

[Link to your Streamlit deployment goes here]

Tech Stack:

Python -- pandas -- NumPy -- scikit-learn -- XGBoost --
SHAP -- Streamlit -- FRED API -- matplotlib -- seaborn --
statsmodels -- joblib

About

Built by Aremu Gideon Marvelous

Economics graduate applying machine learning to macroeconomic forecasting.
This project combines domain knowledge from economics
training with ML engineering to produce a system that
is both technically rigorous and economically interpretable.
Connect on LinkedIn: [Your LinkedIn URL]
