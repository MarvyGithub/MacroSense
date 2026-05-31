# ================================================
# MacroSense -- US Economic Forecasting Dashboard
# Run with: streamlit run app.py
# ================================================

# ------------------------------------------------
# SECTION 1: IMPORT ALL LIBRARIES
# This must always be at the very top of the file
# before anything else runs
# ------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
import os
from dotenv import load_dotenv
import warnings

warnings.filterwarnings('ignore')
from xgboost import XGBRegressor
from fredapi import Fred
from dotenv import load_dotenv

# ------------------------------------------------
# SECTION 2: PAGE CONFIGURATION
# This must be the first Streamlit command
# that runs -- before any other st. command
# ------------------------------------------------
st.set_page_config(
    page_title="MacroSense -- US Economic Forecaster",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------
# SECTION 3: LOAD API KEY
# Tries local .env file first for development
# then falls back to Streamlit Cloud secrets
# for deployment. Same code works in both places.
# ------------------------------------------------

# Get the folder where app.py lives
app_dir  = os.path.dirname(os.path.abspath(__file__))

# Build explicit path to .env file
env_path = os.path.join(app_dir, '.env')

# Load the .env file from that explicit path
load_dotenv(dotenv_path=env_path)

# Try getting key from local .env file first
api_key = os.getenv('FRED_API_KEY')

# If not found locally try Streamlit Cloud secrets
if api_key is None:
    try:
        api_key = st.secrets['FRED_API_KEY']
    except Exception:
        api_key = None

# Stop the app if key is still not found
if api_key is None:
    st.error(
        "FRED API key not found. "
        "If running locally check your .env file. "
        "If running on Streamlit Cloud add your key "
        "under Settings then Secrets."
    )
    st.stop()

# ------------------------------------------------
# SECTION 4: DATA GENERATION FUNCTIONS
# These functions download data from FRED,
# engineer features, and train models.
# The @st.cache_data decorator means each
# function only runs once then stores the
# result in memory for the rest of the session.
# Without caching the app would re-download
# data and retrain models on every single
# user interaction which would be very slow.
# ------------------------------------------------

@st.cache_data
def get_recession_data():
    """
    Downloads the official US recession indicator
    from FRED. Returns 1 during recession months
    and 0 during normal months.
    Used to shade recession periods grey on charts.
    """
    fred      = Fred(api_key=api_key)
    recession = fred.get_series('USREC')
    recession = recession.resample('ME').last()
    return recession


@st.cache_data
def generate_raw_data():
    """
    Downloads all 10 economic series from FRED,
    combines them into one monthly table,
    trims to 1992 where all series are available,
    forward fills GDP quarterly gaps,
    and drops genuinely missing rows at the end.
    Returns the cleaned raw dataframe.
    """
    fred = Fred(api_key=api_key)

    # Download all 10 series
    gdp          = fred.get_series('GDPC1')
    time.sleep(0.5)
    cpi          = fred.get_series('CPIAUCSL')
    time.sleep(0.5)
    unemployment = fred.get_series('UNRATE')
    time.sleep(0.5)
    fedfunds     = fred.get_series('FEDFUNDS')
    time.sleep(0.5)
    m2           = fred.get_series('M2SL')
    time.sleep(0.5)
    yieldcurve   = fred.get_series('T10Y2Y')
    time.sleep(0.5)
    indpro       = fred.get_series('INDPRO')
    time.sleep(0.5)
    retailsales  = fred.get_series('RSAFS')
    time.sleep(0.5)
    sentiment    = fred.get_series('UMCSENT')
    time.sleep(0.5)
    claims       = fred.get_series('ICSA')

    # Combine into one master table
    raw = pd.DataFrame({
        'GDP'          : gdp,
        'CPI'          : cpi,
        'Unemployment' : unemployment,
        'FedFunds'     : fedfunds,
        'M2'           : m2,
        'YieldCurve'   : yieldcurve,
        'IndPro'       : indpro,
        'RetailSales'  : retailsales,
        'Sentiment'    : sentiment,
        'JoblessClaims': claims
    })

    # Align everything to monthly frequency
    raw = raw.resample('ME').last()

    # Trim to 1992 where all series are available
    raw = raw[raw.index >= '1992-01-01']

    # Forward fill quarterly GDP gaps
    raw = raw.ffill()

    # Drop rows that are still missing
    raw = raw.dropna()

    return raw


@st.cache_data
def generate_features(_raw):
    """
    Transforms raw FRED data into machine learning
    ready features. Converts level variables to
    growth rates, creates lag features to give
    the model memory of past conditions, and
    shifts target variables 6 months forward
    so the model learns to predict the future.

    The underscore before _raw tells Streamlit
    not to try to hash this argument for caching
    since dataframes can be large.
    """
    fe = pd.DataFrame(index=_raw.index)

    # Growth rate transformations
    fe['GDP_growth']          = _raw['GDP'].pct_change(4) * 100
    fe['Inflation_yoy']       = _raw['CPI'].pct_change(12) * 100
    fe['Inflation_mom']       = _raw['CPI'].pct_change(1) * 100
    fe['Unemployment']        = _raw['Unemployment']
    fe['Unemp_change_3m']     = _raw['Unemployment'].diff(3)
    fe['Unemp_change_12m']    = _raw['Unemployment'].diff(12)
    fe['FedFunds']            = _raw['FedFunds']
    fe['FedFunds_change_6m']  = _raw['FedFunds'].diff(6)
    fe['YieldCurve']          = _raw['YieldCurve']
    fe['YieldCurve_change']   = _raw['YieldCurve'].diff(3)
    fe['Yield_inverted']      = (_raw['YieldCurve'] < 0).astype(int)
    fe['M2_growth']           = _raw['M2'].pct_change(12) * 100
    fe['IndPro_growth']       = _raw['IndPro'].pct_change(12) * 100
    fe['IndPro_growth_3m']    = _raw['IndPro'].pct_change(3) * 100
    fe['Retail_growth']       = _raw['RetailSales'].pct_change(12) * 100
    fe['Sentiment']           = _raw['Sentiment']
    fe['Sentiment_change_6m'] = _raw['Sentiment'].diff(6)
    fe['Claims_growth']       = _raw['JoblessClaims'].pct_change(12) * 100

    # Lag features -- gives the model memory of past conditions
    cols_to_lag = [
        'GDP_growth', 'Inflation_yoy', 'Unemployment',
        'FedFunds', 'YieldCurve', 'M2_growth',
        'IndPro_growth', 'Retail_growth', 'Sentiment',
        'Claims_growth', 'Yield_inverted',
        'FedFunds_change_6m', 'Unemp_change_3m',
        'Sentiment_change_6m'
    ]

    for col in cols_to_lag:
        for lag in [1, 3, 6, 12]:
            fe[f'{col}_lag{lag}'] = fe[col].shift(lag)

    # Target variables shifted 6 months forward
    fe['target_GDP']          = fe['GDP_growth'].shift(-6)
    fe['target_Inflation']    = fe['Inflation_yoy'].shift(-6)
    fe['target_Unemployment'] = fe['Unemployment'].shift(-6)

    # Drop rows with missing values
    fe_clean = fe.dropna()

    return fe_clean


@st.cache_resource
def train_models(_fe_clean):
    """
    Trains one XGBoost model for each of the
    three target variables using all available
    feature engineered data.

    In deployment we train on all available data
    rather than splitting into train and test
    because we want the model to learn from
    as much history as possible before forecasting.

    The @st.cache_resource decorator is used
    instead of @st.cache_data for models because
    models are objects not data -- cache_resource
    handles them more efficiently.
    """
    target_cols  = ['target_GDP',
                    'target_Inflation',
                    'target_Unemployment']
    feature_cols = [c for c in _fe_clean.columns
                    if c not in target_cols]

    X = _fe_clean[feature_cols]
    y = _fe_clean[target_cols]

    models = {}

    for target in target_cols:
        model = XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
            verbosity=0
        )
        model.fit(X, y[target])
        models[target] = model

    return models, feature_cols

# ------------------------------------------------
# SECTION 5: HELPER FUNCTIONS
# Small reusable functions used across
# multiple pages of the dashboard
# ------------------------------------------------

def add_recession_shading(ax, recession,
                           start_date, end_date):
    """
    Adds grey shading to a matplotlib chart
    during every official US recession period
    between start_date and end_date.
    """
    rec = recession[
        (recession.index >= start_date) &
        (recession.index <= end_date)
    ]

    in_recession = False
    start        = None

    for date, val in rec.items():
        if val == 1 and not in_recession:
            start        = date
            in_recession = True
        elif val == 0 and in_recession:
            ax.axvspan(start, date,
                      alpha=0.2, color='grey')
            in_recession = False


# ------------------------------------------------
# SECTION 6: LOAD AND GENERATE EVERYTHING
# This section runs when the app first loads.
# The spinner shows a loading message to the
# user while data is being downloaded from
# FRED and models are being trained.
# This only happens once per session because
# of the caching decorators above.
# ------------------------------------------------

with st.spinner(
    "Connecting to Federal Reserve database and "
    "loading MacroSense... This takes about "
    "60 seconds on first load."
):
    raw_data  = generate_raw_data()
    fe_clean  = generate_features(raw_data)
    recession = get_recession_data()
    models, feature_cols = train_models(fe_clean)

# Define target columns and display names
target_cols = ['target_GDP',
               'target_Inflation',
               'target_Unemployment']

target_display = {
    'target_GDP'          : 'GDP Growth',
    'target_Inflation'    : 'Inflation Rate',
    'target_Unemployment' : 'Unemployment Rate'
}

# ------------------------------------------------
# SECTION 7: SIDEBAR NAVIGATION
# Everything written to st.sidebar appears
# in the left panel of the dashboard.
# The radio buttons let users switch between
# the five pages of the app.
# ------------------------------------------------

st.sidebar.title("📈 MacroSense")
st.sidebar.markdown(
    "*US Economic Forecasting System*"
)
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["Overview",
     "Economic Dashboard",
     "Model Forecasts",
     "Walk Forward Validation",
     "About This Project"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Data Source**")
st.sidebar.markdown(
    "[FRED -- Federal Reserve Bank of St. Louis]"
    "(https://fred.stlouisfed.org/)"
)
st.sidebar.markdown("**Forecast Horizon:** 6 months ahead")
st.sidebar.markdown("**Primary Model:** XGBoost")
st.sidebar.markdown(
    f"**Data Coverage:** "
    f"{raw_data.index[0].strftime('%b %Y')} to "
    f"{raw_data.index[-1].strftime('%b %Y')}"
)

# ------------------------------------------------
# SECTION 8: PAGE 1 -- OVERVIEW
# Displays when user selects Overview
# from the sidebar navigation
# ------------------------------------------------

if page == "Overview":

    st.title("📈 MacroSense")
    st.subheader(
        "A Machine Learning System for "
        "US Economic Forecasting"
    )

    st.markdown("""
    MacroSense uses Federal Reserve data and machine
    learning to forecast three key US economic
    indicators **6 months into the future**.

    Most organisations make economic decisions
    reactively -- looking at what happened last
    quarter and assuming the near future looks
    similar. MacroSense provides a data-driven
    forward-looking view so organisations can
    prepare for what is coming rather than react
    to what has already happened.
    """)

    st.markdown("---")

    # Current economic snapshot
    st.subheader("Current Economic Snapshot")

    col1, col2, col3 = st.columns(3)

    # GDP growth rate
    gdp_series  = raw_data['GDP'].dropna()
    gdp_current = gdp_series.iloc[-1]
    gdp_prev    = gdp_series.iloc[-5]
    gdp_growth  = ((gdp_current - gdp_prev) /
                    gdp_prev * 100)
    col1.metric(
        label="Real GDP",
        value=f"${gdp_current:,.0f}B",
        delta=f"{gdp_growth:+.1f}% vs 1yr ago"
    )

    # Inflation rate
    cpi_series   = raw_data['CPI'].dropna()
    cpi_current  = cpi_series.iloc[-1]
    cpi_prev     = cpi_series.iloc[-13]
    inflation    = ((cpi_current - cpi_prev) /
                     cpi_prev * 100)
    col2.metric(
        label="Inflation Rate (YoY)",
        value=f"{inflation:.1f}%",
        delta=f"{inflation:+.1f}% annual rate"
    )

    # Unemployment
    unemp_series  = raw_data['Unemployment'].dropna()
    unemp_current = unemp_series.iloc[-1]
    unemp_prev    = unemp_series.iloc[-13]
    unemp_delta   = unemp_current - unemp_prev
    col3.metric(
        label="Unemployment Rate",
        value=f"{unemp_current:.1f}%",
        delta=f"{unemp_delta:+.1f}pp vs 1yr ago"
    )

    st.markdown("---")

    # What MacroSense predicts
    st.subheader("What MacroSense Predicts")

    col1, col2, col3 = st.columns(3)

    col1.info("""
    **GDP Growth Rate**

    Is the US economy growing or contracting?
    MacroSense forecasts the growth rate
    6 months ahead so organisations can plan
    before conditions change.
    """)

    col2.info("""
    **Inflation Rate**

    Are prices rising too fast? The 6-month
    ahead inflation forecast helps businesses
    anticipate cost pressures before they
    materialise.
    """)

    col3.info("""
    **Unemployment Rate**

    Is the labour market tightening or loosening?
    Forward visibility helps with hiring plans
    and workforce strategy.
    """)

    st.markdown("---")

    # Performance summary
    st.subheader("System Performance")

    col1, col2, col3 = st.columns(3)
    col1.success(
        "**GDP Direction Accuracy**\n\n### 88.9%"
    )
    col2.success(
        "**Inflation Direction Accuracy**\n\n### 96.2%"
    )
    col3.success(
        "**Unemployment Direction Accuracy**\n\n### 100%"
    )

    st.caption(
        "Direction accuracy measures how often MacroSense "
        "correctly predicted whether each indicator would "
        "move up or down over the following 6 months. "
        "A random baseline would achieve 50%. "
        "Unemployment accuracy reflects the high persistence "
        "of this series rather than extraordinary model power."
    )

# ------------------------------------------------
# SECTION 9: PAGE 2 -- ECONOMIC DASHBOARD
# Displays when user selects Economic Dashboard
# Shows historical charts for all 10 variables
# with recession shading
# ------------------------------------------------

elif page == "Economic Dashboard":

    st.title("📊 Economic Dashboard")
    st.markdown(
        "Historical trends for all 10 economic "
        "indicators with recession periods shaded grey."
    )

    # Dropdown to select which indicator to display
    indicator = st.selectbox(
        "Select Indicator",
        ["GDP", "CPI", "Unemployment", "FedFunds",
         "M2", "YieldCurve", "IndPro",
         "RetailSales", "Sentiment", "JoblessClaims"]
    )

    # Slider to control how many years to show
    years = st.slider(
        "Years of history to display",
        min_value=5,
        max_value=30,
        value=20
    )

    # Display labels for each indicator
    labels = {
        "GDP"          : ("Real GDP",
                          "Billions USD"),
        "CPI"          : ("Consumer Price Index",
                          "Index"),
        "Unemployment" : ("Unemployment Rate",
                          "%"),
        "FedFunds"     : ("Federal Funds Rate",
                          "%"),
        "M2"           : ("Money Supply M2",
                          "Billions USD"),
        "YieldCurve"   : ("Yield Curve Spread",
                          "Percentage Points"),
        "IndPro"       : ("Industrial Production",
                          "Index"),
        "RetailSales"  : ("Retail Sales",
                          "Millions USD"),
        "Sentiment"    : ("Consumer Sentiment",
                          "Index"),
        "JoblessClaims": ("Weekly Jobless Claims",
                          "Number of People")
    }

    title, ylabel = labels[indicator]

    # Filter series to selected time window
    series = raw_data[indicator].dropna()
    cutoff = series.index[-1] - pd.DateOffset(
        years=years
    )
    series = series[series.index >= cutoff]

    # Plot the chart
    fig, ax = plt.subplots(figsize=(12, 4))

    add_recession_shading(
        ax, recession,
        series.index[0],
        series.index[-1]
    )

    ax.plot(series.index, series.values,
            color='steelblue', linewidth=1.8)
    ax.set_title(title, fontsize=13,
                  fontweight='bold')
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Date')
    ax.grid(alpha=0.3)

    grey_patch = mpatches.Patch(
        color='grey', alpha=0.2,
        label='Recession'
    )
    ax.legend(handles=[grey_patch], fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)

    # Plain English description of each indicator
    descriptions = {
        "GDP"         : "Real GDP measures the total value of all goods and services produced in the US adjusted for inflation. It is the broadest measure of economic health. When GDP grows the economy is expanding. When it shrinks the economy is in recession.",
        "CPI"         : "The Consumer Price Index tracks how much everyday goods and services cost over time. Year over year change gives us the inflation rate. The Federal Reserve targets roughly 2% annual inflation.",
        "Unemployment": "The unemployment rate measures the percentage of working age people actively looking for work but unable to find it. It is a lagging indicator -- it rises after recessions begin not before.",
        "FedFunds"    : "The Federal Funds Rate is set by the Federal Reserve and controls the cost of borrowing throughout the economy. The Fed raises rates to cool inflation and cuts them to stimulate growth.",
        "M2"          : "M2 measures the total amount of money circulating in the US economy. Rapid M2 growth often precedes inflation as too much money chases too few goods.",
        "YieldCurve"  : "The yield curve spread is the difference between 10-year and 2-year Treasury interest rates. When it goes negative -- called inversion -- it has preceded every US recession since 1976 without exception.",
        "IndPro"      : "Industrial Production measures output from US factories, mines, and utilities. It tends to fall before recessions as businesses anticipate slowing demand and cut production early.",
        "RetailSales" : "Retail sales measure consumer spending which drives roughly 70% of US GDP. When consumers stop spending GDP almost always follows downward.",
        "Sentiment"   : "The University of Michigan Consumer Sentiment Index measures how optimistic Americans feel about economic conditions. It is a leading indicator -- it falls before recessions begin as people sense trouble coming.",
        "JoblessClaims": "Weekly initial jobless claims measure how many people filed for unemployment benefits that week. It is the most timely labour market indicator available -- released every Thursday and reacting within weeks of economic conditions changing."
    }

    st.info(descriptions[indicator])

# ------------------------------------------------
# SECTION 10: PAGE 3 -- MODEL FORECASTS
# Displays when user selects Model Forecasts
# Shows MacroSense 6-month ahead predictions
# using the most recent available FRED data
# ------------------------------------------------

elif page == "Model Forecasts":

    st.title("🔮 MacroSense Forecasts")
    st.markdown(
        "6-month ahead forecasts generated from "
        "the most recent available Federal Reserve data."
    )

    st.markdown("---")

    # Get the most recent row of features
    # This is what the model uses to predict
    X_all        = fe_clean[feature_cols]
    latest_row   = X_all.iloc[[-1]]
    latest_date  = X_all.index[-1]
    forecast_date = (latest_date +
                     pd.DateOffset(months=6))

    st.info(
        f"Forecasts based on data available up to: "
        f"**{latest_date.strftime('%B %Y')}**   "
        f"Predicting conditions in: "
        f"**{forecast_date.strftime('%B %Y')}**"
    )

    st.markdown("---")
    st.subheader("6-Month Ahead Forecasts")

    col1, col2, col3 = st.columns(3)

    target_cols_display = {
        'target_GDP'          : ('GDP Growth Rate',   col1),
        'target_Inflation'    : ('Inflation Rate',    col2),
        'target_Unemployment' : ('Unemployment Rate', col3)
    }

    forecasts = {}

    for target, (name, col) in \
            target_cols_display.items():

        if target in models:
            pred = models[target].predict(
                latest_row
            )[0]
            forecasts[target] = pred

            col.metric(
                label=f"{name} -- 6 Month Forecast",
                value=f"{pred:.2f}%"
            )

    st.markdown("---")

    # Historical context chart
    st.subheader("Forecast In Historical Context")
    st.markdown(
        "The red dot shows where MacroSense predicts "
        "the indicator will be in 6 months. The blue "
        "line shows where it has been historically."
    )

    target_choice = st.selectbox(
        "Select indicator for detailed view",
        ["GDP Growth Rate",
         "Inflation Rate",
         "Unemployment Rate"]
    )

    target_map = {
        "GDP Growth Rate"   : 'target_GDP',
        "Inflation Rate"    : 'target_Inflation',
        "Unemployment Rate" : 'target_Unemployment'
    }

    feat_map = {
        "GDP Growth Rate"   : 'GDP_growth',
        "Inflation Rate"    : 'Inflation_yoy',
        "Unemployment Rate" : 'Unemployment'
    }

    selected_target = target_map[target_choice]
    selected_feat   = feat_map[target_choice]

    if selected_feat in fe_clean.columns:

        # Show last 5 years of history
        historical = fe_clean[selected_feat].iloc[-60:]

        fig, ax = plt.subplots(figsize=(12, 4))

        add_recession_shading(
            ax, recession,
            historical.index[0],
            historical.index[-1]
        )

        ax.plot(historical.index,
                historical.values,
                color='steelblue',
                linewidth=1.8,
                label='Historical')

        if selected_target in forecasts:
            ax.scatter(
                [forecast_date],
                [forecasts[selected_target]],
                color='crimson',
                s=150,
                zorder=5,
                label=(
                    f"MacroSense Forecast "
                    f"({forecast_date.strftime('%b %Y')})"
                )
            )
            ax.axvline(
                latest_date,
                color='grey',
                linewidth=1,
                linestyle='--',
                alpha=0.7,
                label='Latest data point'
            )

        ax.set_title(
            f'{target_choice} -- '
            f'Last 5 Years and 6-Month Forecast',
            fontweight='bold'
        )
        ax.set_ylabel(f'{target_choice} (%)')
        ax.set_xlabel('Date')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        plt.tight_layout()

        st.pyplot(fig)

    st.markdown("---")
    st.warning(
        "MacroSense forecasts are generated by machine "
        "learning models trained on historical data. "
        "They are one input into economic analysis -- "
        "not definitive predictions. Economic conditions "
        "can change rapidly due to unforeseen events "
        "that no historical model can anticipate. "
        "Always combine quantitative forecasts with "
        "qualitative judgment and domain expertise."
    )


# ------------------------------------------------
# SECTION 11: PAGE 4 -- WALK FORWARD VALIDATION
# Displays when user selects Walk Forward
# Shows pre-computed validation results
# explaining how the system was tested
# ------------------------------------------------

elif page == "Walk Forward Validation":

    st.title("🔄 Walk Forward Validation")

    st.markdown("""
    Walk forward validation simulated MacroSense
    running live from the late 1990s onward.
    Every month a forecast was made using only
    data genuinely available at that moment.
    The model retrained each month on all
    available history before making the next
    forecast -- exactly how a live deployed
    system would operate.
    """)

    st.markdown("---")

    # Performance metrics
    st.subheader("Overall Performance")

    col1, col2, col3 = st.columns(3)
    col1.metric("GDP Direction Accuracy",
                "88.9%",
                "vs 50% random baseline")
    col2.metric("Inflation Direction Accuracy",
                "96.2%",
                "vs 50% random baseline")
    col3.metric("Unemployment Direction Accuracy",
                "100.0%",
                "vs 50% random baseline")

    st.caption(
        "Direction accuracy measures how often "
        "MacroSense correctly predicted whether each "
        "indicator would move up or down over the "
        "following 6 months."
    )

    st.markdown("---")

    # Breakdown table
    st.subheader("Performance Breakdown")

    breakdown = pd.DataFrame({
        'Indicator'        : [
            'GDP Growth',
            'Inflation Rate',
            'Unemployment Rate'
        ],
        'Overall'          : [
            '88.9%', '96.2%', '100.0%'
        ],
        'During Recessions': [
            '70.0%', '70.0%', '100.0%'
        ],
        'Normal Periods'   : [
            '90.5%', '98.3%', '100.0%'
        ],
        'Pre 2010'         : [
            '88.9%', '91.7%', '100.0%'
        ],
        'Post 2010'        : [
            '88.9%', '97.9%', '100.0%'
        ]
    })

    st.dataframe(breakdown,
                  use_container_width=True)

    st.markdown("---")

    # Explanation of what walk forward means
    st.subheader(
        "Why Walk Forward Validation Matters"
    )

    st.info("""
    A simple train-test split gives one evaluation
    score that may not reflect consistent real-world
    performance. Walk forward validation generates
    hundreds of evaluation scores -- one for every
    month tested -- across stable periods,
    pre-recession periods, crisis periods, and
    recoveries. This gives a far more honest picture
    of how the system would actually perform
    in live deployment.

    The drop in accuracy during recessions -- from
    roughly 90% to 70% for GDP -- is expected and
    honest. Sudden economic shocks are inherently
    harder to predict than normal conditions.
    Any model claiming perfect accuracy during
    recessions should be treated with serious
    skepticism.
    """)

    st.markdown("---")

    # Note about charts
    st.subheader("Validation Charts")
    st.markdown(
        "The full walk forward charts showing actual "
        "versus predicted values across decades of "
        "economic history are available in the project "
        "notebooks on GitHub. The charts folder in the "
        "repository contains all saved visualisations "
        "from the complete validation run."
    )

    col1, col2 = st.columns(2)
    col1.markdown(
        "**GitHub Repository**\n\n"
        "View complete notebooks and charts"
    )
    col2.markdown(
        "**Notebook 06**\n\n"
        "06_walk_forward.ipynb contains the "
        "full validation code and results"
    )

# ------------------------------------------------
# SECTION 12: PAGE 5 -- ABOUT THIS PROJECT
# Displays when user selects About This Project
# Documents methodology, data, and limitations
# ------------------------------------------------

elif page == "About This Project":

    st.title("📋 About MacroSense")

    st.markdown("""
    MacroSense is an end-to-end machine learning
    system built to forecast US economic conditions
    6 months ahead using publicly available
    Federal Reserve data.
    """)

    st.markdown("---")

    st.subheader("The Problem")
    st.markdown("""
    Most organisations make economic decisions
    reactively. They respond to what the economy
    did last quarter rather than preparing for
    what it will do next. MacroSense addresses
    this by providing a systematic data-driven
    early warning system built entirely from
    public Federal Reserve data.
    """)

    st.markdown("---")

    st.subheader("Data Sources")

    data_dict = pd.DataFrame({
        'Variable'   : [
            'GDP', 'CPI', 'Unemployment',
            'Fed Funds Rate', 'Money Supply M2',
            'Yield Curve', 'Industrial Production',
            'Retail Sales', 'Consumer Sentiment',
            'Jobless Claims'
        ],
        'FRED Code'  : [
            'GDPC1', 'CPIAUCSL', 'UNRATE',
            'FEDFUNDS', 'M2SL', 'T10Y2Y',
            'INDPRO', 'RSAFS', 'UMCSENT', 'ICSA'
        ],
        'Type'       : [
            'Target', 'Target', 'Target',
            'Input', 'Input', 'Input',
            'Input', 'Input', 'Input', 'Input'
        ]
    })

    st.dataframe(data_dict,
                  use_container_width=True)

    st.markdown("---")

    st.subheader("Methodology")

    col1, col2 = st.columns(2)

    col1.markdown("""
    **Feature Engineering**
    - Level variables transformed to growth rates
    - Lag features at 1, 3, 6, and 12 months
    - Yield curve inversion binary flag
    - Target variables shifted 6 months forward
    - Chronological 80/20 train-test split
    """)

    col2.markdown("""
    **Models Built**
    - Linear Regression (baseline)
    - Ridge Regression
    - Random Forest (200 trees)
    - XGBoost (300 trees, primary model)
    - Ensemble (average of all four)
    """)

    st.markdown("---")

    st.subheader("Key Findings")

    st.markdown("""
    The yield curve spread is the single most
    powerful predictor in the dataset. It inverted
    before the 2001 Dot-com recession, the 2008
    Financial Crisis, and again starting in 2023.
    Every major US recession since 1976 was preceded
    by yield curve inversion.

    Consumer sentiment leads economic downturns.
    It declined before official recession dates
    in 2001 and 2008 -- consumers sensed trouble
    before economists confirmed it.

    Unemployment is a lagging indicator. It only
    rises after a recession has already begun
    making jobless claims -- which react within
    weeks -- more useful for early warning.
    """)

    st.markdown("---")

    st.subheader("Honest Limitations")
    
    st.warning("""
    1. Novel shocks -- no model trained on historical
       patterns can predict unprecedented events like
       pandemics or novel financial crises

    2. US-centric -- the system uses US data only
       and does not generalise to other economies
       without retraining

    3. Structural breaks -- if the economy behaves
       fundamentally differently from historical
       patterns model performance will degrade

    4. Unemployment accuracy -- the 100% direction
       accuracy reflects series persistence rather
       than extraordinary predictive power

    5. Not financial advice -- MacroSense is for
       research and educational purposes only
    """)

    st.markdown("---")

    st.subheader("Technical Stack")

    col1, col2, col3 = st.columns(3)

    col1.markdown("""
    **Data**
    - Python 3.13
    - pandas
    - FRED API
    - fredapi
    """)

    col2.markdown("""
    **Modelling**
    - scikit-learn
    - XGBoost
    - statsmodels
    - SHAP
    """)

    col3.markdown("""
    **Deployment**
    - Streamlit
    - joblib
    - matplotlib
    - python-dotenv
    """)

    st.markdown("---")

    st.subheader("Built By")

    st.markdown("""
    Aremu Gideon Marvelous
    
    I am an Economics graduate with a strong interest in 
    applying machine learning to real-world economic and 
    financial problems. I am currently preparing for a 
    Masters in Statistics where I plan to deepen my 
    expertise in statistical modelling and data science.
    
    MacroSense reflects my belief that economics domain 
    knowledge and machine learning are most powerful when 
    combined. Understanding why the yield curve inverts 
    before recessions, why unemployment lags behind GDP, 
    and why consumer sentiment leads economic downturns 
    is what separates a meaningful forecasting system 
    from a pure algorithmic exercise.
    
    I am actively seeking data science roles where I can 
    apply this combination of economic intuition and 
    technical skills to solve problems that matter.
    """)
    
    col1, col2 = st.columns(2)
    
    col1.markdown("""
    **Connect With Me**
    - LinkedIn: [Aremu Gideon Marvelous](https://www.linkedin.com/in/gideon-marvelous-5a19b919b?utm_source=share_via&utm_content=profile&utm_medium=member_android)
    - GitHub: [Marvy](https://github.com/MarvyGithub)
    - Email: aremumarvelous@gmail.com
    """)
    
    col2.markdown("""
    **Background**
    - BSc Economics -- University of Ilorin, Ilorin, Nigeria
    - Masters in Statistics -- Incoming
    - Interests: Macroeconomic forecasting, 
      financial ML, time series analysis, Deep Learning, AI
    """)
        
    st.markdown("""
    This project was built as a portfolio
    demonstration of applied machine learning
    in macroeconomic forecasting. It combines
    economics domain knowledge with machine
    learning engineering to produce a system
    that is both technically rigorous and
    economically interpretable.
    
    The complete source code, notebooks, and
    documentation are available on GitHub.
    """)
    
    st.caption(
        "MacroSense is for research and educational "
        "purposes only. Not financial advice."
    )

# ------------------------------------------------
# END OF APP
# ------------------------------------------------
