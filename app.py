# ================================================
# MacroSense -- US Economic Forecasting Dashboard
# Run with: streamlit run app.py
# ================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
import os
from fredapi import Fred
from dotenv import load_dotenv
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings('ignore')

# ------------------------------------------------
# Page Configuration
# ------------------------------------------------
st.set_page_config(
    page_title="MacroSense -- US Economic Forecaster",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------
# Load API Key and Connect to FRED
# ------------------------------------------------
load_dotenv()
api_key = os.getenv('FRED_API_KEY')

@st.cache_data
def load_data():
    """Load our featured dataset from disk"""
    fe = pd.read_csv('data/featured_fred_data.csv',
                      index_col=0,
                      parse_dates=True)
    return fe

@st.cache_data
def load_walkforward_results():
    """Load walk forward validation results"""
    return pd.read_csv('outputs/walkforward_results.csv',
                        header=[0, 1],
                        index_col=0,
                        parse_dates=True)

@st.cache_data
def load_raw_data():
    """Load cleaned raw data for historical charts"""
    return pd.read_csv('data/cleaned_fred_data.csv',
                        index_col=0,
                        parse_dates=True)

@st.cache_resource
def load_models():
    """Load all trained XGBoost models"""
    targets = ['target_GDP',
               'target_Inflation',
               'target_Unemployment']
    models = {}
    for t in targets:
        path = f'outputs/XGBoost_{t}.pkl'
        if os.path.exists(path):
            models[t] = joblib.load(path)
    return models

@st.cache_resource
def load_scaler():
    return joblib.load('outputs/scaler.pkl')

# ------------------------------------------------
# Helper -- Recession Shading
# ------------------------------------------------
@st.cache_data
def get_recession_data():
    fred      = Fred(api_key=api_key)
    recession = fred.get_series('USREC')
    recession = recession.resample('ME').last()
    return recession

def add_recession_shading(ax, recession, 
                           start_date, end_date):
    rec = recession[
        (recession.index >= start_date) &
        (recession.index <= end_date)
    ]
    in_recession = False
    start = None
    for date, val in rec.items():
        if val == 1 and not in_recession:
            start = date
            in_recession = True
        elif val == 0 and in_recession:
            ax.axvspan(start, date,
                      alpha=0.2, color='grey')
            in_recession = False

# ------------------------------------------------
# Load Everything
# ------------------------------------------------
fe_clean  = load_data()
raw_data  = load_raw_data()
models    = load_models()
scaler    = load_scaler()
recession = get_recession_data()

target_cols  = ['target_GDP',
                'target_Inflation',
                'target_Unemployment']
feature_cols = [c for c in fe_clean.columns
                if c not in target_cols]

X = fe_clean[feature_cols]
y = fe_clean[target_cols]

# ------------------------------------------------
# Sidebar
# ------------------------------------------------
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/"
    "thumb/8/8a/Wikimedia_Foundation_logo_-_vertical.svg/"
    "402px-Wikimedia_Foundation_logo_-_vertical.svg.png",
    width=50
)

st.sidebar.title("MacroSense")
st.sidebar.markdown("*US Economic Forecasting System*")
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
st.sidebar.markdown("**Primary Model:** XGBoost Ensemble")
st.sidebar.markdown("**Data Coverage:** 1992 to present")

# ------------------------------------------------
# PAGE 1 -- OVERVIEW
# ------------------------------------------------
if page == "Overview":
    
    st.title("📈 MacroSense")
    st.subheader(
        "A Machine Learning System for US Economic Forecasting"
    )
    
    st.markdown("""
    MacroSense uses 60 years of Federal Reserve data and 
    machine learning to forecast three key US economic 
    indicators **6 months into the future**.
    
    Most organisations make economic decisions based on 
    what happened last quarter. MacroSense gives them a 
    data-driven view of what is coming next.
    """)
    
    st.markdown("---")
    
    # Current snapshot
    st.subheader("Current Economic Snapshot")
    
    col1, col2, col3 = st.columns(3)
    
    # GDP
    gdp_current = raw_data['GDP'].dropna().iloc[-1]
    gdp_prev    = raw_data['GDP'].dropna().iloc[-5]
    gdp_growth  = ((gdp_current - gdp_prev) / 
                    gdp_prev * 100)
    col1.metric(
        "Real GDP",
        f"${gdp_current:,.0f}B",
        f"{gdp_growth:+.1f}% vs 1yr ago"
    )
    
    # Inflation
    cpi_current = raw_data['CPI'].dropna().iloc[-1]
    cpi_prev    = raw_data['CPI'].dropna().iloc[-13]
    inflation   = ((cpi_current - cpi_prev) / 
                    cpi_prev * 100)
    col2.metric(
        "Inflation Rate (YoY)",
        f"{inflation:.1f}%",
        f"{inflation - ((raw_data['CPI'].dropna().iloc[-13] - raw_data['CPI'].dropna().iloc[-25]) / raw_data['CPI'].dropna().iloc[-25] * 100):+.1f}% vs prior year"
    )
    
    # Unemployment
    unemp_current = raw_data['Unemployment'].dropna().iloc[-1]
    unemp_prev    = raw_data['Unemployment'].dropna().iloc[-13]
    col3.metric(
        "Unemployment Rate",
        f"{unemp_current:.1f}%",
        f"{unemp_current - unemp_prev:+.1f}pp vs 1yr ago"
    )
    
    st.markdown("---")
    
    # What MacroSense predicts
    st.subheader("What MacroSense Predicts")
    
    col1, col2, col3 = st.columns(3)
    
    col1.info("""
    **GDP Growth Rate**
    
    Is the US economy growing or 
    contracting? MacroSense forecasts 
    the quarterly growth rate 6 months 
    ahead so organisations can plan 
    before conditions change.
    """)
    
    col2.info("""
    **Inflation Rate**
    
    Are prices rising too fast? 
    The 6-month ahead inflation forecast 
    helps businesses anticipate cost 
    pressures and pricing decisions 
    before they materialise.
    """)
    
    col3.info("""
    **Unemployment Rate**
    
    Is the labour market tightening 
    or loosening? Forward visibility 
    on unemployment helps with hiring 
    plans, wage expectations, and 
    workforce strategy.
    """)
    
    st.markdown("---")
    
    # Performance summary
    st.subheader("System Performance")
    
    col1, col2, col3 = st.columns(3)
    col1.success("**GDP Direction Accuracy**\n\n# 88.9%")
    col2.success("**Inflation Direction Accuracy**\n\n# 96.2%")
    col3.success("**Unemployment Direction Accuracy**\n\n# 100%*")
    
    st.caption(
        "* Unemployment direction accuracy reflects the "
        "high persistence of this series. Unemployment "
        "trends in one direction for extended periods "
        "making directional prediction more straightforward "
        "than for GDP or inflation."
    )

# ------------------------------------------------
# PAGE 2 -- ECONOMIC DASHBOARD
# ------------------------------------------------
elif page == "Economic Dashboard":
    
    st.title("📊 Economic Dashboard")
    st.markdown(
        "Historical trends for all 10 economic indicators "
        "with recession periods shaded grey."
    )
    
    indicator = st.selectbox(
        "Select Indicator",
        ["GDP", "CPI", "Unemployment", "FedFunds",
         "M2", "YieldCurve", "IndPro",
         "RetailSales", "Sentiment", "JoblessClaims"]
    )
    
    years = st.slider(
        "Years of history to display",
        min_value=5,
        max_value=34,
        value=20
    )
    
    labels = {
        "GDP"          : ("Real GDP", "Billions USD"),
        "CPI"          : ("Consumer Price Index", "Index"),
        "Unemployment" : ("Unemployment Rate", "%"),
        "FedFunds"     : ("Federal Funds Rate", "%"),
        "M2"           : ("Money Supply M2", "Billions USD"),
        "YieldCurve"   : ("Yield Curve Spread", "% Points"),
        "IndPro"       : ("Industrial Production", "Index"),
        "RetailSales"  : ("Retail Sales", "Millions USD"),
        "Sentiment"    : ("Consumer Sentiment", "Index"),
        "JoblessClaims": ("Weekly Jobless Claims", "People")
    }
    
    title, ylabel = labels[indicator]
    
    series = raw_data[indicator].dropna()
    cutoff = series.index[-1] - pd.DateOffset(years=years)
    series = series[series.index >= cutoff]
    
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
        color='grey', alpha=0.2, label='Recession'
    )
    ax.legend(handles=[grey_patch], fontsize=9)
    plt.tight_layout()
    
    st.pyplot(fig)
    
    # Indicator description
    descriptions = {
        "GDP": "Real GDP measures the total value of all goods and services produced in the US adjusted for inflation. It is the broadest measure of economic health.",
        "CPI": "The Consumer Price Index tracks how much everyday goods and services cost over time. Year over year change gives us the inflation rate.",
        "Unemployment": "The unemployment rate measures the percentage of working age people actively looking for work but unable to find it.",
        "FedFunds": "The Federal Funds Rate is set by the Federal Reserve and influences borrowing costs throughout the entire economy.",
        "M2": "M2 measures the total amount of money circulating in the US economy including cash and bank deposits.",
        "YieldCurve": "The yield curve spread is the difference between 10-year and 2-year Treasury rates. When it goes negative it has preceded every US recession since 1976.",
        "IndPro": "Industrial Production measures output from factories, mines, and utilities. It tends to fall before recessions as businesses anticipate slowing demand.",
        "RetailSales": "Retail sales measure consumer spending which drives roughly 70% of US GDP.",
        "Sentiment": "The University of Michigan Consumer Sentiment Index measures how optimistic Americans feel about economic conditions.",
        "JoblessClaims": "Weekly initial jobless claims measure how many people filed for unemployment benefits that week -- one of the most timely economic indicators available."
    }
    
    st.info(descriptions[indicator])

# ------------------------------------------------
# PAGE 3 -- MODEL FORECASTS
# ------------------------------------------------
elif page == "Model Forecasts":
    
    st.title("🔮 MacroSense Forecasts")
    st.markdown(
        "6-month ahead forecasts generated by MacroSense "
        "using the most recent available FRED data."
    )
    
    st.markdown("---")
    
    if len(models) < 3:
        st.warning(
            "Some model files were not found. "
            "Please ensure all notebooks have been "
            "run successfully before using this page."
        )
    else:
        # Get most recent features
        latest_features = X.iloc[[-1]]
        latest_date     = X.index[-1]
        
        forecast_date = latest_date + pd.DateOffset(months=6)
        
        st.info(
            f"Forecasts based on data up to: "
            f"**{latest_date.strftime('%B %Y')}**  "
            f"Predicting conditions in: "
            f"**{forecast_date.strftime('%B %Y')}**"
        )
        
        col1, col2, col3 = st.columns(3)
        
        target_info = {
            'target_GDP': {
                'name'  : 'GDP Growth Rate',
                'unit'  : '%',
                'col'   : col1,
                'color' : 'steelblue'
            },
            'target_Inflation': {
                'name'  : 'Inflation Rate',
                'unit'  : '%',
                'col'   : col2,
                'color' : 'crimson'
            },
            'target_Unemployment': {
                'name'  : 'Unemployment Rate',
                'unit'  : '%',
                'col'   : col3,
                'color' : 'darkorange'
            }
        }
        
        forecasts = {}
        
        for target, info in target_info.items():
            if target in models:
                pred = models[target].predict(
                    latest_features
                )[0]
                forecasts[target] = pred
                
                # Current actual value
                current = y[target].dropna().iloc[-7]
                delta   = pred - current
                
                info['col'].metric(
                    label=f"{info['name']} Forecast",
                    value=f"{pred:.2f}{info['unit']}",
                    delta=f"{delta:+.2f}{info['unit']} vs current"
                )
        
        st.markdown("---")
        
        # Historical context chart
        st.subheader("Forecast in Historical Context")
        
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
                    label=f'MacroSense Forecast ({forecast_date.strftime("%b %Y")})'
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
                f'{target_choice} -- Last 5 Years + Forecast',
                fontweight='bold'
            )
            ax.set_ylabel(f'{target_choice} (%)')
            ax.set_xlabel('Date')
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)
            plt.tight_layout()
            
            st.pyplot(fig)
        
        st.markdown("---")
        st.subheader("Important Disclaimer")
        st.warning(
            "MacroSense forecasts are generated by machine "
            "learning models trained on historical data. "
            "They are intended as one input into economic "
            "analysis and decision making -- not as "
            "definitive predictions. Economic conditions "
            "can change rapidly due to unforeseen events "
            "that no historical model can anticipate. "
            "Always combine quantitative forecasts with "
            "qualitative judgment and domain expertise."
        )

# ------------------------------------------------
# PAGE 4 -- WALK FORWARD VALIDATION
# ------------------------------------------------
elif page == "Walk Forward Validation":
    
    st.title("🔄 Walk Forward Validation")
    st.markdown("""
    Walk forward validation simulates MacroSense running 
    live from the late 1990s onward. Every month a forecast 
    is made using only data genuinely available at that 
    moment. The results show how the system would have 
    performed under real world conditions across three 
    decades of economic history including two major 
    recessions and a global pandemic.
    """)
    
    st.markdown("---")
    
    # Performance metrics
    st.subheader("Overall Performance")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("GDP Direction Accuracy",    "88.9%")
    col2.metric("Inflation Direction Accuracy", "96.2%")
    col3.metric("Unemployment Direction Accuracy", "100%")
    
    st.caption(
        "Direction accuracy measures how often MacroSense "
        "correctly predicted whether each indicator would "
        "move up or down over the following 6 months. "
        "A random baseline would achieve 50%."
    )
    
    st.markdown("---")
    
    # Performance breakdown
    st.subheader("Performance Breakdown")
    
    breakdown = pd.DataFrame({
        'Indicator'       : ['GDP Growth',
                             'Inflation Rate',
                             'Unemployment Rate'],
        'Overall'         : ['88.9%', '96.2%', '100.0%'],
        'During Recessions': ['70.0%', '70.0%', '100.0%'],
        'Normal Periods'  : ['90.5%', '98.3%', '100.0%'],
        'Pre 2010'        : ['88.9%', '91.7%', '100.0%'],
        'Post 2010'       : ['88.9%', '97.9%', '100.0%']
    })
    
    st.dataframe(breakdown, use_container_width=True)
    
    st.markdown("---")
    
    # Walk forward charts
    st.subheader("Actual vs Predicted Over Time")
    
    target_choice = st.selectbox(
        "Select indicator",
        ["GDP Growth",
         "Inflation Rate",
         "Unemployment Rate"]
    )
    
    wf_map = {
        "GDP Growth"       : 'target_GDP',
        "Inflation Rate"   : 'target_Inflation',
        "Unemployment Rate": 'target_Unemployment'
    }
    
    selected = wf_map[target_choice]
    
    try:
        wf_data = pd.read_csv(
            'outputs/walkforward_results.csv',
            index_col=0,
            parse_dates=True
        )
        
        actual_col    = f'{selected}_Actual'
        predicted_col = f'{selected}_Predicted'
        error_col     = f'{selected}_Error'
        
        if actual_col in wf_data.columns:
            
            fig, axes = plt.subplots(
                2, 1, figsize=(12, 8)
            )
            
            # Actual vs Predicted
            add_recession_shading(
                axes[0], recession,
                wf_data.index[0],
                wf_data.index[-1]
            )
            
            axes[0].plot(
                wf_data.index,
                wf_data[actual_col],
                label='Actual',
                color='steelblue',
                linewidth=1.8
            )
            axes[0].plot(
                wf_data.index,
                wf_data[predicted_col],
                label='MacroSense Forecast',
                color='crimson',
                linewidth=1.5,
                linestyle='--'
            )
            axes[0].axhline(
                0, color='black',
                linewidth=0.8, alpha=0.4
            )
            axes[0].set_title(
                f'Actual vs Predicted -- {target_choice}',
                fontweight='bold'
            )
            axes[0].set_ylabel(f'{target_choice} (%)')
            axes[0].legend(fontsize=9)
            axes[0].grid(alpha=0.3)
            
            # Error over time
            errors = wf_data[error_col]
            colors = ['crimson' if e < 0 else 'steelblue'
                      for e in errors]
            
            axes[1].bar(
                wf_data.index, errors,
                color=colors, alpha=0.6, width=20
            )
            axes[1].axhline(
                0, color='black', linewidth=0.8
            )
            axes[1].set_title(
                'Forecast Error Over Time',
                fontweight='bold'
            )
            axes[1].set_ylabel('Error (Actual minus Predicted)')
            axes[1].set_xlabel('Date')
            axes[1].grid(alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
        else:
            st.info(
                "Walk forward results use a flat column "
                "structure. Displaying available data."
            )
            st.dataframe(wf_data.head(20))
            
    except Exception as e:
        st.warning(
            f"Walk forward results file could not be "
            f"loaded in expected format. "
            f"Please check outputs/walkforward_results.csv"
        )
    
    st.markdown("---")
    
    st.subheader("What Walk Forward Validation Means")
    st.info("""
    Unlike a simple train test split where the model trains 
    once and tests once, walk forward validation retrains 
    the model hundreds of times -- each time adding one more 
    month of real data before making the next forecast.
    
    This simulates exactly how MacroSense would operate in 
    real deployment. Every month when new FRED data is 
    released the model would retrain and generate a fresh 
    6-month forecast. Walk forward validation proves this 
    approach works consistently across decades of economic 
    history -- not just on one convenient test period.
    
    The drop in accuracy during recessions (from ~90% to 70% 
    for GDP) is expected and honest. Sudden economic shocks 
    are inherently harder to predict than normal economic 
    conditions. Any model claiming perfect accuracy during 
    recessions should be treated with serious skepticism.
    """)

# ------------------------------------------------
# PAGE 5 -- ABOUT THIS PROJECT
# ------------------------------------------------
elif page == "About This Project":
    
    st.title("📋 About MacroSense")
    
    st.markdown("""
    MacroSense is an end-to-end machine learning system 
    built to forecast US economic conditions 6 months 
    ahead using publicly available Federal Reserve data.
    """)
    
    st.markdown("---")
    
    st.subheader("The Problem")
    st.markdown("""
    Most organisations -- banks, companies, government 
    ministries -- make economic decisions reactively. 
    They respond to what the economy did last quarter 
    rather than preparing for what it will do next. 
    MacroSense addresses this by providing a systematic, 
    data-driven early warning system built entirely from 
    public Federal Reserve data.
    """)
    
    st.markdown("---")
    
    st.subheader("Data Sources")
    
    data_dict = pd.DataFrame({
        'Variable'    : ['GDP', 'CPI', 'Unemployment',
                         'Fed Funds Rate', 'Money Supply M2',
                         'Yield Curve', 'Industrial Production',
                         'Retail Sales', 'Consumer Sentiment',
                         'Jobless Claims'],
        'FRED Code'   : ['GDPC1', 'CPIAUCSL', 'UNRATE',
                         'FEDFUNDS', 'M2SL', 'T10Y2Y',
                         'INDPRO', 'RSAFS', 'UMCSENT', 'ICSA'],
        'Type'        : ['Target', 'Target', 'Target',
                         'Input', 'Input', 'Input',
                         'Input', 'Input', 'Input', 'Input'],
        'Coverage'    : ['1947-present', '1947-present',
                         '1948-present', '1954-present',
                         '1959-present', '1976-present',
                         '1919-present', '1992-present',
                         '1952-present', '1967-present']
    })
    
    st.dataframe(data_dict, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("Methodology")
    
    col1, col2 = st.columns(2)
    
    col1.markdown("""
    **Feature Engineering**
    - Level variables transformed to growth rates
    - Lag features at 1, 3, 6, and 12 months
    - Yield curve inversion binary flag
    - Rate of change features for key indicators
    - Target variables shifted 6 months forward
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
    
    st.subheader("Honest Limitations")
    
    st.warning("""
    **What MacroSense cannot do:**
    
    1. Predict sudden unpredictable shocks like pandemics, 
       wars, or financial crises driven by novel mechanisms 
       not present in historical data.
    
    2. Account for structural breaks -- when the economy 
       behaves fundamentally differently from its 
       historical patterns.
    
    3. Replace domain expertise and qualitative judgment. 
       MacroSense is one input into economic analysis, 
       not a substitute for it.
    
    4. Automatically update in real time. The system 
       requires manual retraining when new FRED data 
       is released each month.
    
    5. Generalise to other countries without retraining 
       on country-specific data.
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
    This project was built as a portfolio demonstration 
    of applied machine learning in macroeconomic 
    forecasting. It combines an economics background 
    with machine learning engineering to produce a 
    system that is both technically rigorous and 
    economically interpretable.
    
    The complete source code, notebooks, and 
    documentation are available on GitHub.
    """)
    
    st.caption(
        "MacroSense is for research and educational "
        "purposes only. Not financial advice."
    )