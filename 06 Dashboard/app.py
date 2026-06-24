import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from pathlib import Path
import pandas as pd
import streamlit as st

# ── Page config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Ames Housing — Price Intelligence",
    page_icon="🏠",
    layout="wide"
)

# ── Load data ────────────────────────────────────────────────────
@st.cache_data

def load_data():
    # 1. Get the directory where app.py lives (06 Dashboard)
    app_dir = Path(__file__).parent.resolve()
    
    # 2. Get the repository root directory: 
    repo_root = app_dir.parent
    
    # 3. Build the absolute path to your processed data folder
    # Note: Ensure the folder names match your GitHub casing exactly (e.g., '00 Data' vs '00 data')
    data_dir = repo_root / '00 Data' / 'Processed Data'
    
    # Debugging check: If the directory doesn't exist, print what Streamlit sees
    if not data_dir.exists():
        st.error(f"Directory not found: {data_dir}")
        st.write("Available folders in root:", [p.name for p in repo_root.iterdir() if p.is_dir()])
        return pd.DataFrame(), pd.DataFrame()

    # 4. Load the files
    df   = pd.read_csv(data_dir / 'AmesHousing_cleaned.csv')
    pred = pd.read_csv(data_dir / 'ridge_predictions.csv')
    return df, pred

df, pred = load_data()

# ── Sidebar filters ──────────────────────────────────────────────
st.sidebar.title("🏠 Filters")
st.sidebar.markdown("Use filters to explore the market")

neighborhoods = sorted(df['Neighborhood'].unique())
selected_neighborhoods = st.sidebar.multiselect(
    "Neighborhood",
    options=neighborhoods,
    default=neighborhoods
)

min_price = int(df['SalePrice'].min())
max_price = int(df['SalePrice'].max())
price_range = st.sidebar.slider(
    "Sale Price Range ($)",
    min_value=min_price,
    max_value=max_price,
    value=(min_price, max_price),
    step=5000,
    format="$%d"
)

quality_range = st.sidebar.slider(
    "Overall Quality (1–10)",
    min_value=1, max_value=10,
    value=(1, 10)
)

# ── Apply filters ────────────────────────────────────────────────
mask = (
    df['Neighborhood'].isin(selected_neighborhoods) &
    df['SalePrice'].between(price_range[0], price_range[1]) &
    df['Overall Qual'].between(quality_range[0], quality_range[1])
)
filtered = df[mask]

# ── Header ───────────────────────────────────────────────────────
st.title("🏠 Ames Housing — Real Estate Price Intelligence")
st.markdown(
    "An end-to-end data science project analyzing **2,928 house sales** "
    "in Ames, Iowa. Use the sidebar filters to explore the market."
)
st.markdown("---")

# ── KPI Row ──────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Houses in View",  f"{len(filtered):,}")
k2.metric("Median Price",    f"${filtered['SalePrice'].median():,.0f}")
k3.metric("Mean Price",      f"${filtered['SalePrice'].mean():,.0f}")
k4.metric("Min Price",       f"${filtered['SalePrice'].min():,.0f}")
k5.metric("Model RMSE",      "$18,342",
          help="Ridge Regression average prediction error on test set")

st.markdown("---")

# ── Tabs ─────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📍 Market Overview",
    "🔍 Price Drivers",
    "🤖 Model Performance",
    "📈 Market Trends"
])

# ════════════════════════════════════════════════════════════════
# TAB 1 — MARKET OVERVIEW
# ════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Median Sale Price by Neighborhood")
    st.markdown(
        "Every neighborhood ranked by median sale price. "
        "The red dashed line marks the overall market median."
    )

    nbhd = (
        filtered.groupby('Neighborhood')['SalePrice']
    )
