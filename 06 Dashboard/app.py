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
app_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(app_dir)

```
cleaned_file = os.path.join(
    repo_root,
    "00 Data",
    "Processed Data",
    "AmesHousing_cleaned.csv"
)

prediction_file = os.path.join(
    repo_root,
    "00 Data",
    "Feature Matrix Data",
    "ridge_predictions.csv"
)

if not os.path.exists(cleaned_file):
    st.error(f"Missing file: {cleaned_file}")
    st.stop()

if not os.path.exists(prediction_file):
    st.error(f"Missing file: {prediction_file}")
    st.stop()

df = pd.read_csv(cleaned_file)
pred = pd.read_csv(prediction_file)

return df, pred
```

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
