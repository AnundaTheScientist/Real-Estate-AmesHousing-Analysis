import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ── Page config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Ames Housing — Price Intelligence",
    page_icon="🏠",
    layout="wide"
)

# ── Load data ────────────────────────────────────────────────────
@st.cache_data
def load_data():
    base = os.path.dirname(__file__)
    df   = pd.read_csv(os.path.join(base, 'C:\\Users\\A.K\\DATA_SCIENCE_WORKSPACE\\kaggle_projects\\project_4_amesHousing\\Real Estate-AmesHousing-Analysis\\00 Data\\Processed Data\\AmesHousing_cleaned.csv'))
    pred = pd.read_csv(os.path.join(base, 'C:\\Users\\A.K\\DATA_SCIENCE_WORKSPACE\\kaggle_projects\\project_4_amesHousing\\Real Estate-AmesHousing-Analysis\\00 Data\\Feature Matrix Data\\ridge_predictions.csv'))
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

min_price, max_price = int(df['SalePrice'].min()), int(df['SalePrice'].max())
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

# Apply filters
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

k1.metric("Houses in View",
          f"{len(filtered):,}")
k2.metric("Median Price",
          f"${filtered['SalePrice'].median():,.0f}")
k3.metric("Mean Price",
          f"${filtered['SalePrice'].mean():,.0f}")
k4.metric("Price Range",
          f"${filtered['SalePrice'].min():,.0f} – ${filtered['SalePrice'].max():,.0f}")
k5.metric("Model Dollar RMSE",
          "$18,342",
          help="Ridge Regression average prediction error on test set")

st.markdown("---")

# ── Tab layout ───────────────────────────────────────────────────
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
        "The chart below ranks every neighborhood by median sale price. "
        "The red dashed line marks the overall market median."
    )

    nbhd_price = (
        filtered.groupby('Neighborhood')['SalePrice']
        .agg(['median', 'mean', 'count'])
        .rename(columns={'median': 'Median Price',
                         'mean':   'Mean Price',
                         'count':  'Sales Count'})
        .sort_values('Median Price', ascending=True)
        .reset_index()
    )

    overall_median = filtered['SalePrice'].median()

    fig_nbhd = px.bar(
        nbhd_price,
        x='Median Price',
        y='Neighborhood',
        orientation='h',
        color='Median Price',
        color_continuous_scale='RdYlGn',
        hover_data={'Median Price': ':$,.0f',
                    'Mean Price':   ':$,.0f',
                    'Sales Count':  True},
        title="Neighborhood Median Sale Price"
    )
    fig_nbhd.add_vline(
        x=overall_median,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Market median: ${overall_median:,.0f}",
        annotation_position="top right"
    )
    fig_nbhd.update_layout(
        height=650,
        coloraxis_showscale=False,
        yaxis_title="",
        xaxis_title="Median Sale Price ($)"
    )
    st.plotly_chart(fig_nbhd, use_container_width=True)

    # Two column breakdown
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Top 5 Most Expensive Neighborhoods**")
        top5 = (nbhd_price.sort_values('Median Price', ascending=False)
                           .head(5)[['Neighborhood', 'Median Price', 'Sales Count']]
                           .reset_index(drop=True))
        top5['Median Price'] = top5['Median Price'].apply(lambda x: f"${x:,.0f}")
        st.dataframe(top5, hide_index=True, use_container_width=True)

    with col2:
        st.markdown("**Bottom 5 Most Affordable Neighborhoods**")
        bot5 = (nbhd_price.sort_values('Median Price', ascending=True)
                           .head(5)[['Neighborhood', 'Median Price', 'Sales Count']]
                           .reset_index(drop=True))
        bot5['Median Price'] = bot5['Median Price'].apply(lambda x: f"${x:,.0f}")
        st.dataframe(bot5, hide_index=True, use_container_width=True)

# ════════════════════════════════════════════════════════════════
# TAB 2 — PRICE DRIVERS
# ════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("What Drives House Prices?")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Overall Quality vs Median Price**")
        st.markdown(
            "Overall Quality (1–10) is the strongest single predictor. "
            "Each quality step up adds significant value."
        )
        qual_price = (
            filtered.groupby('Overall Qual')['SalePrice']
            .median().reset_index()
        )
        fig_qual = px.bar(
            qual_price,
            x='Overall Qual',
            y='SalePrice',
            color='SalePrice',
            color_continuous_scale='Blues',
            labels={'SalePrice': 'Median Price ($)',
                    'Overall Qual': 'Quality Rating'},
            title="Quality Rating vs Median Sale Price"
        )
        fig_qual.update_layout(
            coloraxis_showscale=False,
            height=400
        )
        st.plotly_chart(fig_qual, use_container_width=True)

    with col2:
        st.markdown("**Living Area vs Sale Price**")
        st.markdown(
            "Above-ground living area is the #2 price driver. "
            "Each additional square foot adds measurable value."
        )
        fig_area = px.scatter(
            filtered,
            x='Gr Liv Area',
            y='SalePrice',
            color='Overall Qual',
            color_continuous_scale='RdYlGn',
            opacity=0.5,
            trendline='ols',
            labels={'Gr Liv Area': 'Above-Ground Living Area (sq ft)',
                    'SalePrice':   'Sale Price ($)',
                    'Overall Qual': 'Quality'},
            title="Living Area vs Sale Price"
        )
        fig_area.update_layout(height=400)
        st.plotly_chart(fig_area, use_container_width=True)

    st.markdown("---")
    st.markdown("**Price Distribution by House Style**")

    fig_style = px.box(
        filtered,
        x='House Style',
        y='SalePrice',
        color='House Style',
        labels={'SalePrice': 'Sale Price ($)',
                'House Style': 'House Style'},
        title="Sale Price Distribution by House Style"
    )
    fig_style.update_layout(
        height=450,
        showlegend=False,
        xaxis_title="House Style",
        yaxis_title="Sale Price ($)"
    )
    st.plotly_chart(fig_style, use_container_width=True)

    st.markdown("---")

    st.markdown("**Price by Number of Bedrooms Above Ground**")
    bed_price = (
        filtered.groupby('Bedroom AbvGr')['SalePrice']
        .agg(['median', 'count'])
        .rename(columns={'median': 'Median Price', 'count': 'Count'})
        .reset_index()
    )
    fig_bed = px.bar(
        bed_price,
        x='Bedroom AbvGr',
        y='Median Price',
        text='Count',
        labels={'Bedroom AbvGr': 'Bedrooms (Above Ground)',
                'Median Price': 'Median Price ($)'},
        title="Median Price by Bedroom Count"
    )
    fig_bed.update_traces(
        texttemplate='n=%{text}',
        textposition='outside'
    )
    fig_bed.update_layout(height=380)
    st.plotly_chart(fig_bed, use_container_width=True)

# ════════════════════════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Ridge Regression — Model Performance")
    st.markdown(
        "Our recommended model predicts house prices within **±$18,342** on average. "
        "Points on the red line represent a perfect prediction."
    )

    # KPI row for model
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Model",         "Ridge Regression")
    m2.metric("Test R²",       "0.9486",  help="94.9% of price variance explained")
    m3.metric("CV RMSE",       "0.1217",  help="5-fold cross-validated RMSE on log scale")
    m4.metric("Dollar RMSE",   "$18,342", help="Average prediction error in dollars")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Actual vs Predicted Price**")
        min_val = min(pred['Actual_Price'].min(),
                      pred['Predicted_Price'].min())
        max_val = max(pred['Actual_Price'].max(),
                      pred['Predicted_Price'].max())

        fig_pred = px.scatter(
            pred,
            x='Actual_Price',
            y='Predicted_Price',
            color='Residual_Dollar',
            color_continuous_scale='RdYlGn',
            opacity=0.6,
            labels={'Actual_Price':    'Actual Sale Price ($)',
                    'Predicted_Price': 'Predicted Sale Price ($)',
                    'Residual_Dollar': 'Residual ($)'},
            title="Actual vs Predicted Sale Price"
        )
        fig_pred.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='red', dash='dash', width=1.5),
            name='Perfect prediction'
        ))
        fig_pred.update_layout(height=440)
        st.plotly_chart(fig_pred, use_container_width=True)

    with col2:
        st.markdown("**Residual Distribution**")
        st.markdown(
            "A well-behaved model has residuals centered at zero "
            "with no systematic pattern."
        )
        fig_resid = px.histogram(
            pred,
            x='Residual_Dollar',
            nbins=50,
            color_discrete_sequence=['#1D9E75'],
            labels={'Residual_Dollar': 'Residual: Actual − Predicted ($)'},
            title="Distribution of Prediction Errors"
        )
        fig_resid.add_vline(
            x=0,
            line_dash="dash",
            line_color="red",
            annotation_text="Zero error"
        )
        fig_resid.update_layout(height=440)
        st.plotly_chart(fig_resid, use_container_width=True)

    # Model comparison table
    st.markdown("---")
    st.markdown("**All Models Compared**")

    comparison = pd.DataFrame({
        'Model': ['Linear Regression', 'Ridge Regression',
                  'Lasso Regression', 'Gradient Boosting'],
        'Test R²':     [0.9400, 0.9486, 0.9500, 0.9452],
        'CV RMSE':     [0.1311, 0.1217, 0.1235, 0.1249],
        'Dollar RMSE': ['$19,100', '$18,342', '$18,594', '$20,721'],
        'Verdict':     ['Baseline', '⭐ Recommended',
                        'Best R² / Lean model', 'Overfit on this dataset']
    })
    st.dataframe(comparison, hide_index=True, use_container_width=True)

# ════════════════════════════════════════════════════════════════
# TAB 4 — MARKET TRENDS
# ════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Market Trends Over Time")

    yr_price = (
        df.groupby('Yr Sold')['SalePrice']
        .agg(['median', 'count'])
        .rename(columns={'median': 'Median Price', 'count': 'Sales'})
        .reset_index()
    )

    fig_trend = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=('Median Sale Price by Year',
                        'Number of Sales by Year'),
        vertical_spacing=0.12
    )

    fig_trend.add_trace(
        go.Scatter(
            x=yr_price['Yr Sold'],
            y=yr_price['Median Price'],
            mode='lines+markers',
            line=dict(color='#1D9E75', width=2.5),
            marker=dict(size=8),
            name='Median Price',
            fill='tozeroy',
            fillcolor='rgba(29,158,117,0.1)'
        ),
        row=1, col=1
    )

    fig_trend.add_trace(
        go.Bar(
            x=yr_price['Yr Sold'],
            y=yr_price['Sales'],
            marker_color='#534AB7',
            name='Sales Count'
        ),
        row=2, col=1
    )

    fig_trend.update_layout(
        height=520,
        showlegend=False,
        xaxis2_title="Year Sold"
    )
    fig_trend.update_yaxes(title_text="Median Price ($)", row=1)
    fig_trend.update_yaxes(title_text="Number of Sales", row=2)

    st.plotly_chart(fig_trend, use_container_width=True)

    st.info(
        "📉 The dip in 2008–2009 reflects the U.S. financial crisis. "
        "Sales volume and median prices both declined during this period, "
        "recovering by 2010. This context is important when presenting "
        "model predictions to clients."
    )

    st.markdown("---")
    st.markdown("**Sale Price Distribution — Full Dataset**")

    fig_dist = px.histogram(
        df,
        x='SalePrice',
        nbins=60,
        color_discrete_sequence=['#534AB7'],
        labels={'SalePrice': 'Sale Price ($)'},
        title="Overall Sale Price Distribution"
    )
    fig_dist.add_vline(
        x=df['SalePrice'].median(),
        line_dash="dash", line_color="red",
        annotation_text=f"Median: ${df['SalePrice'].median():,.0f}"
    )
    fig_dist.add_vline(
        x=df['SalePrice'].mean(),
        line_dash="dot", line_color="orange",
        annotation_text=f"Mean: ${df['SalePrice'].mean():,.0f}"
    )
    fig_dist.update_layout(height=380)
    st.plotly_chart(fig_dist, use_container_width=True)

# ── Footer ───────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:grey; font-size:13px;'>"
    "Ames Housing Price Intelligence · Built with Python, Streamlit & Plotly · "
    "Data: Ames Housing Dataset (De Cock, 2011)"
    "</div>",
    unsafe_allow_html=True
)