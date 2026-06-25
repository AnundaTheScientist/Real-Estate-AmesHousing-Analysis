import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# ── Page config — MUST be first Streamlit command ───────────────
st.set_page_config(
    page_title="Ames Housing Analytics Hub",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global CSS — dark premium theme ─────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0F1117;
    color: #EAEAEA;
}
[data-testid="stSidebar"] {
    background-color: #1A1D27;
    border-right: 1px solid #2A2D3E;
}
[data-testid="stSidebar"] * {
    color: #EAEAEA !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] button {
    color: #8A8FA8 !important;
    font-size: 13px;
    font-weight: 500;
    letter-spacing: 0.04em;
    padding: 10px 20px;
    border-bottom: 2px solid transparent;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #00C49A !important;
    border-bottom: 2px solid #00C49A !important;
}

/* ── KPI cards ── */
.kpi-card {
    background: #1A1D27;
    border: 1px solid #2A2D3E;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    transition: border-color 0.2s ease;
}
.kpi-card:hover { border-color: #00C49A; }
.kpi-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.12em;
    color: #8A8FA8;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.kpi-value {
    font-size: 28px;
    font-weight: 700;
    color: #EAEAEA;
    font-variant-numeric: tabular-nums;
    letter-spacing: -0.02em;
}
.kpi-delta {
    font-size: 12px;
    color: #00C49A;
    margin-top: 4px;
}

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #1A1D27 0%, #0F1117 60%, #0d1f1a 100%);
    border: 1px solid #2A2D3E;
    border-left: 4px solid #00C49A;
    border-radius: 12px;
    padding: 32px 36px;
    margin-bottom: 24px;
}
.hero-eyebrow {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.16em;
    color: #00C49A;
    text-transform: uppercase;
    margin-bottom: 10px;
}
.hero-title {
    font-size: 36px;
    font-weight: 800;
    color: #EAEAEA;
    letter-spacing: -0.03em;
    line-height: 1.15;
    margin-bottom: 8px;
}
.hero-subtitle {
    font-size: 15px;
    color: #8A8FA8;
    line-height: 1.6;
    max-width: 640px;
}
.hero-stat {
    font-size: 13px;
    color: #8A8FA8;
    margin-top: 16px;
}
.hero-stat span {
    color: #F5A623;
    font-weight: 600;
}

/* ── Section headers ── */
.section-header {
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.1em;
    color: #8A8FA8;
    text-transform: uppercase;
    border-bottom: 1px solid #2A2D3E;
    padding-bottom: 8px;
    margin-bottom: 16px;
    margin-top: 8px;
}

/* ── Insight callout ── */
.insight-box {
    background: #1A1D27;
    border: 1px solid #2A2D3E;
    border-left: 3px solid #F5A623;
    border-radius: 8px;
    padding: 14px 18px;
    font-size: 13px;
    color: #8A8FA8;
    line-height: 1.6;
    margin: 12px 0;
}
.insight-box strong { color: #EAEAEA; }

/* ── Model badge ── */
.model-badge {
    display: inline-block;
    background: rgba(0,196,154,0.12);
    color: #00C49A;
    border: 1px solid rgba(0,196,154,0.3);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.06em;
    margin-bottom: 12px;
}

/* ── Sidebar filter labels ── */
.sidebar-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.1em;
    color: #8A8FA8;
    text-transform: uppercase;
    margin-bottom: 4px;
}

/* ── Plotly charts: transparent background ── */
[data-testid="stPlotlyChart"] > div {
    border-radius: 10px;
    overflow: hidden;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid #2A2D3E !important;
    border-radius: 8px;
    overflow: hidden;
}

/* ── Metrics override ── */
[data-testid="stMetric"] {
    background: #1A1D27;
    border: 1px solid #2A2D3E;
    border-radius: 10px;
    padding: 14px 18px;
}
[data-testid="stMetricLabel"] { color: #8A8FA8 !important; font-size: 11px !important; }
[data-testid="stMetricValue"] { color: #EAEAEA !important; }
[data-testid="stMetricDelta"] { color: #00C49A !important; }

/* ── Hide Streamlit branding ── */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# DATA & MODEL
# ════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    app_dir   = Path(__file__).resolve().parent
    repo_root = app_dir.parent
    csv_path  = repo_root / "00 Data" / "Processed Data" / "AmesHousing_cleaned.csv"
    return pd.read_csv(csv_path)

@st.cache_data
def build_predictions(_df):
    data = _df.copy()
    y    = np.log1p(data['SalePrice'])
    drop_cols = ['SalePrice', 'Order', 'PID']
    data = data.drop(columns=[c for c in drop_cols if c in data.columns])
    data = pd.get_dummies(data, drop_first=True)
    data = data.fillna(data.median(numeric_only=True))
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(data)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42)
    ridge = RidgeCV(alphas=[0.1, 1, 10, 50, 100, 200], cv=5)
    ridge.fit(X_train, y_train)
    y_pred   = ridge.predict(X_test)
    pred_df  = pd.DataFrame({
        'Actual_Price'    : np.expm1(y_test.values),
        'Predicted_Price' : np.expm1(y_pred),
        'Residual_Dollar' : np.expm1(y_test.values) - np.expm1(y_pred)
    })
    metrics = {
        'r2'          : round(r2_score(y_test, y_pred), 4),
        'dollar_rmse' : round(np.sqrt(mean_squared_error(
                            np.expm1(y_test.values), np.expm1(y_pred))), 0),
        'alpha'       : round(ridge.alpha_, 1)
    }
    return pred_df, metrics

# ── Plotly dark theme helper ─────────────────────────────────────
def dark_layout(fig, height=420, title=None):
    fig.update_layout(
        height=height,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,29,39,1)',
        font=dict(color='#8A8FA8', size=12),
        title=dict(text=title, font=dict(color='#EAEAEA', size=14), x=0) if title else None,
        margin=dict(l=16, r=16, t=40 if title else 16, b=16),
        xaxis=dict(gridcolor='#2A2D3E', linecolor='#2A2D3E', tickcolor='#2A2D3E'),
        yaxis=dict(gridcolor='#2A2D3E', linecolor='#2A2D3E', tickcolor='#2A2D3E'),
        legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='#2A2D3E')
    )
    return fig

# ── Load ─────────────────────────────────────────────────────────
df              = load_data()
pred, metrics   = build_predictions(df)

# ════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding:16px 0 8px;'>
      <div style='font-size:18px;font-weight:800;color:#EAEAEA;'>🏡 Ames Analytics</div>
      <div style='font-size:11px;color:#8A8FA8;letter-spacing:0.08em;'>PRICE INTELLIGENCE HUB</div>
    </div>
    <hr style='border-color:#2A2D3E;margin:8px 0 20px;'>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">Neighborhood</div>', unsafe_allow_html=True)
    all_neighborhoods = sorted(df['Neighborhood'].unique())
    selected_nbhd = st.multiselect(
        label="neighborhood_select",
        options=all_neighborhoods,
        default=all_neighborhoods,
        label_visibility="collapsed"
    )

    st.markdown('<div class="sidebar-label" style="margin-top:16px;">Sale Price Range</div>',
                unsafe_allow_html=True)
    min_p, max_p = int(df['SalePrice'].min()), int(df['SalePrice'].max())
    price_range = st.slider("price_slider", min_p, max_p, (min_p, max_p),
                             step=5000, format="$%d",
                             label_visibility="collapsed")

    st.markdown('<div class="sidebar-label" style="margin-top:16px;">Overall Quality</div>',
                unsafe_allow_html=True)
    quality_range = st.slider("quality_slider", 1, 10, (1, 10),
                               label_visibility="collapsed")

    st.markdown('<div class="sidebar-label" style="margin-top:16px;">Year Built</div>',
                unsafe_allow_html=True)
    yr_min, yr_max = int(df['Year Built'].min()), int(df['Year Built'].max())
    year_range = st.slider("year_slider", yr_min, yr_max, (yr_min, yr_max),
                            label_visibility="collapsed")

    st.markdown("""
    <hr style='border-color:#2A2D3E;margin:20px 0 12px;'>
    <div style='font-size:11px;color:#8A8FA8;line-height:1.7;'>
      <b style='color:#EAEAEA;'>Dataset</b><br>
      Ames Housing · De Cock (2011)<br>
      2,928 sales · 82 features<br><br>
      <b style='color:#EAEAEA;'>Model</b><br>
      Ridge Regression<br>
      R² 0.9486 · RMSE $18,342
    </div>
    """, unsafe_allow_html=True)

# ── Apply filters ────────────────────────────────────────────────
mask = (
    df['Neighborhood'].isin(selected_nbhd) &
    df['SalePrice'].between(price_range[0], price_range[1]) &
    df['Overall Qual'].between(quality_range[0], quality_range[1]) &
    df['Year Built'].between(year_range[0], year_range[1])
)
filtered = df[mask]
n        = len(filtered)

# ════════════════════════════════════════════════════════════════
# HERO BANNER
# ════════════════════════════════════════════════════════════════
top_nbhd = (df.groupby('Neighborhood')['SalePrice']
              .median().idxmax())

st.markdown(f"""
<div class="hero">
  <div class="hero-eyebrow">Real Estate · Ames, Iowa · 2006–2010</div>
  <div class="hero-title">Housing Price<br>Intelligence Hub</div>
  <div class="hero-subtitle">
    Machine learning analysis of 2,928 residential sales.
    Ridge regression model explains <strong style='color:#00C49A;'>94.9%</strong>
    of price variance with a mean error of
    <strong style='color:#00C49A;'>$18,342</strong>.
  </div>
  <div class="hero-stat">
    Showing <span>{n:,} houses</span> matching your filters ·
    Top neighborhood by median price: <span>{top_nbhd}</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# KPI CARDS
# ════════════════════════════════════════════════════════════════
med   = filtered['SalePrice'].median()
mean  = filtered['SalePrice'].mean()
hi    = filtered['SalePrice'].max()
lo    = filtered['SalePrice'].min()
top5n = (filtered.groupby('Neighborhood')['SalePrice']
                  .median().idxmax() if n > 0 else "—")

c1, c2, c3, c4, c5 = st.columns(5)
for col, label, value, delta in [
    (c1, "Houses in View",    f"{n:,}",          None),
    (c2, "Median Sale Price", f"${med:,.0f}",     None),
    (c3, "Mean Sale Price",   f"${mean:,.0f}",    None),
    (c4, "Price Range",       f"${lo:,.0f}",      f"↑ ${hi:,.0f}"),
    (c5, "Top Neighborhood",  top5n,              None),
]:
    with col:
        delta_html = f'<div class="kpi-delta">{delta}</div>' if delta else ""
        col.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{value}</div>
          {delta_html}
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📍  Market Overview",
    "🔍  Price Drivers",
    "🤖  Model Performance",
    "📈  Market Trends"
])

# ────────────────────────────────────────────────────────────────
# TAB 1 — MARKET OVERVIEW
# ────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">Neighborhood Price Landscape</div>',
                unsafe_allow_html=True)

    nbhd = (
        filtered.groupby('Neighborhood')['SalePrice']
        .agg(['median', 'mean', 'count', 'std'])
        .rename(columns={'median':'Median','mean':'Mean',
                         'count':'Sales','std':'Std Dev'})
        .sort_values('Median', ascending=True)
        .reset_index()
    )

    fig_nbhd = px.bar(
        nbhd, x='Median', y='Neighborhood', orientation='h',
        color='Median', color_continuous_scale=[
            [0,   '#1A1D27'],
            [0.4, '#0d6e57'],
            [1.0, '#00C49A']
        ],
        hover_data={'Median':':.0f', 'Mean':':.0f',
                    'Sales':True, 'Std Dev':':.0f'},
        custom_data=['Mean','Sales','Std Dev']
    )
    fig_nbhd.update_traces(
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Median: $%{x:,.0f}<br>"
            "Mean: $%{customdata[0]:,.0f}<br>"
            "Sales: %{customdata[1]}<br>"
            "Std Dev: $%{customdata[2]:,.0f}"
            "<extra></extra>"
        )
    )
    overall_med = filtered['SalePrice'].median()
    fig_nbhd.add_vline(
        x=overall_med, line_dash="dash", line_color="#F5A623",
        annotation_text=f"  Median ${overall_med:,.0f}",
        annotation_font_color="#F5A623",
        annotation_font_size=11
    )
    fig_nbhd = dark_layout(fig_nbhd, height=660)
    fig_nbhd.update_layout(
        coloraxis_showscale=False,
        xaxis_title="Median Sale Price ($)",
        yaxis_title=""
    )
    st.plotly_chart(fig_nbhd, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Top 5 Premium Neighborhoods</div>',
                    unsafe_allow_html=True)
        top5 = (nbhd.sort_values('Median', ascending=False)
                     .head(5)[['Neighborhood','Median','Sales']]
                     .reset_index(drop=True))
        top5['Median'] = top5['Median'].apply(lambda x: f"${x:,.0f}")
        st.dataframe(top5, hide_index=True, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Top 5 Affordable Neighborhoods</div>',
                    unsafe_allow_html=True)
        bot5 = (nbhd.sort_values('Median', ascending=True)
                     .head(5)[['Neighborhood','Median','Sales']]
                     .reset_index(drop=True))
        bot5['Median'] = bot5['Median'].apply(lambda x: f"${x:,.0f}")
        st.dataframe(bot5, hide_index=True, use_container_width=True)

    st.markdown('<div class="section-header" style="margin-top:24px;">Price Distribution</div>',
                unsafe_allow_html=True)
    fig_dist = px.histogram(
        filtered, x='SalePrice', nbins=60,
        color_discrete_sequence=['#00C49A'],
        labels={'SalePrice':'Sale Price ($)'}
    )
    fig_dist.update_traces(marker_line_width=0, opacity=0.8)
    fig_dist.add_vline(
        x=filtered['SalePrice'].median(), line_dash="dash",
        line_color="#F5A623",
        annotation_text=f"  Median ${filtered['SalePrice'].median():,.0f}",
        annotation_font_color="#F5A623"
    )
    fig_dist.add_vline(
        x=filtered['SalePrice'].mean(), line_dash="dot",
        line_color="#8A8FA8",
        annotation_text=f"  Mean ${filtered['SalePrice'].mean():,.0f}",
        annotation_font_color="#8A8FA8"
    )
    fig_dist = dark_layout(fig_dist, height=340,
                            title="Sale Price Distribution")
    st.plotly_chart(fig_dist, use_container_width=True)

# ────────────────────────────────────────────────────────────────
# TAB 2 — PRICE DRIVERS
# ────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">What moves the price needle?</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
      <strong>Key finding:</strong> Overall Quality rating and Above-Ground Living Area
      are the two strongest predictors of sale price — together they explain the
      majority of price variance across all neighborhoods.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Quality Rating vs Median Price</div>',
                    unsafe_allow_html=True)
        qual = (filtered.groupby('Overall Qual')['SalePrice']
                        .agg(['median','count'])
                        .rename(columns={'median':'Median','count':'Sales'})
                        .reset_index())
        fig_q = px.bar(
            qual, x='Overall Qual', y='Median',
            color='Median',
            color_continuous_scale=[[0,'#1A1D27'],[1,'#00C49A']],
            text=qual['Median'].apply(lambda x: f"${x/1000:.0f}K"),
            labels={'Overall Qual':'Quality (1–10)',
                    'Median':'Median Price ($)'}
        )
        fig_q.update_traces(textposition='outside',
                             textfont_color='#8A8FA8')
        fig_q = dark_layout(fig_q, height=380,
                             title="Overall Quality vs Median Sale Price")
        fig_q.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_q, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Living Area vs Sale Price</div>',
                    unsafe_allow_html=True)
        fig_scat = px.scatter(
            filtered, x='Gr Liv Area', y='SalePrice',
            color='Overall Qual',
            color_continuous_scale='Tealgrn',
            opacity=0.55,
            trendline='ols',
            labels={'Gr Liv Area':'Living Area (sq ft)',
                    'SalePrice':'Sale Price ($)',
                    'Overall Qual':'Quality'}
        )
        fig_scat = dark_layout(fig_scat, height=380,
                                title="Living Area vs Sale Price")
        st.plotly_chart(fig_scat, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="section-header">Price by House Style</div>',
                    unsafe_allow_html=True)
        fig_box = px.box(
            filtered, x='House Style', y='SalePrice',
            color='House Style',
            color_discrete_sequence=['#00C49A','#F5A623','#534AB7',
                                     '#D85A30','#4A90D9','#9B59B6',
                                     '#2ECC71','#E74C3C'],
            labels={'SalePrice':'Sale Price ($)','House Style':'Style'}
        )
        fig_box.update_layout(showlegend=False)
        fig_box = dark_layout(fig_box, height=380,
                               title="Sale Price by House Style")
        st.plotly_chart(fig_box, use_container_width=True)

    with col4:
        st.markdown('<div class="section-header">Bedrooms vs Median Price</div>',
                    unsafe_allow_html=True)
        bed = (filtered.groupby('Bedroom AbvGr')['SalePrice']
                       .agg(['median','count'])
                       .rename(columns={'median':'Median','count':'n'})
                       .reset_index())
        fig_bed = px.bar(
            bed, x='Bedroom AbvGr', y='Median',
            text=bed['n'].apply(lambda x: f"n={x}"),
            color='Median',
            color_continuous_scale=[[0,'#1A1D27'],[1,'#F5A623']],
            labels={'Bedroom AbvGr':'Bedrooms Above Ground',
                    'Median':'Median Price ($)'}
        )
        fig_bed.update_traces(textposition='outside',
                               textfont_color='#8A8FA8')
        fig_bed = dark_layout(fig_bed, height=380,
                               title="Median Price by Bedroom Count")
        fig_bed.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_bed, use_container_width=True)

    # Correlation heatmap
    st.markdown('<div class="section-header" style="margin-top:8px;">Top Feature Correlations with SalePrice</div>',
                unsafe_allow_html=True)
    num_cols = filtered.select_dtypes(include='number').columns
    corr_series = (filtered[num_cols].corr()['SalePrice']
                                     .drop('SalePrice')
                                     .abs()
                                     .sort_values(ascending=False)
                                     .head(12))
    fig_corr = px.bar(
        x=corr_series.values,
        y=corr_series.index,
        orientation='h',
        color=corr_series.values,
        color_continuous_scale=[[0,'#1A1D27'],[1,'#00C49A']],
        labels={'x':'Correlation with SalePrice','y':'Feature'}
    )
    fig_corr = dark_layout(fig_corr, height=380,
                            title="Feature Correlation with Sale Price")
    fig_corr.update_layout(
        coloraxis_showscale=False,
        yaxis={'categoryorder':'total ascending'}
    )
    st.plotly_chart(fig_corr, use_container_width=True)

# ────────────────────────────────────────────────────────────────
# TAB 3 — MODEL PERFORMANCE
# ────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("""
    <div class="model-badge">⭐ RECOMMENDED MODEL</div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="section-header">Ridge Regression — Performance Dashboard</div>',
                unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        with st.container(border=True):
            st.metric("Model", "Ridge Regression")
    with m2:
        with st.container(border=True):
            st.metric("Best Alpha", str(metrics['alpha']))
    with m3:
        with st.container(border=True):
            st.metric("Test R²", str(metrics['r2']),
                      delta="94.9% variance explained")
    with m4:
        with st.container(border=True):
            st.metric("Dollar RMSE", f"${metrics['dollar_rmse']:,.0f}",
                      delta="avg prediction error")

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Actual vs Predicted Price</div>',
                    unsafe_allow_html=True)
        min_v = float(min(pred['Actual_Price'].min(),
                          pred['Predicted_Price'].min()))
        max_v = float(max(pred['Actual_Price'].max(),
                          pred['Predicted_Price'].max()))
        fig_ap = px.scatter(
            pred, x='Actual_Price', y='Predicted_Price',
            color='Residual_Dollar',
            color_continuous_scale='RdYlGn',
            opacity=0.65,
            labels={'Actual_Price':'Actual Price ($)',
                    'Predicted_Price':'Predicted Price ($)',
                    'Residual_Dollar':'Residual ($)'}
        )
        fig_ap.add_trace(go.Scatter(
            x=[min_v, max_v], y=[min_v, max_v],
            mode='lines',
            line=dict(color='#F5A623', dash='dash', width=1.5),
            name='Perfect prediction',
            showlegend=True
        ))
        fig_ap = dark_layout(fig_ap, height=420,
                              title="Actual vs Predicted Sale Price")
        st.plotly_chart(fig_ap, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Prediction Error Distribution</div>',
                    unsafe_allow_html=True)
        fig_res = px.histogram(
            pred, x='Residual_Dollar', nbins=50,
            color_discrete_sequence=['#00C49A'],
            labels={'Residual_Dollar':'Residual: Actual − Predicted ($)'}
        )
        fig_res.update_traces(opacity=0.8)
        fig_res.add_vline(
            x=0, line_dash="dash", line_color="#F5A623",
            annotation_text="  Zero error",
            annotation_font_color="#F5A623"
        )
        within_20k = (pred['Residual_Dollar'].abs() <= 20000).mean() * 100
        fig_res.add_annotation(
            x=0, y=1, xref='paper', yref='paper',
            text=f"{within_20k:.0f}% of predictions within ±$20K",
            showarrow=False,
            font=dict(color='#8A8FA8', size=11),
            xanchor='left', yanchor='top'
        )
        fig_res = dark_layout(fig_res, height=420,
                               title="Prediction Error Distribution")
        st.plotly_chart(fig_res, use_container_width=True)

    st.markdown('<div class="section-header" style="margin-top:8px;">All Models Compared</div>',
                unsafe_allow_html=True)

    comparison = pd.DataFrame({
        'Model'       : ['Linear Regression','Ridge Regression',
                         'Lasso Regression','Gradient Boosting'],
        'Test R²'     : [0.9400, 0.9486, 0.9500, 0.9452],
        'CV RMSE'     : [0.1311, 0.1217, 0.1235, 0.1249],
        'Dollar RMSE' : ['$19,100','$18,342','$18,594','$20,721'],
        'Verdict'     : ['Baseline','⭐ Recommended',
                         'Best R² · Lean','Overfit on dataset']
    })
    st.dataframe(comparison, hide_index=True, use_container_width=True)

    st.markdown("""
    <div class="insight-box" style="margin-top:12px;">
      <strong>Why Ridge over Gradient Boosting?</strong>
      Gradient Boosting achieved a train RMSE of 0.0610 vs a CV RMSE of 0.1249 —
      a gap of 0.064 that signals overfitting on this 2,928-row dataset.
      Ridge's train/test gap is only 0.008, meaning it generalises reliably
      to houses it has never seen.
    </div>
    """, unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────
# TAB 4 — MARKET TRENDS
# ────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">Market Dynamics 2006–2010</div>',
                unsafe_allow_html=True)

    yr = (df.groupby('Yr Sold')['SalePrice']
            .agg(['median','count'])
            .rename(columns={'median':'Median','count':'Sales'})
            .reset_index())

    fig_trend = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=('Median Sale Price by Year',
                        'Number of Sales by Year'),
        vertical_spacing=0.14,
        row_heights=[0.6, 0.4]
    )
    fig_trend.add_trace(
        go.Scatter(
            x=yr['Yr Sold'], y=yr['Median'],
            mode='lines+markers',
            line=dict(color='#00C49A', width=2.5),
            marker=dict(size=9, color='#00C49A',
                        line=dict(width=2, color='#0F1117')),
            fill='tozeroy', fillcolor='rgba(0,196,154,0.08)',
            name='Median Price'
        ), row=1, col=1
    )
    fig_trend.add_trace(
        go.Bar(
            x=yr['Yr Sold'], y=yr['Sales'],
            marker_color='#F5A623',
            marker_opacity=0.75,
            name='Sales Count'
        ), row=2, col=1
    )
    fig_trend.update_layout(
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,29,39,1)',
        font=dict(color='#8A8FA8'),
        showlegend=False,
        margin=dict(l=16, r=16, t=40, b=16)
    )
    for axis in ['xaxis','xaxis2','yaxis','yaxis2']:
        fig_trend.update_layout(**{
            axis: dict(gridcolor='#2A2D3E', linecolor='#2A2D3E')
        })
    fig_trend.update_yaxes(title_text="Median Price ($)", row=1)
    fig_trend.update_yaxes(title_text="Sales Count", row=2)
    st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
      <strong>2008–2009 Financial Crisis:</strong> Both median price and sales
      volume declined during the market downturn, with volume falling more
      sharply than price. Recovery was visible by 2010.
      This context anchors any valuation conversation with clients operating
      in cyclical real estate markets.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Price by Month Sold</div>',
                    unsafe_allow_html=True)
        mo = (df.groupby('Mo Sold')['SalePrice']
                 .median().reset_index()
                 .rename(columns={'SalePrice':'Median'}))
        mo['Month'] = ['Jan','Feb','Mar','Apr','May','Jun',
                        'Jul','Aug','Sep','Oct','Nov','Dec'][
                            :len(mo)]
        fig_mo = px.line(
            mo, x='Month', y='Median',
            markers=True,
            labels={'Median':'Median Sale Price ($)'},
            color_discrete_sequence=['#00C49A']
        )
        fig_mo.update_traces(
            line_width=2.5,
            marker=dict(size=8, color='#00C49A',
                        line=dict(width=2, color='#0F1117'))
        )
        fig_mo = dark_layout(fig_mo, height=320,
                              title="Seasonality — Median Price by Month")
        st.plotly_chart(fig_mo, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Year Built Distribution</div>',
                    unsafe_allow_html=True)
        fig_yr = px.histogram(
            filtered, x='Year Built', nbins=40,
            color_discrete_sequence=['#F5A623'],
            labels={'Year Built':'Year Built'}
        )
        fig_yr.update_traces(opacity=0.8)
        fig_yr = dark_layout(fig_yr, height=320,
                              title="Housing Stock Age Distribution")
        st.plotly_chart(fig_yr, use_container_width=True)

# ── Footer ───────────────────────────────────────────────────────
st.markdown("""
<div style='
    text-align:center;
    padding: 32px 0 16px;
    font-size: 12px;
    color: #2A2D3E;
    letter-spacing: 0.08em;
    border-top: 1px solid #2A2D3E;
    margin-top: 24px;
'>
  AMES HOUSING ANALYTICS HUB &nbsp;·&nbsp;
  PYTHON · STREAMLIT · PLOTLY · SCIKIT-LEARN &nbsp;·&nbsp;
  DATA: DE COCK (2011)
</div>
""", unsafe_allow_html=True)
