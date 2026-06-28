<div align="center">

# Ames Housing — Real Estate Price Intelligence

### End-to-end machine learning project predicting residential sale prices  
### across 28 neighborhoods in Ames, Iowa (2006–2010)

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live_Dashboard-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://real-estate-ameshousing-analysis.streamlit.app)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Ridge_Regression-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-7_Notebooks-F37626?style=flat&logo=jupyter&logoColor=white)](https://jupyter.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=flat)](LICENSE)

---

**[ Launch Interactive Dashboard](https://real-estate-ameshousing-analysis.streamlit.app)** &nbsp;·&nbsp;
**[ View Notebooks](#-project-structure)** &nbsp;·&nbsp;
**[ Key Findings](#-key-findings)**

</div>

---

## Project Overview

This project applies the full **CRISP-DM data science methodology** to 2,928 residential property sales in Ames, Iowa — answering one business question:

> *"What makes a house sell for more — and how accurately can we predict any given property's sale price?"*

The result is a production-ready machine learning pipeline with an interactive Streamlit dashboard that real estate clients can use without reading a line of code.

| | |
|---|---|
| **Dataset** | Ames Housing · De Cock (2011) · [Kaggle](https://www.kaggle.com/datasets/prevek18/ames-housing-dataset) |
| **Records** | 2,928 residential sales |
| **Features** | 82 variables → 238 engineered features |
| **Target** | Sale Price (log-transformed for modeling) |
| **Best model** | Ridge Regression · R² = 0.9486 · Dollar RMSE = $18,342 |
| **Dashboard** | Live on Streamlit Community Cloud |

---

## Interactive Dashboard

> Click the image below to launch the live dashboard

[![Dashboard Preview](06%20Dashboard/dashboard_preview.png)](https://real-estate-ameshousing-analysis.streamlit.app)

**Features:**
-  Filter by neighborhood, price range, quality, and year built
-  28-neighborhood price landscape with hover details  
-  Live Ridge Regression model predictions
-  Market trend analysis 2006–2010 including 2008 crisis dip
-  Built with Python · Streamlit · Plotly · scikit-learn

---

## Key Findings

### 1 — Overall Quality is the #1 Price Driver
A house rated 8/10 sells for nearly **twice** the price of one rated 5/10.  
Quality upgrades deliver the highest ROI of any single improvement.

### 2 — Neighborhood Creates a 3× Price Spread
| Tier | Neighborhoods | Median Price |
|---|---|---|
| Premium | NridgHt, NoRidge, StoneBr | > $300,000 |
| Mid-market | CollgCr, Somerst, NWAmes | $150,000–$220,000 |
| Affordable | MeadowV, IDOTRR, BrDale | < $100,000 |

### 3 — Size Amplifies Quality, Not the Other Way Around
The highest sale prices belong to houses that are *both* large *and* high quality.  
Maximising square footage alone without quality improvements yields diminishing returns.

### 4 — Spring and Summer Command Higher Prices
April–July consistently shows higher median prices than autumn and winter —  
timing listings to the spring market adds measurable value.

### 5 — Our Model Predicts Within ±$18,342
For a house listed at $200,000, predictions typically fall within **$182K–$218K** —  
sufficient for pre-listing valuation, acquisition screening, and portfolio estimates.

---

## Model Results

| Model | Train RMSE | Test RMSE | CV RMSE | Dollar RMSE | Verdict |
|---|---|---|---|---|---|
| Linear Regression | 0.1022 | 0.1038 | 0.1311 | $19,100 | Baseline |
| **Ridge Regression** | **0.1045** | **0.0961** | **0.1217** | **$18,342** | ⭐ **Recommended** |
| Lasso Regression | 0.1050 | 0.0948 | 0.1235 | $18,594 | Best R² · lean model |
| Gradient Boosting | 0.0610 | 0.0992 | 0.1249 | $20,721 | Overfit on this dataset |

**Why Ridge over Gradient Boosting?**  
Gradient Boosting's train/CV gap of 0.064 signals overfitting on this 2,928-row dataset.  
Ridge's gap of 0.008 means reliable generalisation to houses it has never seen.

---

## Project Structure

```
Real-Estate-AmesHousing-Analysis/
│
├── 00 Data/
│   ├── AmesHousing.csv                  ← Raw dataset (Kaggle)
│   └── Processed Data/
│       ├── AmesHousing_cleaned.csv      ← Phase 3 output
│       ├── X_features.csv               ← Phase 4 output — feature matrix
│       ├── y_target.csv                 ← Phase 4 output — log(SalePrice)
│       └── scaler.pkl                   ← Fitted StandardScaler
│
├── 01 EDA/
│   └── 01_eda_ameshousing.ipynb         ← Data understanding & visualisation
│
├── 02 Data Cleaning/
│   └── 02_data_cleaning.ipynb           ← Missing values, outliers, type fixes
│
├── 03 Feature Engineering/
│   └── 03_feature_engineering.ipynb     ← Encoding, new features, scaling
│
├── 04 Regression/
│   ├── 04_regression_modeling.ipynb     ← Four models trained & evaluated
│   ├── model_linear.pkl
│   ├── model_ridge.pkl
│   ├── model_lasso.pkl
│   └── model_gb.pkl
│
├── 05 Model Comparison/
│   └── 05_model_comparison.ipynb        ← Formal evaluation & chart outputs
│
├── 06 Dashboard/
│   └── app.py                           ← Streamlit dashboard (live)
│
├── 07 Insight - Video/
│   ├── 07_insights.ipynb                ← Stakeholder findings notebook
│   └── insights_report.md               ← Client-facing report (no code)
│
├── requirements.txt
└── README.md
```

---

## Methodology — CRISP-DM Pipeline

```
Business Understanding → Data Understanding → Data Preparation
        → Modeling → Evaluation → Deployment
```

| Phase | Notebook | Key Output |
|---|---|---|
| 1 · Setup | — | Folder structure · Git · venv |
| 2 · EDA | `01_eda_ameshousing.ipynb` | 8 insight charts · skewness identified |
| 3 · Cleaning | `02_data_cleaning.ipynb` | 0 nulls · 2 outliers removed |
| 4 · Features | `03_feature_engineering.ipynb` | 238 features · log target · scaled |
| 5 · Modeling | `04_regression_modeling.ipynb` | 4 models · Dollar RMSE per model |
| 6 · Evaluation | `05_model_comparison.ipynb` | Ridge selected · comparison charts |
| 7 · Dashboard | `06 Dashboard/app.py` | Live Streamlit app |
| 8 · Insights | `07_insights.ipynb` | 5 findings · client report |

---

## Feature Engineering Highlights

| Feature | Source columns | Rationale |
|---|---|---|
| `House Age` | `Yr Sold` − `Year Built` | Captures depreciation |
| `Remod Age` | `Yr Sold` − `Year Remod/Add` | Recency of improvements |
| `Total SF` | `Gr Liv Area` + basement SF | True total usable space |
| `Total Baths` | All bath columns × 0.5/1.0 | Single bathroom signal |
| `Has Pool` / `Has Garage` / `Has Fireplace` | Area/count columns | Binary amenity flags |
| Ordinal encoding | 10 quality columns | Preserves Poor→Excellent order |
| One-hot encoding | 43 categorical columns | 195 binary features |
| Log transform | `SalePrice` | Reduces skew 1.76 → 0.12 |

---

## Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/AnundaTheScientist/Real-Estate-AmesHousing-Analysis.git
cd Real-Estate-AmesHousing-Analysis

# 2. Create and activate virtual environment
python -m venv venv
source venv/Scripts/activate    # Windows Git Bash
# source venv/bin/activate      # Mac / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the dashboard
cd "06 Dashboard"
streamlit run app.py

# 5. Or run notebooks in order
# Open 01 EDA → 02 Data Cleaning → 03 Feature Engineering
# → 04 Regression → 05 Model Comparison → 07 Insight - Video
```

---

## Requirements

```
pandas
numpy
streamlit
plotly
scikit-learn
matplotlib
seaborn
scipy
joblib
```

---

## Data Source

**Ames Housing Dataset** — Dean De Cock (2011)  
*"Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project"*  
Journal of Statistics Education · Volume 19, Number 3  
Retrieved from [Kaggle](https://www.kaggle.com/datasets/prevek18/ames-housing-dataset)

---

## 👤 Author

**Anunda The Scientist** — Data Scientist  
*Specialising in real estate analytics and predictive modeling*

[![GitHub](https://img.shields.io/badge/GitHub-AnundaTheScientist-181717?style=flat&logo=github)](https://github.com/AnundaTheScientist)

---

<div align="center">

*Built with Python · scikit-learn · Streamlit · Plotly*  
*CRISP-DM methodology · Ames Housing Dataset · De Cock (2011)*

</div>
