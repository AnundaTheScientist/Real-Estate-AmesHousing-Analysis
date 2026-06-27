# Ames Housing Market — Price Intelligence Report

**Prepared by:** Anunda The Scientist  
**Dataset:** Ames, Iowa Residential Sales 2006–2010 (2,928 transactions)  
**Model:** Ridge Regression · R² = 0.9486 · Dollar RMSE = $18,342

---

## Executive Summary

This analysis examined 2,928 residential property sales in Ames, Iowa
to identify the key drivers of sale price and build a machine learning
model capable of predicting property values with an average error of
**$18,342**.

Five findings stand above the rest:

1. **Overall build quality is the #1 price driver** — a quality score
   of 8 commands nearly double the price of a score of 5
2. **Neighborhood creates a 3× price spread** — from $80K to $315K
   median across the city's 28 neighborhoods
3. **The mid-market ($100K–$250K) dominates by volume** — where
   liquidity is highest and days-on-market lowest
4. **Size amplifies quality** — the highest prices belong to large
   houses with high quality scores; neither alone is sufficient
5. **The market dipped in 2008 and recovered by 2010** — in line
   with national trends from the financial crisis

---

## Key Charts

![Market Overview](C:\Users\A.K\DATA_SCIENCE_WORKSPACE\kaggle_projects\project_4_amesHousing\Real Estate-AmesHousing-Analysis\07 Insight - Video\Report\insight_market_overview.png)
![Quality vs Price](C:\Users\A.K\DATA_SCIENCE_WORKSPACE\kaggle_projects\project_4_amesHousing\Real Estate-AmesHousing-Analysis\07 Insight - Video\Report\insight_quality_vs_price.png)
![Neighborhood Ranking](C:\Users\A.K\DATA_SCIENCE_WORKSPACE\kaggle_projects\project_4_amesHousing\Real Estate-AmesHousing-Analysis\07 Insight - Video\Report\insight_neighborhood_ranking.png)
![Size and Quality](C:\Users\A.K\DATA_SCIENCE_WORKSPACE\kaggle_projects\project_4_amesHousing\Real Estate-AmesHousing-Analysis\07 Insight - Video\Report\insight_size_quality.png)
![Model Performance](C:\Users\A.K\DATA_SCIENCE_WORKSPACE\kaggle_projects\project_4_amesHousing\Real Estate-AmesHousing-Analysis\07 Insight - Video\Report\insight_model_performance.png)

---

## Strategic Recommendations

| Priority | Action | Evidence |
|---|---|---|
| 1 | Focus acquisitions on NridgHt, NoRidge, StoneBr | 3 highest median neighborhoods |
| 2 | Target quality score 7–9 for renovations | Steepest price increase per quality point |
| 3 | Time listings to spring/summer | April–July median prices highest |
| 4 | Use model predictions for pre-listing pricing checks | ±$18,342 average accuracy |
| 5 | Avoid low quality stock in bottom-tier neighborhoods | Double risk from quality + location |

---

## Model Performance

Our recommended model — Ridge Regression — was selected from four candidates
based on cross-validated reliability rather than raw test performance.

| Model | Test R² | Dollar RMSE |
|---|---|---|
| Linear Regression | 0.9400 | $19,100 |
| **Ridge Regression** | **0.9486** | **$18,342** |
| Lasso Regression | 0.9500 | $18,594 |
| Gradient Boosting | 0.9452 | $20,721 |

Gradient Boosting was excluded as recommended model despite competitive
test scores because it showed signs of overfitting — a gap of 0.064 between
its training and cross-validated RMSE versus only 0.008 for Ridge.

---

*Full methodology, code, and interactive dashboard available at:*  
*[github.com/AnundaTheScientist/Real-Estate-AmesHousing-Analysis](https://github.com/AnundaTheScientist/Real-Estate-AmesHousing-Analysis)*
