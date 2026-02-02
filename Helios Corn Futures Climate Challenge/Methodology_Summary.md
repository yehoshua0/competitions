# Methodology Summary

**Helios Corn Futures Climate Challenge**

## Submission Reference

- **Kaggle Username:** aaaml007
- **Team Name:** yehoshua
- **Submission Name:** Helios Corn Futures Climate Challenge | Version 2
- **Public Score:** 63.64000
- **Private Score:** 53.66000

## 1. Data Sources

We strictly used the official competition dataset provided by Helios.

- `corn_climate_risk_futures_daily_master.csv`: Primary dataset containing climate risk counts and futures prices.
- `corn_regional_market_share.csv`: Used to weight climate risks by regional production importance.

**External Data/Tools:**

- No external datasets were used.
- Standard Python data science stack (`pandas`, `numpy`, `scikit-learn`) was used for processing and modeling.

## 2. Key Feature Engineering Steps & Rationale

Our approach focused on creating economically meaningful signals from the raw climate risk counts. The pipeline, implemented in `notebooks/helios-corn-futures-climate-challenge.ipynb`, generates the following feature categories:

1.  **Weighted Risk Scores:**
    - Raw risk counts (Low/Medium/High) were aggregated into a single `risk_score` for each category (`heat_stress`, `drought`, etc.).
    - These scores were weighted by `percent_country_production` to prioritize risks in major production regions.

2.  **Temporal Aggregation (Smoothing & Trends):**
    - **Moving Averages (7, 14, 30, 60, 90 days):** To capture sustained climate trends rather than daily noise.
    - **Rolling Max:** To detect peak stress events in the recent past.
    - **Exponential Moving Averages (EMA):** Applied to give more weight to recent weather events.

3.  **Lag Features (7 to 90 days):**
    - Created shifted versions of risk scores to model the delayed impact of weather on crop yields and market sentiment.

4.  **Volatility Measures:**
    - Rolling standard deviation of risk scores (14-46 days) to proxy for "climate instability" and market uncertainty.

5.  **Cumulative Stress:**
    - Rolling sums (30-90 days) to model the accumulation of adverse conditions (e.g., prolonged drought).

**Rationale:** This diverse set of features captures both acute shocks (max/volatility) and chronic stressors (cumulative/moving averages), allowing the model to detect the complex, non-linear relationships between climate anomalies and futures pricing.

## 3. Anti-Gaming Compliance

I confirm that **NO** `futures_*` columns (price, volume, open interest) or their derivatives were used to generate any `climate_risk_*` features.

- All `climate_risk` features are derived solely from weather/risk columns and static metadata (market share).
- The target variable (futures price changes) was strictly isolated from the feature engineering process.
