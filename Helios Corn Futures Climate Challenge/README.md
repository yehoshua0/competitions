# Helios Corn Futures Climate Challenge

> Turn weather wisdom into trading gold! Use Helios AI's climate data to decode the weather signals behind corn futures and outsmart the markets.

## 📌 Overview

The **Helios Corn Futures Climate Challenge** tasks participants with leveraging proprietary climate risk data to predict movements in corn futures prices. The goal is to develop innovative methods that enhance the correlation between weather patterns and commodity markets, transforming raw climate risk signals into actionable market intelligence.

---

## 📊 Evaluation Metric: CFCS

Ranking is determined by the **Climate-Futures Correlation Score (CFCS)**, a composite metric that rewards both the strength and breadth of climate-market relationships:

$$CFCS = (0.5 \times Avg\_Sig\_Corr\_Score) + (0.3 \times Max\_Corr\_Score) + (0.2 \times Sig\_Count\_Score)$$

*   **Avg_Sig_Corr_Score (50%):** Average of significant correlations (absolute value $\ge 0.5$).
*   **Max_Corr_Score (30%):** The maximum absolute correlation found.
*   **Sig_Count_Score (20%):** Percentage of correlations with an absolute value $\ge 0.5$.

---

## 📂 Dataset Details

The competition provides two categories of comprehensive data:

### 1. Climate Risk Data (Proprietary)
*   **Risk Classifications:** Daily assessments (Low, Medium, High) based on crop-specific thresholds.
*   **Risk Categories:** Heat Stress, Cold Stress, Drought, and Excess Precipitation.
*   **Geographies:** Major global corn-growing regions with regional aggregation and production share context.

### 2. Futures Market Data
*   **Commodities:** Continuous pricing for Corn (ZC), Wheat (ZW), and Soybeans (ZS).
*   **Technical Indicators:** Daily returns, volatility, moving averages, etc.
*   **Market Structure:** Term spreads and cross-commodity relationships.

---

## 🛠️ Repository Structure

```text
Helios Corn Futures Climate Challenge/
├── data/           # Climate and Futures datasets
├── notebooks/      # Correlation analysis & modeling
├── src/            # Signal generation and correlation logic
├── scripts/        # Data processing scripts
├── submissions/    # Final CFCS scores and submissions
└── README.md       # Competition report
```

---

## 🚀 Getting Started

1.  **Exploration:** Start with the `notebooks/` directory to analyze core correlations.
2.  **Signal Generation:** Implement custom aggregations in `src/` to boost the CFCS components.
3.  **Validation:** Use the CFCS formula to validate local results before submission.

---
*Reference: [Kaggle Competition Page](https://www.kaggle.com/competitions/forecasting-the-future-the-helios-corn-climate-challenge)*
