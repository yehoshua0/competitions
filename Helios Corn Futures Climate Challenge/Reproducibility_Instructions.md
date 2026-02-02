# Reproducibility Instructions

## Overview

This solution represents the winning submission for the Helios Corn Futures Climate Challenge by Team **aaaml007 / yehoshua**.

**Submission Version:** Version 2
**Score:** 63.64000 (Public) / 53.66000 (Private)

## Kaggle Notebook Verification

The entire solution was developed and executed within the Kaggle Notebooks environment. The most reliable way to reproduce the results is to run the provided notebook directly on Kaggle.

### 1. Setup on Kaggle

1.  **Create a Notebook:**
    - Go to the [Helios Corn Futures Climate Challenge](https://www.kaggle.com/competitions/forecasting-the-future-the-helios-corn-climate-challenge/code) competition page.
    - Click **"New Notebook"**.

2.  **Import Code:**
    - In the notebook editor, go to **File > Import Notebook**.
    - Upload the `notebooks/helios-corn-futures-climate-challenge.ipynb` file included in this package.

3.  **Data Configuration:**
    - Ensure the competition dataset is added. It should be available at:
      `/kaggle/input/forecasting-the-future-the-helios-corn-climate-challenge/`
    - No external datasets are required.

### 2. Environment Settings

- **Docker Image:** Latest available Kaggle Python image.
- **Accelerator:** CPU (Standard) is sufficient.
- **Internet:** Disabled (Not required).

### 3. Execution Steps

1.  **Run All Cells:**
    - Click **Run > Run All** in the notebook toolbar.
    - The feature engineering pipeline is efficient and should complete in under 5 minutes.

2.  **Output:**
    - The notebook will generate a `submission.csv` file in the `/kaggle/working/` directory.

### 4. Verification

- Compare the generated `submission.csv` with the `BestSubmission.csv` provided in this package.
- They should be identical (or match within floating-point tolerance).

## Feature Engineering Pipeline

The notebook implements the following pipeline:

1.  **Risk Scoring:** Aggregates `climate_risk_*` counts into weighted scores based on `corn_regional_market_share.csv`.
2.  **Temporal Features:** Computes Rolling Means, Max, and EMAs (7, 14, 30, 60, 90 days).
3.  **Lag Features:** Generates shifted features to capture delayed market reactions.
4.  **Volatility:** Calculates rolling standard deviations.
5.  **CFCS Optimization:** Selects features based on the Climate-Futures Correlation Score.
