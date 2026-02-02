# -*- coding: utf-8 -*-
"""
Helios Corn Futures Climate Challenge
Team: aaaml007 / yehoshua
"""

import os
import sys
import platform
import warnings
import pandas as pd
import numpy as np
from scipy import stats

# Filter warnings
warnings.filterwarnings('ignore')

# Configuration
RISK_CATEGORIES = ['heat_stress', 'unseasonably_cold', 'excess_precip', 'drought']
SIGNIFICANCE_THRESHOLD = 0.5
REQUIRED_ROWS = 219161

# Default paths (can be overridden via environment variables or args)
# If running on Kaggle, these paths are standard.
# If running locally, you might need to adjust them.
DATA_PATH = os.environ.get('HELIOS_DATA_PATH', '/kaggle/input/forecasting-the-future-the-helios-corn-climate-challenge/')
OUTPUT_PATH = os.environ.get('HELIOS_OUTPUT_PATH', 'submissions/')

def setup_environment():
    """Sets up the environment, ensuring directories exist."""
    print(f"Python version: {platform.python_version()}")
    pd.options.display.max_columns = 100
    pd.options.display.max_rows = 100
    
    if not os.path.exists(OUTPUT_PATH):
        try:
            os.makedirs(OUTPUT_PATH)
        except OSError:
            pass # Might be /kaggle/working/ which exists
            
    print("✅ Libraries loaded & Environment configured")

def load_data():
    """Loads the competition data."""
    print(f"Loading data from: {DATA_PATH}")
    try:
        df = pd.read_csv(os.path.join(DATA_PATH, 'corn_climate_risk_futures_daily_master.csv'))
        df['date_on'] = pd.to_datetime(df['date_on'])
        market_share_df = pd.read_csv(os.path.join(DATA_PATH, 'corn_regional_market_share.csv'))
        
        print(f"📊 Dataset: {len(df):,} rows")
        print(f"📅 Date range: {df['date_on'].min()} to {df['date_on'].max()}")
        print(f"🌍 Countries: {df['country_name'].nunique()}")
        print(f"📍 Regions: {df['region_name'].nunique()}")
        
        return df, market_share_df
    except FileNotFoundError as e:
        print(f"❌ Error loading data: {e}")
        print("Please ensure DATA_PATH is correct.")
        sys.exit(1)

def compute_cfcs(df, verbose=True):
    """
    Compute CFCS score for a dataframe.
    CFCS = (0.5 × Avg_Sig_Corr) + (0.3 × Max_Corr) + (0.2 × Sig_Count%)
    """
    climate_cols = [c for c in df.columns if c.startswith("climate_risk_")]
    futures_cols = [c for c in df.columns if c.startswith("futures_")]
    
    correlations = []
    
    # Pre-compute to avoid repeated filtering
    # Optimization: processing country by country
    for country in df['country_name'].unique():
        df_country = df[df['country_name'] == country]
        
        for month in df_country['date_on_month'].unique():
            df_month = df_country[df_country['date_on_month'] == month]
            
            # Vectorized correlation check if possible, but keeping original logic for reproducibility
            if len(df_month) < 2: continue

            # Filter columns with variance (std > 0)
            valid_clim = [c for c in climate_cols if df_month[c].std() > 0]
            valid_fut = [c for c in futures_cols if df_month[c].std() > 0]
            
            if not valid_clim or not valid_fut: continue

            # Bulk correlation
            # We want correlation between each clim and each fut.
            # Only taking corr[0,1] implies comparing specific pairs? 
            # Original code: for clim... for fut... corr().iloc[0,1]
            for clim in valid_clim:
                for fut in valid_fut:
                    corr = df_month[[clim, fut]].corr().iloc[0, 1]
                    correlations.append(corr)
    
    correlations = pd.Series(correlations).dropna()
    abs_corrs = correlations.abs()
    sig_corrs = abs_corrs[abs_corrs >= SIGNIFICANCE_THRESHOLD]
    
    avg_sig = sig_corrs.mean() if len(sig_corrs) > 0 else 0
    max_corr = abs_corrs.max() if len(abs_corrs) > 0 else 0
    sig_pct = len(sig_corrs) / len(correlations) * 100 if len(correlations) > 0 else 0
    
    avg_sig_score = min(100, avg_sig * 100)
    max_score = min(100, max_corr * 100)
    
    cfcs = (0.5 * avg_sig_score) + (0.3 * max_score) + (0.2 * sig_pct)
    
    result = {
        'cfcs': round(cfcs, 2),
        'avg_sig_corr': round(avg_sig, 4),
        'max_corr': round(max_corr, 4),
        'sig_count': len(sig_corrs),
        'total': len(correlations),
        'sig_pct': round(sig_pct, 4),
        'n_features': len(climate_cols)
    }
    
    if verbose:
        print(f"CFCS: {result['cfcs']} | Sig: {result['sig_count']}/{result['total']} ({result['sig_pct']:.2f}%) | Features: {result['n_features']}")
    
    return result

def analyze_feature_contributions(df, climate_cols, futures_cols):
    """
    Analyze contribution of each climate feature.
    Returns DataFrame with sig_count, max_corr, etc for each feature.
    """
    feature_stats = {col: {'sig_count': 0, 'total': 0, 'max_corr': 0, 'sig_corrs': []} 
                     for col in climate_cols}
    
    for country in df['country_name'].unique():
        df_country = df[df['country_name'] == country]
        
        for month in df_country['date_on_month'].unique():
            df_month = df_country[df_country['date_on_month'] == month]
            
            if len(df_month) < 2: continue

            valid_clim = [c for c in climate_cols if df_month[c].std() > 0]
            valid_fut = [c for c in futures_cols if df_month[c].std() > 0]

            for clim in valid_clim:
                for fut in valid_fut:
                    corr = df_month[[clim, fut]].corr().iloc[0, 1]
                    
                    feature_stats[clim]['total'] += 1
                    
                    if abs(corr) >= SIGNIFICANCE_THRESHOLD:
                        feature_stats[clim]['sig_count'] += 1
                        feature_stats[clim]['sig_corrs'].append(abs(corr))
                    
                    if abs(corr) > feature_stats[clim]['max_corr']:
                        feature_stats[clim]['max_corr'] = abs(corr)
    
    results = []
    for col, stats in feature_stats.items():
        avg_sig = np.mean(stats['sig_corrs']) if stats['sig_corrs'] else 0
        results.append({
            'feature': col,
            'sig_count': stats['sig_count'],
            'total': stats['total'],
            'sig_pct': stats['sig_count'] / stats['total'] * 100 if stats['total'] > 0 else 0,
            'max_corr': round(stats['max_corr'], 4),
            'avg_sig_corr': round(avg_sig, 4)
        })
    
    return pd.DataFrame(results).sort_values('sig_count', ascending=False)

def engineer_features(df, market_share_df):
    """Executes the complete feature engineering pipeline."""
    
    print("\n--- 🔧 Phase 1: Base Feature Engineering ---")
    merged_df = df.copy()
    merged_df['day_of_year'] = merged_df['date_on'].dt.dayofyear
    merged_df['quarter'] = merged_df['date_on'].dt.quarter
    
    merged_df = merged_df.merge(
        market_share_df[['region_id', 'percent_country_production']], 
        on='region_id', how='left'
    )
    merged_df['percent_country_production'] = merged_df['percent_country_production'].fillna(1.0)
    
    ALL_NEW_FEATURES = []
    print("✅ Base setup complete")
    
    # Base Risk Scores
    for risk_type in RISK_CATEGORIES:
        low_col = f'climate_risk_cnt_locations_{risk_type}_risk_low'
        med_col = f'climate_risk_cnt_locations_{risk_type}_risk_medium' 
        high_col = f'climate_risk_cnt_locations_{risk_type}_risk_high'
        
        total = merged_df[low_col] + merged_df[med_col] + merged_df[high_col]
        risk_score = (merged_df[med_col] + 2 * merged_df[high_col]) / (total + 1e-6)
        weighted = risk_score * (merged_df['percent_country_production'] / 100)
        
        merged_df[f'climate_risk_{risk_type}_score'] = risk_score
        merged_df[f'climate_risk_{risk_type}_weighted'] = weighted
        ALL_NEW_FEATURES.extend([f'climate_risk_{risk_type}_score', f'climate_risk_{risk_type}_weighted'])

    print(f"✅ Base risk scores: {len(ALL_NEW_FEATURES)} features")
    
    print("\n--- 🔧 Phase 2: Advanced Rolling Features ---")
    merged_df = merged_df.sort_values(['region_id', 'date_on'])
    
    for window in [7, 14, 30, 60, 90]:
        for risk_type in RISK_CATEGORIES:
            score_col = f'climate_risk_{risk_type}_score'
            
            # Moving Average
            ma_col = f'climate_risk_{risk_type}_ma_{window}d'
            merged_df[ma_col] = (
                merged_df.groupby('region_id')[score_col]
                .transform(lambda x: x.rolling(window, min_periods=1).mean())
            )
            ALL_NEW_FEATURES.append(ma_col)
            
            # Rolling Max
            max_col = f'climate_risk_{risk_type}_max_{window}d'
            merged_df[max_col] = (
                merged_df.groupby('region_id')[score_col]
                .transform(lambda x: x.rolling(window, min_periods=1).max())
            )
            ALL_NEW_FEATURES.append(max_col)

    print(f"✅ Rolling features: {len(ALL_NEW_FEATURES)} total")
    
    print("\n--- 🔧 Phase 3: Lag Features ---")
    for lag in [7, 14, 30, 60, 90]:
        for risk_type in RISK_CATEGORIES:
            score_col = f'climate_risk_{risk_type}_score'
            lag_col = f'climate_risk_{risk_type}_lag_{lag}d'
            merged_df[lag_col] = merged_df.groupby('region_id')[score_col].shift(lag)
            ALL_NEW_FEATURES.append(lag_col)
    print(f"✅ Lag features added: {len(ALL_NEW_FEATURES)} total")
    
    print("\n--- 🔧 Phase 4: EMA Features ---")
    for span in [14, 30, 46]:
        for risk_type in RISK_CATEGORIES:
            score_col = f'climate_risk_{risk_type}_score'
            ema_col = f'climate_risk_{risk_type}_ema_{span}d'
            merged_df[ema_col] = (
                merged_df.groupby('region_id')[score_col]
                .transform(lambda x: x.ewm(span=span, min_periods=1).mean())
            )
            ALL_NEW_FEATURES.append(ema_col)
    print(f"✅ EMA features added: {len(ALL_NEW_FEATURES)} total")

    print("\n--- 🔧 Phase 5: Volatility Features ---")
    for window in [14, 30, 46]:
        for risk_type in RISK_CATEGORIES:
            score_col = f'climate_risk_{risk_type}_score'
            vol_col = f'climate_risk_{risk_type}_vol_{window}d'
            merged_df[vol_col] = (
                merged_df.groupby('region_id')[score_col]
                .transform(lambda x: x.rolling(window, min_periods=2).std())
            )
            ALL_NEW_FEATURES.append(vol_col)
    print(f"✅ Volatility features added: {len(ALL_NEW_FEATURES)} total")

    print("\n--- 🔧 Phase 6: Cumulative Stress Features ---")
    for window in [30, 60, 90]:
        for risk_type in RISK_CATEGORIES:
            score_col = f'climate_risk_{risk_type}_score'
            cum_col = f'climate_risk_{risk_type}_cumsum_{window}d'
            merged_df[cum_col] = (
                merged_df.groupby('region_id')[score_col]
                .transform(lambda x: x.rolling(window, min_periods=1).sum())
            )
            ALL_NEW_FEATURES.append(cum_col)
    print(f"✅ Cumulative features added: {len(ALL_NEW_FEATURES)} total")

    print("\n--- 🔧 Phase 7: Non-linear Features ---")
    for risk_type in RISK_CATEGORIES:
        score_col = f'climate_risk_{risk_type}_score'
        
        sq_col = f'climate_risk_{risk_type}_squared'
        merged_df[sq_col] = merged_df[score_col] ** 2
        ALL_NEW_FEATURES.append(sq_col)
        
        log_col = f'climate_risk_{risk_type}_log'
        merged_df[log_col] = np.log1p(merged_df[score_col])
        ALL_NEW_FEATURES.append(log_col)
    print(f"✅ Non-linear features added: {len(ALL_NEW_FEATURES)} total")
    
    print("\n--- 🔧 Phase 8: Interaction Features ---")
    score_cols = [f'climate_risk_{r}_score' for r in RISK_CATEGORIES]
    
    merged_df['climate_risk_temperature_stress'] = merged_df[['climate_risk_heat_stress_score', 'climate_risk_unseasonably_cold_score']].max(axis=1)
    ALL_NEW_FEATURES.append('climate_risk_temperature_stress')
    
    merged_df['climate_risk_precipitation_stress'] = merged_df[['climate_risk_excess_precip_score', 'climate_risk_drought_score']].max(axis=1)
    ALL_NEW_FEATURES.append('climate_risk_precipitation_stress')
    
    merged_df['climate_risk_overall_stress'] = merged_df[score_cols].max(axis=1)
    ALL_NEW_FEATURES.append('climate_risk_overall_stress')
    
    merged_df['climate_risk_combined_stress'] = merged_df[score_cols].sum(axis=1)
    ALL_NEW_FEATURES.append('climate_risk_combined_stress')
    
    merged_df['climate_risk_precip_drought_diff'] = merged_df['climate_risk_excess_precip_score'] - merged_df['climate_risk_drought_score']
    ALL_NEW_FEATURES.append('climate_risk_precip_drought_diff')
    
    merged_df['climate_risk_temp_diff'] = merged_df['climate_risk_heat_stress_score'] - merged_df['climate_risk_unseasonably_cold_score']
    ALL_NEW_FEATURES.append('climate_risk_temp_diff')
    
    merged_df['climate_risk_precip_drought_ratio'] = merged_df['climate_risk_excess_precip_score'] / (merged_df['climate_risk_drought_score'] + 0.01)
    ALL_NEW_FEATURES.append('climate_risk_precip_drought_ratio')
    print(f"✅ Interaction features added: {len(ALL_NEW_FEATURES)} total")
    
    print("\n--- 🔧 Phase 9: Seasonal Features ---")
    merged_df['climate_risk_season_sin'] = np.sin(2 * np.pi * merged_df['day_of_year'] / 365)
    merged_df['climate_risk_season_cos'] = np.cos(2 * np.pi * merged_df['day_of_year'] / 365)
    ALL_NEW_FEATURES.extend(['climate_risk_season_sin', 'climate_risk_season_cos'])
    
    growing_season_weight = merged_df['quarter'].map({1: 0.5, 2: 1.0, 3: 1.0, 4: 0.5})
    for risk_type in ['drought', 'excess_precip']:
        score_col = f'climate_risk_{risk_type}_score'
        seasonal_col = f'climate_risk_{risk_type}_seasonal'
        merged_df[seasonal_col] = merged_df[score_col] * growing_season_weight
        ALL_NEW_FEATURES.append(seasonal_col)
    print(f"✅ Seasonal features added: {len(ALL_NEW_FEATURES)} total")

    print("\n--- 🔧 Phase 10: Momentum Features ---")
    for risk_type in RISK_CATEGORIES:
        score_col = f'climate_risk_{risk_type}_score'
        
        c1 = f'climate_risk_{risk_type}_change_1d'
        merged_df[c1] = merged_df.groupby('region_id')[score_col].diff(1)
        ALL_NEW_FEATURES.append(c1)
        
        c7 = f'climate_risk_{risk_type}_change_7d'
        merged_df[c7] = merged_df.groupby('region_id')[score_col].diff(7)
        ALL_NEW_FEATURES.append(c7)
        
        acc = f'climate_risk_{risk_type}_acceleration'
        merged_df[acc] = merged_df.groupby('region_id')[c1].diff(1)
        ALL_NEW_FEATURES.append(acc)
    print(f"✅ Momentum features added: {len(ALL_NEW_FEATURES)} total")
    
    print("\n--- 🔧 Phase 11: Country Aggregations ---")
    for risk_type in RISK_CATEGORIES:
        score_col = f'climate_risk_{risk_type}_score'
        weighted_col = f'climate_risk_{risk_type}_weighted'
        
        country_agg = merged_df.groupby(['country_name', 'date_on']).agg({
            score_col: ['mean', 'max', 'std'],
            weighted_col: 'sum',
            'percent_country_production': 'sum'
        }).round(4)
        
        country_agg.columns = [f'country_{risk_type}_{"_".join(col).strip()}' for col in country_agg.columns]
        country_agg = country_agg.reset_index()
        
        new_cols = [c for c in country_agg.columns if c not in ['country_name', 'date_on']]
        ALL_NEW_FEATURES.extend(new_cols)
        
        merged_df = merged_df.merge(country_agg, on=['country_name', 'date_on'], how='left')
    print(f"✅ Country aggregations added: {len(ALL_NEW_FEATURES)} total")

    return merged_df, ALL_NEW_FEATURES

def handle_nans_and_validation(merged_df, valid_ids_source_df, all_new_features):
    """
    Handles NaN values and filters to valid IDs using the approach from the sample submission.
    """
    # 1. Identify valid IDs by simulating sample submission's approach
    print("\n📊 Identifying valid IDs (simulating sample submission)...")
    
    # Simulate minimal pipeline to get IDs
    temp_df = valid_ids_source_df.copy() # Assume this is raw or minimally processed
    
    # We need to replicate the exact steps that LEAD to the dropna() in the original notebook 
    # to get the exact same valid_ids. 
    # In the original notebook, 'temp_df' is loaded fresh.
    # For efficiency, we assume valid_ids_source_df IS that fresh load or we reconstruct it.
    
    # ... (Replicating the minimal logic for valid_ids as per original script)
    # The original script re-reads the file to be safe. We will use a copy of the input.
    
    # Minimal feature eng for ID determination
    temp_df['day_of_year'] = temp_df['date_on'].dt.dayofyear
    
    # Sort for rolling operations
    temp_df = temp_df.sort_values(['region_id', 'date_on'])
    
    # Create rolling features (7, 14, 30 days) - minimal set that causes NaNs in sample sub
    for window in [7, 14, 30]:
        for risk_type in RISK_CATEGORIES:
            # We assume risk scores exist or create dummy ones if needed, 
            # BUT the original script creates them. Let's assume we need to create them.
            # Simplified for ID extraction:
            score_col = f'climate_risk_{risk_type}_score'
            # We need these columns to exist.
            # If they don't (because we passed raw df), we calculate them quickly.
            # For strictness, let's reuse the calculated ones from merged_df but just take the relevant columns?
            # No, because lag will introduce NaNs differently.
            pass

    # Actually, the original script logic for valid_ids is specific:
    # It re-calculates everything on temp_df up to step Phase 11, then does `temp_df.dropna()['ID']`.
    # This implies that users ONLY appear in the final set if they survive ALL feature engineering NaNs 
    # (like rolling windows at the start of the series).
    # To reliably get the same IDs, we should probably just use the `merged_df` we just built
    # BEFORE filling 0s.
    # BUT, the original script does:
    # 1. Calc valid IDs on a separate temp_df (re-doing work).
    # 2. Fill 0s in MAIN `merged_df`.
    # 3. Filter `merged_df` by `valid_ids`.
    
    # To save time but maintain correctness, we can implement the logic on the MAIN df
    # by identifying which rows WOULD be dropped.
    # However, let's stick to the script's logic to be "nickel".
    
    # We will simulate the "valid_ids" derived from the main `merged_df` if we HAD NOT filled NaNs yet?
    # The original script re-calculates to be absolutely sure.
    # Let's try to derive it from `merged_df` before fillna. 
    # Any row with NaN in the columns that were created in temp_df logic would be dropped.
    # The temp_df logic went up to Phase 11 (Country Aggs).
    
    # So, `valid_ids` are rows in `merged_df` that have NO NaNs in the columns created up to Phase 11.
    # Let's use `merged_df.dropna()['ID']` from the current state (since we haven't filled 0 yet).
    valid_ids = merged_df.dropna()['ID'].tolist()
    print(f"📊 Valid IDs: {len(valid_ids):,}")

    # 2. Fill all engineered features with 0
    print("📊 Filling engineered features with 0...")
    for col in all_new_features:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].fillna(0)
            
    # Also fill any remaining NaN in climate_risk columns
    climate_cols = [c for c in merged_df.columns if c.startswith('climate_risk_')]
    for col in climate_cols:
        if merged_df[col].isna().any():
            merged_df[col] = merged_df[col].fillna(0)

    # 3. Filter to valid IDs
    print("📊 Filtering to valid IDs...")
    futures_cols = [c for c in merged_df.columns if c.startswith('futures_')]
    baseline_df = merged_df.dropna(subset=futures_cols)
    baseline_df = baseline_df[baseline_df['ID'].isin(valid_ids)]
    
    return baseline_df

def main():
    setup_environment()
    
    # Load Data
    df, market_share_df = load_data()
    
    # Feature Engineering
    merged_df, created_features = engineer_features(df, market_share_df)
    
    # Handle NaNs and Valid rows
    # Note: We pass df again as source for valid IDs if we wanted to be strict,
    # but using merged_df.dropna() effectively does the same if the feature set is identical.
    baseline_df = handle_nans_and_validation(merged_df, df, created_features)
    
    print(f"📊 After NaN handling: {len(baseline_df):,} rows")
    if len(baseline_df) != REQUIRED_ROWS:
        print(f"⚠️ Row count mismatch! Expected {REQUIRED_ROWS}, got {len(baseline_df)}")
    
    # Feature Selection (Phase 12)
    print("\n--- 📊 Phase 12: Feature Analysis and Selection ---")
    climate_cols = [c for c in baseline_df.columns if c.startswith('climate_risk_')]
    futures_cols = [c for c in baseline_df.columns if c.startswith('futures_')]
    
    # Sampling for speed if needed (but script implies full run)
    feature_analysis = analyze_feature_contributions(baseline_df, climate_cols, futures_cols)
    
    # Identify features to remove
    zero_sig_features = feature_analysis[feature_analysis['sig_count'] == 0]['feature'].tolist()
    
    # Keep original cnt_locations columns
    original_cols = [c for c in zero_sig_features if 'cnt_locations' in c]
    features_to_remove = [c for c in zero_sig_features if c not in original_cols]
    
    print(f"Removing {len(features_to_remove)} features with 0 significant correlations.")
    
    # Create Optimized Dataset (Phase 13)
    optimized_df = baseline_df.drop(columns=features_to_remove, errors='ignore')
    
    # Score Comparison (Phase 14)
    print("\n--- 📊 Phase 14: Score Comparison ---")
    print("Computing Baseline CFCS...")
    baseline_score = compute_cfcs(baseline_df, verbose=False) # Too verbose otherwise
    print(f"Baseline: {baseline_score['cfcs']}")
    
    print("Computing Optimized CFCS...")
    optimized_score = compute_cfcs(optimized_df, verbose=False)
    print(f"Optimized: {optimized_score['cfcs']}")
    
    # Final Selection
    if optimized_score['cfcs'] >= baseline_score['cfcs']:
        best_df = optimized_df
        best_score = optimized_score
        best_name = 'optimized'
    else:
        best_df = baseline_df
        best_score = baseline_score
        best_name = 'baseline'
        
    print(f"🏆 Best version: {best_name} (CFCS: {best_score['cfcs']})")
    
    # Save Submission (Phase 15)
    print("\n--- 📊 Phase 15: Final Submission ---")
    
    # Final check
    submission = best_df.copy()
    if submission.isnull().sum().sum() > 0:
        submission = submission.fillna(0)
        
    validation_passed = (len(submission) == REQUIRED_ROWS) and\
                        ('ID' in submission.columns) and\
                        (submission.isnull().sum().sum() == 0)
                        
    if not validation_passed:
        print("❌ Validation Failed")
    else:
        print("✅ Validation Passed")
        
    output_file = os.path.join(OUTPUT_PATH, 'submission.csv')
    submission.to_csv(output_file, index=False)
    print(f"📁 Saved: {output_file}")

if __name__ == "__main__":
    main()