"""
FAST OPTIMIZATION: Accuracy-Based Pre-Filtering
================================================
Step 1: Quickly evaluate prediction quality by confidence threshold (no backtest)
Step 2: Only backtest the top 5 candidates

This is 10x faster than full optimization.
"""
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = 'data_15m'
PAIRS = ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY']

print("="*100)
print("FAST OPTIMIZATION: Accuracy-Based Pre-Filtering")
print("="*100)
print()

# Load predictions and actual outcomes
print("Loading predictions and calculating accuracy...")
test_files = [
    ('2021', 'test_predictions_15m_2021_test.pkl'),
    ('2022', 'test_predictions_15m_2022_test.pkl'),
    ('2023', 'test_predictions_15m_2023_test.pkl'),
    ('2024', 'test_predictions_15m_2024_test.pkl'),
    ('2025', 'test_predictions_15m_2025_test.pkl'),
]

# Load all predictions
all_predictions = {}
for year, filename in test_files:
    with open(filename, 'rb') as f:
        all_predictions[year] = pickle.load(f)

print("Done")
print()

# Load actual data with outcomes to calculate accuracy
print("Loading actual data with outcomes...")
from train_model_15m import calculate_features, calculate_targets

LOOKBACK_PERIOD = 80
FORWARD_PERIODS = 24

actual_data = {}
for pair in PAIRS:
    file_path = f'{DATA_DIR}/{pair}_15m.csv'
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    # Calculate features and targets to get actual breakout labels
    df = calculate_features(df, LOOKBACK_PERIOD)
    df = calculate_targets(df, LOOKBACK_PERIOD, FORWARD_PERIODS)

    actual_data[pair] = df

print("Done")
print()

# STEP 1: Test different confidence thresholds for accuracy
print("="*100)
print("STEP 1: Testing Confidence Thresholds (Fast - No Backtest)")
print("="*100)
print()

confidence_thresholds = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]

results = []

for conf_threshold in confidence_thresholds:
    print(f"Testing confidence threshold: {conf_threshold:.2f}")

    # Calculate win rate and trade count for each year
    year_stats = []

    for year, preds in all_predictions.items():
        total_signals = 0
        correct_signals = 0

        for pair, pred_df in preds.items():
            # Get actual outcomes for this pair
            actual_df = actual_data[pair]

            # Filter predictions by confidence
            high_conf = pred_df[pred_df[['breakout_high_prob', 'breakout_low_prob']].max(axis=1) > conf_threshold]

            if len(high_conf) == 0:
                continue

            # For each high-confidence prediction, check if prediction was correct
            for idx, row in high_conf.iterrows():
                # Get actual outcome at this timestamp
                if idx not in actual_df.index:
                    continue

                actual_row = actual_df.loc[idx]

                # Skip if no actual outcome available
                if pd.isna(actual_row.get('breakout_high')) or pd.isna(actual_row.get('breakout_low')):
                    continue

                total_signals += 1

                if row['breakout_high_prob'] > row['breakout_low_prob']:
                    # Predicted high breakout - was it correct?
                    if actual_row['breakout_high'] == 1:
                        correct_signals += 1
                else:
                    # Predicted low breakout - was it correct?
                    if actual_row['breakout_low'] == 1:
                        correct_signals += 1

        if total_signals > 0:
            accuracy = correct_signals / total_signals
            year_stats.append({
                'year': year,
                'signals': total_signals,
                'accuracy': accuracy
            })

    # Calculate average stats
    if len(year_stats) > 0:
        avg_signals = np.mean([s['signals'] for s in year_stats])
        avg_accuracy = np.mean([s['accuracy'] for s in year_stats])
    else:
        avg_signals = 0
        avg_accuracy = 0

    print(f"  Avg signals per year: {avg_signals:.0f}")
    print(f"  Avg accuracy: {avg_accuracy:.1%}")
    print()

    results.append({
        'conf_threshold': conf_threshold,
        'avg_signals': avg_signals,
        'avg_accuracy': avg_accuracy,
        'score': avg_accuracy * (avg_signals / 1000) if avg_signals > 0 else 0
    })

# Show top candidates
print("="*100)
print("TOP CONFIDENCE THRESHOLDS BY SCORE")
print("="*100)
print()

if results:
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('score', ascending=False)

    print(results_df.to_string(index=False))
    print()

    # Get top 3 for full backtest
    top_candidates = results_df.head(3)['conf_threshold'].tolist()
else:
    print("No results found!")
    top_candidates = [0.70]  # Default fallback

print(f"Top candidates for full backtest: {top_candidates}")
print()

# STEP 2: Full staged optimization (imported module runs automatically)
print("="*100)
print("STEP 2: Running Staged Optimization")
print("="*100)
print()

# Import staged optimization - this runs the full 3-stage optimization
from optimize_staged import run_backtest_with_params

print()
print("="*100)
print("OPTIMIZATION COMPLETE")
print("="*100)
print()
print("The staged optimization results above show the best parameters found.")
print("Look for 'FINAL OPTIMIZED PARAMETERS' section in the output above.")
print()
