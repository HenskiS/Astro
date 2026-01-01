"""
ANALYZE PREDICTION ACCURACY
============================
Evaluate how accurately the model predicts breakouts vs actual outcomes
"""
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("PREDICTION ACCURACY ANALYSIS")
print("="*100)
print()

# Load predictions
print("Loading predictions...")
with open('model_predictions_quarterly.pkl', 'rb') as f:
    all_predictions = pickle.load(f)

# Load raw price data
DATA_DIR = 'data'
PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY']

all_raw_data = {}
for pair in PAIRS:
    file_path = f'{DATA_DIR}/{pair}_1day_with_spreads.csv'
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    all_raw_data[pair] = df

print(f"Loaded {len(all_predictions)} quarters")
print()

# For each prediction, check if breakout actually occurred
results = []

for quarter_name, quarter_preds in sorted(all_predictions.items()):
    for pair, pred_df in quarter_preds.items():
        raw_df = all_raw_data[pair]

        for date, row in pred_df.iterrows():
            # Get the next 10 days of actual price data
            future_dates = raw_df[raw_df.index > date].head(10)

            if len(future_dates) < 10:
                continue  # Not enough future data

            entry_price = row['close']
            breakout_high = row['high_20d']
            breakout_low = row['low_20d']

            # Check if highs/lows were broken in next 10 days
            future_high = future_dates['high'].max()
            future_low = future_dates['low'].min()

            broke_high = future_high >= breakout_high
            broke_low = future_low <= breakout_low

            # Model's prediction
            pred_high_prob = row['breakout_high_prob']
            pred_low_prob = row['breakout_low_prob']

            # Determine which side model favored
            if pred_high_prob > pred_low_prob:
                model_direction = 'high'
                model_confidence = pred_high_prob
                outcome = broke_high
            else:
                model_direction = 'low'
                model_confidence = pred_low_prob
                outcome = broke_low

            results.append({
                'quarter': quarter_name,
                'pair': pair,
                'date': date,
                'model_direction': model_direction,
                'model_confidence': model_confidence,
                'outcome': outcome,
                'broke_high': broke_high,
                'broke_low': broke_low,
                'pred_high_prob': pred_high_prob,
                'pred_low_prob': pred_low_prob
            })

results_df = pd.DataFrame(results)
print(f"Total predictions analyzed: {len(results_df):,}")
print()

# Overall accuracy
print("="*100)
print("OVERALL PREDICTION ACCURACY")
print("="*100)
print()
accuracy = results_df['outcome'].mean()
print(f"Model Accuracy:           {accuracy:.1%}")
print(f"Correct Predictions:      {results_df['outcome'].sum():,}")
print(f"Incorrect Predictions:    {(~results_df['outcome']).sum():,}")
print()

# Baseline (random guessing)
print("Baseline Comparisons:")
print(f"  High breakout rate:     {results_df['broke_high'].mean():.1%}")
print(f"  Low breakout rate:      {results_df['broke_low'].mean():.1%}")
print(f"  Both broke:             {(results_df['broke_high'] & results_df['broke_low']).mean():.1%}")
print(f"  Neither broke:          {(~results_df['broke_high'] & ~results_df['broke_low']).mean():.1%}")
print()

# Accuracy by confidence level
print("="*100)
print("ACCURACY BY CONFIDENCE LEVEL")
print("="*100)
print()
conf_bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
conf_labels = ['50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
results_df['conf_bin'] = pd.cut(results_df['model_confidence'], bins=conf_bins, labels=conf_labels)

print("Confidence | Predictions | Accuracy | Expected")
print("-" * 55)
for label in conf_labels:
    subset = results_df[results_df['conf_bin'] == label]
    if len(subset) > 0:
        accuracy = subset['outcome'].mean()
        expected = subset['model_confidence'].mean()
        count = len(subset)
        print(f"{label:10s} | {count:>11,} | {accuracy:>7.1%} | {expected:>7.1%}")
print()

# Calibration analysis
print("="*100)
print("MODEL CALIBRATION")
print("="*100)
print()
print("Is the model well-calibrated? (predicted probability ~ actual outcome rate)")
print()

# Group by 5% confidence buckets
results_df['conf_bucket'] = (results_df['model_confidence'] * 20).astype(int) * 5
calibration = results_df.groupby('conf_bucket').agg({
    'outcome': ['count', 'mean'],
    'model_confidence': 'mean'
})
calibration.columns = ['Count', 'Actual_Rate', 'Predicted_Rate']
calibration = calibration[calibration['Count'] >= 10]  # Only buckets with 10+ predictions

print("Predicted % | Actual % | Count    | Calibration Error")
print("-" * 60)
for bucket in sorted(calibration.index):
    row = calibration.loc[bucket]
    predicted = row['Predicted_Rate']
    actual = row['Actual_Rate']
    count = row['Count']
    error = actual - predicted
    print(f"{predicted:>10.1%} | {actual:>7.1%} | {count:>7,.0f} | {error:>+8.1%}")
print()

# Direction-specific accuracy
print("="*100)
print("ACCURACY BY PREDICTED DIRECTION")
print("="*100)
print()
for direction in ['high', 'low']:
    subset = results_df[results_df['model_direction'] == direction]
    if len(subset) > 0:
        accuracy = subset['outcome'].mean()
        avg_conf = subset['model_confidence'].mean()
        print(f"{direction.upper():4s} predictions: {len(subset):>6,} | Accuracy: {accuracy:.1%} | Avg Confidence: {avg_conf:.1%}")
print()

# Pair-specific accuracy
print("="*100)
print("ACCURACY BY PAIR")
print("="*100)
print()
print("Pair    | Predictions | Accuracy | Avg Confidence")
print("-" * 60)
for pair in sorted(results_df['pair'].unique()):
    subset = results_df[results_df['pair'] == pair]
    accuracy = subset['outcome'].mean()
    avg_conf = subset['model_confidence'].mean()
    print(f"{pair:7s} | {len(subset):>11,} | {accuracy:>7.1%} | {avg_conf:>13.1%}")
print()

# Yearly trends
print("="*100)
print("ACCURACY BY YEAR")
print("="*100)
print()
results_df['year'] = results_df['quarter'].str[:4]
print("Year | Predictions | Accuracy | Avg Confidence")
print("-" * 55)
for year in sorted(results_df['year'].unique()):
    subset = results_df[results_df['year'] == year]
    accuracy = subset['outcome'].mean()
    avg_conf = subset['model_confidence'].mean()
    print(f"{year} | {len(subset):>11,} | {accuracy:>7.1%} | {avg_conf:>13.1%}")
print()

# High confidence subset (>= 70%)
print("="*100)
print("HIGH CONFIDENCE PREDICTIONS (>=70%)")
print("="*100)
print()
high_conf = results_df[results_df['model_confidence'] >= 0.70]
print(f"Total high confidence predictions: {len(high_conf):,} ({len(high_conf)/len(results_df):.1%} of all)")
print(f"Accuracy on high confidence:       {high_conf['outcome'].mean():.1%}")
print(f"Average confidence:                {high_conf['model_confidence'].mean():.1%}")
print()

print("="*100)
