"""
Check Feature Importance from Trained Models
"""
import pickle
import pandas as pd
from pathlib import Path

# Load models
models_file = Path('production_trader/models_15m.pkl')

if not models_file.exists():
    print("Models file not found!")
    exit(1)

with open(models_file, 'rb') as f:
    trained_models = pickle.load(f)

# Feature columns
feature_cols = [
    'dist_to_high', 'dist_to_low', 'range_80p',
    'price_vs_ema12', 'price_vs_ema26', 'price_vs_ema50', 'price_vs_ema100',
    'macd', 'macd_signal', 'macd_hist',
    'rsi_14',
    'atr_pct',
    'volume_ratio',
    'return_1p', 'return_4p', 'return_16p', 'return_96p',
    'spread_pct', 'spread_ratio',
    'hour', 'minute_slot', 'day_of_week',
    'asian_session', 'european_session', 'us_session', 'session_overlap',
    'friday_close', 'sunday_open'
]

print("=" * 80)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 80)
print()

# Analyze each pair
for pair in ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD']:
    if pair not in trained_models:
        continue

    print(f"\n{pair}:")
    print("-" * 80)

    # Get feature importances from both models
    model_high = trained_models[pair]['model_high']
    model_low = trained_models[pair]['model_low']

    importances_high = model_high.feature_importances_
    importances_low = model_low.feature_importances_

    # Average across both models
    avg_importance = (importances_high + importances_low) / 2

    # Create DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'High Model': importances_high,
        'Low Model': importances_low,
        'Average': avg_importance
    })

    # Sort by average importance
    importance_df = importance_df.sort_values('Average', ascending=False)

    # Show top 10
    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10).to_string(index=False))

print("\n" + "=" * 80)
print("OVERALL TOP FEATURES (averaged across all pairs)")
print("=" * 80)

# Calculate overall importance across all pairs
all_importances = []
for pair in trained_models.keys():
    model_high = trained_models[pair]['model_high']
    model_low = trained_models[pair]['model_low']
    avg_imp = (model_high.feature_importances_ + model_low.feature_importances_) / 2
    all_importances.append(avg_imp)

overall_importance = pd.DataFrame(all_importances).mean(axis=0)
overall_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': overall_importance
}).sort_values('Importance', ascending=False)

print("\n" + overall_df.head(15).to_string(index=False))
print()
