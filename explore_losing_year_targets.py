"""
EXPLORE PREDICTION TARGETS FOR LOSING YEARS
===========================================
Identify patterns that work when breakout strategy fails.

Losing years: 2018, 2020, 2021, 2022, 2024
Goal: Find mean reversion, range-bound, or other patterns that work in choppy markets
"""
import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("EXPLORING TARGETS FOR LOSING YEARS (2018, 2020, 2021, 2022, 2024)")
print("="*100)
print()

DATA_DIR = 'data'
PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY']
LOSING_YEARS = [2018, 2020, 2021, 2022, 2024]

def calculate_features(df):
    """Calculate technical features"""
    # Returns
    df['return_1d'] = df['close'].pct_change()
    df['return_3d'] = df['close'].pct_change(3)
    df['return_5d'] = df['close'].pct_change(5)
    df['return_10d'] = df['close'].pct_change(10)

    # Moving averages
    for period in [10, 20, 50]:
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        df[f'price_to_ema_{period}'] = df['close'] / df[f'ema_{period}'] - 1

    # Volatility
    df['volatility_10d'] = df['return_1d'].rolling(10).std()
    df['volatility_20d'] = df['return_1d'].rolling(20).std()

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * bb_std
    df['bb_lower'] = df['bb_middle'] - 2 * bb_std
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

    # Range features
    df['high_20d'] = df['high'].rolling(20).max()
    df['low_20d'] = df['low'].rolling(20).min()
    df['range_20d'] = (df['high_20d'] - df['low_20d']) / df['close']
    df['position_in_range'] = (df['close'] - df['low_20d']) / (df['high_20d'] - df['low_20d'] + 1e-10)

    return df


def create_targets(df):
    """Create various prediction targets to test"""
    targets = {}

    # === MEAN REVERSION TARGETS ===

    # 1. Mean reversion to EMA20 (5-day horizon)
    future_close_5d = df['close'].shift(-5)
    ema_20 = df['ema_20']
    # If price is above EMA, will it move down toward EMA?
    # If price is below EMA, will it move up toward EMA?
    distance_from_ema = df['close'] / ema_20 - 1
    future_distance = future_close_5d / ema_20 - 1
    # Target: Distance to EMA reduces (mean reversion happens)
    targets['mean_revert_ema20_5d'] = (abs(future_distance) < abs(distance_from_ema) * 0.7).astype(int)

    # 2. Bollinger Band bounce (3-day horizon)
    # If price near BB edges, will it bounce back?
    future_close_3d = df['close'].shift(-3)
    bb_position = df['bb_position']
    future_bb_position = (future_close_3d - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
    # Near lower band (< 0.2) -> will bounce up
    # Near upper band (> 0.8) -> will bounce down
    targets['bb_bounce_3d'] = (
        ((bb_position < 0.2) & (future_bb_position > 0.3)) |
        ((bb_position > 0.8) & (future_bb_position < 0.7))
    ).astype(int)

    # 3. Oversold/Overbought reversion (5-day)
    rsi = df['rsi']
    future_close = df['close'].shift(-5)
    price_change_5d = (future_close - df['close']) / df['close']
    # Oversold (RSI < 30) -> price goes up
    # Overbought (RSI > 70) -> price goes down
    targets['rsi_reversion_5d'] = (
        ((rsi < 30) & (price_change_5d > 0)) |
        ((rsi > 70) & (price_change_5d < 0))
    ).astype(int)

    # === RANGE-BOUND TARGETS ===

    # 4. Stay in range (5-day horizon)
    future_high_5d = df['high'].rolling(5).max().shift(-5)
    future_low_5d = df['low'].rolling(5).min().shift(-5)
    current_high_20d = df['high_20d']
    current_low_20d = df['low_20d']
    targets['range_bound_5d'] = (
        (future_high_5d <= current_high_20d * 1.002) &
        (future_low_5d >= current_low_20d * 0.998)
    ).astype(int)

    # 5. Stay in tight range (10-day horizon)
    future_high_10d = df['high'].rolling(10).max().shift(-10)
    future_low_10d = df['low'].rolling(10).min().shift(-10)
    future_range = (future_high_10d - future_low_10d) / df['close']
    targets['tight_range_10d'] = (future_range < df['range_20d'] * 0.8).astype(int)

    # === VOLATILITY TARGETS ===

    # 6. Volatility contraction (10-day)
    current_vol = df['volatility_10d']
    future_vol = df['volatility_10d'].shift(-10)
    targets['vol_contraction_10d'] = (future_vol < current_vol * 0.8).astype(int)

    # 7. Volatility expansion (5-day)
    current_vol_5d = df['volatility_10d']
    future_vol_5d = df['volatility_10d'].shift(-5)
    targets['vol_expansion_5d'] = (future_vol_5d > current_vol_5d * 1.2).astype(int)

    # === FAILED BREAKOUT TARGETS ===

    # 8. Failed upward breakout (5-day)
    # Price approaches 20-day high but fails to break through
    distance_to_high = (df['high_20d'] - df['close']) / df['close']
    future_high_5d = df['high'].rolling(5).max().shift(-5)
    targets['failed_breakout_up_5d'] = (
        (distance_to_high < 0.01) &  # Very close to 20-day high
        (future_high_5d < df['high_20d'] * 1.002) &  # Doesn't break through
        (future_close_5d < df['close'])  # Price falls back
    ).astype(int)

    # 9. Failed downward breakout (5-day)
    distance_to_low = (df['close'] - df['low_20d']) / df['close']
    future_low_5d = df['low'].rolling(5).min().shift(-5)
    targets['failed_breakout_down_5d'] = (
        (distance_to_low < 0.01) &  # Very close to 20-day low
        (future_low_5d > df['low_20d'] * 0.998) &  # Doesn't break through
        (future_close_5d > df['close'])  # Price bounces back
    ).astype(int)

    # === CONSOLIDATION TARGETS ===

    # 10. Price consolidates after big move (7-day)
    recent_move = abs(df['return_5d'])
    future_range_7d = (df['high'].rolling(7).max().shift(-7) - df['low'].rolling(7).min().shift(-7)) / df['close']
    targets['consolidation_7d'] = (
        (recent_move > 0.015) &  # Had a big move (>1.5%)
        (future_range_7d < 0.01)  # Then consolidates in tight range
    ).astype(int)

    return df, targets


# Load and process all data
print("Loading data for all pairs...")
all_results = []

for pair in PAIRS:
    file_path = os.path.join(DATA_DIR, f'{pair}_1day_oanda.csv')
    if not os.path.exists(file_path):
        continue

    df = pd.read_csv(file_path, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df['year'] = pd.to_datetime(df['date']).dt.year

    # Calculate features
    df = calculate_features(df)

    # Create targets
    df, targets = create_targets(df)

    # Test each target
    for target_name, target_series in targets.items():
        df[target_name] = target_series

        # Split by losing years vs winning years
        losing_year_data = df[df['year'].isin(LOSING_YEARS)].copy()
        winning_year_data = df[~df['year'].isin(LOSING_YEARS) & (df['year'] >= 2016) & (df['year'] <= 2025)].copy()

        # Drop NaN
        losing_year_data = losing_year_data.dropna(subset=[target_name])
        winning_year_data = winning_year_data.dropna(subset=[target_name])

        if len(losing_year_data) > 100 and len(winning_year_data) > 100:
            # Calculate base rate and accuracy
            losing_base_rate = losing_year_data[target_name].mean()
            winning_base_rate = winning_year_data[target_name].mean()

            # A good target has:
            # 1. High base rate in losing years (happens often)
            # 2. Predictable pattern (not just 50/50)

            all_results.append({
                'pair': pair,
                'target': target_name,
                'losing_year_rate': losing_base_rate,
                'winning_year_rate': winning_base_rate,
                'losing_year_count': len(losing_year_data),
                'winning_year_count': len(winning_year_data),
                'difference': losing_base_rate - winning_base_rate
            })

print()
print("="*100)
print("ANALYSIS: TARGETS THAT WORK BETTER IN LOSING YEARS")
print("="*100)
print()

# Convert to DataFrame for analysis
results_df = pd.DataFrame(all_results)

# Aggregate by target across all pairs
target_summary = results_df.groupby('target').agg({
    'losing_year_rate': 'mean',
    'winning_year_rate': 'mean',
    'difference': 'mean',
    'losing_year_count': 'sum'
}).reset_index()

# Sort by difference (targets that occur more in losing years)
target_summary = target_summary.sort_values('difference', ascending=False)

print("Targets ranked by how much MORE they occur in LOSING years vs WINNING years:")
print()
print(f"{'Target':<30} {'Losing Yr %':>12} {'Winning Yr %':>13} {'Difference':>12} {'Samples':>10}")
print("-" * 90)

for _, row in target_summary.iterrows():
    print(f"{row['target']:<30} {row['losing_year_rate']:>11.1%} {row['winning_year_rate']:>12.1%} "
          f"{row['difference']:>11.1%} {row['losing_year_count']:>10.0f}")

print()
print("="*100)
print("TOP CANDIDATES FOR COMPLEMENT STRATEGY")
print("="*100)
print()

# Identify best targets: high rate in losing years, low in winning years
top_targets = target_summary[
    (target_summary['losing_year_rate'] > 0.3) &  # Happens >30% of time in losing years
    (target_summary['difference'] > 0.05)  # At least 5% more common in losing years
].head(5)

if len(top_targets) > 0:
    print("These patterns are MORE COMMON in losing years (when breakouts fail):")
    print()
    for _, row in top_targets.iterrows():
        print(f"✓ {row['target']}")
        print(f"  - Occurs {row['losing_year_rate']:.1%} in losing years vs {row['winning_year_rate']:.1%} in winning years")
        print(f"  - Difference: +{row['difference']:.1%}")
        print()

    print("RECOMMENDATION:")
    print("Train models on these targets and trade them alongside breakout strategy.")
    print("They should perform well when breakouts fail (choppy/range-bound markets).")
else:
    print("No strong patterns found. Try different target definitions or timeframes.")

print()
print("="*100)
print("NEXT STEPS")
print("="*100)
print()
print("1. Train XGBoost models on top 2-3 targets above")
print("2. Backtest these targets in losing years (2018, 2020, 2021, 2022, 2024)")
print("3. If profitable, combine with breakout strategy for all-weather approach")
print()
