"""
TEST RANGE-BOUND TARGET VARIATIONS
===================================
The range_bound target showed +89% advantage in losing years.
Test variations to find the optimal definition before full backtest.

Variations:
1. range_bound_3d: 3-day horizon (faster mean reversion)
2. range_bound_5d: 5-day horizon (original)
3. range_bound_7d: 7-day horizon (longer consolidation)
4. tight_range: Future range contracts (volatility compression)
5. failed_breakout: Price approaches range edge but bounces back
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
import os
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("TESTING RANGE-BOUND VARIATIONS")
print("="*100)
print()

DATA_DIR = 'data'
PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']  # Just 4 pairs for speed
LOSING_YEARS = [2018, 2020, 2021]
WINNING_YEARS = [2019]

INITIAL_CAPITAL = 100000
RISK_PER_TRADE = 0.005
MIN_CONFIDENCE = 0.70

def calculate_features(df):
    """Technical features"""
    df['return_1d'] = df['close'].pct_change()
    df['return_3d'] = df['close'].pct_change(3)
    df['return_5d'] = df['close'].pct_change(5)
    df['return_10d'] = df['close'].pct_change(10)

    for period in [10, 20, 50]:
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        df[f'price_to_ema_{period}'] = df['close'] / df[f'ema_{period}'] - 1

    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']

    df['volatility_10d'] = df['return_1d'].rolling(10).std()
    df['volatility_20d'] = df['return_1d'].rolling(20).std()

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df['atr'] = ranges.max(axis=1).rolling(14).mean()
    df['atr_pct'] = df['atr'] / df['close']

    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * bb_std
    df['bb_lower'] = df['bb_middle'] - 2 * bb_std
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

    df['momentum_10d'] = df['close'] - df['close'].shift(10)
    df['momentum_20d'] = df['close'] - df['close'].shift(20)

    df['high_20d'] = df['high'].rolling(20).max()
    df['low_20d'] = df['low'].rolling(20).min()
    df['range_20d'] = (df['high_20d'] - df['low_20d']) / df['close']
    df['position_in_range'] = (df['close'] - df['low_20d']) / (df['high_20d'] - df['low_20d'] + 1e-10)

    return df

def create_range_targets(df):
    """Create 5 range-bound target variations"""
    targets = {}

    # 1. Range-bound 3-day
    future_high_3d = df['high'].rolling(3).max().shift(-3)
    future_low_3d = df['low'].rolling(3).min().shift(-3)
    targets['range_bound_3d'] = (
        (future_high_3d <= df['high_20d'] * 1.002) &
        (future_low_3d >= df['low_20d'] * 0.998)
    ).astype(int)

    # 2. Range-bound 5-day (original)
    future_high_5d = df['high'].rolling(5).max().shift(-5)
    future_low_5d = df['low'].rolling(5).min().shift(-5)
    targets['range_bound_5d'] = (
        (future_high_5d <= df['high_20d'] * 1.002) &
        (future_low_5d >= df['low_20d'] * 0.998)
    ).astype(int)

    # 3. Range-bound 7-day
    future_high_7d = df['high'].rolling(7).max().shift(-7)
    future_low_7d = df['low'].rolling(7).min().shift(-7)
    targets['range_bound_7d'] = (
        (future_high_7d <= df['high_20d'] * 1.002) &
        (future_low_7d >= df['low_20d'] * 0.998)
    ).astype(int)

    # 4. Tight range (volatility contraction)
    future_high_10d = df['high'].rolling(10).max().shift(-10)
    future_low_10d = df['low'].rolling(10).min().shift(-10)
    future_range = (future_high_10d - future_low_10d) / df['close']
    targets['tight_range'] = (future_range < df['range_20d'] * 0.8).astype(int)

    # 5. Failed breakout (approaches edge but bounces)
    distance_to_high = (df['high_20d'] - df['close']) / df['close']
    distance_to_low = (df['close'] - df['low_20d']) / df['close']
    future_close_5d = df['close'].shift(-5)

    targets['failed_breakout'] = (
        (
            # Near top edge, fails to break, falls back
            (distance_to_high < 0.01) &
            (future_high_5d < df['high_20d'] * 1.002) &
            (future_close_5d < df['close'])
        ) |
        (
            # Near bottom edge, fails to break, bounces back
            (distance_to_low < 0.01) &
            (future_low_5d > df['low_20d'] * 0.998) &
            (future_close_5d > df['close'])
        )
    ).astype(int)

    return targets

FEATURE_COLS = [
    'return_1d', 'return_3d', 'return_5d', 'return_10d',
    'ema_10', 'ema_20', 'ema_50',
    'price_to_ema_10', 'price_to_ema_20', 'price_to_ema_50',
    'macd', 'macd_signal', 'macd_diff',
    'rsi', 'atr', 'atr_pct',
    'volatility_10d', 'volatility_20d',
    'bb_middle', 'bb_upper', 'bb_lower', 'bb_position', 'bb_width',
    'momentum_10d', 'momentum_20d',
    'high_20d', 'low_20d', 'range_20d', 'position_in_range'
]

def train_model(target_name, all_pairs_data):
    """Train XGBoost model"""
    X_train_list = []
    y_train_list = []

    for pair_data in all_pairs_data.values():
        train_data = pair_data[(pair_data['year'] >= 2016) & (pair_data['year'] <= 2017)].copy()
        train_data = train_data.dropna(subset=FEATURE_COLS + [target_name])

        if len(train_data) > 50:
            X_train_list.append(train_data[FEATURE_COLS])
            y_train_list.append(train_data[target_name])

    if len(X_train_list) == 0:
        return None

    X_train = pd.concat(X_train_list)
    y_train = pd.concat(y_train_list)

    model = xgb.XGBClassifier(
        n_estimators=30,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    )

    model.fit(X_train, y_train, verbose=False)
    return model

def backtest_simple(target_name, model, all_pairs_data, test_years):
    """Simple backtest"""
    capital = INITIAL_CAPITAL
    trades = []

    for pair, pair_data in all_pairs_data.items():
        test_data = pair_data[pair_data['year'].isin(test_years)].copy()
        test_data = test_data.dropna(subset=FEATURE_COLS + [target_name])

        for idx, row in test_data.iterrows():
            X = row[FEATURE_COLS].values.reshape(1, -1)
            prob = model.predict_proba(X)[0, 1]

            if prob > MIN_CONFIDENCE:
                actual = row[target_name]
                profit_pct = 0.005 if actual == 1 else -0.003

                risk_amount = capital * RISK_PER_TRADE
                position_value = risk_amount / 0.02
                profit_dollars = profit_pct * position_value
                capital += profit_dollars

                trades.append({'year': row['year'], 'profit_pct': profit_pct})

    return capital, trades

# Load data
print("Loading data (4 pairs)...")
all_pairs_data = {}

for pair in PAIRS:
    file_path = os.path.join(DATA_DIR, f'{pair}_1day_oanda.csv')
    if not os.path.exists(file_path):
        continue

    df = pd.read_csv(file_path, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df['year'] = pd.to_datetime(df['date']).dt.year
    df = df[(df['year'] >= 2016) & (df['year'] <= 2021)].copy()

    df = calculate_features(df)
    targets = create_range_targets(df)
    for target_name, target_series in targets.items():
        df[target_name] = target_series

    all_pairs_data[pair] = df

print(f"Loaded {len(all_pairs_data)} pairs")
print()

# Test variations
print("="*100)
print("TESTING RANGE-BOUND VARIATIONS (2018-2021)")
print("="*100)
print()

target_names = ['range_bound_3d', 'range_bound_5d', 'range_bound_7d', 'tight_range', 'failed_breakout']
results = []

for target_name in target_names:
    print(f"{target_name}:")

    model = train_model(target_name, all_pairs_data)
    if model is None:
        print("  Skipped")
        continue

    # Test
    losing_cap, losing_trades = backtest_simple(target_name, model, all_pairs_data, LOSING_YEARS)
    losing_return = (losing_cap - INITIAL_CAPITAL) / INITIAL_CAPITAL
    losing_win = sum(1 for t in losing_trades if t['profit_pct'] > 0) / len(losing_trades) if losing_trades else 0

    winning_cap, winning_trades = backtest_simple(target_name, model, all_pairs_data, WINNING_YEARS)
    winning_return = (winning_cap - INITIAL_CAPITAL) / INITIAL_CAPITAL
    winning_win = sum(1 for t in winning_trades if t['profit_pct'] > 0) / len(winning_trades) if winning_trades else 0

    print(f"  Losing yrs: {losing_return:+6.1%} ({len(losing_trades):>3} trades, {losing_win:>3.0%} win)")
    print(f"  Winning yr: {winning_return:+6.1%} ({len(winning_trades):>3} trades, {winning_win:>3.0%} win)")
    print()

    results.append({
        'target': target_name,
        'losing_return': losing_return,
        'losing_trades': len(losing_trades),
        'losing_win': losing_win,
        'winning_return': winning_return,
        'winning_trades': len(winning_trades),
        'winning_win': winning_win,
        'difference': losing_return - winning_return
    })

# Summary
print("="*100)
print("COMPARISON")
print("="*100)
print()

results_df = pd.DataFrame(results).sort_values('difference', ascending=False)

print(f"{'Target':<20} {'Losing Yrs':>11} {'Trades':>7} {'Win%':>5}   {'Win Year':>11} {'Trades':>7} {'Win%':>5}   {'Edge':>7}")
print("-" * 90)

for _, row in results_df.iterrows():
    print(f"{row['target']:<20} {row['losing_return']:>10.1%} {row['losing_trades']:>7.0f} {row['losing_win']:>4.0%}   "
          f"{row['winning_return']:>10.1%} {row['winning_trades']:>7.0f} {row['winning_win']:>4.0%}   "
          f"{row['difference']:>6.1%}")

print()
print("="*100)
print("RECOMMENDATION")
print("="*100)
print()

best = results_df.iloc[0]
print(f"Best variation: {best['target']}")
print(f"  Advantage in losing years: {best['difference']:+.1%}")
print(f"  Losing years: {best['losing_return']:+.1%} ({best['losing_trades']:.0f} trades, {best['losing_win']:.0%} win)")
print(f"  Winning year: {best['winning_return']:+.1%} ({best['winning_trades']:.0f} trades, {best['winning_win']:.0%} win)")
print()

if best['difference'] > 0.50:
    print("STRONG COMPLEMENT: This target significantly outperforms in losing years.")
    print("Next step: Full backtest with proper risk management (stops, position sizing).")
else:
    print("MARGINAL: Small edge, may not be worth implementing.")
