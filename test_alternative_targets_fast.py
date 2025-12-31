"""
FAST TEST: ALTERNATIVE TARGETS IN LOSING YEARS
===============================================
Train models for 4 promising targets, test on 2018-2021 only.

Targets:
1. Mean reversion to EMA20
2. Range-bound (stay in range)
3. RSI reversion
4. Volatility contraction

Test Period: 2018-2021 (4 years)
- Losing years: 2018, 2020, 2021
- Winning year: 2019
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
import os
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("FAST TEST: ALTERNATIVE TARGETS")
print("="*100)
print()

DATA_DIR = 'data'
PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY']
LOSING_YEARS = [2018, 2020, 2021]
WINNING_YEARS = [2019]
TEST_YEARS = [2018, 2019, 2020, 2021]

INITIAL_CAPITAL = 100000
RISK_PER_TRADE = 0.005
MIN_CONFIDENCE = 0.70

def calculate_features(df):
    """Calculate technical features"""
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

def create_targets(df):
    """Create 4 key targets"""
    targets = {}

    future_close_5d = df['close'].shift(-5)

    # 1. Mean reversion to EMA20
    ema_20 = df['ema_20']
    distance_from_ema = df['close'] / ema_20 - 1
    future_distance = future_close_5d / ema_20 - 1
    targets['mean_revert_ema20'] = (abs(future_distance) < abs(distance_from_ema) * 0.7).astype(int)

    # 2. Range-bound
    future_high_5d = df['high'].rolling(5).max().shift(-5)
    future_low_5d = df['low'].rolling(5).min().shift(-5)
    targets['range_bound'] = (
        (future_high_5d <= df['high_20d'] * 1.002) &
        (future_low_5d >= df['low_20d'] * 0.998)
    ).astype(int)

    # 3. RSI reversion
    rsi = df['rsi']
    price_change_5d = (future_close_5d - df['close']) / df['close']
    targets['rsi_reversion'] = (
        ((rsi < 30) & (price_change_5d > 0)) |
        ((rsi > 70) & (price_change_5d < 0))
    ).astype(int)

    # 4. Volatility contraction
    current_vol = df['volatility_10d']
    future_vol = df['volatility_10d'].shift(-10)
    targets['vol_contraction'] = (future_vol < current_vol * 0.8).astype(int)

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
    """Train fast XGBoost model"""
    print(f"  Training '{target_name}'...")

    X_train_list = []
    y_train_list = []

    for pair_data in all_pairs_data.values():
        # Train on 2016-2017 only (fast)
        train_data = pair_data[(pair_data['year'] >= 2016) & (pair_data['year'] <= 2017)].copy()
        train_data = train_data.dropna(subset=FEATURE_COLS + [target_name])

        if len(train_data) > 50:
            X_train_list.append(train_data[FEATURE_COLS])
            y_train_list.append(train_data[target_name])

    if len(X_train_list) == 0:
        return None

    X_train = pd.concat(X_train_list)
    y_train = pd.concat(y_train_list)

    # Fast XGBoost settings
    model = xgb.XGBClassifier(
        n_estimators=30,      # Much faster
        max_depth=3,          # Shallower trees
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    )

    model.fit(X_train, y_train, verbose=False)

    y_pred = model.predict(X_train)
    accuracy = accuracy_score(y_train, y_pred)
    print(f"    Accuracy: {accuracy:.1%}, samples: {len(y_train)}")

    return model

def backtest_simple(target_name, model, all_pairs_data, test_years):
    """Simple backtest - just track if predictions are profitable"""
    capital = INITIAL_CAPITAL
    trades = []

    for pair, pair_data in all_pairs_data.items():
        test_data = pair_data[pair_data['year'].isin(test_years)].copy()
        test_data = test_data.dropna(subset=FEATURE_COLS + [target_name])

        for idx, row in test_data.iterrows():
            # Get prediction
            X = row[FEATURE_COLS].values.reshape(1, -1)
            prob = model.predict_proba(X)[0, 1]

            if prob > MIN_CONFIDENCE:
                # Simple P&L: if prediction correct, gain 0.5%, else lose 0.3%
                actual = row[target_name]
                if actual == 1:
                    profit_pct = 0.005
                else:
                    profit_pct = -0.003

                risk_amount = capital * RISK_PER_TRADE
                position_value = risk_amount / 0.02
                profit_dollars = profit_pct * position_value
                capital += profit_dollars

                trades.append({
                    'year': row['year'],
                    'profit_pct': profit_pct
                })

    return capital, trades

# Load data
print("Loading data...")
all_pairs_data = {}

for pair in PAIRS:
    file_path = os.path.join(DATA_DIR, f'{pair}_1day_oanda.csv')
    if not os.path.exists(file_path):
        continue

    df = pd.read_csv(file_path, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df['year'] = pd.to_datetime(df['date']).dt.year

    # Only keep data we need (2016-2021)
    df = df[(df['year'] >= 2016) & (df['year'] <= 2021)].copy()

    df = calculate_features(df)
    targets = create_targets(df)
    for target_name, target_series in targets.items():
        df[target_name] = target_series

    all_pairs_data[pair] = df

print(f"Loaded {len(all_pairs_data)} pairs")
print()

# Test 4 targets
print("="*100)
print("TRAINING AND TESTING 4 KEY TARGETS (2018-2021)")
print("="*100)
print()

target_names = ['mean_revert_ema20', 'range_bound', 'rsi_reversion', 'vol_contraction']
results = []

for target_name in target_names:
    print(f"{target_name}:")

    model = train_model(target_name, all_pairs_data)
    if model is None:
        print("  Skipped - insufficient data")
        continue

    # Test on losing years
    losing_cap, losing_trades = backtest_simple(target_name, model, all_pairs_data, LOSING_YEARS)
    losing_return = (losing_cap - INITIAL_CAPITAL) / INITIAL_CAPITAL
    losing_win_rate = sum(1 for t in losing_trades if t['profit_pct'] > 0) / len(losing_trades) if losing_trades else 0

    # Test on winning year
    winning_cap, winning_trades = backtest_simple(target_name, model, all_pairs_data, WINNING_YEARS)
    winning_return = (winning_cap - INITIAL_CAPITAL) / INITIAL_CAPITAL
    winning_win_rate = sum(1 for t in winning_trades if t['profit_pct'] > 0) / len(winning_trades) if winning_trades else 0

    print(f"  Losing years (2018,2020,2021): {losing_return:+.1%} ({len(losing_trades)} trades, {losing_win_rate:.0%} win)")
    print(f"  Winning year (2019):            {winning_return:+.1%} ({len(winning_trades)} trades, {winning_win_rate:.0%} win)")
    print()

    results.append({
        'target': target_name,
        'losing_return': losing_return,
        'losing_trades': len(losing_trades),
        'losing_win_rate': losing_win_rate,
        'winning_return': winning_return,
        'winning_trades': len(winning_trades),
        'winning_win_rate': winning_win_rate,
        'difference': losing_return - winning_return
    })

# Summary
print("="*100)
print("RESULTS")
print("="*100)
print()

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('difference', ascending=False)

print(f"{'Target':<22} {'Losing Yrs':>12} {'Trades':>7} {'Win%':>5}   {'Win Year':>12} {'Trades':>7} {'Win%':>5}   {'Diff':>7}")
print("-" * 95)

for _, row in results_df.iterrows():
    print(f"{row['target']:<22} {row['losing_return']:>11.1%} {row['losing_trades']:>7.0f} {row['losing_win_rate']:>4.0%}   "
          f"{row['winning_return']:>11.1%} {row['winning_trades']:>7.0f} {row['winning_win_rate']:>4.0%}   "
          f"{row['difference']:>6.1%}")

print()
print("="*100)
print("CONCLUSION")
print("="*100)
print()

best = results_df.iloc[0]
if best['difference'] > 0.10:
    print(f"✓ FOUND COMPLEMENT: '{best['target']}' performs {best['difference']:+.1%} better in losing years!")
    print(f"  This could help offset breakout strategy losses in choppy markets.")
elif best['difference'] > 0:
    print(f"MARGINAL: '{best['target']}' performs {best['difference']:+.1%} better in losing years.")
    print(f"  Small edge, may not be worth the complexity.")
else:
    print("NO COMPLEMENT FOUND: All targets perform worse (or equal) in losing years.")
    print("Losing years are fundamentally harder to predict across all patterns.")
    print()
    print("Recommendation: Accept 50% win rate, focus on position sizing and stops.")
