"""
COMPREHENSIVE TARGET PROFITABILITY TEST
========================================
Test ALL alternative targets with actual trading to find profitable patterns.

Tests 10 targets:
1. mean_revert_ema20 - Mean reversion to EMA20
2. bb_bounce - Bollinger Band bounce
3. rsi_reversion - Oversold/overbought reversions
4. range_bound - Stay in range
5. tight_range - Volatility contraction
6. vol_contraction - Volatility decreases
7. vol_expansion - Volatility increases
8. failed_breakout_up - Failed upward breakout
9. failed_breakout_down - Failed downward breakout
10. consolidation - Price consolidates after big move

Goal: Find targets that are PROFITABLE in losing years (2018, 2020, 2021)
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("COMPREHENSIVE TARGET PROFITABILITY TEST")
print("="*100)
print()

DATA_DIR = 'data'
PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY']
LOSING_YEARS = [2018, 2020, 2021]
WINNING_YEARS = [2019]

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

def create_all_targets(df):
    """Create 10 prediction targets"""
    targets = {}

    future_close_3d = df['close'].shift(-3)
    future_close_5d = df['close'].shift(-5)
    future_close_7d = df['close'].shift(-7)

    # 1. Mean reversion to EMA20
    ema_20 = df['ema_20']
    distance_from_ema = df['close'] / ema_20 - 1
    future_distance = future_close_5d / ema_20 - 1
    targets['mean_revert_ema20'] = (abs(future_distance) < abs(distance_from_ema) * 0.7).astype(int)

    # 2. Bollinger Band bounce
    bb_position = df['bb_position']
    future_bb_position = (future_close_3d - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
    targets['bb_bounce'] = (
        ((bb_position < 0.2) & (future_bb_position > 0.3)) |
        ((bb_position > 0.8) & (future_bb_position < 0.7))
    ).astype(int)

    # 3. RSI reversion
    rsi = df['rsi']
    price_change_5d = (future_close_5d - df['close']) / df['close']
    targets['rsi_reversion'] = (
        ((rsi < 30) & (price_change_5d > 0)) |
        ((rsi > 70) & (price_change_5d < 0))
    ).astype(int)

    # 4. Range-bound
    future_high_5d = df['high'].rolling(5).max().shift(-5)
    future_low_5d = df['low'].rolling(5).min().shift(-5)
    targets['range_bound'] = (
        (future_high_5d <= df['high_20d'] * 1.002) &
        (future_low_5d >= df['low_20d'] * 0.998)
    ).astype(int)

    # 5. Tight range (volatility contraction in price)
    future_high_10d = df['high'].rolling(10).max().shift(-10)
    future_low_10d = df['low'].rolling(10).min().shift(-10)
    future_range = (future_high_10d - future_low_10d) / df['close']
    targets['tight_range'] = (future_range < df['range_20d'] * 0.8).astype(int)

    # 6. Volatility contraction (vol decreases)
    current_vol = df['volatility_10d']
    future_vol = df['volatility_10d'].shift(-10)
    targets['vol_contraction'] = (future_vol < current_vol * 0.8).astype(int)

    # 7. Volatility expansion (vol increases)
    targets['vol_expansion'] = (future_vol > current_vol * 1.2).astype(int)

    # 8. Failed upward breakout
    distance_to_high = (df['high_20d'] - df['close']) / df['close']
    targets['failed_breakout_up'] = (
        (distance_to_high < 0.01) &
        (future_high_5d < df['high_20d'] * 1.002) &
        (future_close_5d < df['close'])
    ).astype(int)

    # 9. Failed downward breakout
    distance_to_low = (df['close'] - df['low_20d']) / df['close']
    targets['failed_breakout_down'] = (
        (distance_to_low < 0.01) &
        (future_low_5d > df['low_20d'] * 0.998) &
        (future_close_5d > df['close'])
    ).astype(int)

    # 10. Consolidation after big move
    recent_move = abs(df['return_5d'])
    future_range_7d = (df['high'].rolling(7).max().shift(-7) - df['low'].rolling(7).min().shift(-7)) / df['close']
    targets['consolidation'] = (
        (recent_move > 0.015) &
        (future_range_7d < 0.01)
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

    # Fast model
    model = xgb.XGBClassifier(
        n_estimators=50,
        max_depth=4,
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
    """Simple backtest - fixed gains per correct/wrong prediction"""
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
                if actual == 1:
                    profit_pct = 0.005  # +0.5%
                else:
                    profit_pct = -0.003  # -0.3%

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
print("Loading data for 8 pairs...")
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
    targets = create_all_targets(df)
    for target_name, target_series in targets.items():
        df[target_name] = target_series

    all_pairs_data[pair] = df

print(f"Loaded {len(all_pairs_data)} pairs")
print()

# Test all targets
print("="*100)
print("TRAINING AND TESTING ALL TARGETS")
print("="*100)
print()

target_names = [
    'mean_revert_ema20', 'bb_bounce', 'rsi_reversion', 'range_bound',
    'tight_range', 'vol_contraction', 'vol_expansion',
    'failed_breakout_up', 'failed_breakout_down', 'consolidation'
]

results = []

for target_name in target_names:
    print(f"Testing {target_name}...")

    model = train_model(target_name, all_pairs_data)
    if model is None:
        print(f"  Skipped - insufficient data")
        print()
        continue

    # Test on losing years
    losing_cap, losing_trades = backtest_simple(target_name, model, all_pairs_data, LOSING_YEARS)
    losing_return = (losing_cap - INITIAL_CAPITAL) / INITIAL_CAPITAL
    losing_win = sum(1 for t in losing_trades if t['profit_pct'] > 0) / len(losing_trades) if losing_trades else 0

    # Test on winning year
    winning_cap, winning_trades = backtest_simple(target_name, model, all_pairs_data, WINNING_YEARS)
    winning_return = (winning_cap - INITIAL_CAPITAL) / INITIAL_CAPITAL
    winning_win = sum(1 for t in winning_trades if t['profit_pct'] > 0) / len(winning_trades) if winning_trades else 0

    print(f"  Losing years: {losing_return:+6.1%} ({len(losing_trades):>4} trades, {losing_win:>3.0%} win)")
    print(f"  Winning year: {winning_return:+6.1%} ({len(winning_trades):>4} trades, {winning_win:>3.0%} win)")
    print()

    results.append({
        'target': target_name,
        'losing_return': losing_return,
        'losing_trades': len(losing_trades),
        'losing_win': losing_win,
        'winning_return': winning_return,
        'winning_trades': len(winning_trades),
        'winning_win': winning_win,
        'edge': losing_return - winning_return
    })

# Summary
print("="*100)
print("RESULTS RANKED BY EDGE IN LOSING YEARS")
print("="*100)
print()

results_df = pd.DataFrame(results).sort_values('edge', ascending=False)

print(f"{'Target':<22} {'Losing Yrs':>11} {'Trades':>7} {'Win%':>5}   {'Win Year':>11} {'Trades':>7} {'Win%':>5}   {'Edge':>7}")
print("-" * 95)

for _, row in results_df.iterrows():
    print(f"{row['target']:<22} {row['losing_return']:>10.1%} {row['losing_trades']:>7.0f} {row['losing_win']:>4.0%}   "
          f"{row['winning_return']:>10.1%} {row['winning_trades']:>7.0f} {row['winning_win']:>4.0%}   "
          f"{row['edge']:>6.1%}")

print()
print("="*100)
print("TOP CANDIDATES")
print("="*100)
print()

# Find targets with strong edge in losing years
top_targets = results_df[results_df['edge'] > 0.30].head(3)

if len(top_targets) > 0:
    print("These targets show strong EDGE in losing years:")
    print()
    for idx, row in top_targets.iterrows():
        print(f"{idx+1}. {row['target']}")
        print(f"   Edge: {row['edge']:+.1%} (losing yrs: {row['losing_return']:+.1%}, winning yr: {row['winning_return']:+.1%})")
        print(f"   Trades: {row['losing_trades']:.0f} in losing years, {row['winning_trades']:.0f} in winning year")
        print(f"   Win rate: {row['losing_win']:.0%} in losing years, {row['winning_win']:.0%} in winning year")
        print()

    print("NEXT STEPS:")
    print("1. Test top target with PROPER backtest (real entry/exit logic)")
    print("2. If profitable with real trading, test on full 2016-2025 period")
    print("3. Consider combining with breakout strategy")
else:
    print("No targets found with >30% edge in losing years.")
    print()
    # Show best 3 anyway
    print("Top 3 targets by edge:")
    for idx, row in results_df.head(3).iterrows():
        print(f"  {row['target']}: {row['edge']:+.1%} edge")
        print(f"    Losing years: {row['losing_return']:+.1%} ({row['losing_trades']:.0f} trades)")
        print(f"    Winning year: {row['winning_return']:+.1%} ({row['winning_trades']:.0f} trades)")
        print()

print()
print("="*100)
print("IMPORTANT NOTE")
print("="*100)
print()
print("These results use SIMPLIFIED backtest (fixed +0.5%/-0.3% per trade).")
print("High returns here don't guarantee profitability with real trading logic.")
print("Always validate top candidates with proper backtest before trusting results.")
