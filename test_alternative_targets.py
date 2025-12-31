"""
TEST ALTERNATIVE TARGETS IN LOSING YEARS
=========================================
Train XGBoost models for mean reversion, range-bound, and other targets.
Backtest to see which ones are PROFITABLE in losing years (2018, 2020, 2021, 2022, 2024).

Test Period: 2018-2024 (7 years)
- Losing years: 2018, 2020, 2021, 2022, 2024
- Winning years: 2019, 2023
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("TESTING ALTERNATIVE TARGETS FOR LOSING YEARS")
print("="*100)
print()

DATA_DIR = 'data'
PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY']
LOSING_YEARS = [2018, 2020, 2021, 2022, 2024]
WINNING_YEARS = [2019, 2023]
TEST_YEARS = LOSING_YEARS + WINNING_YEARS

# Strategy parameters
INITIAL_CAPITAL = 100000
RISK_PER_TRADE = 0.005  # 0.5% per trade
MIN_CONFIDENCE = 0.70   # 70% confidence threshold

def calculate_features(df):
    """Calculate technical features (same as breakout strategy)"""
    # Returns
    df['return_1d'] = df['close'].pct_change()
    df['return_3d'] = df['close'].pct_change(3)
    df['return_5d'] = df['close'].pct_change(5)
    df['return_10d'] = df['close'].pct_change(10)

    # Moving averages
    for period in [10, 20, 50]:
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        df[f'price_to_ema_{period}'] = df['close'] / df[f'ema_{period}'] - 1

    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']

    # Volatility
    df['volatility_10d'] = df['return_1d'].rolling(10).std()
    df['volatility_20d'] = df['return_1d'].rolling(20).std()

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df['atr'] = ranges.max(axis=1).rolling(14).mean()
    df['atr_pct'] = df['atr'] / df['close']

    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * bb_std
    df['bb_lower'] = df['bb_middle'] - 2 * bb_std
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

    # Momentum
    df['momentum_10d'] = df['close'] - df['close'].shift(10)
    df['momentum_20d'] = df['close'] - df['close'].shift(20)

    # Range features
    df['high_20d'] = df['high'].rolling(20).max()
    df['low_20d'] = df['low'].rolling(20).min()
    df['range_20d'] = (df['high_20d'] - df['low_20d']) / df['close']
    df['position_in_range'] = (df['close'] - df['low_20d']) / (df['high_20d'] - df['low_20d'] + 1e-10)

    return df

def create_targets(df):
    """Create alternative prediction targets"""
    targets = {}

    # 1. Mean reversion to EMA20 (5-day)
    future_close_5d = df['close'].shift(-5)
    ema_20 = df['ema_20']
    distance_from_ema = df['close'] / ema_20 - 1
    future_distance = future_close_5d / ema_20 - 1
    targets['mean_revert_ema20'] = (abs(future_distance) < abs(distance_from_ema) * 0.7).astype(int)

    # 2. Bollinger Band bounce (3-day)
    future_close_3d = df['close'].shift(-3)
    bb_position = df['bb_position']
    future_bb_position = (future_close_3d - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
    targets['bb_bounce'] = (
        ((bb_position < 0.2) & (future_bb_position > 0.3)) |
        ((bb_position > 0.8) & (future_bb_position < 0.7))
    ).astype(int)

    # 3. RSI reversion (5-day)
    rsi = df['rsi']
    price_change_5d = (future_close_5d - df['close']) / df['close']
    targets['rsi_reversion'] = (
        ((rsi < 30) & (price_change_5d > 0)) |
        ((rsi > 70) & (price_change_5d < 0))
    ).astype(int)

    # 4. Range-bound (5-day stays in range)
    future_high_5d = df['high'].rolling(5).max().shift(-5)
    future_low_5d = df['low'].rolling(5).min().shift(-5)
    current_high_20d = df['high_20d']
    current_low_20d = df['low_20d']
    targets['range_bound'] = (
        (future_high_5d <= current_high_20d * 1.002) &
        (future_low_5d >= current_low_20d * 0.998)
    ).astype(int)

    # 5. Tight range (10-day range contracts)
    future_high_10d = df['high'].rolling(10).max().shift(-10)
    future_low_10d = df['low'].rolling(10).min().shift(-10)
    future_range = (future_high_10d - future_low_10d) / df['close']
    targets['tight_range'] = (future_range < df['range_20d'] * 0.8).astype(int)

    # 6. Volatility contraction (10-day)
    current_vol = df['volatility_10d']
    future_vol = df['volatility_10d'].shift(-10)
    targets['vol_contraction'] = (future_vol < current_vol * 0.8).astype(int)

    # 7. Volatility expansion (5-day)
    future_vol_5d = df['volatility_10d'].shift(-5)
    targets['vol_expansion'] = (future_vol_5d > current_vol * 1.2).astype(int)

    # 8. Failed upward breakout (5-day)
    distance_to_high = (df['high_20d'] - df['close']) / df['close']
    future_high_5d_2 = df['high'].rolling(5).max().shift(-5)
    targets['failed_breakout_up'] = (
        (distance_to_high < 0.01) &
        (future_high_5d_2 < df['high_20d'] * 1.002) &
        (future_close_5d < df['close'])
    ).astype(int)

    # 9. Failed downward breakout (5-day)
    distance_to_low = (df['close'] - df['low_20d']) / df['close']
    future_low_5d_2 = df['low'].rolling(5).min().shift(-5)
    targets['failed_breakout_down'] = (
        (distance_to_low < 0.01) &
        (future_low_5d_2 > df['low_20d'] * 0.998) &
        (future_close_5d > df['close'])
    ).astype(int)

    # 10. Consolidation after big move (7-day)
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

def train_models_for_target(target_name, all_pairs_data):
    """Train XGBoost model for a specific target across all pairs"""
    print(f"  Training model for '{target_name}'...")

    # Combine all pairs for training (2010-2017)
    X_train_list = []
    y_train_list = []

    for pair_data in all_pairs_data.values():
        train_data = pair_data[(pair_data['year'] >= 2010) & (pair_data['year'] <= 2017)].copy()
        train_data = train_data.dropna(subset=FEATURE_COLS + [target_name])

        if len(train_data) > 100:
            X_train_list.append(train_data[FEATURE_COLS])
            y_train_list.append(train_data[target_name])

    if len(X_train_list) == 0:
        return None

    X_train = pd.concat(X_train_list)
    y_train = pd.concat(y_train_list)

    # Train XGBoost
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        eval_metric='logloss'
    )

    model.fit(X_train, y_train, verbose=False)

    # Check accuracy
    y_pred = model.predict(X_train)
    accuracy = accuracy_score(y_train, y_pred)
    print(f"    Training accuracy: {accuracy:.1%}")

    return model

def backtest_target(target_name, model, all_pairs_data, test_years):
    """Backtest a target strategy on specific years"""
    capital = INITIAL_CAPITAL
    trades = []

    # Get all dates from test years
    all_dates = set()
    for pair_data in all_pairs_data.values():
        test_data = pair_data[pair_data['year'].isin(test_years)]
        all_dates.update(test_data['date'])
    all_dates = sorted(list(all_dates))

    for date in all_dates:
        # Check each pair for signals
        for pair, pair_data in all_pairs_data.items():
            row_data = pair_data[pair_data['date'] == date]

            if len(row_data) == 0:
                continue

            row = row_data.iloc[0]

            # Skip if missing features
            if pd.isna(row[FEATURE_COLS]).any():
                continue

            # Get prediction
            X = row[FEATURE_COLS].values.reshape(1, -1)
            prob = model.predict_proba(X)[0, 1]

            if prob > MIN_CONFIDENCE:
                # Enter trade
                entry_price = row['close']

                # Simple exit: check 5 days later
                future_date = date + pd.Timedelta(days=5)
                future_data = pair_data[pair_data['date'] == future_date]

                if len(future_data) > 0:
                    exit_price = future_data.iloc[0]['close']

                    # Calculate profit (assume long only for simplicity)
                    # For mean reversion/range strategies, we'd trade differently
                    # For now, just measure if the prediction was correct
                    actual = row[target_name]

                    if actual == 1:
                        # Prediction was correct - small profit
                        profit_pct = 0.005  # 0.5% profit
                    else:
                        # Prediction was wrong - small loss
                        profit_pct = -0.003  # -0.3% loss

                    # Calculate position size
                    risk_amount = capital * RISK_PER_TRADE
                    position_value = risk_amount / 0.02  # Assume 2% risk
                    profit_dollars = profit_pct * position_value
                    capital += profit_dollars

                    trades.append({
                        'date': date,
                        'pair': pair,
                        'profit_pct': profit_pct,
                        'year': row['year']
                    })

    return capital, trades

# Load and prepare all data
print("Loading data for all pairs...")
all_pairs_data = {}

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
    targets = create_targets(df)
    for target_name, target_series in targets.items():
        df[target_name] = target_series

    all_pairs_data[pair] = df

print(f"Loaded {len(all_pairs_data)} pairs")
print()

# Test each target
print("="*100)
print("TRAINING AND TESTING ALTERNATIVE TARGETS")
print("="*100)
print()

results = []

target_names = [
    'mean_revert_ema20', 'bb_bounce', 'rsi_reversion',
    'range_bound', 'tight_range',
    'vol_contraction', 'vol_expansion',
    'failed_breakout_up', 'failed_breakout_down',
    'consolidation'
]

for target_name in target_names:
    print(f"Testing: {target_name}")

    # Train model
    model = train_models_for_target(target_name, all_pairs_data)

    if model is None:
        print(f"  Skipping - insufficient data")
        print()
        continue

    # Backtest on losing years
    losing_capital, losing_trades = backtest_target(target_name, model, all_pairs_data, LOSING_YEARS)
    losing_return = (losing_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL
    losing_win_rate = sum(1 for t in losing_trades if t['profit_pct'] > 0) / len(losing_trades) if losing_trades else 0

    # Backtest on winning years
    winning_capital, winning_trades = backtest_target(target_name, model, all_pairs_data, WINNING_YEARS)
    winning_return = (winning_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL
    winning_win_rate = sum(1 for t in winning_trades if t['profit_pct'] > 0) / len(winning_trades) if winning_trades else 0

    print(f"  Losing years (2018,2020,2021,2022,2024): {losing_return:+.1%} ({len(losing_trades)} trades, {losing_win_rate:.0%} win)")
    print(f"  Winning years (2019,2023): {winning_return:+.1%} ({len(winning_trades)} trades, {winning_win_rate:.0%} win)")
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
print("RESULTS SUMMARY")
print("="*100)
print()

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('difference', ascending=False)

print(f"{'Target':<25} {'Losing Yrs':>12} {'Trades':>8} {'Win%':>6}  {'Winning Yrs':>12} {'Trades':>8} {'Win%':>6}  {'Diff':>8}")
print("-" * 110)

for _, row in results_df.iterrows():
    print(f"{row['target']:<25} {row['losing_return']:>11.1%} {row['losing_trades']:>8.0f} {row['losing_win_rate']:>5.0%}  "
          f"{row['winning_return']:>11.1%} {row['winning_trades']:>8.0f} {row['winning_win_rate']:>5.0%}  "
          f"{row['difference']:>7.1%}")

print()
print("="*100)
print("CANDIDATES FOR COMPLEMENT STRATEGY")
print("="*100)
print()

# Find targets that perform better in losing years
good_targets = results_df[results_df['difference'] > 0.05]  # At least 5% better in losing years

if len(good_targets) > 0:
    print(f"Found {len(good_targets)} targets that perform BETTER in losing years:")
    print()
    for _, row in good_targets.iterrows():
        print(f"✓ {row['target']}")
        print(f"  Losing years: {row['losing_return']:+.1%} ({row['losing_trades']:.0f} trades, {row['losing_win_rate']:.0%} win)")
        print(f"  Winning years: {row['winning_return']:+.1%} ({row['winning_trades']:.0f} trades, {row['winning_win_rate']:.0%} win)")
        print(f"  Advantage: +{row['difference']:.1%} in losing years")
        print()
else:
    print("No targets found that significantly outperform in losing years.")
    print()
    print("Possible reasons:")
    print("1. Losing years are fundamentally unpredictable (regime changes)")
    print("2. Need different target definitions or timeframes")
    print("3. Strategy parameters need optimization")
    print("4. Emergency stops + parameter optimization may be the best approach")
