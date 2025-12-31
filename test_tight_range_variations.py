"""
TEST TIGHT_RANGE VARIATIONS
============================
Based on analysis:
- Movements are TINY (avg +0.23% at bottom, -0.05% at top)
- 0.7% target is too large
- Many time exits (36%)

Test:
1. Tighter targets (0.3%, 0.4%, 0.5%) with matched stops
2. No stops, time exit only (various targets)
3. Asymmetric: small target, no stop
4. Higher confidence thresholds
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("TESTING TIGHT_RANGE VARIATIONS")
print("="*100)
print()

DATA_DIR = 'data'
PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY']
LOSING_YEARS = [2018, 2020, 2021]
WINNING_YEARS = [2019]

INITIAL_CAPITAL = 100000
RISK_PER_TRADE = 0.005
MIN_CONFIDENCE = 0.70
COOLDOWN_DAYS = 1
MAX_HOLD_DAYS = 5

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

def create_tight_range_target(df):
    """Create tight_range target"""
    future_high_10d = df['high'].rolling(10).max().shift(-10)
    future_low_10d = df['low'].rolling(10).min().shift(-10)
    future_range = (future_high_10d - future_low_10d) / df['close']

    df['tight_range'] = (future_range < df['range_20d'] * 0.8).astype(int)

    return df

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

def train_model(all_pairs_data):
    """Train tight_range model"""
    X_train_list = []
    y_train_list = []

    for pair_data in all_pairs_data.values():
        train_data = pair_data[(pair_data['year'] >= 2016) & (pair_data['year'] <= 2017)].copy()
        train_data = train_data.dropna(subset=FEATURE_COLS + ['tight_range'])

        if len(train_data) > 50:
            X_train_list.append(train_data[FEATURE_COLS])
            y_train_list.append(train_data['tight_range'])

    X_train = pd.concat(X_train_list)
    y_train = pd.concat(y_train_list)

    model = xgb.XGBClassifier(
        n_estimators=100,
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

def backtest_variation(model, all_pairs_data, test_years, target_pct, stop_pct):
    """
    Backtest with specific target/stop configuration
    stop_pct = None means no stop loss
    """
    capital = INITIAL_CAPITAL
    trades = []
    open_positions = {}
    last_trade_date = {}

    # Combine all data
    all_data = []
    for pair, pair_data in all_pairs_data.items():
        test_data = pair_data[pair_data['year'].isin(test_years)].copy()
        test_data = test_data.dropna(subset=FEATURE_COLS + ['tight_range'])
        test_data['pair'] = pair
        all_data.append(test_data)

    combined_data = pd.concat(all_data).sort_values('date').reset_index(drop=True)

    for idx, row in combined_data.iterrows():
        pair = row['pair']
        current_date = row['date']
        current_price = row['close']

        # Entry
        if pair not in open_positions:
            # Cooldown
            if pair in last_trade_date:
                days_since_last = (current_date - last_trade_date[pair]).days
                if days_since_last < COOLDOWN_DAYS:
                    continue

            # Get prediction
            X = row[FEATURE_COLS].values.reshape(1, -1)
            prob = model.predict_proba(X)[0, 1]

            if prob > MIN_CONFIDENCE:
                position_in_range = row['position_in_range']

                # Entry at edges
                if position_in_range < 0.3:
                    direction = 'long'
                elif position_in_range > 0.7:
                    direction = 'short'
                else:
                    continue

                # Calculate prices
                if direction == 'long':
                    target_price = current_price * (1 + target_pct)
                    stop_price = current_price * (1 + stop_pct) if stop_pct is not None else None
                else:
                    target_price = current_price * (1 - target_pct)
                    stop_price = current_price * (1 - stop_pct) if stop_pct is not None else None

                # Position sizing
                risk_amount = capital * RISK_PER_TRADE
                if stop_pct is not None:
                    stop_distance = abs(current_price * abs(stop_pct))
                else:
                    # No stop: use target as risk proxy
                    stop_distance = abs(current_price * abs(target_pct))
                position_size = risk_amount / stop_distance

                open_positions[pair] = {
                    'entry_date': current_date,
                    'entry_price': current_price,
                    'direction': direction,
                    'target_price': target_price,
                    'stop_price': stop_price,
                    'position_size': position_size,
                    'days_held': 0
                }

        # Check exits
        if pair in open_positions:
            pos = open_positions[pair]
            pos['days_held'] += 1

            exit_triggered = False
            exit_reason = None

            # 1. Target
            if pos['direction'] == 'long' and current_price >= pos['target_price']:
                exit_triggered = True
                exit_reason = 'target'
            elif pos['direction'] == 'short' and current_price <= pos['target_price']:
                exit_triggered = True
                exit_reason = 'target'

            # 2. Stop (if exists)
            if not exit_triggered and pos['stop_price'] is not None:
                if pos['direction'] == 'long' and current_price <= pos['stop_price']:
                    exit_triggered = True
                    exit_reason = 'stop'
                elif pos['direction'] == 'short' and current_price >= pos['stop_price']:
                    exit_triggered = True
                    exit_reason = 'stop'

            # 3. Time
            if not exit_triggered and pos['days_held'] >= MAX_HOLD_DAYS:
                exit_triggered = True
                exit_reason = 'time'

            if exit_triggered:
                if pos['direction'] == 'long':
                    pnl_pct = (current_price - pos['entry_price']) / pos['entry_price']
                else:
                    pnl_pct = (pos['entry_price'] - current_price) / pos['entry_price']

                profit_dollars = pnl_pct * pos['position_size'] * pos['entry_price']
                capital += profit_dollars

                trades.append({
                    'pnl_pct': pnl_pct,
                    'exit_reason': exit_reason
                })

                last_trade_date[pair] = current_date
                del open_positions[pair]

    # Close remaining
    for pair, pos in open_positions.items():
        exit_price = combined_data[combined_data['pair'] == pair].iloc[-1]['close']

        if pos['direction'] == 'long':
            pnl_pct = (exit_price - pos['entry_price']) / pos['entry_price']
        else:
            pnl_pct = (pos['entry_price'] - exit_price) / pos['entry_price']

        profit_dollars = pnl_pct * pos['position_size'] * pos['entry_price']
        capital += profit_dollars

        trades.append({
            'pnl_pct': pnl_pct,
            'exit_reason': 'final'
        })

    # Calculate metrics
    if len(trades) > 0:
        trades_df = pd.DataFrame(trades)
        total_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL
        num_trades = len(trades_df)
        win_rate = (trades_df['pnl_pct'] > 0).sum() / num_trades

        return {
            'total_return': total_return,
            'num_trades': num_trades,
            'win_rate': win_rate
        }
    else:
        return None

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
    df = df[(df['year'] >= 2016) & (df['year'] <= 2021)].copy()

    df = calculate_features(df)
    df = create_tight_range_target(df)

    all_pairs_data[pair] = df

print(f"Loaded {len(all_pairs_data)} pairs")
print()

# Train model
print("Training model...")
model = train_model(all_pairs_data)
print()

# Test configurations
print("="*100)
print("TESTING CONFIGURATIONS")
print("="*100)
print()

configs = [
    # Tighter targets with matched stops
    (0.003, -0.003, "Tight: 0.3% target, -0.3% stop"),
    (0.004, -0.004, "Tight: 0.4% target, -0.4% stop"),
    (0.005, -0.005, "Tight: 0.5% target, -0.5% stop"),

    # No stops (time exit only)
    (0.003, None, "No stop: 0.3% target, time exit"),
    (0.004, None, "No stop: 0.4% target, time exit"),
    (0.005, None, "No stop: 0.5% target, time exit"),
    (0.006, None, "No stop: 0.6% target, time exit"),
    (0.007, None, "No stop: 0.7% target, time exit"),

    # Original (for comparison)
    (0.007, -0.007, "Original: 0.7% / -0.7%"),
]

results = []

for target_pct, stop_pct, label in configs:
    losing_result = backtest_variation(model, all_pairs_data, LOSING_YEARS, target_pct, stop_pct)
    winning_result = backtest_variation(model, all_pairs_data, WINNING_YEARS, target_pct, stop_pct)

    if losing_result and winning_result:
        edge = losing_result['total_return'] - winning_result['total_return']

        results.append({
            'config': label,
            'target': target_pct,
            'stop': stop_pct,
            'losing_return': losing_result['total_return'],
            'losing_trades': losing_result['num_trades'],
            'losing_winrate': losing_result['win_rate'],
            'winning_return': winning_result['total_return'],
            'winning_trades': winning_result['num_trades'],
            'winning_winrate': winning_result['win_rate'],
            'edge': edge
        })

# Print results
results_df = pd.DataFrame(results).sort_values('edge', ascending=False)

print("\nResults sorted by edge (losing - winning):")
print("-" * 100)
print(f"{'Config':<35} {'Losing Yrs':<12} {'Trades':<8} {'Win%':<8} {'Win Year':<12} {'Trades':<8} {'Win%':<8} {'Edge':<8}")
print("-" * 100)

for _, row in results_df.iterrows():
    print(f"{row['config']:<35} "
          f"{row['losing_return']:>10.1%}  "
          f"{row['losing_trades']:>6}  "
          f"{row['losing_winrate']:>6.0%}  "
          f"{row['winning_return']:>10.1%}  "
          f"{row['winning_trades']:>6}  "
          f"{row['winning_winrate']:>6.0%}  "
          f"{row['edge']:>7.1%}")

print()
print("="*100)
print("BEST CONFIGURATION")
print("="*100)
print()

best = results_df.iloc[0]
print(f"Config: {best['config']}")
print(f"Losing years: {best['losing_return']:+.1%} ({best['losing_trades']} trades, {best['losing_winrate']:.0%} win)")
print(f"Winning year: {best['winning_return']:+.1%} ({best['winning_trades']} trades, {best['winning_winrate']:.0%} win)")
print(f"Edge: {best['edge']:+.1%}")
print()

if best['edge'] > 0.10 and best['losing_return'] > 0.10:
    print("SUCCESS! Found profitable tight_range scalping configuration!")
elif best['losing_return'] > 0:
    print("Profitable but marginal edge")
else:
    print("Still unprofitable - tight_range may not be tradeable directly")
    print("Consider using as filter for breakout strategy instead")
