"""
ANALYZE SCALP DISCONNECT
=========================
Why is win rate only 52% when prediction accuracy is 85%?

Key questions:
1. Of the 85% correct tight_range predictions, how many would we ENTER (at edges)?
2. For entries we take, what's the actual directional outcome?
3. Is mean reversion (long at bottom, short at top) the right approach?
4. What if we did the OPPOSITE direction?
5. What if we entered ANYWHERE (not just edges)?
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("ANALYZING SCALP ENTRY DISCONNECT")
print("="*100)
print()

DATA_DIR = 'data'
PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY']
TEST_YEARS = [2018, 2020, 2021]  # Losing years
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
    print("Training tight_range model on 2016-2017...")

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
    print(f"  Trained on {len(y_train)} samples")
    print()

    return model

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
model = train_model(all_pairs_data)

# Analyze predictions
print("="*100)
print("ANALYZING PREDICTIONS")
print("="*100)
print()

all_predictions = []

for pair, pair_data in all_pairs_data.items():
    test_data = pair_data[pair_data['year'].isin(TEST_YEARS)].copy()
    test_data = test_data.dropna(subset=FEATURE_COLS + ['tight_range'])

    for idx in test_data.index[:-5]:  # Need future data
        row = test_data.loc[idx]

        # Get prediction
        X = row[FEATURE_COLS].values.reshape(1, -1)
        prob = model.predict_proba(X)[0, 1]

        if prob > MIN_CONFIDENCE:
            # Prediction made
            actual_tight_range = row['tight_range']
            prediction_correct = (actual_tight_range == 1)

            # Would we ENTER this trade? (edge filter from scalp strategy)
            position_in_range = row['position_in_range']
            at_bottom_edge = position_in_range < 0.3
            at_top_edge = position_in_range > 0.7
            would_enter = at_bottom_edge or at_top_edge

            # If entered, what direction?
            if at_bottom_edge:
                direction = 'long'
            elif at_top_edge:
                direction = 'short'
            else:
                direction = None

            # Get future price action (next 5 days for 0.7% target)
            future_prices = []
            for i in range(1, 6):
                if idx+i in test_data.index:
                    future_prices.append(test_data.loc[idx+i, 'close'])

            if len(future_prices) >= 5:
                entry_price = row['close']

                # Check outcomes for different strategies
                outcomes = {}

                # 1. Mean reversion (our current approach)
                if direction == 'long':
                    target_price = entry_price * 1.007
                    stop_price = entry_price * 0.993
                    outcome = None
                    for price in future_prices:
                        if price >= target_price:
                            outcome = 'win'
                            break
                        elif price <= stop_price:
                            outcome = 'loss'
                            break
                    if outcome is None:
                        outcome = 'time_exit'
                    outcomes['mean_revert'] = outcome

                elif direction == 'short':
                    target_price = entry_price * 0.993
                    stop_price = entry_price * 1.007
                    outcome = None
                    for price in future_prices:
                        if price <= target_price:
                            outcome = 'win'
                            break
                        elif price >= stop_price:
                            outcome = 'loss'
                            break
                    if outcome is None:
                        outcome = 'time_exit'
                    outcomes['mean_revert'] = outcome

                # 2. Momentum (opposite direction)
                if direction == 'long':
                    # Instead do SHORT
                    target_price = entry_price * 0.993
                    stop_price = entry_price * 1.007
                    outcome = None
                    for price in future_prices:
                        if price <= target_price:
                            outcome = 'win'
                            break
                        elif price >= stop_price:
                            outcome = 'loss'
                            break
                    if outcome is None:
                        outcome = 'time_exit'
                    outcomes['momentum'] = outcome

                elif direction == 'short':
                    # Instead do LONG
                    target_price = entry_price * 1.007
                    stop_price = entry_price * 0.993
                    outcome = None
                    for price in future_prices:
                        if price >= target_price:
                            outcome = 'win'
                            break
                        elif price <= stop_price:
                            outcome = 'loss'
                            break
                    if outcome is None:
                        outcome = 'time_exit'
                    outcomes['momentum'] = outcome

                # 3. Just track price movement
                final_price = future_prices[-1]
                price_change = (final_price - entry_price) / entry_price

                all_predictions.append({
                    'pair': pair,
                    'prob': prob,
                    'correct_prediction': prediction_correct,
                    'would_enter': would_enter,
                    'at_bottom': at_bottom_edge,
                    'at_top': at_top_edge,
                    'direction': direction,
                    'position_in_range': position_in_range,
                    'mean_revert_outcome': outcomes.get('mean_revert'),
                    'momentum_outcome': outcomes.get('momentum'),
                    'price_change_5d': price_change
                })

results = pd.DataFrame(all_predictions)

print(f"Total Predictions: {len(results)}")
print(f"Prediction Accuracy: {results['correct_prediction'].mean():.1%}")
print()

# Filter by entry
print("="*100)
print("ENTRY FILTER ANALYSIS")
print("="*100)
print()

entered = results[results['would_enter']]
not_entered = results[~results['would_enter']]

print(f"Would Enter (at edges): {len(entered)} ({len(entered)/len(results):.1%})")
print(f"Would NOT Enter (middle): {len(not_entered)} ({len(not_entered)/len(results):.1%})")
print()

print("Prediction Accuracy by Entry:")
print(f"  At edges: {entered['correct_prediction'].mean():.1%}")
print(f"  At middle: {not_entered['correct_prediction'].mean():.1%}")
print()

# Analyze entered trades
print("="*100)
print("ENTERED TRADES - DIRECTION ANALYSIS")
print("="*100)
print()

correct_and_entered = entered[entered['correct_prediction']]

print(f"Correct predictions that we'd enter: {len(correct_and_entered)}")
print()

# Mean reversion results
mean_revert_results = correct_and_entered['mean_revert_outcome'].value_counts()
print("MEAN REVERSION (Current Strategy):")
print(f"  Wins: {mean_revert_results.get('win', 0)}")
print(f"  Losses: {mean_revert_results.get('loss', 0)}")
print(f"  Time exits: {mean_revert_results.get('time_exit', 0)}")
if len(correct_and_entered) > 0:
    win_rate = mean_revert_results.get('win', 0) / len(correct_and_entered)
    print(f"  Win Rate: {win_rate:.1%}")
print()

# Momentum results (opposite)
momentum_results = correct_and_entered['momentum_outcome'].value_counts()
print("MOMENTUM (Opposite Direction):")
print(f"  Wins: {momentum_results.get('win', 0)}")
print(f"  Losses: {momentum_results.get('loss', 0)}")
print(f"  Time exits: {momentum_results.get('time_exit', 0)}")
if len(correct_and_entered) > 0:
    win_rate = momentum_results.get('win', 0) / len(correct_and_entered)
    print(f"  Win Rate: {win_rate:.1%}")
print()

# Direction bias
print("DIRECTIONAL BIAS (5-day price change):")
at_bottom = correct_and_entered[correct_and_entered['at_bottom']]
at_top = correct_and_entered[correct_and_entered['at_top']]

print(f"  At bottom edge (position < 0.3):")
print(f"    N = {len(at_bottom)}")
print(f"    Avg price change: {at_bottom['price_change_5d'].mean():+.2%}")
print(f"    % moved up: {(at_bottom['price_change_5d'] > 0).sum() / len(at_bottom):.1%}" if len(at_bottom) > 0 else "")
print()

print(f"  At top edge (position > 0.7):")
print(f"    N = {len(at_top)}")
print(f"    Avg price change: {at_top['price_change_5d'].mean():+.2%}")
print(f"    % moved down: {(at_top['price_change_5d'] < 0).sum() / len(at_top):.1%}" if len(at_top) > 0 else "")
print()

print("="*100)
print("MIDDLE ENTRIES - WHAT IF WE ENTERED ANYWHERE?")
print("="*100)
print()

# What if we entered ALL correct predictions?
print(f"Correct predictions NOT entered (middle positions): {len(not_entered[not_entered['correct_prediction']])}")
print()

# For middle positions, simulate entering long if in bottom half, short if top half
middle_entries = []
for idx, row in not_entered[not_entered['correct_prediction']].iterrows():
    pos = row['position_in_range']
    if pos < 0.5:
        direction = 'long'
    else:
        direction = 'short'

    # Would mean reversion work from middle?
    if direction == 'long' and row['price_change_5d'] > 0.007:
        outcome = 'win'
    elif direction == 'short' and row['price_change_5d'] < -0.007:
        outcome = 'win'
    else:
        outcome = 'loss'

    middle_entries.append(outcome)

if middle_entries:
    middle_win_rate = middle_entries.count('win') / len(middle_entries)
    print(f"Hypothetical win rate from middle entries: {middle_win_rate:.1%}")
    print()

print("="*100)
print("CONCLUSIONS")
print("="*100)
print()

# Overall stats
all_entered = entered[entered['correct_prediction']]
if len(all_entered) > 0:
    mean_revert_wr = mean_revert_results.get('win', 0) / len(all_entered)
    momentum_wr = momentum_results.get('win', 0) / len(all_entered)

    print(f"Model accuracy: {results['correct_prediction'].mean():.1%}")
    print(f"% of predictions we'd enter: {len(entered)/len(results):.1%}")
    print(f"Accuracy of entered predictions: {entered['correct_prediction'].mean():.1%}")
    print()
    print(f"Win rate - Mean reversion: {mean_revert_wr:.1%}")
    print(f"Win rate - Momentum: {momentum_wr:.1%}")
    print()

    if mean_revert_wr > momentum_wr:
        print("Mean reversion is BETTER")
    else:
        print("Momentum is BETTER")
    print()

    if mean_revert_wr < 0.6:
        print("Problem: Even with correct predictions, win rate is low!")
        print("Root cause: Tight range means SMALL movements, hard to hit targets before stops")
        print()
        print("Possible solutions:")
        print("  1. Even tighter targets (<0.7%)")
        print("  2. Wider stops (asymmetric)")
        print("  3. No stop loss, time exit only")
        print("  4. Enter ALL predictions (not just edges)")
        print("  5. Don't trade tight_range directly - use as filter only")
