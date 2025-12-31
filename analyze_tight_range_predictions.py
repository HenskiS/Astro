"""
ANALYZE TIGHT_RANGE PREDICTIONS
================================
Deep dive into WHEN and HOW tight_range predictions are correct.

Questions to answer:
1. What % of predictions are correct?
2. WHEN does the range tighten? (Day 1-3? Day 5-7? Day 8-10?)
3. What's the price action pattern during tight_range periods?
4. What's the optimal entry timing?
5. What's the typical price movement range during tight_range?
6. When predictions fail, why? (breakout happens instead?)
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("ANALYZING TIGHT_RANGE PREDICTIONS")
print("="*100)
print()

DATA_DIR = 'data'
PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY']
TEST_YEARS = [2018, 2019, 2020, 2021]

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
    df['future_range_10d'] = future_range

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

def analyze_predictions(model, all_pairs_data, test_years):
    """
    Analyze tight_range predictions in detail

    For each prediction, track:
    - Was it correct?
    - Price movement during the 10-day window
    - When did range tighten (if it did)?
    - Max gain/loss available
    - Optimal entry/exit points
    """
    print("="*100)
    print(f"ANALYZING PREDICTIONS FOR {test_years}")
    print("="*100)
    print()

    analysis_results = []

    for pair, pair_data in all_pairs_data.items():
        test_data = pair_data[pair_data['year'].isin(test_years)].copy()
        test_data = test_data.dropna(subset=FEATURE_COLS + ['tight_range', 'future_range_10d'])

        for idx in test_data.index[:-10]:  # Need 10 days of future data
            row = test_data.loc[idx]

            # Get prediction
            X = row[FEATURE_COLS].values.reshape(1, -1)
            prob = model.predict_proba(X)[0, 1]

            if prob > MIN_CONFIDENCE:
                # Prediction made! Now analyze what happens
                actual = row['tight_range']
                prediction_correct = (actual == 1)

                # Get next 10 days of data
                future_data = test_data.loc[idx:idx+10]

                if len(future_data) >= 10:
                    entry_price = row['close']
                    original_range = row['range_20d']

                    # Track price movement each day
                    daily_returns = []
                    daily_ranges = []
                    for i in range(1, 11):
                        if idx+i in test_data.index:
                            future_row = test_data.loc[idx+i]
                            daily_return = (future_row['close'] - entry_price) / entry_price

                            # Calculate range from prediction date to this day
                            window_data = test_data.loc[idx:idx+i]
                            window_high = window_data['high'].max()
                            window_low = window_data['low'].min()
                            window_range = (window_high - window_low) / entry_price

                            daily_returns.append(daily_return)
                            daily_ranges.append(window_range)

                    # Calculate metrics
                    max_gain = max(daily_returns) if daily_returns else 0
                    max_loss = min(daily_returns) if daily_returns else 0
                    final_return = daily_returns[-1] if daily_returns else 0

                    # When did range tighten (if ever)?
                    range_tightened_day = None
                    for i, r in enumerate(daily_ranges):
                        if r < original_range * 0.8:
                            range_tightened_day = i + 1
                            break

                    analysis_results.append({
                        'pair': pair,
                        'date': row['date'],
                        'prediction_prob': prob,
                        'actual': actual,
                        'correct': prediction_correct,
                        'max_gain': max_gain,
                        'max_loss': max_loss,
                        'final_return': final_return,
                        'original_range': original_range,
                        'future_range': row['future_range_10d'],
                        'range_contraction': (original_range - row['future_range_10d']) / original_range if row['future_range_10d'] == row['future_range_10d'] else 0,
                        'range_tightened_day': range_tightened_day,
                        'daily_returns': daily_returns,
                        'daily_ranges': daily_ranges
                    })

    return pd.DataFrame(analysis_results)

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
results_df = analyze_predictions(model, all_pairs_data, TEST_YEARS)

# Summary statistics
print("="*100)
print("PREDICTION ACCURACY")
print("="*100)
print()

total_predictions = len(results_df)
correct_predictions = results_df['correct'].sum()
accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

print(f"Total Predictions (>70% confidence): {total_predictions}")
print(f"Correct Predictions: {correct_predictions}")
print(f"Accuracy: {accuracy:.1%}")
print()

# Separate correct vs incorrect
correct_df = results_df[results_df['correct'] == True]
incorrect_df = results_df[results_df['correct'] == False]

print("="*100)
print("CORRECT PREDICTIONS - PRICE ACTION ANALYSIS")
print("="*100)
print()

if len(correct_df) > 0:
    print(f"Sample Size: {len(correct_df)} correct predictions")
    print()

    print("Price Movement Statistics:")
    print(f"  Avg Max Gain: {correct_df['max_gain'].mean():+.2%}")
    print(f"  Avg Max Loss: {correct_df['max_loss'].mean():+.2%}")
    print(f"  Avg Final Return (Day 10): {correct_df['final_return'].mean():+.2%}")
    print(f"  Avg Range Contraction: {correct_df['range_contraction'].mean():.1%}")
    print()

    # When does range actually tighten?
    tightened_df = correct_df[correct_df['range_tightened_day'].notna()]
    if len(tightened_df) > 0:
        avg_tighten_day = tightened_df['range_tightened_day'].mean()
        print(f"Range Tightening Timing:")
        print(f"  {len(tightened_df)}/{len(correct_df)} predictions ({len(tightened_df)/len(correct_df):.0%}) had range tighten")
        print(f"  Avg day when tightening occurred: Day {avg_tighten_day:.1f}")
        print()

        # Distribution of when tightening occurs
        print("  When does range tighten?")
        for day_range in [(1, 3), (4, 6), (7, 10)]:
            count = ((tightened_df['range_tightened_day'] >= day_range[0]) &
                     (tightened_df['range_tightened_day'] <= day_range[1])).sum()
            pct = count / len(tightened_df) * 100
            print(f"    Day {day_range[0]}-{day_range[1]}: {count} ({pct:.0f}%)")
    print()

    # Average daily price path
    print("Average Daily Price Movement:")
    for day in range(1, 11):
        day_returns = [r[day-1] for r in correct_df['daily_returns'] if len(r) >= day]
        if day_returns:
            avg_return = np.mean(day_returns)
            print(f"  Day {day:2d}: {avg_return:+.3%}")
    print()

print("="*100)
print("INCORRECT PREDICTIONS - WHY THEY FAILED")
print("="*100)
print()

if len(incorrect_df) > 0:
    print(f"Sample Size: {len(incorrect_df)} incorrect predictions")
    print()

    print("Price Movement Statistics:")
    print(f"  Avg Max Gain: {incorrect_df['max_gain'].mean():+.2%}")
    print(f"  Avg Max Loss: {incorrect_df['max_loss'].mean():+.2%}")
    print(f"  Avg Final Return (Day 10): {incorrect_df['final_return'].mean():+.2%}")
    print(f"  Avg Range Change: {incorrect_df['range_contraction'].mean():+.1%} (expected <0, actual >{0 if incorrect_df['range_contraction'].mean() > 0 else '<'}0)")
    print()

    # Did range expand instead?
    range_expanded = (incorrect_df['future_range'] > incorrect_df['original_range']).sum()
    print(f"  Range expanded: {range_expanded}/{len(incorrect_df)} ({range_expanded/len(incorrect_df):.0%})")
    print(f"  Avg expansion: {(incorrect_df['future_range'] / incorrect_df['original_range'] - 1).mean():+.1%}")
    print()

print("="*100)
print("TRADING OPPORTUNITIES")
print("="*100)
print()

# Identify profitable patterns
print("Correct Predictions - Trading Stats:")
if len(correct_df) > 0:
    # Could we profitably trade these?
    can_profit_1pct = ((correct_df['max_gain'] > 0.01) | (correct_df['max_loss'] < -0.01)).sum()
    can_profit_05pct = ((correct_df['max_gain'] > 0.005) | (correct_df['max_loss'] < -0.005)).sum()

    print(f"  Predictions with >1% move: {can_profit_1pct}/{len(correct_df)} ({can_profit_1pct/len(correct_df):.0%})")
    print(f"  Predictions with >0.5% move: {can_profit_05pct}/{len(correct_df)} ({can_profit_05pct/len(correct_df):.0%})")
    print()

    # Mean reversion opportunity
    mean_reversion_opps = ((correct_df['max_gain'] > 0.005) & (correct_df['final_return'].abs() < correct_df['max_gain'].abs() * 0.5)).sum()
    print(f"  Mean reversion setups (gain then revert): {mean_reversion_opps}/{len(correct_df)} ({mean_reversion_opps/len(correct_df):.0%})")
    print()

print("="*100)
print("CONCLUSION")
print("="*100)
print()

print(f"Model Accuracy: {accuracy:.1%}")
print()

if len(correct_df) > 0:
    avg_max_move = max(abs(correct_df['max_gain'].mean()), abs(correct_df['max_loss'].mean()))
    print(f"Typical Max Move During Tight Range: {avg_max_move:.2%}")
    print(f"Typical Final Return: {correct_df['final_return'].mean():+.2%}")
    print()

    if avg_max_move < 0.01:
        print("INSIGHT: During tight_range, moves are VERY SMALL (<1%)")
        print("This explains why mean reversion strategies fail - there's not enough movement to profit.")
        print()
        print("Recommendation: Don't trade tight_range predictions directly.")
        print("Consider using them as FILTERS instead (avoid breakouts when tight_range predicted).")
    else:
        print("INSIGHT: There IS tradeable movement during tight_range periods")
        print(f"Optimal strategy timing:")
        if len(tightened_df) > 0:
            print(f"  - Range typically tightens by Day {avg_tighten_day:.0f}")
            print(f"  - Enter mean reversion trades AFTER tightening confirms")
            print(f"  - Target small gains ({avg_max_move:.2%}), tight stops")
