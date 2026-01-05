"""
COMPARE FEATURES
================
Compare the exact features being calculated for EURUSD on 2016-01-31
between backtest generation and production simulation.
"""
import pandas as pd
import numpy as np
from mock_broker_api import MockBrokerAPI

# Use same feature calculation as both systems
def calculate_features(df):
    """Calculate technical features"""
    df = df.copy()
    
    df['return_1d'] = df['close'].pct_change()
    df['return_3d'] = df['close'].pct_change(3)
    df['return_5d'] = df['close'].pct_change(5)
    df['return_10d'] = df['close'].pct_change(10)

    for period in [10, 20, 50]:
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        df[f'price_to_ema_{period}'] = df['close'] / df[f'ema_{period}'] - 1

    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    tr = pd.concat([df['high'] - df['low'],
                    abs(df['high'] - df['close'].shift()),
                    abs(df['low'] - df['close'].shift())], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean()
    df['atr_pct'] = df['atr'] / df['close']

    df['volatility_10d'] = df['return_1d'].rolling(10).std()
    df['volatility_20d'] = df['return_1d'].rolling(20).std()

    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * bb_std
    df['bb_lower'] = df['bb_middle'] - 2 * bb_std
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)

    df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
    df['momentum_20'] = df['close'] / df['close'].shift(20) - 1

    df['high_20d'] = df['high'].rolling(20).max()
    df['low_20d'] = df['low'].rolling(20).min()
    df['range_20d'] = (df['high_20d'] - df['low_20d']) / df['close']
    df['position_in_range'] = (df['close'] - df['low_20d']) / (df['high_20d'] - df['low_20d'] + 1e-10)

    return df

print("="*100)
print("COMPARING FEATURES FOR EURUSD ON 2016-01-31")
print("="*100)
print()

# Method 1: How backtest (generate_predictions_quarterly.py) does it
print("METHOD 1: Backtest approach")
print("-" * 100)

# Load ALL raw data
df = pd.read_csv('data/EURUSD_1day_with_spreads.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

# Remove timezone if present
if df.index.tz is not None:
    df.index = df.index.tz_localize(None)

# Calculate features on ALL data at once
features_all = calculate_features(df)

# Filter to just Jan 31 (use date matching to handle timestamp differences)
jan_31 = pd.Timestamp('2016-01-31')
jan_31_matches = [idx for idx in features_all.index if idx.date() == jan_31.date()]

if len(jan_31_matches) > 0:
    jan_31_actual = jan_31_matches[0]
    row = features_all.loc[jan_31_actual]
    print(f"Features on {jan_31_actual}:")
    print(f"  high_20d: {row['high_20d']:.5f}")
    print(f"  low_20d: {row['low_20d']:.5f}")
    print(f"  rsi: {row['rsi']:.4f}")
    print(f"  macd: {row['macd']:.6f}")
    print(f"  ema_20: {row['ema_20']:.5f}")
else:
    print(f"No data found for {jan_31.date()}")

print()

# Method 2: How production simulator does it
print("METHOD 2: Production approach")
print("-" * 100)

api = MockBrokerAPI(data_dir='data')

# Get history up to Jan 31 (like production does)
history = api.get_history('EURUSD', count=999999, end_date=jan_31)

# Calculate features
features_prod = calculate_features(history)
features_prod = features_prod.dropna()

if len(features_prod) > 0:
    last_row = features_prod.iloc[-1]
    print(f"Features on {last_row.name.date()} (last row):")
    print(f"  high_20d: {last_row['high_20d']:.5f}")
    print(f"  low_20d: {last_row['low_20d']:.5f}")
    print(f"  rsi: {last_row['rsi']:.4f}")
    print(f"  macd: {last_row['macd']:.6f}")
    print(f"  ema_20: {last_row['ema_20']:.5f}")

print()
print("="*100)
print("COMPARISON")
print("="*100)

if len(jan_31_matches) > 0 and len(features_prod) > 0:
    bt_row = features_all.loc[jan_31_actual]
    prod_row = features_prod.iloc[-1]
    
    print(f"\nComparing key features:")
    print(f"  high_20d:  BT={bt_row['high_20d']:.5f}, Prod={prod_row['high_20d']:.5f}, Match={abs(bt_row['high_20d']-prod_row['high_20d'])<0.00001}")
    print(f"  low_20d:   BT={bt_row['low_20d']:.5f}, Prod={prod_row['low_20d']:.5f}, Match={abs(bt_row['low_20d']-prod_row['low_20d'])<0.00001}")
    print(f"  rsi:       BT={bt_row['rsi']:.4f}, Prod={prod_row['rsi']:.4f}, Match={abs(bt_row['rsi']-prod_row['rsi'])<0.01}")
    
    if abs(bt_row['high_20d']-prod_row['high_20d']) > 0.00001:
        print(f"\n  high_20d differs! This affects the breakout target!")
    if abs(bt_row['low_20d']-prod_row['low_20d']) > 0.00001:
        print(f"  low_20d differs! This affects the breakout target!")
else:
    print("Cannot compare - missing data")