"""Check if the 80-period high/low targets are realistic"""
import pandas as pd
import numpy as np

df = pd.read_csv('data_15m/EURUSD_15m.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

# Calculate 80-period high/low
df['high_80'] = df['high'].rolling(80).max()
df['low_80'] = df['low'].rolling(80).min()
df['range_80'] = df['high_80'] - df['low_80']

df_clean = df.dropna()

print("80-period target statistics (EURUSD):")
print(f"  Avg 80p range: {df_clean['range_80'].mean():.5f}")
print(f"  Avg price: {df_clean['close'].mean():.5f}")
print(f"  Range as % of price: {100*df_clean['range_80'].mean()/df_clean['close'].mean():.3f}%")
print()

# Check hit probability
print("Probability of hitting 80p high/low in next N bars:")
for forward_bars in [4, 8, 16, 24]:
    # Calculate future high/low
    future_high = df_clean['high'].shift(-1).rolling(forward_bars).max()
    future_low = df_clean['low'].shift(-1).rolling(forward_bars).min()

    # Check if they break 80p levels
    hit_high = (future_high > df_clean['high_80']).mean()
    hit_low = (future_low < df_clean['low_80']).mean()

    print(f"  {forward_bars:2d} bars: breakout_high={hit_high:.1%}, breakout_low={hit_low:.1%}")

print()
print("This shows the BASE RATE of breakouts (what % of time they happen)")
print("The model predicts these with 65% precision when confidence > 0.80")
