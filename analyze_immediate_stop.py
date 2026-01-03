"""
Analyze impact of adding -5% immediate stop to 15m strategy
"""
import pandas as pd
import numpy as np
from backtest_15m_optimized import load_and_prepare_data, calculate_features
import pickle

# Load models
print("Loading models...")
with open('models_15m.pkl', 'rb') as f:
    models = pickle.load(f)

# Load and prepare data
print("Loading data...")
df = load_and_prepare_data()

# Calculate features
print("Calculating features...")
df = calculate_features(df)
df = df.dropna()

# Use 2023 data for analysis (recent data)
df_2023 = df[df.index.year == 2023].copy()

print(f"\nAnalyzing {len(df_2023)} bars from 2023...")

# Track trades
trades_no_stop = []
trades_with_stop = []

RISK_PER_TRADE = 0.004
initial_capital = 500
capital_no_stop = initial_capital
capital_with_stop = initial_capital

# Simple simulation
positions_no_stop = []
positions_with_stop = []

for pair in ['EURUSD', 'USDJPY', 'GBPUSD']:
    if pair not in df_2023['pair'].values:
        continue

    df_pair = df_2023[df_2023['pair'] == pair].copy()

    for i in range(100, len(df_pair) - 24):  # Need 24 bars ahead
        current_bar = df_pair.iloc[i]

        # Get predictions
        X = current_bar[['confidence_high', 'confidence_low']].values.reshape(1, -1)

        if current_bar['confidence_high'] > 0.70:
            direction = 'long'
            target = current_bar['high_80p'] * 1.005

            # Check next 24 bars
            future_bars = df_pair.iloc[i+1:i+25]

            if len(future_bars) < 24:
                continue

            # Entry at next bar open
            entry = future_bars.iloc[0]['ask_open']

            # Track for both scenarios
            hit_target = False
            hit_minus_5_stop = False
            worst_drawdown = 0

            for j, bar in future_bars.iterrows():
                # Check profit
                intraday_high = bar['bid_high']
                intraday_low = bar['bid_low']

                # Check target
                if intraday_high >= target:
                    hit_target = True
                    profit_at_target = (target - entry) / entry
                    break

                # Check -5% immediate stop
                profit_at_low = (intraday_low - entry) / entry
                worst_drawdown = min(worst_drawdown, profit_at_low)

                if profit_at_low <= -0.05:
                    hit_minus_5_stop = True
                    break

            # Record result
            if hit_target:
                trades_no_stop.append({
                    'pair': pair,
                    'direction': direction,
                    'result': 'target',
                    'worst_drawdown': worst_drawdown,
                    'stopped_at_5pct': hit_minus_5_stop
                })

                if hit_minus_5_stop:
                    trades_with_stop.append({
                        'result': 'stopped',
                        'lost_winner': True
                    })
                else:
                    trades_with_stop.append({
                        'result': 'target',
                        'lost_winner': False
                    })

# Analysis
print("\n" + "="*60)
print("ANALYSIS: Impact of -5% Immediate Stop")
print("="*60)

winners_no_stop = [t for t in trades_no_stop if t['result'] == 'target']
lost_winners = [t for t in winners_no_stop if t['stopped_at_5pct']]

print(f"\nTotal winning trades (no stop): {len(winners_no_stop)}")
print(f"Winners that hit -5% before target: {len(lost_winners)} ({len(lost_winners)/len(winners_no_stop)*100:.1f}%)")

if winners_no_stop:
    drawdowns = [t['worst_drawdown'] for t in winners_no_stop]
    print(f"\nWorst drawdown on winning trades:")
    print(f"  Mean: {np.mean(drawdowns)*100:.2f}%")
    print(f"  Median: {np.median(drawdowns)*100:.2f}%")
    print(f"  25th percentile: {np.percentile(drawdowns, 25)*100:.2f}%")
    print(f"  75th percentile: {np.percentile(drawdowns, 75)*100:.2f}%")
    print(f"  Min (worst): {np.min(drawdowns)*100:.2f}%")

print("\n" + "="*60)
print("INTERPRETATION:")
print("="*60)
print("A -5% immediate stop would cut off ~X% of winning trades")
print("that temporarily dip below -5% before recovering to hit targets.")
print("\nFor peace of mind, this is a reasonable safety net, but")
print("it will reduce overall returns by stopping out some eventual winners.")
