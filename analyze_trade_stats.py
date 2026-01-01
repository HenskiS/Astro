"""
ANALYZE TRADE STATISTICS
=========================
Deep dive into trade metrics: hold times, win/loss sizes, concurrent positions, exit reasons
"""
import pandas as pd
import numpy as np
import glob
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("DETAILED TRADE ANALYSIS")
print("="*100)
print()

# Load all trade files
print("Loading trades...")
trade_files = sorted(glob.glob('trades/backtest_*.csv'))
all_trades = []

for file in trade_files:
    df = pd.read_csv(file)
    if len(df) > 0:
        all_trades.append(df)

if len(all_trades) == 0:
    print("ERROR: No trade files found!")
    exit(1)

trades_df = pd.concat(all_trades, ignore_index=True)
trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])

print(f"Total trades: {len(trades_df)}")
print()

# Separate winners and losers
winners = trades_df[trades_df['profit_pct'] > 0]
losers = trades_df[trades_df['profit_pct'] <= 0]

print("="*100)
print("WIN/LOSS ANALYSIS")
print("="*100)
print()
print(f"Total Trades:     {len(trades_df):,}")
print(f"Winners:          {len(winners):,} ({len(winners)/len(trades_df):.1%})")
print(f"Losers:           {len(losers):,} ({len(losers)/len(trades_df):.1%})")
print()
print(f"Average Win:      {winners['profit_pct'].mean():.2%}")
print(f"Average Loss:     {losers['profit_pct'].mean():.2%}")
print(f"Median Win:       {winners['profit_pct'].median():.2%}")
print(f"Median Loss:      {losers['profit_pct'].median():.2%}")
print(f"Max Win:          {winners['profit_pct'].max():.2%}")
print(f"Max Loss:         {losers['profit_pct'].min():.2%}")
print()
print(f"Win/Loss Ratio:   {abs(winners['profit_pct'].mean() / losers['profit_pct'].mean()):.2f}x")
print(f"Expectancy:       {trades_df['profit_pct'].mean():.2%} per trade")
print()

# Hold time analysis
print("="*100)
print("HOLD TIME ANALYSIS")
print("="*100)
print()
print("All Trades:")
print(f"  Average:        {trades_df['days_held'].mean():.1f} days")
print(f"  Median:         {trades_df['days_held'].median():.0f} days")
print(f"  Min:            {trades_df['days_held'].min():.0f} days")
print(f"  Max:            {trades_df['days_held'].max():.0f} days")
print()
print("Winners:")
print(f"  Average:        {winners['days_held'].mean():.1f} days")
print(f"  Median:         {winners['days_held'].median():.0f} days")
print()
print("Losers:")
print(f"  Average:        {losers['days_held'].mean():.1f} days")
print(f"  Median:         {losers['days_held'].median():.0f} days")
print()

# Exit reason analysis
print("="*100)
print("EXIT REASON BREAKDOWN")
print("="*100)
print()
exit_reasons = trades_df['exit_reason'].value_counts()
for reason, count in exit_reasons.items():
    pct = count / len(trades_df) * 100
    avg_profit = trades_df[trades_df['exit_reason'] == reason]['profit_pct'].mean()
    print(f"{reason:20s} {count:>5,} ({pct:>5.1f}%)  |  Avg P/L: {avg_profit:>+6.2%}")
print()

# Ladder analysis
print("="*100)
print("LADDER PERFORMANCE")
print("="*100)
print()
ladder_stats = trades_df.groupby('ladder_hits').agg({
    'profit_pct': ['count', 'mean'],
    'days_held': 'mean'
})
print("Ladder Hits | Trades | Avg Profit | Avg Hold Time")
print("-" * 60)
for ladder_level in sorted(trades_df['ladder_hits'].unique()):
    count = trades_df[trades_df['ladder_hits'] == ladder_level].shape[0]
    avg_profit = trades_df[trades_df['ladder_hits'] == ladder_level]['profit_pct'].mean()
    avg_days = trades_df[trades_df['ladder_hits'] == ladder_level]['days_held'].mean()
    print(f"     {ladder_level}      | {count:>6,} | {avg_profit:>+9.2%} | {avg_days:>6.1f} days")
print()

# Concurrent positions analysis
print("="*100)
print("CONCURRENT POSITIONS ANALYSIS")
print("="*100)
print()

# Build daily position count
all_dates = []
for _, trade in trades_df.iterrows():
    date_range = pd.date_range(trade['entry_date'], trade['exit_date'], freq='D')
    all_dates.extend(date_range)

daily_counts = pd.Series(all_dates).value_counts().sort_index()

print(f"Average Concurrent Positions:  {daily_counts.mean():.1f}")
print(f"Median Concurrent Positions:   {daily_counts.median():.0f}")
print(f"Max Concurrent Positions:      {daily_counts.max():.0f}")
print(f"Min Concurrent Positions:      {daily_counts.min():.0f}")
print()

# Distribution
print("Position Count Distribution:")
print("-" * 40)
bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
for i in range(len(bins)-1):
    count = ((daily_counts >= bins[i]) & (daily_counts < bins[i+1])).sum()
    pct = count / len(daily_counts) * 100
    print(f"  {bins[i]:>3}-{bins[i+1]:>3} positions: {count:>5,} days ({pct:>5.1f}%)")
count = (daily_counts >= bins[-1]).sum()
pct = count / len(daily_counts) * 100
print(f"  {bins[-1]:>3}+    positions: {count:>5,} days ({pct:>5.1f}%)")
print()

# Pair-level analysis
print("="*100)
print("PERFORMANCE BY PAIR")
print("="*100)
print()
pair_stats = trades_df.groupby('pair').agg({
    'profit_pct': ['count', 'mean'],
    'days_held': 'mean'
})
pair_stats.columns = ['Trades', 'Avg_Profit', 'Avg_Hold']
pair_stats = pair_stats.sort_values('Avg_Profit', ascending=False)

print("Pair    | Trades | Win Rate | Avg Profit | Avg Hold")
print("-" * 60)
for pair in pair_stats.index:
    pair_trades = trades_df[trades_df['pair'] == pair]
    count = len(pair_trades)
    win_rate = (pair_trades['profit_pct'] > 0).sum() / count
    avg_profit = pair_trades['profit_pct'].mean()
    avg_hold = pair_trades['days_held'].mean()
    print(f"{pair:7s} | {count:>6,} | {win_rate:>7.1%} | {avg_profit:>+9.2%} | {avg_hold:>6.1f}d")
print()

# Direction analysis
print("="*100)
print("LONG VS SHORT PERFORMANCE")
print("="*100)
print()
for direction in ['long', 'short']:
    dir_trades = trades_df[trades_df['direction'] == direction]
    if len(dir_trades) == 0:
        continue
    win_rate = (dir_trades['profit_pct'] > 0).sum() / len(dir_trades)
    avg_profit = dir_trades['profit_pct'].mean()
    avg_hold = dir_trades['days_held'].mean()
    print(f"{direction.upper():5s}: {len(dir_trades):>5,} trades | Win Rate: {win_rate:.1%} | Avg P/L: {avg_profit:>+6.2%} | Avg Hold: {avg_hold:.1f}d")
print()

# Profit distribution
print("="*100)
print("PROFIT DISTRIBUTION")
print("="*100)
print()
bins = [-1.0, -0.04, -0.02, -0.01, 0, 0.005, 0.01, 0.015, 0.02, 0.03, 1.0]
labels = ['<-4%', '-4 to -2%', '-2 to -1%', '-1 to 0%', '0 to 0.5%', '0.5 to 1%', '1 to 1.5%', '1.5 to 2%', '2 to 3%', '>3%']
trades_df['profit_bin'] = pd.cut(trades_df['profit_pct'], bins=bins, labels=labels)
profit_dist = trades_df['profit_bin'].value_counts().sort_index()
print("Profit Range  | Count     | Percentage")
print("-" * 45)
for label in labels:
    if label in profit_dist.index:
        count = profit_dist[label]
        pct = count / len(trades_df) * 100
        print(f"{label:12s} | {count:>7,} | {pct:>6.1f}%")
print()

print("="*100)
