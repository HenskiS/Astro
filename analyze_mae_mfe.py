"""
MAXIMUM ADVERSE/FAVORABLE EXCURSION ANALYSIS
=============================================
Analyze how far positions move against us (MAE) and in our favor (MFE)
This helps determine optimal stop loss levels without cutting winners
"""
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("MAXIMUM ADVERSE/FAVORABLE EXCURSION ANALYSIS")
print("="*100)
print()

# Load data
print("Loading data...")
with open('model_predictions_quarterly.pkl', 'rb') as f:
    all_predictions = pickle.load(f)

DATA_DIR = 'data'
PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY']

all_raw_data = {}
for pair in PAIRS:
    file_path = f'{DATA_DIR}/{pair}_1day_with_spreads.csv'
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    all_raw_data[pair] = df

print("Data loaded")
print()

# Parameters (optimized)
MIN_CONFIDENCE = 0.65

# Collect all trades with their full price trajectory
print("Analyzing trade trajectories...")
trades = []

for quarter_name, quarter_preds in sorted(all_predictions.items()):
    for pair, pred_df in quarter_preds.items():
        raw_df = all_raw_data[pair]

        for date, row in pred_df.iterrows():
            breakout_high_prob = row['breakout_high_prob']
            breakout_low_prob = row['breakout_low_prob']
            max_prob = max(breakout_high_prob, breakout_low_prob)

            if max_prob <= MIN_CONFIDENCE:
                continue

            entry_price = row['close']

            if breakout_high_prob > breakout_low_prob:
                direction = 'long'
                target = row['high_20d'] * 1.005
            else:
                direction = 'short'
                target = row['low_20d'] * 0.995

            # Get next 30 days of price action
            future_dates = raw_df[raw_df.index > date].head(30)
            if len(future_dates) < 5:
                continue

            # Track excursions
            max_favorable = 0  # Best profit achieved
            max_adverse = 0    # Worst loss experienced

            hit_target = False
            exit_day = None

            for i, (future_date, future_row) in enumerate(future_dates.iterrows(), 1):
                if direction == 'long':
                    high_profit = (future_row['high'] - entry_price) / entry_price
                    low_profit = (future_row['low'] - entry_price) / entry_price

                    max_favorable = max(max_favorable, high_profit)
                    max_adverse = min(max_adverse, low_profit)

                    if future_row['high'] >= target:
                        hit_target = True
                        exit_day = i
                        break
                else:  # short
                    high_profit = (entry_price - future_row['low']) / entry_price
                    low_profit = (entry_price - future_row['high']) / entry_price

                    max_favorable = max(max_favorable, high_profit)
                    max_adverse = min(max_adverse, low_profit)

                    if future_row['low'] <= target:
                        hit_target = True
                        exit_day = i
                        break

            # Classify as winner or loser
            if hit_target:
                outcome = 'winner'
                final_profit = max_favorable
            else:
                outcome = 'loser'
                final_profit = max_adverse if max_adverse < 0 else 0

            trades.append({
                'pair': pair,
                'date': date,
                'direction': direction,
                'confidence': max_prob,
                'max_favorable': max_favorable,
                'max_adverse': max_adverse,
                'outcome': outcome,
                'final_profit': final_profit,
                'exit_day': exit_day if hit_target else len(future_dates)
            })

trades_df = pd.DataFrame(trades)
print(f"Analyzed {len(trades_df):,} trades")
print()

# Separate winners and losers
winners = trades_df[trades_df['outcome'] == 'winner']
losers = trades_df[trades_df['outcome'] == 'loser']

print("="*100)
print("MAXIMUM ADVERSE EXCURSION (MAE) - How far do trades move AGAINST us?")
print("="*100)
print()

print(f"Total trades: {len(trades_df):,}")
print(f"Winners: {len(winners):,} ({len(winners)/len(trades_df):.1%})")
print(f"Losers: {len(losers):,} ({len(losers)/len(trades_df):.1%})")
print()

# MAE statistics
print("MAE Statistics (Maximum Adverse Excursion):")
print("-" * 80)
print(f"{'Category':<20} {'Count':<10} {'Mean MAE':<12} {'Median MAE':<12} {'90th %ile':<12} {'95th %ile':<12}")
print("-" * 80)

for category, subset in [('All Trades', trades_df), ('Winners', winners), ('Losers', losers)]:
    mean_mae = subset['max_adverse'].mean()
    median_mae = subset['max_adverse'].median()
    p90_mae = subset['max_adverse'].quantile(0.10)  # 10th percentile (most negative)
    p95_mae = subset['max_adverse'].quantile(0.05)  # 5th percentile (most negative)

    print(f"{category:<20} {len(subset):<10,} {mean_mae:<12.2%} {median_mae:<12.2%} {p90_mae:<12.2%} {p95_mae:<12.2%}")

print()

# Key question: What stop loss would preserve winners?
print("="*100)
print("STOP LOSS ANALYSIS - What stop loss preserves all/most winners?")
print("="*100)
print()

stop_levels = [-0.01, -0.015, -0.02, -0.025, -0.03, -0.035, -0.04, -0.05]

print("Stop Loss | Winners Cut | Winners Kept | Losers Cut | Net Effect")
print("-" * 80)

for stop in stop_levels:
    # Winners that would hit this stop
    winners_cut = winners[winners['max_adverse'] <= stop]
    winners_kept = winners[winners['max_adverse'] > stop]

    # Losers that would hit this stop
    losers_cut = losers[losers['max_adverse'] <= stop]

    # Calculate value
    winners_cut_value = winners_cut['final_profit'].sum()
    winners_kept_value = winners_kept['final_profit'].sum()
    losers_cut_value = losers_cut['max_adverse'].sum()  # Negative values

    net_effect = winners_kept_value + losers_cut_value - winners_cut_value

    print(f"{stop:>+7.1%}   | {len(winners_cut):>11,} | {len(winners_kept):>12,} | {len(losers_cut):>10,} | {net_effect:>+10.2%}")

print()

# Distribution of MAE for winners
print("="*100)
print("WINNER MAE DISTRIBUTION - How far underwater do winning trades go?")
print("="*100)
print()

bins = [float('-inf'), -0.05, -0.04, -0.03, -0.025, -0.02, -0.015, -0.01, -0.005, 0]
labels = ['<-5%', '-5 to -4%', '-4 to -3%', '-3 to -2.5%', '-2.5 to -2%', '-2 to -1.5%', '-1.5 to -1%', '-1 to -0.5%', '>-0.5%']

winners_mae_bins = pd.cut(winners['max_adverse'], bins=bins, labels=labels)
mae_dist = winners_mae_bins.value_counts().sort_index()

print("MAE Range      | Winners | Percentage | Cumulative")
print("-" * 60)
cumulative = 0
for label in labels:
    if label in mae_dist.index:
        count = mae_dist[label]
        pct = count / len(winners) * 100
        cumulative += pct
        print(f"{label:<14} | {count:>7,} | {pct:>9.1f}% | {cumulative:>9.1f}%")
    else:
        print(f"{label:<14} | {0:>7,} | {0:>9.1f}% | {cumulative:>9.1f}%")

print()

# MFE analysis
print("="*100)
print("MAXIMUM FAVORABLE EXCURSION (MFE) - How far do trades move IN FAVOR?")
print("="*100)
print()

print("MFE Statistics (Maximum Favorable Excursion):")
print("-" * 80)
print(f"{'Category':<20} {'Mean MFE':<12} {'Median MFE':<12} {'90th %ile':<12} {'95th %ile':<12}")
print("-" * 80)

for category, subset in [('All Trades', trades_df), ('Winners', winners), ('Losers', losers)]:
    mean_mfe = subset['max_favorable'].mean()
    median_mfe = subset['max_favorable'].median()
    p90_mfe = subset['max_favorable'].quantile(0.90)
    p95_mfe = subset['max_favorable'].quantile(0.95)

    print(f"{category:<20} {mean_mfe:<12.2%} {median_mfe:<12.2%} {p90_mfe:<12.2%} {p95_mfe:<12.2%}")

print()

# Behavior patterns
print("="*100)
print("BEHAVIOR PATTERNS - First move analysis")
print("="*100)
print()

# Check: Do winners typically go against us first, or with us first?
winners_underwater_first = winners[winners['max_adverse'] < -0.005]  # More than 0.5% underwater
winners_profitable_first = winners[winners['max_adverse'] >= -0.005]  # Never significantly underwater

print(f"Winners that went underwater first (>0.5%): {len(winners_underwater_first):,} ({len(winners_underwater_first)/len(winners):.1%})")
print(f"Winners that stayed profitable: {len(winners_profitable_first):,} ({len(winners_profitable_first)/len(winners):.1%})")
print()

print("Average characteristics:")
print(f"  Winners underwater first: MAE={winners_underwater_first['max_adverse'].mean():.2%}, MFE={winners_underwater_first['max_favorable'].mean():.2%}")
print(f"  Winners profitable: MAE={winners_profitable_first['max_adverse'].mean():.2%}, MFE={winners_profitable_first['max_favorable'].mean():.2%}")
print()

# Exit timing
print("="*100)
print("EXIT TIMING - When do trades exit?")
print("="*100)
print()

print(f"Average days to exit:")
print(f"  Winners: {winners['exit_day'].mean():.1f} days (median: {winners['exit_day'].median():.0f})")
print(f"  Losers: {losers['exit_day'].mean():.1f} days (median: {losers['exit_day'].median():.0f})")
print()

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Maximum Adverse/Favorable Excursion Analysis', fontsize=16, fontweight='bold')

# Plot 1: MAE vs Final Profit (Winners)
ax1 = axes[0, 0]
ax1.scatter(winners['max_adverse'] * 100, winners['final_profit'] * 100, alpha=0.3, s=10, color='green')
ax1.axvline(x=-4, color='red', linestyle='--', label='Current emergency stop (-4%)')
ax1.axvline(x=-2, color='orange', linestyle='--', label='Potential tighter stop (-2%)')
ax1.set_xlabel('Max Adverse Excursion (%)')
ax1.set_ylabel('Final Profit (%)')
ax1.set_title('Winners: MAE vs Final Profit')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 2: MAE vs Final Profit (Losers)
ax2 = axes[0, 1]
ax2.scatter(losers['max_adverse'] * 100, losers['final_profit'] * 100, alpha=0.3, s=10, color='red')
ax2.axvline(x=-4, color='red', linestyle='--', label='Current emergency stop (-4%)')
ax2.axvline(x=-2, color='orange', linestyle='--', label='Potential tighter stop (-2%)')
ax2.set_xlabel('Max Adverse Excursion (%)')
ax2.set_ylabel('Final Profit (%)')
ax2.set_title('Losers: MAE vs Final Profit')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Plot 3: MAE distribution
ax3 = axes[1, 0]
ax3.hist(winners['max_adverse'] * 100, bins=50, alpha=0.7, color='green', label='Winners')
ax3.hist(losers['max_adverse'] * 100, bins=50, alpha=0.7, color='red', label='Losers')
ax3.axvline(x=-4, color='red', linestyle='--', linewidth=2, label='Current emergency stop')
ax3.axvline(x=-2, color='orange', linestyle='--', linewidth=2, label='Potential tighter stop')
ax3.set_xlabel('Max Adverse Excursion (%)')
ax3.set_ylabel('Count')
ax3.set_title('MAE Distribution: Winners vs Losers')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: MFE distribution
ax4 = axes[1, 1]
ax4.hist(winners['max_favorable'] * 100, bins=50, alpha=0.7, color='green', label='Winners')
ax4.hist(losers['max_favorable'] * 100, bins=50, alpha=0.7, color='red', label='Losers')
ax4.set_xlabel('Max Favorable Excursion (%)')
ax4.set_ylabel('Count')
ax4.set_title('MFE Distribution: Winners vs Losers')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mae_mfe_analysis.png', dpi=150, bbox_inches='tight')
print("Visualization saved to mae_mfe_analysis.png")
print()

print("="*100)
print("KEY INSIGHTS")
print("="*100)
print()

# Calculate key insights
winners_cut_at_2pct = winners[winners['max_adverse'] <= -0.02]
winners_cut_at_3pct = winners[winners['max_adverse'] <= -0.03]

print(f"1. Current -4% emergency stop preserves {len(winners[winners['max_adverse'] > -0.04]):,} of {len(winners):,} winners ({len(winners[winners['max_adverse'] > -0.04])/len(winners):.1%})")
print(f"2. A -3% stop would preserve {len(winners[winners['max_adverse'] > -0.03]):,} winners ({len(winners[winners['max_adverse'] > -0.03])/len(winners):.1%})")
print(f"3. A -2% stop would preserve {len(winners[winners['max_adverse'] > -0.02]):,} winners ({len(winners[winners['max_adverse'] > -0.02])/len(winners):.1%})")
print()
print(f"4. {len(winners_underwater_first):,} winners ({len(winners_underwater_first)/len(winners):.1%}) go underwater >0.5% before winning")
print(f"5. Average winner MAE: {winners['max_adverse'].mean():.2%} (50% go worse than {winners['max_adverse'].median():.2%})")
print(f"6. Average loser MAE: {losers['max_adverse'].mean():.2%} (they get stopped appropriately)")
print()
print("="*100)
