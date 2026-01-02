"""
TIME-BASED BEHAVIOR ANALYSIS
=============================
Analyze how winners vs losers behave over time
Can we predict early if a trade will win or lose?
"""
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("TIME-BASED BEHAVIOR ANALYSIS")
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

# Parameters
MIN_CONFIDENCE = 0.65

# Collect daily progression for each trade
print("Analyzing daily trade progression...")
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

            # Get next 20 days
            future_dates = raw_df[raw_df.index > date].head(20)
            if len(future_dates) < 10:
                continue

            # Track daily P&L
            daily_pl = []
            hit_target = False

            for i, (future_date, future_row) in enumerate(future_dates.iterrows(), 1):
                if direction == 'long':
                    close_pl = (future_row['close'] - entry_price) / entry_price
                    high_pl = (future_row['high'] - entry_price) / entry_price

                    if future_row['high'] >= target:
                        hit_target = True
                        daily_pl.append(close_pl)
                        break
                else:
                    close_pl = (entry_price - future_row['close']) / entry_price
                    low_pl = (entry_price - future_row['low']) / entry_price

                    if future_row['low'] <= target:
                        hit_target = True
                        daily_pl.append(close_pl)
                        break

                daily_pl.append(close_pl)

            outcome = 'winner' if hit_target else 'loser'

            # Pad to 20 days
            while len(daily_pl) < 20:
                daily_pl.append(daily_pl[-1] if len(daily_pl) > 0 else 0)

            trade_data = {
                'pair': pair,
                'date': date,
                'direction': direction,
                'confidence': max_prob,
                'outcome': outcome,
                'exit_day': len(daily_pl) if hit_target else 20
            }

            # Add daily P&L
            for day in range(1, 21):
                trade_data[f'day_{day}'] = daily_pl[day-1] if day-1 < len(daily_pl) else daily_pl[-1]

            trades.append(trade_data)

trades_df = pd.DataFrame(trades)
print(f"Analyzed {len(trades_df):,} trades")
print()

winners = trades_df[trades_df['outcome'] == 'winner']
losers = trades_df[trades_df['outcome'] == 'loser']

print(f"Winners: {len(winners):,} ({len(winners)/len(trades_df):.1%})")
print(f"Losers: {len(losers):,} ({len(losers)/len(trades_df):.1%})")
print()

# Calculate average trajectory
print("="*100)
print("AVERAGE P&L TRAJECTORY BY DAY")
print("="*100)
print()

day_cols = [f'day_{i}' for i in range(1, 21)]

print("Day | Winners Avg | Winners Median | Losers Avg | Losers Median | Difference")
print("-" * 90)
for i in range(1, 21):
    col = f'day_{i}'
    winner_avg = winners[col].mean()
    winner_med = winners[col].median()
    loser_avg = losers[col].mean()
    loser_med = losers[col].median()
    diff = winner_avg - loser_avg

    print(f"{i:>3} | {winner_avg:>+10.2%} | {winner_med:>+13.2%} | {loser_avg:>+9.2%} | {loser_med:>+12.2%} | {diff:>+9.2%}")

print()

# Early indicators
print("="*100)
print("EARLY INDICATORS - Can we predict outcomes early?")
print("="*100)
print()

# Check performance by day 3, 5, 7
for check_day in [3, 5, 7]:
    col = f'day_{check_day}'

    # Winners underwater at check day
    winners_underwater = winners[winners[col] < 0]
    winners_profitable = winners[winners[col] >= 0]

    # Losers underwater at check day
    losers_underwater = losers[losers[col] < 0]
    losers_profitable = losers[losers[col] >= 0]

    print(f"By Day {check_day}:")
    print(f"  Winners: {len(winners_profitable):>5,} profitable ({len(winners_profitable)/len(winners):>5.1%}), "
          f"{len(winners_underwater):>5,} underwater ({len(winners_underwater)/len(winners):>5.1%})")
    print(f"  Losers:  {len(losers_profitable):>5,} profitable ({len(losers_profitable)/len(losers):>5.1%}), "
          f"{len(losers_underwater):>5,} underwater ({len(losers_underwater)/len(losers):>5.1%})")
    print(f"  Avg P&L: Winners {winners[col].mean():>+6.2%}, Losers {losers[col].mean():>+6.2%} (diff: {winners[col].mean() - losers[col].mean():>+6.2%})")
    print()

# Probability of winning based on early P&L
print("="*100)
print("CONDITIONAL PROBABILITY - Win rate based on early P&L")
print("="*100)
print()

for check_day in [3, 5, 7]:
    col = f'day_{check_day}'

    # Bins: Deep loss, loss, small loss, small profit, profit, big profit
    bins = [float('-inf'), -0.02, -0.01, -0.005, 0, 0.005, 0.01, float('inf')]
    labels = ['<-2%', '-2 to -1%', '-1 to -0.5%', '-0.5 to 0%', '0 to 0.5%', '0.5 to 1%', '>1%']

    trades_df[f'day_{check_day}_bin'] = pd.cut(trades_df[col], bins=bins, labels=labels)

    print(f"Day {check_day} P&L | Total | Winners | Losers | Win Rate")
    print("-" * 70)

    for label in labels:
        subset = trades_df[trades_df[f'day_{check_day}_bin'] == label]
        if len(subset) > 0:
            subset_winners = subset[subset['outcome'] == 'winner']
            win_rate = len(subset_winners) / len(subset)
            print(f"{label:<14} | {len(subset):>5,} | {len(subset_winners):>7,} | {len(subset) - len(subset_winners):>6,} | {win_rate:>7.1%}")

    print()

# Time-based stop rules
print("="*100)
print("TIME-BASED STOP ANALYSIS")
print("="*100)
print()

# Test: If underwater by X% on day Y, exit
stop_rules = [
    (5, -0.015, "Day 5: -1.5%"),
    (5, -0.02, "Day 5: -2.0%"),
    (7, -0.015, "Day 7: -1.5%"),
    (7, -0.02, "Day 7: -2.0%"),
    (10, -0.02, "Day 10: -2.0%"),
    (10, -0.025, "Day 10: -2.5%"),
]

print("Rule            | Winners Cut | Losers Cut | Winners Cut Value | Losers Cut Value | Net Effect")
print("-" * 100)

for day, threshold, label in stop_rules:
    col = f'day_{day}'

    winners_cut = winners[winners[col] <= threshold]
    losers_cut = losers[losers[col] <= threshold]

    winners_cut_value = winners_cut[col].sum()
    losers_cut_value = losers_cut[col].sum()

    # Estimate final value (winners would have won, losers would have lost more)
    winners_opportunity_cost = len(winners_cut) * 0.01  # Avg winner profits ~1%
    losers_saved = abs(losers_cut_value) - (len(losers_cut) * 0.03)  # Losers would lose ~3% more

    net_effect = losers_saved - winners_opportunity_cost

    print(f"{label:<15} | {len(winners_cut):>11,} | {len(losers_cut):>10,} | "
          f"{winners_cut_value:>+16.2%} | {losers_cut_value:>+15.2%} | {net_effect:>+9.2%}")

print()

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Time-Based Behavior Analysis', fontsize=16, fontweight='bold')

# Plot 1: Average trajectory
ax1 = axes[0, 0]
days = range(1, 21)
winner_avg = [winners[f'day_{i}'].mean() * 100 for i in days]
loser_avg = [losers[f'day_{i}'].mean() * 100 for i in days]

ax1.plot(days, winner_avg, 'g-', linewidth=2, label='Winners', marker='o', markersize=4)
ax1.plot(days, loser_avg, 'r-', linewidth=2, label='Losers', marker='o', markersize=4)
ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax1.axhline(y=-4, color='red', linestyle='--', alpha=0.5, label='Emergency stop (-4%)')
ax1.set_xlabel('Days Since Entry')
ax1.set_ylabel('Average P&L (%)')
ax1.set_title('Average P&L Trajectory')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Median trajectory
ax2 = axes[0, 1]
winner_med = [winners[f'day_{i}'].median() * 100 for i in days]
loser_med = [losers[f'day_{i}'].median() * 100 for i in days]

ax2.plot(days, winner_med, 'g-', linewidth=2, label='Winners', marker='o', markersize=4)
ax2.plot(days, loser_med, 'r-', linewidth=2, label='Losers', marker='o', markersize=4)
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax2.axhline(y=-4, color='red', linestyle='--', alpha=0.5, label='Emergency stop (-4%)')
ax2.set_xlabel('Days Since Entry')
ax2.set_ylabel('Median P&L (%)')
ax2.set_title('Median P&L Trajectory')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Distribution at day 5
ax3 = axes[1, 0]
ax3.hist(winners['day_5'] * 100, bins=50, alpha=0.7, color='green', label='Winners')
ax3.hist(losers['day_5'] * 100, bins=50, alpha=0.7, color='red', label='Losers')
ax3.axvline(x=0, color='black', linestyle='-', linewidth=2)
ax3.axvline(x=-2, color='orange', linestyle='--', linewidth=2, label='Potential day 5 stop')
ax3.set_xlabel('P&L at Day 5 (%)')
ax3.set_ylabel('Count')
ax3.set_title('P&L Distribution at Day 5')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Win rate by early P&L
ax4 = axes[1, 1]
day5_bins = pd.cut(trades_df['day_5'] * 100, bins=20)
win_rate_by_pl = trades_df.groupby(day5_bins)['outcome'].apply(lambda x: (x == 'winner').mean() * 100)
bin_centers = [(interval.left + interval.right) / 2 for interval in win_rate_by_pl.index]

ax4.plot(bin_centers, win_rate_by_pl.values, 'b-', linewidth=2, marker='o')
ax4.axhline(y=66.6, color='gray', linestyle='--', alpha=0.5, label='Overall win rate (66.6%)')
ax4.axvline(x=0, color='black', linestyle='-', alpha=0.3)
ax4.set_xlabel('P&L at Day 5 (%)')
ax4.set_ylabel('Win Rate (%)')
ax4.set_title('Win Rate by Day 5 P&L')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('time_behavior_analysis.png', dpi=150, bbox_inches='tight')
print("Visualization saved to time_behavior_analysis.png")
print()

print("="*100)
print("KEY INSIGHTS")
print("="*100)
print()

# Calculate divergence point
diverge_day = None
for i in range(1, 11):
    col = f'day_{i}'
    winner_avg = winners[col].mean()
    loser_avg = losers[col].mean()
    diff = winner_avg - loser_avg

    if diff >= 0.005:  # 0.5% difference
        diverge_day = i
        break

print(f"1. Winners and losers diverge by day {diverge_day} (>{winner_avg - loser_avg:.1%} difference)")
print(f"2. By day 5, winners average {winners['day_5'].mean():+.2%}, losers average {losers['day_5'].mean():+.2%}")
print(f"3. {len(winners[winners['day_5'] < 0])} winners ({len(winners[winners['day_5'] < 0])/len(winners):.1%}) are underwater on day 5")
print(f"4. {len(losers[losers['day_5'] < 0])} losers ({len(losers[losers['day_5'] < 0])/len(losers):.1%}) are underwater on day 5")
print()

# Find best time-based rule
best_rule = max(stop_rules, key=lambda x: (
    len(losers[losers[f'day_{x[0]}'] <= x[1]]) - len(winners[winners[f'day_{x[0]}'] <= x[1]])
))
best_day, best_thresh, best_label = best_rule
print(f"5. Best time-based rule: {best_label}")
print(f"   Would cut {len(winners[winners[f'day_{best_day}'] <= best_thresh]):,} winners and {len(losers[losers[f'day_{best_day}'] <= best_thresh]):,} losers")
print()
print("="*100)
