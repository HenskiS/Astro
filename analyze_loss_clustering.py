"""
ANALYZE LOSS CLUSTERING
Check if losses cluster in time or are randomly distributed
"""
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("LOSS CLUSTERING ANALYSIS")
print("="*100)
print()

# Load predictions
print("Loading predictions...")
with open('model_predictions.pkl', 'rb') as f:
    all_predictions = pickle.load(f)

def analyze_clustering(predictions_df):
    """Analyze win/loss clustering for a single pair/period"""
    # Ensure date is a column
    if 'date' not in predictions_df.columns and 'date' in predictions_df.index.names:
        predictions_df = predictions_df.reset_index()
    elif 'date' in predictions_df.index.names:
        predictions_df = predictions_df.reset_index(drop=True)

    df = predictions_df.copy()

    # Calculate predictions
    df['max_prob'] = df[['breakout_high_prob', 'breakout_low_prob']].max(axis=1)
    df['prediction'] = df.apply(
        lambda row: 'HIGH' if row['breakout_high_prob'] > row['breakout_low_prob'] else 'LOW',
        axis=1
    )

    # Filter to high confidence only (>70%)
    df = df[df['max_prob'] > 0.70].copy()
    df = df.sort_values('date').reset_index(drop=True)

    # Calculate if prediction was correct
    results = []
    for idx in range(len(df) - 10):
        current = df.iloc[idx]
        future_10d = df.iloc[idx:min(idx+10, len(df))]

        if current['prediction'] == 'HIGH':
            hit = (future_10d['high'].max() > current['high_20d'])
        else:
            hit = (future_10d['low'].min() < current['low_20d'])

        results.append({
            'date': current['date'],
            'correct': hit,
            'confidence': current['max_prob']
        })

    if len(results) == 0:
        return None

    results_df = pd.DataFrame(results)

    # Calculate streaks
    results_df['outcome'] = results_df['correct'].map({True: 'W', False: 'L'})
    results_df['streak_id'] = (results_df['outcome'] != results_df['outcome'].shift()).cumsum()

    streaks = results_df.groupby('streak_id').agg({
        'outcome': 'first',
        'date': ['first', 'count']
    }).reset_index(drop=True)
    streaks.columns = ['outcome', 'start_date', 'length']

    # Analyze loss streaks
    loss_streaks = streaks[streaks['outcome'] == 'L']
    win_streaks = streaks[streaks['outcome'] == 'W']

    # Time between losses
    loss_dates = results_df[results_df['outcome'] == 'L']['date'].values
    if len(loss_dates) > 1:
        time_between_losses = []
        for i in range(1, len(loss_dates)):
            days = (pd.Timestamp(loss_dates[i]) - pd.Timestamp(loss_dates[i-1])).days
            time_between_losses.append(days)
    else:
        time_between_losses = []

    return {
        'total_predictions': len(results_df),
        'num_correct': results_df['correct'].sum(),
        'num_incorrect': (~results_df['correct']).sum(),
        'accuracy': results_df['correct'].mean(),
        'loss_streaks': loss_streaks,
        'win_streaks': win_streaks,
        'time_between_losses': time_between_losses,
        'results_df': results_df
    }

# Analyze all pairs and periods
all_stats = []

for period_name, period_preds in all_predictions.items():
    print(f"\n{period_name}:")
    print("-" * 80)

    for pair, predictions_df in period_preds.items():
        stats = analyze_clustering(predictions_df)
        if stats is None:
            continue

        loss_streaks = stats['loss_streaks']
        win_streaks = stats['win_streaks']

        print(f"\n  {pair}:")
        print(f"    Predictions: {stats['total_predictions']}")
        print(f"    Accuracy: {stats['accuracy']:.1%} ({stats['num_correct']}/{stats['total_predictions']})")
        print(f"    Losses: {stats['num_incorrect']}")

        if len(loss_streaks) > 0:
            print(f"    Loss streaks:")
            print(f"      Total: {len(loss_streaks)}")
            print(f"      Avg length: {loss_streaks['length'].mean():.1f}")
            print(f"      Max length: {loss_streaks['length'].max()}")
            print(f"      Longest streaks: {sorted(loss_streaks['length'].values, reverse=True)[:3]}")

            # Clustering metric: what % of losses are in streaks of 2+?
            losses_in_streaks = loss_streaks[loss_streaks['length'] >= 2]['length'].sum()
            pct_clustered = losses_in_streaks / stats['num_incorrect'] * 100 if stats['num_incorrect'] > 0 else 0
            print(f"      {pct_clustered:.1f}% of losses are in streaks of 2+")

        if len(stats['time_between_losses']) > 0:
            print(f"    Time between losses:")
            print(f"      Avg: {np.mean(stats['time_between_losses']):.1f} days")
            print(f"      Median: {np.median(stats['time_between_losses']):.1f} days")
            print(f"      Min: {np.min(stats['time_between_losses'])} days")

        all_stats.append({
            'period': period_name,
            'pair': pair,
            **{k: v for k, v in stats.items() if k not in ['loss_streaks', 'win_streaks', 'results_df', 'time_between_losses']},
            'max_loss_streak': loss_streaks['length'].max() if len(loss_streaks) > 0 else 0,
            'avg_loss_streak': loss_streaks['length'].mean() if len(loss_streaks) > 0 else 0,
            'pct_losses_clustered': losses_in_streaks / stats['num_incorrect'] * 100 if stats['num_incorrect'] > 0 and len(loss_streaks) > 0 else 0,
            'avg_days_between_losses': np.mean(stats['time_between_losses']) if len(stats['time_between_losses']) > 0 else None
        })

# Overall statistics
print()
print("="*100)
print("OVERALL CLUSTERING STATISTICS")
print("="*100)
print()

stats_df = pd.DataFrame(all_stats)

print(f"Total predictions analyzed: {stats_df['total_predictions'].sum()}")
print(f"Overall accuracy: {stats_df['num_correct'].sum() / stats_df['total_predictions'].sum():.1%}")
print(f"Total losses: {stats_df['num_incorrect'].sum()}")
print()

print("Loss streak distribution:")
print(f"  Average max loss streak per pair/period: {stats_df['max_loss_streak'].mean():.1f}")
print(f"  Worst loss streak: {stats_df['max_loss_streak'].max():.0f} consecutive losses")
print(f"  Average length of loss streaks: {stats_df['avg_loss_streak'].mean():.2f}")
print()

print("Loss clustering:")
print(f"  Average % of losses in streaks of 2+: {stats_df['pct_losses_clustered'].mean():.1f}%")
print(f"  (This means {stats_df['pct_losses_clustered'].mean():.1f}% of losses come in clusters)")
print()

if stats_df['avg_days_between_losses'].notna().any():
    print("Time between losses:")
    print(f"  Average: {stats_df['avg_days_between_losses'].mean():.1f} days")
    print(f"  Range: {stats_df['avg_days_between_losses'].min():.1f} - {stats_df['avg_days_between_losses'].max():.1f} days")
print()

# Worst clustering cases
print("Pairs/periods with most clustering (highest % losses in streaks):")
worst_clustering = stats_df.nlargest(10, 'pct_losses_clustered')[['period', 'pair', 'accuracy', 'num_incorrect', 'max_loss_streak', 'pct_losses_clustered']]
for idx, row in worst_clustering.iterrows():
    print(f"  {row['period']:10} {row['pair']:8} - {row['accuracy']:.1%} accuracy, "
          f"{row['num_incorrect']:.0f} losses, max streak={row['max_loss_streak']:.0f}, "
          f"{row['pct_losses_clustered']:.1f}% clustered")

print()
print("Pairs/periods with longest loss streaks:")
worst_streaks = stats_df.nlargest(10, 'max_loss_streak')[['period', 'pair', 'accuracy', 'num_incorrect', 'max_loss_streak', 'pct_losses_clustered']]
for idx, row in worst_streaks.iterrows():
    print(f"  {row['period']:10} {row['pair']:8} - {row['accuracy']:.1%} accuracy, "
          f"{row['num_incorrect']:.0f} losses, max streak={row['max_loss_streak']:.0f}")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Loss streak length distribution
ax1 = axes[0, 0]
ax1.hist(stats_df['max_loss_streak'], bins=range(0, int(stats_df['max_loss_streak'].max())+2),
         edgecolor='black', alpha=0.7)
ax1.set_xlabel('Maximum Loss Streak Length')
ax1.set_ylabel('Frequency')
ax1.set_title('Distribution of Maximum Loss Streaks')
ax1.grid(True, alpha=0.3)

# 2. Clustering percentage
ax2 = axes[0, 1]
ax2.hist(stats_df['pct_losses_clustered'], bins=20, edgecolor='black', alpha=0.7, color='red')
ax2.set_xlabel('% of Losses in Streaks of 2+')
ax2.set_ylabel('Frequency')
ax2.set_title('Loss Clustering Distribution')
ax2.axvline(stats_df['pct_losses_clustered'].mean(), color='black', linestyle='--',
            linewidth=2, label=f'Mean: {stats_df["pct_losses_clustered"].mean():.1f}%')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Accuracy vs loss clustering
ax3 = axes[1, 0]
scatter = ax3.scatter(stats_df['accuracy']*100, stats_df['pct_losses_clustered'],
                     s=50, alpha=0.6, c=stats_df['max_loss_streak'], cmap='Reds')
ax3.set_xlabel('Accuracy (%)')
ax3.set_ylabel('% Losses Clustered')
ax3.set_title('Accuracy vs Loss Clustering')
ax3.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax3, label='Max Loss Streak')

# 4. Time between losses
ax4 = axes[1, 1]
valid_times = stats_df['avg_days_between_losses'].dropna()
if len(valid_times) > 0:
    ax4.hist(valid_times, bins=20, edgecolor='black', alpha=0.7, color='green')
    ax4.set_xlabel('Average Days Between Losses')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Time Between Losses Distribution')
    ax4.axvline(valid_times.mean(), color='black', linestyle='--',
                linewidth=2, label=f'Mean: {valid_times.mean():.1f} days')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('loss_clustering_analysis.png', dpi=300, bbox_inches='tight')
print()
print("Visualization saved to loss_clustering_analysis.png")
print()

# Statistical test for clustering
print("="*100)
print("CLUSTERING SIGNIFICANCE TEST")
print("="*100)
print()

# If losses were random, we'd expect them to be evenly distributed
# Calculate expected vs actual clustering
total_predictions = stats_df['total_predictions'].sum()
total_losses = stats_df['num_incorrect'].sum()
loss_rate = total_losses / total_predictions

# Expected consecutive losses if random
expected_consecutive = loss_rate * loss_rate * total_losses
actual_consecutive = stats_df['pct_losses_clustered'].mean() / 100 * total_losses

print(f"Loss rate: {loss_rate:.1%}")
print(f"If losses were random:")
print(f"  Expected consecutive pairs: ~{expected_consecutive:.0f}")
print(f"Actual observations:")
print(f"  Actual consecutive losses: ~{actual_consecutive:.0f}")
print(f"  Ratio: {actual_consecutive / expected_consecutive:.2f}x more clustering than random")
print()

if actual_consecutive / expected_consecutive > 1.5:
    print("⚠️  SIGNIFICANT CLUSTERING DETECTED")
    print("Losses cluster more than expected by chance.")
    print("This suggests:")
    print("  - Strategy may struggle during specific market conditions")
    print("  - Drawdowns could be deeper than random losses would suggest")
    print("  - Consider regime detection or dynamic position sizing")
else:
    print("✓ Losses appear roughly randomly distributed")
    print("No significant clustering beyond what chance would predict.")

stats_df.to_csv('loss_clustering_stats.csv', index=False)
print()
print("Full statistics saved to loss_clustering_stats.csv")
