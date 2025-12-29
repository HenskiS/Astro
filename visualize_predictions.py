"""
PREDICTION VISUALIZER
Examine individual predictions and actual price movements
"""
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("PREDICTION VISUALIZER")
print("="*100)
print()

# Load predictions
print("Loading predictions...")
with open('model_predictions.pkl', 'rb') as f:
    all_predictions = pickle.load(f)

print(f"Loaded predictions for {len(all_predictions)} periods")
print()

# Let user select period and pair
print("Available periods:")
for i, period_name in enumerate(all_predictions.keys(), 1):
    print(f"  {i}. {period_name}")

period_choice = input("\nSelect period (1-5): ").strip()
period_name = list(all_predictions.keys())[int(period_choice) - 1]
period_preds = all_predictions[period_name]

print(f"\nAvailable pairs in {period_name}:")
pairs = list(period_preds.keys())
for i, pair in enumerate(pairs, 1):
    print(f"  {i}. {pair}")

pair_choice = input("\nSelect pair (1-8): ").strip()
selected_pair = pairs[int(pair_choice) - 1]

predictions_df = period_preds[selected_pair]
print(f"\nLoaded {len(predictions_df)} predictions for {selected_pair} in {period_name}")
print()

# Ensure date is a column, not index
if 'date' not in predictions_df.columns and 'date' in predictions_df.index.names:
    predictions_df = predictions_df.reset_index()
elif 'date' in predictions_df.index.names:
    # Date is both in index and columns, just reset without the index
    predictions_df = predictions_df.reset_index(drop=True)

# Filter to predictions with high confidence (>70%)
predictions_df['max_prob'] = predictions_df[['breakout_high_prob', 'breakout_low_prob']].max(axis=1)
predictions_df['prediction'] = predictions_df.apply(
    lambda row: 'HIGH' if row['breakout_high_prob'] > row['breakout_low_prob'] else 'LOW',
    axis=1
)

# Calculate if prediction was correct (looking ahead 10 days)
predictions_df = predictions_df.sort_values('date').reset_index(drop=True)

def check_prediction_accuracy(df, idx):
    """Check if prediction at index idx was correct"""
    if idx >= len(df) - 10:
        return None, None

    current = df.iloc[idx]
    future_10d = df.iloc[idx:idx+10]

    if current['prediction'] == 'HIGH':
        # Check if high was broken in next 10 days
        hit = (future_10d['high'].max() > current['high_20d'])
        target_level = current['high_20d']
    else:
        # Check if low was broken in next 10 days
        hit = (future_10d['low'].min() < current['low_20d'])
        target_level = current['low_20d']

    return hit, target_level

predictions_df['prediction_correct'] = None
predictions_df['target_level'] = None

for idx in range(len(predictions_df) - 10):
    correct, target = check_prediction_accuracy(predictions_df, idx)
    predictions_df.at[idx, 'prediction_correct'] = correct
    predictions_df.at[idx, 'target_level'] = target

# Filter to high confidence predictions only
high_conf_df = predictions_df[predictions_df['max_prob'] > 0.70].copy()
high_conf_df = high_conf_df[high_conf_df['prediction_correct'].notna()].copy()

print(f"Found {len(high_conf_df)} high confidence (>70%) predictions with outcomes")
correct_count = high_conf_df['prediction_correct'].sum()
accuracy = correct_count / len(high_conf_df) * 100
print(f"Accuracy: {correct_count}/{len(high_conf_df)} = {accuracy:.1f}%")
print()

# Interactive visualization
class PredictionViewer:
    def __init__(self, predictions_df, full_df):
        self.predictions = predictions_df.reset_index(drop=True)
        self.full_df = full_df
        self.current_idx = 0

        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(14, 10))
        self.fig.subplots_adjust(bottom=0.15)

        # Create navigation buttons
        ax_prev = plt.axes([0.3, 0.05, 0.1, 0.04])
        ax_next = plt.axes([0.6, 0.05, 0.1, 0.04])
        ax_rand = plt.axes([0.45, 0.05, 0.1, 0.04])

        self.btn_prev = Button(ax_prev, 'Previous')
        self.btn_next = Button(ax_next, 'Next')
        self.btn_rand = Button(ax_rand, 'Random')

        self.btn_prev.on_clicked(self.prev)
        self.btn_next.on_clicked(self.next)
        self.btn_rand.on_clicked(self.random)

        # Filter buttons
        ax_correct = plt.axes([0.05, 0.05, 0.1, 0.04])
        ax_wrong = plt.axes([0.16, 0.05, 0.1, 0.04])
        ax_all = plt.axes([0.85, 0.05, 0.1, 0.04])

        self.btn_correct = Button(ax_correct, 'Correct Only')
        self.btn_wrong = Button(ax_wrong, 'Wrong Only')
        self.btn_all = Button(ax_all, 'All')

        self.btn_correct.on_clicked(self.filter_correct)
        self.btn_wrong.on_clicked(self.filter_wrong)
        self.btn_all.on_clicked(self.filter_all)

        self.filter_mode = 'all'
        self.update_plot()

    def filter_correct(self, event):
        self.predictions = high_conf_df[high_conf_df['prediction_correct'] == True].reset_index(drop=True)
        self.current_idx = 0
        self.filter_mode = 'correct'
        self.update_plot()

    def filter_wrong(self, event):
        self.predictions = high_conf_df[high_conf_df['prediction_correct'] == False].reset_index(drop=True)
        self.current_idx = 0
        self.filter_mode = 'wrong'
        self.update_plot()

    def filter_all(self, event):
        self.predictions = high_conf_df.copy()
        self.current_idx = 0
        self.filter_mode = 'all'
        self.update_plot()

    def prev(self, event):
        self.current_idx = max(0, self.current_idx - 1)
        self.update_plot()

    def next(self, event):
        self.current_idx = min(len(self.predictions) - 1, self.current_idx + 1)
        self.update_plot()

    def random(self, event):
        self.current_idx = np.random.randint(0, len(self.predictions))
        self.update_plot()

    def update_plot(self):
        if len(self.predictions) == 0:
            self.ax1.clear()
            self.ax2.clear()
            self.ax1.text(0.5, 0.5, 'No predictions matching filter',
                         ha='center', va='center', transform=self.ax1.transAxes)
            plt.draw()
            return

        pred_row = self.predictions.iloc[self.current_idx]

        # Get data around prediction
        pred_date = pred_row['date']
        pred_idx = self.full_df[self.full_df['date'] == pred_date].index[0]

        # Get 50 days before and 10 days after
        start_idx = max(0, pred_idx - 50)
        end_idx = min(len(self.full_df), pred_idx + 11)

        window_df = self.full_df.iloc[start_idx:end_idx].copy()

        # Clear axes
        self.ax1.clear()
        self.ax2.clear()

        # Plot 1: Full price context
        dates = window_df['date']
        self.ax1.plot(dates, window_df['close'], 'k-', linewidth=1.5, label='Close')
        self.ax1.plot(dates, window_df['high'], 'gray', alpha=0.3, linewidth=0.5)
        self.ax1.plot(dates, window_df['low'], 'gray', alpha=0.3, linewidth=0.5)

        # Plot 20d range at prediction time
        self.ax1.axhline(y=pred_row['high_20d'], color='red', linestyle='--', alpha=0.5, label='20D High')
        self.ax1.axhline(y=pred_row['low_20d'], color='blue', linestyle='--', alpha=0.5, label='20D Low')

        # Mark prediction point
        self.ax1.axvline(x=pred_date, color='green', linestyle='-', linewidth=2, alpha=0.7, label='Prediction')

        # Highlight target level
        if pred_row['prediction'] == 'HIGH':
            self.ax1.axhline(y=pred_row['target_level'], color='red', linestyle='-', linewidth=2,
                           alpha=0.8, label=f"Target: {pred_row['target_level']:.5f}")
        else:
            self.ax1.axhline(y=pred_row['target_level'], color='blue', linestyle='-', linewidth=2,
                           alpha=0.8, label=f"Target: {pred_row['target_level']:.5f}")

        self.ax1.set_title(f"{selected_pair} - {period_name} | Prediction #{self.current_idx+1}/{len(self.predictions)} ({self.filter_mode})",
                          fontsize=14, fontweight='bold')
        self.ax1.set_ylabel('Price', fontsize=12)
        self.ax1.legend(loc='upper left')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.tick_params(axis='x', rotation=45)

        # Plot 2: Zoom on prediction + 10 days
        future_window = window_df[window_df['date'] >= pred_date].iloc[:11]

        self.ax2.plot(future_window['date'], future_window['high'], 'g^', markersize=4, label='High')
        self.ax2.plot(future_window['date'], future_window['low'], 'rv', markersize=4, label='Low')
        self.ax2.plot(future_window['date'], future_window['close'], 'ko-', linewidth=2, markersize=6, label='Close')

        # Target level
        if pred_row['prediction'] == 'HIGH':
            self.ax2.axhline(y=pred_row['target_level'], color='red', linestyle='-', linewidth=2, alpha=0.8)
        else:
            self.ax2.axhline(y=pred_row['target_level'], color='blue', linestyle='-', linewidth=2, alpha=0.8)

        # Prediction details
        result_text = "✓ CORRECT" if pred_row['prediction_correct'] else "✗ WRONG"
        result_color = 'green' if pred_row['prediction_correct'] else 'red'

        info_text = (
            f"Date: {pred_date.strftime('%Y-%m-%d')}\n"
            f"Prediction: {pred_row['prediction']} BREAKOUT\n"
            f"Confidence: {pred_row['max_prob']:.1%}\n"
            f"High Prob: {pred_row['breakout_high_prob']:.1%}\n"
            f"Low Prob: {pred_row['breakout_low_prob']:.1%}\n"
            f"Entry: {pred_row['close']:.5f}\n"
            f"Target: {pred_row['target_level']:.5f}\n"
            f"Result: {result_text}"
        )

        self.ax2.text(0.02, 0.98, info_text, transform=self.ax2.transAxes,
                     fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                     family='monospace')

        # Add result indicator
        self.ax2.text(0.98, 0.98, result_text, transform=self.ax2.transAxes,
                     fontsize=16, verticalalignment='top', horizontalalignment='right',
                     fontweight='bold', color=result_color,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        self.ax2.set_title('Next 10 Days After Prediction', fontsize=12, fontweight='bold')
        self.ax2.set_xlabel('Date', fontsize=12)
        self.ax2.set_ylabel('Price', fontsize=12)
        self.ax2.legend(loc='lower left')
        self.ax2.grid(True, alpha=0.3)
        self.ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.draw()

# Create viewer
viewer = PredictionViewer(high_conf_df, predictions_df)

print("="*100)
print("CONTROLS:")
print("  - Previous/Next: Navigate through predictions")
print("  - Random: Jump to random prediction")
print("  - Correct Only: Show only correct predictions")
print("  - Wrong Only: Show only incorrect predictions")
print("  - All: Show all predictions")
print("="*100)

plt.show()
