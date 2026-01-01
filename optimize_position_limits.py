"""
OPTIMIZE POSITION LIMITS
=========================
Test different position limit combinations to find optimal risk/reward balance
"""
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from backtest_quarterly_ladder_corrected import Position, INITIAL_CAPITAL, RISK_PER_TRADE, MIN_CONFIDENCE
from backtest_quarterly_ladder_corrected import EMERGENCY_STOP_LOSS_PCT, EMERGENCY_STOP_DAYS
from backtest_quarterly_ladder_corrected import TRAILING_STOP_TRIGGER, TRAILING_STOP_PCT
from backtest_quarterly_ladder_corrected import LADDER_LEVELS, LADDER_SCALE_PCT

print("="*100)
print("OPTIMIZING POSITION LIMITS")
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

print()

def run_backtest_with_limits(max_total, max_per_pair):
    """Run full backtest with specific position limits"""
    capital = INITIAL_CAPITAL
    carried_positions = []

    for quarter_name, quarter_preds in sorted(all_predictions.items()):
        # Get trading days
        all_trading_days = set()
        prediction_dates = set()

        for pair_df in quarter_preds.values():
            prediction_dates.update(pair_df.index)

        for pair, pair_df in all_raw_data.items():
            if len(prediction_dates) > 0:
                min_date = min(prediction_dates)
                max_date = max(prediction_dates)
                trading_days = pair_df[(pair_df.index >= min_date) & (pair_df.index <= max_date)].index
                all_trading_days.update(trading_days)

        all_trading_days = sorted(list(all_trading_days))

        for date in all_trading_days:
            # Get prices
            prices_dict = {}
            for pair, pair_df in all_raw_data.items():
                if date in pair_df.index:
                    row = pair_df.loc[date]
                    prices_dict[pair] = {'high': row['high'], 'low': row['low'], 'close': row['close']}

            # Update positions
            positions_to_close = []
            for position in carried_positions:
                if position.pair not in prices_dict:
                    continue
                high = prices_dict[position.pair]['high']
                low = prices_dict[position.pair]['low']
                close = prices_dict[position.pair]['close']
                exit_info = position.update(date, high, low, close)
                if exit_info is not None:
                    positions_to_close.append((position, exit_info))

            # Close positions
            for position, exit_info in positions_to_close:
                exit_reason, exit_price, current_profit = exit_info

                if position.direction == 'long':
                    raw_profit = (exit_price - position.entry_price) / position.entry_price
                else:
                    raw_profit = (position.entry_price - exit_price) / position.entry_price

                profit_pct = position.calculate_blended_profit(raw_profit)
                profit_dollars = profit_pct * (position.original_size * position.entry_price)

                capital += profit_dollars
                carried_positions.remove(position)

            # Open new positions
            if date not in prediction_dates:
                continue

            if len(carried_positions) >= max_total:
                continue

            for pair, pair_df in quarter_preds.items():
                if date not in pair_df.index:
                    continue

                pair_positions = [p for p in carried_positions if p.pair == pair]
                if len(pair_positions) >= max_per_pair:
                    continue

                row = pair_df.loc[date]

                breakout_high_prob = row['breakout_high_prob']
                breakout_low_prob = row['breakout_low_prob']
                max_prob = max(breakout_high_prob, breakout_low_prob)

                if max_prob <= MIN_CONFIDENCE:
                    continue

                assumed_risk_pct = 0.02
                risk_amount = capital * RISK_PER_TRADE
                price = row['close']
                position_size = risk_amount / (price * assumed_risk_pct)

                if breakout_high_prob > breakout_low_prob:
                    direction = 'long'
                    breakout_level = row['high_20d']
                    target = breakout_level * 1.005
                else:
                    direction = 'short'
                    breakout_level = row['low_20d']
                    target = breakout_level * 0.995

                position = Position(pair, date, price, direction, position_size, target, max_prob)
                carried_positions.append(position)

    return capital


# Test different combinations
print("Testing position limit combinations...")
print()

results = []

# Test combinations
total_limits = [40, 50, 60, 70, 80]
per_pair_limits = [8, 10, 12]

for max_total in total_limits:
    for max_per_pair in per_pair_limits:
        final_capital = run_backtest_with_limits(max_total, max_per_pair)
        total_return = (final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL

        results.append({
            'max_total': max_total,
            'max_per_pair': max_per_pair,
            'final_capital': final_capital,
            'total_return': total_return
        })

        print(f"Total: {max_total:2d} | Per-pair: {max_per_pair:2d} | Final: ${final_capital:>8,.0f} | Return: {total_return:>7.1%}")

print()
print("="*100)
print("BEST COMBINATIONS")
print("="*100)
print()

# Sort by return
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('total_return', ascending=False)

print("Top 5 by Total Return:")
for i, row in results_df.head(5).iterrows():
    print(f"  {row['max_total']:2.0f} total, {row['max_per_pair']:2.0f} per-pair: ${row['final_capital']:>8,.0f} ({row['total_return']:>7.1%})")

print()
print("="*100)
