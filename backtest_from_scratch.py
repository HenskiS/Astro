"""
TRADING FROM SCRATCH
====================
Forget optimization - let's design a strategy from first principles.

Key insight: Model has 86% directional accuracy at 0.80 confidence
Question: How should we trade this?

Start simple, iterate based on what we learn.
"""
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = 'data_15m'
PAIRS = ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY']
INITIAL_CAPITAL = 500


# ============================================================================
# RAW MODEL TRADING
# ============================================================================
def strategy_raw_model(predictions, raw_data):
    """
    RAW: Just follow the model exactly
    - Model predicts: "breakout to X will happen in next 24 bars"
    - We: Enter now, target X, exit after 24 bars
    - No stops, no early exits, just pure model following
    """
    all_dates = sorted(set([d for p in predictions.values() for d in p.index]))

    capital = INITIAL_CAPITAL
    trades = []
    positions = []  # Can hold multiple positions

    for date in all_dates:
        # Update existing positions
        positions_to_close = []
        for pos in positions:
            pair = pos['pair']
            if date not in raw_data[pair].index:
                continue

            row = raw_data[pair].loc[date]
            pos['bars_held'] += 1

            # Check if target hit
            if pos['direction'] == 'long':
                if row['bid_high'] >= pos['target']:
                    # Target hit!
                    exit_price = pos['target']
                    profit_pct = (exit_price - pos['entry_price']) / pos['entry_price']
                    exit_reason = 'target_hit'
                    positions_to_close.append((pos, profit_pct, exit_reason))
                    continue
            else:
                if row['ask_low'] <= pos['target']:
                    # Target hit!
                    exit_price = pos['target']
                    profit_pct = (pos['entry_price'] - exit_price) / pos['entry_price']
                    exit_reason = 'target_hit'
                    positions_to_close.append((pos, profit_pct, exit_reason))
                    continue

            # Time exit after prediction window (24 bars)
            if pos['bars_held'] >= 24:
                if pos['direction'] == 'long':
                    exit_price = row['bid_close']
                    profit_pct = (exit_price - pos['entry_price']) / pos['entry_price']
                else:
                    exit_price = row['ask_close']
                    profit_pct = (pos['entry_price'] - exit_price) / pos['entry_price']
                exit_reason = 'time_exit'
                positions_to_close.append((pos, profit_pct, exit_reason))

        # Close positions
        for pos, profit_pct, exit_reason in positions_to_close:
            profit_dollars = profit_pct * pos['size']
            capital += profit_dollars
            trades.append({
                'profit_pct': profit_pct,
                'exit_reason': exit_reason,
                'bars_held': pos['bars_held'],
                'confidence': pos['confidence']
            })
            positions.remove(pos)

        # Look for new signals
        for pair in PAIRS:
            if date not in predictions[pair].index:
                continue
            if date not in raw_data[pair].index:
                continue

            pred = predictions[pair].loc[date]
            max_prob = max(pred['breakout_high_prob'], pred['breakout_low_prob'])

            # Only take high confidence signals (where model is 86% accurate)
            if max_prob <= 0.80:
                continue

            # Enter position following model's prediction
            row = raw_data[pair].loc[date]
            if pred['breakout_high_prob'] > pred['breakout_low_prob']:
                direction = 'long'
                entry_price = row['ask_open']
                target = pred['high_80p']  # Model's predicted breakout level
            else:
                direction = 'short'
                entry_price = row['bid_open']
                target = pred['low_80p']  # Model's predicted breakout level

            position_size = capital * 0.01  # 1% of capital

            positions.append({
                'pair': pair,
                'direction': direction,
                'entry_price': entry_price,
                'target': target,
                'size': position_size,
                'bars_held': 0,
                'confidence': max_prob
            })

    if len(trades) == 0:
        return None

    trades_df = pd.DataFrame(trades)
    winners = trades_df[trades_df['profit_pct'] > 0]
    target_hits = trades_df[trades_df['exit_reason'] == 'target_hit']

    return {
        'final_capital': capital,
        'total_return': (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL,
        'total_trades': len(trades_df),
        'win_rate': len(winners) / len(trades_df),
        'target_hit_rate': len(target_hits) / len(trades_df),
        'avg_profit': trades_df['profit_pct'].mean(),
        'avg_winner': winners['profit_pct'].mean() if len(winners) > 0 else 0,
        'avg_loser': trades_df[trades_df['profit_pct'] < 0]['profit_pct'].mean() if len(trades_df[trades_df['profit_pct'] < 0]) > 0 else 0,
        'trades': trades_df
    }


# ============================================================================
# 3X TARGET STRATEGY
# ============================================================================
def strategy_3x_target(predictions, raw_data):
    """
    3X: Target 3x the initial predicted level
    - Initial target confirms breakout started
    - But price typically moves 3x further
    - So just target 3x from entry
    """
    all_dates = sorted(set([d for p in predictions.values() for d in p.index]))

    capital = INITIAL_CAPITAL
    trades = []
    positions = []

    for date in all_dates:
        # Update existing positions
        positions_to_close = []
        for pos in positions:
            pair = pos['pair']
            if date not in raw_data[pair].index:
                continue

            row = raw_data[pair].loc[date]
            pos['bars_held'] += 1

            # Calculate 2x target from entry
            if pos['direction'] == 'long':
                # 2x the distance to initial target
                target_2x = pos['entry_price'] + 2 * (pos['initial_target'] - pos['entry_price'])

                # Check if 2x target hit
                if row['bid_high'] >= target_2x:
                    exit_price = target_2x
                    profit_pct = (exit_price - pos['entry_price']) / pos['entry_price']
                    exit_reason = '2x_target'
                    positions_to_close.append((pos, profit_pct, exit_reason))
                    continue
            else:
                # 2x the distance to initial target
                target_2x = pos['entry_price'] - 2 * (pos['entry_price'] - pos['initial_target'])

                # Check if 2x target hit
                if row['ask_low'] <= target_2x:
                    exit_price = target_2x
                    profit_pct = (pos['entry_price'] - exit_price) / pos['entry_price']
                    exit_reason = '2x_target'
                    positions_to_close.append((pos, profit_pct, exit_reason))
                    continue

            # Time exit after 24 bars
            if pos['bars_held'] >= 24:
                if pos['direction'] == 'long':
                    exit_price = row['bid_close']
                    profit_pct = (exit_price - pos['entry_price']) / pos['entry_price']
                else:
                    exit_price = row['ask_close']
                    profit_pct = (pos['entry_price'] - exit_price) / pos['entry_price']
                exit_reason = 'time_exit'
                positions_to_close.append((pos, profit_pct, exit_reason))

        # Close positions
        for pos, profit_pct, exit_reason in positions_to_close:
            profit_dollars = profit_pct * pos['size']
            capital += profit_dollars
            trades.append({
                'profit_pct': profit_pct,
                'exit_reason': exit_reason,
                'bars_held': pos['bars_held'],
                'confidence': pos['confidence']
            })
            positions.remove(pos)

        # Look for new signals
        for pair in PAIRS:
            if date not in predictions[pair].index:
                continue
            if date not in raw_data[pair].index:
                continue

            pred = predictions[pair].loc[date]
            max_prob = max(pred['breakout_high_prob'], pred['breakout_low_prob'])

            if max_prob <= 0.80:
                continue

            row = raw_data[pair].loc[date]
            if pred['breakout_high_prob'] > pred['breakout_low_prob']:
                direction = 'long'
                entry_price = row['ask_open']
                initial_target = pred['high_80p']
            else:
                direction = 'short'
                entry_price = row['bid_open']
                initial_target = pred['low_80p']

            position_size = capital * 0.01

            positions.append({
                'pair': pair,
                'direction': direction,
                'entry_price': entry_price,
                'initial_target': initial_target,
                'size': position_size,
                'bars_held': 0,
                'confidence': max_prob
            })

    if len(trades) == 0:
        return None

    trades_df = pd.DataFrame(trades)
    winners = trades_df[trades_df['profit_pct'] > 0]
    target_hits = trades_df[trades_df['exit_reason'] == '2x_target']

    return {
        'final_capital': capital,
        'total_return': (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL,
        'total_trades': len(trades_df),
        'win_rate': len(winners) / len(trades_df),
        'target_hit_rate': len(target_hits) / len(trades_df),
        'avg_profit': trades_df['profit_pct'].mean(),
        'avg_winner': winners['profit_pct'].mean() if len(winners) > 0 else 0,
        'avg_loser': trades_df[trades_df['profit_pct'] < 0]['profit_pct'].mean() if len(trades_df[trades_df['profit_pct'] < 0]) > 0 else 0,
        'trades': trades_df
    }


# ============================================================================
# SIMPLEST POSSIBLE STRATEGY
# ============================================================================
def strategy_v1_simple(predictions, raw_data):
    """
    V1: Dead simple
    - Take all signals > 0.80 confidence
    - Fixed position size (1% of capital)
    - Fixed stop loss (-2%)
    - Fixed take profit (+2%)
    - No fancy exits
    """
    all_dates = sorted(set([d for p in predictions.values() for d in p.index]))

    capital = INITIAL_CAPITAL
    trades = []
    position = None  # Can only hold 1 position at a time

    for date in all_dates:
        # Update existing position
        if position is not None:
            pair = position['pair']
            if date not in raw_data[pair].index:
                continue

            row = raw_data[pair].loc[date]
            position['bars_held'] += 1

            if position['direction'] == 'long':
                current_price = row['bid_close']
                profit_pct = (current_price - position['entry_price']) / position['entry_price']
            else:
                current_price = row['ask_close']
                profit_pct = (position['entry_price'] - current_price) / position['entry_price']

            # Exit conditions
            exit_reason = None
            if profit_pct >= 0.02:  # +2% take profit
                exit_reason = 'take_profit'
            elif profit_pct <= -0.02:  # -2% stop loss
                exit_reason = 'stop_loss'
            elif position['bars_held'] >= 48:  # 12 hours max hold
                exit_reason = 'time_exit'

            if exit_reason:
                profit_dollars = profit_pct * position['size']
                capital += profit_dollars
                trades.append({
                    'profit_pct': profit_pct,
                    'exit_reason': exit_reason,
                    'bars_held': position['bars_held']
                })
                position = None

        # Look for new signals
        if position is None:
            for pair in PAIRS:
                if date not in predictions[pair].index:
                    continue
                if date not in raw_data[pair].index:
                    continue

                pred = predictions[pair].loc[date]
                max_prob = max(pred['breakout_high_prob'], pred['breakout_low_prob'])

                # Only take high confidence signals
                if max_prob <= 0.80:
                    continue

                # Enter position
                row = raw_data[pair].loc[date]
                if pred['breakout_high_prob'] > pred['breakout_low_prob']:
                    direction = 'long'
                    entry_price = row['ask_open']
                else:
                    direction = 'short'
                    entry_price = row['bid_open']

                position_size = capital * 0.01  # 1% of capital

                position = {
                    'pair': pair,
                    'direction': direction,
                    'entry_price': entry_price,
                    'size': position_size,
                    'bars_held': 0,
                    'confidence': max_prob
                }
                break  # Only take first signal

    if len(trades) == 0:
        return None

    trades_df = pd.DataFrame(trades)
    winners = trades_df[trades_df['profit_pct'] > 0]

    return {
        'final_capital': capital,
        'total_return': (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL,
        'total_trades': len(trades_df),
        'win_rate': len(winners) / len(trades_df),
        'avg_profit': trades_df['profit_pct'].mean(),
        'avg_winner': winners['profit_pct'].mean() if len(winners) > 0 else 0,
        'avg_loser': trades_df[trades_df['profit_pct'] < 0]['profit_pct'].mean() if len(trades_df[trades_df['profit_pct'] < 0]) > 0 else 0,
        'trades': trades_df
    }


# ============================================================================
# MAIN
# ============================================================================
print("="*100)
print("TRADING FROM SCRATCH")
print("="*100)
print()

print("Loading data...")
all_raw_data = {}
for pair in PAIRS:
    df = pd.read_csv(f'{DATA_DIR}/{pair}_15m.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    all_raw_data[pair] = df

test_files = [
    ('2021 Test', 'test_predictions_15m_2021_test.pkl'),
    ('2022 Test', 'test_predictions_15m_2022_test.pkl'),
    ('2023 Test', 'test_predictions_15m_2023_test.pkl'),
    ('2024 Test', 'test_predictions_15m_2024_test.pkl'),
    ('2025 Test', 'test_predictions_15m_2025_test.pkl'),
]

print()
print("="*100)
print("STRATEGY: 2X TARGET")
print("="*100)
print("Rules:")
print("  - Take all signals > 0.80 confidence")
print("  - Target = 2x the initial predicted move")
print("  - Rationale: Median move is 2.9x, so 2x should hit ~65% of time")
print("  - Exit when 2x target hit OR after 24 bars")
print("  - 1% position size, can hold multiple positions")
print()

for name, filename in test_files:
    try:
        with open(filename, 'rb') as f:
            preds = pickle.load(f)

        result = strategy_3x_target(preds, all_raw_data)
        if result:
            print(f"{name}:")
            print(f"  Return: {result['total_return']:>7.1%} | Win Rate: {result['win_rate']:>5.1%} | 2x Hit: {result['target_hit_rate']:>5.1%} | Trades: {result['total_trades']:>4}")
            print(f"  Avg P&L: {result['avg_profit']:>6.2%} | Avg Win: {result['avg_winner']:>6.2%} | Avg Loss: {result['avg_loser']:>6.2%}")
            print()
    except FileNotFoundError:
        print(f"{name}: File not found")

print()
print("="*100)
print("WHAT DO WE LEARN?")
print("="*100)
print()
print("From the 2x target strategy:")
print("  1. 2x Hit Rate: Should be ~65% (median is 2.9x)")
print("  2. Win Rate: Should be higher than 3x's 69%")
print("  3. Avg Winner: Should be ~0.18% (double the 1x target)")
print()
print("Comparison:")
print("  - Raw 1x: 94% win rate, 0.08% avg win, 0.6-1.1% return")
print("  - 3x target: 69% win rate, 0.18% avg win, 0.4-0.9% return")
print("  - 2x target: ??? (testing now)")
print()
print("Next: If 2x works well, test scaled exits (50% at 1x, 50% at 3x)")
print()
