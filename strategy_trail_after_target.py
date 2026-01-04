"""
TRAILING STOP AFTER TARGET HIT
===============================
Strategy:
1. Enter on high confidence signal (>0.80)
2. When 1x target hits (94% of trades), activate trailing stop
3. Trail at 75% of max favorable move from target hit
4. This captures the continuation while locking in profit

Theory: The 1x target confirms the breakout is real. Then trail to ride the momentum.
"""
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = 'data_15m'
PAIRS = ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY']
INITIAL_CAPITAL = 500


def strategy_trail_after_target(predictions, raw_data):
    """
    Trail after 1x target hit
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

            if pos['direction'] == 'long':
                # Check if 1x target hit (activates trailing)
                if not pos.get('trailing_active') and row['bid_high'] >= pos['initial_target']:
                    pos['trailing_active'] = True
                    pos['trailing_stop'] = pos['initial_target']  # Start trail at target
                    pos['peak_price'] = row['bid_high']
                    continue  # Don't exit yet, start trailing

                # Update trailing stop if active
                if pos.get('trailing_active'):
                    # Update peak
                    if row['bid_high'] > pos['peak_price']:
                        pos['peak_price'] = row['bid_high']
                        # Trail at 75% of move from target
                        new_stop = pos['initial_target'] + 0.75 * (pos['peak_price'] - pos['initial_target'])
                        pos['trailing_stop'] = max(pos['trailing_stop'], new_stop)

                    # Check if stopped out
                    if row['bid_low'] <= pos['trailing_stop']:
                        exit_price = pos['trailing_stop']
                        profit_pct = (exit_price - pos['entry_price']) / pos['entry_price']
                        positions_to_close.append((pos, profit_pct, 'trailing_stop'))
                        continue

            else:  # short
                # Check if 1x target hit (activates trailing)
                if not pos.get('trailing_active') and row['ask_low'] <= pos['initial_target']:
                    pos['trailing_active'] = True
                    pos['trailing_stop'] = pos['initial_target']  # Start trail at target
                    pos['peak_price'] = row['ask_low']
                    continue  # Don't exit yet, start trailing

                # Update trailing stop if active
                if pos.get('trailing_active'):
                    # Update peak
                    if row['ask_low'] < pos['peak_price']:
                        pos['peak_price'] = row['ask_low']
                        # Trail at 75% of move from target
                        new_stop = pos['initial_target'] - 0.75 * (pos['initial_target'] - pos['peak_price'])
                        pos['trailing_stop'] = min(pos['trailing_stop'], new_stop)

                    # Check if stopped out
                    if row['ask_high'] >= pos['trailing_stop']:
                        exit_price = pos['trailing_stop']
                        profit_pct = (pos['entry_price'] - exit_price) / pos['entry_price']
                        positions_to_close.append((pos, profit_pct, 'trailing_stop'))
                        continue

            # Time exit after 24 bars (failsafe)
            if pos['bars_held'] >= 24:
                if pos['direction'] == 'long':
                    exit_price = row['bid_close']
                    profit_pct = (exit_price - pos['entry_price']) / pos['entry_price']
                else:
                    exit_price = row['ask_close']
                    profit_pct = (pos['entry_price'] - exit_price) / pos['entry_price']
                positions_to_close.append((pos, profit_pct, 'time_exit'))

        # Close positions
        for pos, profit_pct, exit_reason in positions_to_close:
            profit_dollars = profit_pct * pos['size']
            capital += profit_dollars
            trades.append({
                'profit_pct': profit_pct,
                'exit_reason': exit_reason,
                'bars_held': pos['bars_held'],
                'confidence': pos['confidence'],
                'trailing_activated': pos.get('trailing_active', False)
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

            position_size = capital * 0.01  # 1% of capital

            positions.append({
                'pair': pair,
                'direction': direction,
                'entry_price': entry_price,
                'initial_target': initial_target,
                'size': position_size,
                'bars_held': 0,
                'confidence': max_prob,
                'trailing_active': False
            })

    if len(trades) == 0:
        return None

    trades_df = pd.DataFrame(trades)
    winners = trades_df[trades_df['profit_pct'] > 0]
    trailing_trades = trades_df[trades_df['trailing_activated'] == True]

    return {
        'final_capital': capital,
        'total_return': (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL,
        'total_trades': len(trades_df),
        'win_rate': len(winners) / len(trades_df),
        'trailing_rate': len(trailing_trades) / len(trades_df),
        'avg_profit': trades_df['profit_pct'].mean(),
        'avg_winner': winners['profit_pct'].mean() if len(winners) > 0 else 0,
        'avg_loser': trades_df[trades_df['profit_pct'] < 0]['profit_pct'].mean() if len(trades_df[trades_df['profit_pct'] < 0]) > 0 else 0,
        'trades': trades_df
    }


# ============================================================================
# MAIN
# ============================================================================
print("="*100)
print("STRATEGY: TRAIL AFTER TARGET HIT")
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
print("Strategy Rules:")
print("  1. Enter on signals > 0.80 confidence")
print("  2. When 1x target hits, activate trailing stop")
print("  3. Trail at 75% of max favorable from target")
print("  4. Exit on trail stop hit OR after 24 bars")
print("  5. 1% position size, multiple positions allowed")
print()

results = []

for name, filename in test_files:
    try:
        with open(filename, 'rb') as f:
            preds = pickle.load(f)

        result = strategy_trail_after_target(preds, all_raw_data)
        if result:
            results.append(result)
            print(f"{name}:")
            print(f"  Return: {result['total_return']:>7.1%} | Win Rate: {result['win_rate']:>5.1%} | Trail Rate: {result['trailing_rate']:>5.1%} | Trades: {result['total_trades']:>4}")
            print(f"  Avg P&L: {result['avg_profit']:>6.2%} | Avg Win: {result['avg_winner']:>6.2%} | Avg Loss: {result['avg_loser']:>6.2%}")
            print()
    except FileNotFoundError:
        print(f"{name}: File not found")

print()
print("="*100)
print("COMPARISON TO RAW 1X")
print("="*100)
print()
print("Raw 1x baseline: 0.6-1.1% return, 94% win rate, 0.08% avg win")
print()

if len(results) > 0:
    avg_return = np.mean([r['total_return'] for r in results])
    avg_win_rate = np.mean([r['win_rate'] for r in results])
    avg_winner = np.mean([r['avg_winner'] for r in results])

    print(f"Trail strategy: {avg_return:.1%} avg return, {avg_win_rate:.1%} win rate, {avg_winner:.2%} avg win")
    print()

    if avg_return > 0.009:  # Better than 0.9% average
        print("[SUCCESS] Trailing stop improves returns!")
        print(f"  Capturing more of the move: {avg_winner:.2%} vs 0.08% (1x)")
    else:
        print("[MIXED] Similar or worse than baseline")
        print("  May need to adjust trail percentage or add other rules")

print()
