"""
TEST 10% POSITION SIZING WITH DRAWDOWN
========================================
Quick test to see drawdown at 10% sizing
"""
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = 'data_15m'
PAIRS = ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY']
INITIAL_CAPITAL = 500
POSITION_PCT = 0.10  # 10%


def strategy_with_equity(predictions, raw_data):
    """Trail after 1x target hit with 10% position sizing"""
    all_dates = sorted(set([d for p in predictions.values() for d in p.index]))

    capital = INITIAL_CAPITAL
    trades = []
    positions = []
    equity_curve = []

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
                if not pos.get('trailing_active') and row['bid_high'] >= pos['initial_target']:
                    pos['trailing_active'] = True
                    pos['trailing_stop'] = pos['initial_target']
                    pos['peak_price'] = row['bid_high']
                    continue

                if pos.get('trailing_active'):
                    if row['bid_high'] > pos['peak_price']:
                        pos['peak_price'] = row['bid_high']
                        new_stop = pos['initial_target'] + 0.75 * (pos['peak_price'] - pos['initial_target'])
                        pos['trailing_stop'] = max(pos['trailing_stop'], new_stop)

                    if row['bid_low'] <= pos['trailing_stop']:
                        exit_price = pos['trailing_stop']
                        profit_pct = (exit_price - pos['entry_price']) / pos['entry_price']
                        positions_to_close.append((pos, profit_pct, 'trailing_stop'))
                        continue

            else:  # short
                if not pos.get('trailing_active') and row['ask_low'] <= pos['initial_target']:
                    pos['trailing_active'] = True
                    pos['trailing_stop'] = pos['initial_target']
                    pos['peak_price'] = row['ask_low']
                    continue

                if pos.get('trailing_active'):
                    if row['ask_low'] < pos['peak_price']:
                        pos['peak_price'] = row['ask_low']
                        new_stop = pos['initial_target'] - 0.75 * (pos['initial_target'] - pos['peak_price'])
                        pos['trailing_stop'] = min(pos['trailing_stop'], new_stop)

                    if row['ask_high'] >= pos['trailing_stop']:
                        exit_price = pos['trailing_stop']
                        profit_pct = (pos['entry_price'] - exit_price) / pos['entry_price']
                        positions_to_close.append((pos, profit_pct, 'trailing_stop'))
                        continue

            # Time exit after 24 bars
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
                'date': date,
                'profit_pct': profit_pct,
                'profit_dollars': profit_dollars,
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

            position_size = capital * POSITION_PCT

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

        # Record equity
        equity_curve.append({'date': date, 'capital': capital})

    if len(trades) == 0:
        return None

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_curve)

    # Calculate drawdown
    equity_df['peak'] = equity_df['capital'].cummax()
    equity_df['drawdown'] = (equity_df['capital'] - equity_df['peak']) / equity_df['peak']
    max_drawdown = equity_df['drawdown'].min()

    return {
        'final_capital': capital,
        'total_return': (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL,
        'total_trades': len(trades_df),
        'max_drawdown': max_drawdown,
        'equity': equity_df
    }


# ============================================================================
# MAIN
# ============================================================================
print("="*100)
print("TESTING 10% POSITION SIZING WITH DRAWDOWN")
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
print("Results with 10% position sizing:")
print()

results = []
for name, filename in test_files:
    try:
        with open(filename, 'rb') as f:
            preds = pickle.load(f)

        result = strategy_with_equity(preds, all_raw_data)
        if result:
            results.append(result)
            print(f"{name}:")
            print(f"  Return: {result['total_return']:>7.1%} | Max DD: {result['max_drawdown']:>6.1%} | Trades: {result['total_trades']:>5,}")
    except FileNotFoundError:
        print(f"{name}: File not found")

print()
print("="*100)
print("SUMMARY")
print("="*100)
print()

if len(results) > 0:
    avg_return = np.mean([r['total_return'] for r in results])
    avg_max_dd = np.mean([r['max_drawdown'] for r in results])
    worst_dd = min([r['max_drawdown'] for r in results])

    print(f"Average Return:    {avg_return:>7.1%}")
    print(f"Average Max DD:    {avg_max_dd:>7.1%}")
    print(f"Worst Max DD:      {worst_dd:>7.1%}")
    print(f"Return/DD Ratio:   {abs(avg_return / avg_max_dd):>7.2f}")
    print()
    print("Comparison to 1% sizing:")
    print(f"  1% sizing: ~1.1% return, ~-2% max DD (estimated)")
    print(f"  10% sizing: {avg_return:.1%} return, {avg_max_dd:.1%} max DD")
    print()
    print(f"Returns increased: {avg_return / 0.011:.1f}x")
    print(f"Drawdown increased: {abs(avg_max_dd / -0.02):.1f}x (estimated)")
    print()

print()
