"""
TEST OANDA NATIVE TRAILING STOPS
=================================
Compare current dynamic trailing stop (75% from target)
vs Oanda-native fixed-pip trailing stops.

Current: Trail at 75% of gain above target (dynamic)
Oanda: Fixed pip distance from current price (simple)
"""
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = 'data_15m'
PAIRS = ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY']
INITIAL_CAPITAL = 500
POSITION_PCT = 0.10

print("="*100)
print("TESTING OANDA NATIVE TRAILING STOPS")
print("="*100)
print()


def pip_value(pair, price):
    """Get pip value for the pair"""
    if 'JPY' in pair:
        return 0.01  # JPY pairs: 1 pip = 0.01
    else:
        return 0.0001  # Other pairs: 1 pip = 0.0001


def strategy_dynamic_trail(predictions, raw_data):
    """Current strategy: Dynamic trail at 75% from target"""
    all_dates = sorted(set([d for p in predictions.values() for d in p.index]))

    capital = INITIAL_CAPITAL
    positions = []
    trades = []

    for date in all_dates:
        positions_to_close = []

        for pos in positions:
            pair = pos['pair']
            if date not in raw_data[pair].index:
                continue

            row = raw_data[pair].loc[date]
            pos['bars_held'] += 1

            # Emergency stop (5%)
            if pos['direction'] == 'long':
                emergency_stop = pos['entry_price'] * 0.95
                if row['bid_low'] <= emergency_stop:
                    exit_price = emergency_stop
                    profit_pct = (exit_price - pos['entry_price']) / pos['entry_price']
                    positions_to_close.append((pos, profit_pct, 'emergency_stop'))
                    continue
            else:
                emergency_stop = pos['entry_price'] * 1.05
                if row['ask_high'] >= emergency_stop:
                    exit_price = emergency_stop
                    profit_pct = (pos['entry_price'] - exit_price) / pos['entry_price']
                    positions_to_close.append((pos, profit_pct, 'emergency_stop'))
                    continue

            # Dynamic trailing stop
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
            else:
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

            # Time exit
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
            positions.remove(pos)
            trades.append({
                'profit_pct': profit_pct,
                'profit_dollars': profit_dollars,
                'exit_reason': exit_reason,
                'bars_held': pos['bars_held']
            })

        # New signals
        for pair in PAIRS:
            if date not in predictions[pair].index or date not in raw_data[pair].index:
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
                'trailing_active': False
            })

    return capital, pd.DataFrame(trades)


def strategy_fixed_pip_trail(predictions, raw_data, trail_pips):
    """Oanda-style: Fixed pip distance from current price"""
    all_dates = sorted(set([d for p in predictions.values() for d in p.index]))

    capital = INITIAL_CAPITAL
    positions = []
    trades = []

    for date in all_dates:
        positions_to_close = []

        for pos in positions:
            pair = pos['pair']
            if date not in raw_data[pair].index:
                continue

            row = raw_data[pair].loc[date]
            pos['bars_held'] += 1
            pip = pip_value(pair, pos['entry_price'])

            # Emergency stop (5%)
            if pos['direction'] == 'long':
                emergency_stop = pos['entry_price'] * 0.95
                if row['bid_low'] <= emergency_stop:
                    exit_price = emergency_stop
                    profit_pct = (exit_price - pos['entry_price']) / pos['entry_price']
                    positions_to_close.append((pos, profit_pct, 'emergency_stop'))
                    continue
            else:
                emergency_stop = pos['entry_price'] * 1.05
                if row['ask_high'] >= emergency_stop:
                    exit_price = emergency_stop
                    profit_pct = (pos['entry_price'] - exit_price) / pos['entry_price']
                    positions_to_close.append((pos, profit_pct, 'emergency_stop'))
                    continue

            # Fixed pip trailing stop (activates when target hit)
            if pos['direction'] == 'long':
                if not pos.get('trailing_active') and row['bid_high'] >= pos['initial_target']:
                    pos['trailing_active'] = True
                    # Initialize stop at current high minus trail_pips
                    pos['trailing_stop'] = row['bid_high'] - (trail_pips * pip)
                    pos['peak_price'] = row['bid_high']
                    continue

                if pos.get('trailing_active'):
                    # Update peak and stop if new high
                    if row['bid_high'] > pos['peak_price']:
                        pos['peak_price'] = row['bid_high']
                        # Stop trails at fixed distance
                        pos['trailing_stop'] = pos['peak_price'] - (trail_pips * pip)

                    if row['bid_low'] <= pos['trailing_stop']:
                        exit_price = pos['trailing_stop']
                        profit_pct = (exit_price - pos['entry_price']) / pos['entry_price']
                        positions_to_close.append((pos, profit_pct, 'trailing_stop'))
                        continue
            else:
                if not pos.get('trailing_active') and row['ask_low'] <= pos['initial_target']:
                    pos['trailing_active'] = True
                    pos['trailing_stop'] = row['ask_low'] + (trail_pips * pip)
                    pos['peak_price'] = row['ask_low']
                    continue

                if pos.get('trailing_active'):
                    if row['ask_low'] < pos['peak_price']:
                        pos['peak_price'] = row['ask_low']
                        pos['trailing_stop'] = pos['peak_price'] + (trail_pips * pip)

                    if row['ask_high'] >= pos['trailing_stop']:
                        exit_price = pos['trailing_stop']
                        profit_pct = (pos['entry_price'] - exit_price) / pos['entry_price']
                        positions_to_close.append((pos, profit_pct, 'trailing_stop'))
                        continue

            # Time exit
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
            positions.remove(pos)
            trades.append({
                'profit_pct': profit_pct,
                'profit_dollars': profit_dollars,
                'exit_reason': exit_reason,
                'bars_held': pos['bars_held']
            })

        # New signals (same as dynamic)
        for pair in PAIRS:
            if date not in predictions[pair].index or date not in raw_data[pair].index:
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
                'trailing_active': False
            })

    return capital, pd.DataFrame(trades)


# Load data
print("Loading data...")
all_raw_data = {}
for pair in PAIRS:
    df = pd.read_csv(f'{DATA_DIR}/{pair}_15m.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    all_raw_data[pair] = df

print("Loading predictions...")
with open('test_predictions_15m_continuous.pkl', 'rb') as f:
    predictions = pickle.load(f)

print()

# Test dynamic strategy (current)
print("="*100)
print("TESTING CURRENT DYNAMIC STRATEGY")
print("="*100)
print()
final_capital_dynamic, trades_dynamic = strategy_dynamic_trail(predictions, all_raw_data)

winners_dyn = trades_dynamic[trades_dynamic['profit_pct'] > 0]
losers_dyn = trades_dynamic[trades_dynamic['profit_pct'] <= 0]

print(f"Final Capital: ${final_capital_dynamic:,.2f}")
print(f"Total Return: {(final_capital_dynamic/INITIAL_CAPITAL - 1):.1%}")
print(f"Win Rate: {len(winners_dyn)/len(trades_dynamic):.1%}")
print(f"Avg Win: {winners_dyn['profit_pct'].mean():.2%}")
print(f"Avg Loss: {losers_dyn['profit_pct'].mean():.2%}")
print(f"Profit Factor: {winners_dyn['profit_dollars'].sum() / abs(losers_dyn['profit_dollars'].sum()):.2f}")
print()

# Test various fixed-pip distances
print("="*100)
print("TESTING OANDA FIXED-PIP TRAILING STOPS")
print("="*100)
print()

pip_distances = [5, 10, 15, 20, 25, 30]
results = []

for pips in pip_distances:
    print(f"Testing {pips}-pip trail...")
    final_capital, trades_df = strategy_fixed_pip_trail(predictions, all_raw_data, pips)

    winners = trades_df[trades_df['profit_pct'] > 0]
    losers = trades_df[trades_df['profit_pct'] <= 0]

    total_return = (final_capital / INITIAL_CAPITAL - 1)
    win_rate = len(winners) / len(trades_df) if len(trades_df) > 0 else 0
    profit_factor = winners['profit_dollars'].sum() / abs(losers['profit_dollars'].sum()) if len(losers) > 0 else float('inf')

    results.append({
        'pips': pips,
        'final_capital': final_capital,
        'total_return': total_return,
        'win_rate': win_rate,
        'avg_win': winners['profit_pct'].mean(),
        'avg_loss': losers['profit_pct'].mean(),
        'profit_factor': profit_factor,
        'trades': len(trades_df)
    })

    print(f"  Final: ${final_capital:,.2f} | Return: {total_return:+.1%} | Win Rate: {win_rate:.1%}")
    print()

# Summary comparison
print("="*100)
print("COMPARISON SUMMARY")
print("="*100)
print()

print(f"{'Strategy':<20} {'Final Capital':>15} {'Return':>10} {'Win Rate':>10} {'Profit Factor':>15}")
print("-" * 100)
print(f"{'Dynamic (Current)':<20} ${final_capital_dynamic:>14,.0f} {(final_capital_dynamic/INITIAL_CAPITAL - 1):>9.1%} {len(winners_dyn)/len(trades_dynamic):>9.1%} {winners_dyn['profit_dollars'].sum() / abs(losers_dyn['profit_dollars'].sum()):>14.2f}")

for r in results:
    print(f"{f'{r['pips']}-pip trail':<20} ${r['final_capital']:>14,.0f} {r['total_return']:>9.1%} {r['win_rate']:>9.1%} {r['profit_factor']:>14.2f}")

print()
print("="*100)
print("RECOMMENDATION")
print("="*100)
print()

best_fixed = max(results, key=lambda x: x['final_capital'])
perf_diff = (best_fixed['final_capital'] / final_capital_dynamic - 1) * 100

if perf_diff >= -5:  # Within 5% of dynamic
    print(f"✅ OANDA NATIVE STOPS WORK!")
    print(f"   Best: {best_fixed['pips']}-pip trail")
    print(f"   Performance: {perf_diff:+.1f}% vs dynamic strategy")
    print(f"   Recommendation: Use Oanda's native {best_fixed['pips']}-pip trailing stop")
else:
    print(f"❌ OANDA NATIVE STOPS UNDERPERFORM")
    print(f"   Best fixed-pip: {best_fixed['pips']} pips")
    print(f"   Performance: {perf_diff:+.1f}% vs dynamic strategy")
    print(f"   Recommendation: Implement custom dynamic trailing stop logic")

print()
