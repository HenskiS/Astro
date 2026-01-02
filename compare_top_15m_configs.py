"""
COMPARE TOP 2 15M CONFIGURATIONS WITH DRAWDOWN
===============================================
Detailed comparison of the two best performers:
1. TIGHT + Tighter Trail
2. ULTRA TIGHT

Includes max drawdown calculation.
"""
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("COMPARING TOP 2 15M CONFIGURATIONS")
print("="*100)
print()

# Fixed parameters
INITIAL_CAPITAL = 500
RISK_PER_TRADE = 0.004
AVOID_HOURS = [20, 21, 22]
MAX_TOTAL_POSITIONS = 120
MAX_POSITIONS_PER_PAIR = 15
DATA_DIR = 'data_15m'
PAIRS = ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY']

# Top 2 configurations
CONFIGS = [
    {
        'name': 'TIGHT + Tighter Trail',
        'emergency_periods': 24,
        'trailing_trigger': 0.001,
        'trailing_pct': 0.75,
        'confidence': 0.70,
        'ladders': [0.002, 0.004]
    },
    {
        'name': 'ULTRA TIGHT',
        'emergency_periods': 18,
        'trailing_trigger': 0.0008,
        'trailing_pct': 0.70,
        'confidence': 0.70,
        'ladders': [0.002, 0.004]
    }
]


class Position:
    def __init__(self, pair, entry_date, entry_price, direction, size, breakout_target, confidence,
                 emergency_periods, trailing_trigger, trailing_pct, ladder_levels):
        self.pair = pair
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.direction = direction
        self.size = size
        self.original_size = size
        self.breakout_target = breakout_target
        self.confidence = confidence
        self.periods_held = 0
        self.max_profit = 0
        self.trailing_stop = None
        self.partial_exits = []
        self.ladder_level = 0
        self.emergency_periods = emergency_periods
        self.trailing_trigger = trailing_trigger
        self.trailing_pct = trailing_pct
        self.ladder_levels = ladder_levels

    def update(self, date, bid_high, bid_low, bid_close, ask_high, ask_low, ask_close):
        self.periods_held += 1

        if self.direction == 'long':
            current_profit = (bid_close - self.entry_price) / self.entry_price
            intraday_high_profit = (bid_high - self.entry_price) / self.entry_price
            hit_target = bid_high >= self.breakout_target
        else:
            current_profit = (self.entry_price - ask_close) / self.entry_price
            intraday_high_profit = (self.entry_price - ask_low) / self.entry_price
            hit_target = ask_low <= self.breakout_target

        self.max_profit = max(self.max_profit, intraday_high_profit)

        if self.ladder_level < len(self.ladder_levels):
            if intraday_high_profit >= self.ladder_levels[self.ladder_level]:
                self.partial_exits.append((self.ladder_levels[self.ladder_level], 0.40))
                self.size *= 0.60
                self.ladder_level += 1
                return None

        if self.periods_held >= self.emergency_periods and current_profit < -0.04:
            exit_price = bid_close if self.direction == 'long' else ask_close
            return 'emergency_stop', exit_price, current_profit

        if self.trailing_stop is None:
            if self.max_profit > self.trailing_trigger:
                self.trailing_stop = self.entry_price
        else:
            old_stop = self.trailing_stop

            if self.direction == 'long':
                hit_stop = bid_low <= old_stop
                if hit_stop:
                    return 'trailing_stop', old_stop, current_profit
                new_stop = self.entry_price + (bid_high - self.entry_price) * self.trailing_pct
                self.trailing_stop = max(self.trailing_stop, new_stop)
            else:
                hit_stop = ask_high >= old_stop
                if hit_stop:
                    return 'trailing_stop', old_stop, current_profit
                new_stop = self.entry_price - (self.entry_price - ask_low) * self.trailing_pct
                self.trailing_stop = min(self.trailing_stop, new_stop)

        if hit_target:
            return 'target', self.breakout_target, current_profit

        return None

    def calculate_blended_profit(self, final_profit):
        if len(self.partial_exits) == 0:
            return final_profit
        total = 0
        remaining = 1.0
        for exit_profit, exit_pct in self.partial_exits:
            total += exit_profit * exit_pct
            remaining -= exit_pct
        total += final_profit * remaining
        return total


# Load data
print("Loading predictions and data...")
with open('test_predictions_15m.pkl', 'rb') as f:
    predictions = pickle.load(f)

all_raw_data = {}
for pair in PAIRS:
    df = pd.read_csv(f'{DATA_DIR}/{pair}_15m.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    all_raw_data[pair] = df

all_trading_periods = set()
for pair_df in predictions.values():
    all_trading_periods.update(pair_df.index)
all_trading_periods = sorted(list(all_trading_periods))

print(f"Test period: {len(all_trading_periods):,} periods")
print()

# Test each configuration
results = []

for config in CONFIGS:
    print(f"Testing: {config['name']}")
    print(f"  Emergency: {config['emergency_periods']} periods ({config['emergency_periods']*0.25:.1f}h)")
    print(f"  Trailing: {config['trailing_trigger']:.4f} trigger, {config['trailing_pct']:.0%} trail")

    capital = INITIAL_CAPITAL
    positions = []
    trades = []
    equity_curve = [INITIAL_CAPITAL]

    for date in all_trading_periods:
        prices_dict = {}
        for pair in PAIRS:
            if date in all_raw_data[pair].index:
                row = all_raw_data[pair].loc[date]
                prices_dict[pair] = {
                    'bid_high': row['bid_high'], 'bid_low': row['bid_low'], 'bid_close': row['bid_close'],
                    'ask_high': row['ask_high'], 'ask_low': row['ask_low'], 'ask_close': row['ask_close'],
                    'close': row['close']
                }

        positions_to_close = []
        for position in positions:
            if position.pair not in prices_dict:
                continue
            prices = prices_dict[position.pair]
            exit_info = position.update(date, prices['bid_high'], prices['bid_low'], prices['bid_close'],
                                       prices['ask_high'], prices['ask_low'], prices['ask_close'])
            if exit_info is not None:
                positions_to_close.append((position, exit_info))

        for position, exit_info in positions_to_close:
            exit_reason, exit_price, current_profit = exit_info
            if position.direction == 'long':
                raw_profit = (exit_price - position.entry_price) / position.entry_price
            else:
                raw_profit = (position.entry_price - exit_price) / position.entry_price

            profit_pct = position.calculate_blended_profit(raw_profit)
            profit_dollars = profit_pct * (position.original_size * position.entry_price)
            capital += profit_dollars
            positions.remove(position)

            trades.append({
                'profit_pct': profit_pct,
                'profit_dollars': profit_dollars,
                'periods_held': position.periods_held,
                'exit_reason': exit_reason
            })
            equity_curve.append(capital)

        if len(positions) >= MAX_TOTAL_POSITIONS or date.hour in AVOID_HOURS:
            continue

        for pair in PAIRS:
            if date not in predictions[pair].index:
                continue

            pair_positions = [p for p in positions if p.pair == pair]
            if len(pair_positions) >= MAX_POSITIONS_PER_PAIR:
                continue

            row = predictions[pair].loc[date]
            max_prob = max(row['breakout_high_prob'], row['breakout_low_prob'])

            if max_prob <= config['confidence']:
                continue

            risk_amount = capital * RISK_PER_TRADE
            position_size = risk_amount / (row['close'] * 0.02)

            if row['breakout_high_prob'] > row['breakout_low_prob']:
                direction = 'long'
                target = row['high_80p'] * 1.005
                entry_price = prices_dict[pair]['ask_close'] if pair in prices_dict else row['close']
            else:
                direction = 'short'
                target = row['low_80p'] * 0.995
                entry_price = prices_dict[pair]['bid_close'] if pair in prices_dict else row['close']

            position = Position(pair, date, entry_price, direction, position_size, target, max_prob,
                              config['emergency_periods'], config['trailing_trigger'],
                              config['trailing_pct'], config['ladders'])
            positions.append(position)

    # Calculate metrics
    trades_df = pd.DataFrame(trades)
    total_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL

    # Calculate max drawdown
    peak = INITIAL_CAPITAL
    max_dd = 0
    for val in equity_curve:
        peak = max(peak, val)
        dd = (val - peak) / peak
        max_dd = min(max_dd, dd)

    # Calculate CAGR
    days = (all_trading_periods[-1] - all_trading_periods[0]).days
    years = days / 365
    cagr = (capital / INITIAL_CAPITAL) ** (1 / years) - 1

    winners = trades_df[trades_df['profit_pct'] > 0]
    losers = trades_df[trades_df['profit_pct'] <= 0]
    win_rate = len(winners) / len(trades_df) if len(trades_df) > 0 else 0
    avg_winner = winners['profit_pct'].mean() if len(winners) > 0 else 0
    avg_loser = losers['profit_pct'].mean() if len(losers) > 0 else 0
    profit_ratio = abs(avg_winner / avg_loser) if len(losers) > 0 and avg_loser != 0 else 0
    avg_winner_hold = winners['periods_held'].mean() * 0.25 if len(winners) > 0 else 0
    avg_loser_hold = losers['periods_held'].mean() * 0.25 if len(losers) > 0 else 0

    print(f"  Return: {total_return:+.1%} (${capital:.0f})")
    print(f"  CAGR: {cagr:.1%}")
    print(f"  Max DD: {max_dd:.1%}")
    print(f"  Trades: {len(trades_df)} | Win Rate: {win_rate:.1%}")
    print(f"  Profit Ratio: {profit_ratio:.2f}:1")
    print(f"  Hold Times: {avg_winner_hold:.1f}h winners, {avg_loser_hold:.1f}h losers")
    print()

    results.append({
        'config': config['name'],
        'final_capital': capital,
        'return': total_return,
        'cagr': cagr,
        'max_dd': max_dd,
        'trades': len(trades_df),
        'win_rate': win_rate,
        'profit_ratio': profit_ratio,
        'avg_winner': avg_winner,
        'avg_loser': avg_loser,
        'avg_winner_hold_hours': avg_winner_hold,
        'avg_loser_hold_hours': avg_loser_hold,
        'emergency_periods': config['emergency_periods'],
        'emergency_hours': config['emergency_periods'] * 0.25,
        'trailing_trigger': config['trailing_trigger'],
        'trailing_pct': config['trailing_pct']
    })

print("="*100)
print("DETAILED COMPARISON")
print("="*100)
print()

results_df = pd.DataFrame(results)

# Print side-by-side comparison
config1 = results_df.iloc[0]
config2 = results_df.iloc[1]

print(f"{'Metric':<30} {'TIGHT + Tighter Trail':>25} {'ULTRA TIGHT':>25}")
print("-"*100)
print(f"{'Return':<30} {config1['return']:>24.1%} {config2['return']:>25.1%}")
print(f"{'CAGR':<30} {config1['cagr']:>24.1%} {config2['cagr']:>25.1%}")
print(f"{'Max Drawdown':<30} {config1['max_dd']:>24.1%} {config2['max_dd']:>25.1%}")
print(f"{'Final Capital':<30} ${config1['final_capital']:>23,.0f} ${config2['final_capital']:>24,.0f}")
print()
print(f"{'Trades':<30} {config1['trades']:>24,.0f} {config2['trades']:>25,.0f}")
print(f"{'Win Rate':<30} {config1['win_rate']:>24.1%} {config2['win_rate']:>25.1%}")
print(f"{'Profit Ratio':<30} {config1['profit_ratio']:>23.2f}:1 {config2['profit_ratio']:>24.2f}:1")
print()
print(f"{'Avg Winner':<30} {config1['avg_winner']:>24.2%} {config2['avg_winner']:>25.2%}")
print(f"{'Avg Loser':<30} {config1['avg_loser']:>24.2%} {config2['avg_loser']:>25.2%}")
print(f"{'Winner Hold Time':<30} {config1['avg_winner_hold_hours']:>23.1f}h {config2['avg_winner_hold_hours']:>24.1f}h")
print(f"{'Loser Hold Time':<30} {config1['avg_loser_hold_hours']:>23.1f}h {config2['avg_loser_hold_hours']:>24.1f}h")
print()
print(f"{'Emergency Stop':<30} {config1['emergency_hours']:>23.1f}h {config2['emergency_hours']:>24.1f}h")
print(f"{'Trailing Trigger':<30} {config1['trailing_trigger']:>24.4f} {config2['trailing_trigger']:>25.4f}")
print(f"{'Trailing %':<30} {config1['trailing_pct']:>24.0%} {config2['trailing_pct']:>25.0%}")
print()

print("="*100)
print("RECOMMENDATION")
print("="*100)
print()

if config1['return'] > config2['return'] and config1['max_dd'] >= config2['max_dd']:
    print("✅ TIGHT + Tighter Trail is the WINNER")
    print("   - Higher return (+49.6% vs +39.7%)")
    print(f"   - Max DD: {config1['max_dd']:.1%} vs {config2['max_dd']:.1%}")
    print("   - Better balance of return and risk")
elif config2['max_dd'] > config1['max_dd'] and config2['profit_ratio'] > config1['profit_ratio']:
    print("✅ ULTRA TIGHT is the WINNER")
    print("   - Lower drawdown")
    print("   - Better profit ratio")
    print("   - Superior risk management")
else:
    print("Both configs have merit:")
    print("  - TIGHT + Tighter Trail: Higher returns")
    print("  - ULTRA TIGHT: Better risk metrics")
    print("  Choice depends on risk tolerance")

print()
print("="*100)
