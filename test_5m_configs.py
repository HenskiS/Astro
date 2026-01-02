"""
TEST 3 KEY 5M CONFIGURATIONS
=============================
Much faster - test only 3 promising parameter sets
"""
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("TESTING 3 KEY 5M CONFIGURATIONS")
print("="*100)
print()

# Fixed parameters
INITIAL_CAPITAL = 500
RISK_PER_TRADE = 0.004
AVOID_HOURS = [20, 21, 22]
MAX_TOTAL_POSITIONS = 120  # 15 per pair * 8 pairs
MAX_POSITIONS_PER_PAIR = 15
DATA_DIR = 'data_5m'
PAIRS = ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY']  # 8 pairs

# Test only the best config (VERY FAST)
CONFIGS = [
    {
        'name': 'VERY FAST (Scalping) - 8 PAIRS',
        'emergency_periods': 24,  # 2 hours
        'trailing_trigger': 0.0005,  # 0.05%
        'trailing_pct': 0.75,
        'confidence': 0.70,
        'ladders': [0.001, 0.002]
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
with open('test_predictions_5m.pkl', 'rb') as f:
    predictions = pickle.load(f)

all_raw_data = {}
for pair in PAIRS:
    df = pd.read_csv(f'{DATA_DIR}/{pair}_5m.csv')
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
    print(f"  Emergency: {config['emergency_periods']} periods ({config['emergency_periods']*5/60:.1f}h)")
    print(f"  Trailing: {config['trailing_trigger']:.4f} trigger, {config['trailing_pct']:.0%} trail")
    print(f"  Confidence: {config['confidence']:.2f}")
    print(f"  Ladders: {config['ladders']}")

    capital = INITIAL_CAPITAL
    positions = []
    trades = []

    for period_idx, date in enumerate(all_trading_periods):
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
                'periods_held': position.periods_held,
                'exit_reason': exit_reason
            })

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
                target = row['high_240p'] * 1.005
                entry_price = prices_dict[pair]['ask_close'] if pair in prices_dict else row['close']
            else:
                direction = 'short'
                target = row['low_240p'] * 0.995
                entry_price = prices_dict[pair]['bid_close'] if pair in prices_dict else row['close']

            position = Position(pair, date, entry_price, direction, position_size, target, max_prob,
                              config['emergency_periods'], config['trailing_trigger'],
                              config['trailing_pct'], config['ladders'])
            positions.append(position)

        if (period_idx + 1) % 2000 == 0:
            print(f"  Progress: {period_idx+1:,}/{len(all_trading_periods):,} ({(period_idx+1)/len(all_trading_periods)*100:.0f}%)")

    # Calculate metrics
    trades_df = pd.DataFrame(trades)
    total_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL
    winners = trades_df[trades_df['profit_pct'] > 0]
    losers = trades_df[trades_df['profit_pct'] <= 0]
    win_rate = len(winners) / len(trades_df) if len(trades_df) > 0 else 0
    avg_winner = winners['profit_pct'].mean() if len(winners) > 0 else 0
    avg_loser = losers['profit_pct'].mean() if len(losers) > 0 else 0
    profit_ratio = abs(avg_winner / avg_loser) if len(losers) > 0 and avg_loser != 0 else 0
    avg_winner_hold = winners['periods_held'].mean() * 5 / 60 if len(winners) > 0 else 0
    avg_loser_hold = losers['periods_held'].mean() * 5 / 60 if len(losers) > 0 else 0

    print(f"  Results: ${capital:.0f} ({total_return:+.1%})")
    print(f"  Trades: {len(trades_df)} | Win Rate: {win_rate:.1%}")
    print(f"  Winners: {avg_winner:+.2%} ({avg_winner_hold:.1f}h) | Losers: {avg_loser:+.2%} ({avg_loser_hold:.1f}h)")
    print(f"  Profit Ratio: {profit_ratio:.2f}:1")
    print()

    results.append({
        'config': config['name'],
        'final_capital': capital,
        'return': total_return,
        'trades': len(trades_df),
        'win_rate': win_rate,
        'profit_ratio': profit_ratio,
        'avg_winner_hold_hours': avg_winner_hold,
        'avg_loser_hold_hours': avg_loser_hold
    })

print("="*100)
print("RESULTS SUMMARY")
print("="*100)
print()

results_df = pd.DataFrame(results)
for _, row in results_df.iterrows():
    print(f"{row['config']}:")
    print(f"  Return: {row['return']:+.1%} (${row['final_capital']:.0f})")
    print(f"  Trades: {row['trades']:.0f} | Win Rate: {row['win_rate']:.1%} | Ratio: {row['profit_ratio']:.2f}:1")
    print(f"  Hold Times: {row['avg_winner_hold_hours']:.1f}h winners, {row['avg_loser_hold_hours']:.1f}h losers")
    print()

print("="*100)
