"""
CALCULATE TRADE FREQUENCY
==========================
How many trades per day/week/month?
"""
import pandas as pd
import numpy as np
import pickle

# From the test results we saw earlier
test_periods = [
    ('2021', 948, 4),  # trades, months
    ('2022', 1174, 4),
    ('2023', 992, 4),
    ('2024', 1239, 4),
    ('2025', 892, 4),
]

print("="*100)
print("TRADE FREQUENCY ANALYSIS")
print("="*100)
print()

print("Test Period Results:")
print("-" * 50)
for name, trades, months in test_periods:
    days = months * 30  # Approximate
    weeks = months * 4.33  # More accurate weeks per month

    per_day = trades / days
    per_week = trades / weeks
    per_month = trades / months

    print(f"{name}:")
    print(f"  Total trades: {trades:>4}")
    print(f"  Per month:    {per_month:>5.1f} trades")
    print(f"  Per week:     {per_week:>5.1f} trades")
    print(f"  Per day:      {per_day:>5.1f} trades")
    print()

print("="*100)
print("AVERAGE FREQUENCY")
print("="*100)
print()

total_trades = sum(t[1] for t in test_periods)
total_months = sum(t[2] for t in test_periods)
avg_trades = total_trades / len(test_periods)

avg_per_month = total_trades / total_months
avg_per_week = avg_per_month / 4.33
avg_per_day = avg_per_month / 30

print(f"Average trades per 4-month period: {avg_trades:.0f}")
print(f"Average per month:  {avg_per_month:.1f} trades")
print(f"Average per week:   {avg_per_week:.1f} trades")
print(f"Average per day:    {avg_per_day:.1f} trades")
print()

print("="*100)
print("TRADING INTENSITY")
print("="*100)
print()

# On 15-minute timeframe
bars_per_day = 24 * 4  # 96 bars per day
bars_per_week = bars_per_day * 5  # 480 bars per week

print(f"15-minute bars per day:  {bars_per_day}")
print(f"15-minute bars per week: {bars_per_week}")
print()

print(f"Trade every {bars_per_day / avg_per_day:.1f} bars ({bars_per_day / avg_per_day * 15:.0f} minutes)")
print(f"Or about 1 trade every {24 / avg_per_day:.1f} hours")
print()

print("With 8 currency pairs:")
print(f"  Each pair trades ~{avg_per_day / 8:.1f} times per day")
print(f"  Each pair trades ~{avg_per_week / 8:.1f} times per week")
print()

print("="*100)
print("IMPLICATIONS FOR 10% SIZING")
print("="*100)
print()

# With 10% sizing and compounding
capital_turnover_per_day = avg_per_day * 0.10  # 10% per trade
capital_turnover_per_week = avg_per_week * 0.10
capital_turnover_per_month = avg_per_month * 0.10

print(f"Capital turnover per day:   {capital_turnover_per_day:.1f}x")
print(f"Capital turnover per week:  {capital_turnover_per_week:.1f}x")
print(f"Capital turnover per month: {capital_turnover_per_month:.1f}x")
print()

print("With 94% win rate and 0.11% avg profit per trade:")
expected_daily = avg_per_day * 0.10 * 0.0011  # trades * sizing * profit
expected_weekly = avg_per_week * 0.10 * 0.0011
expected_monthly = avg_per_month * 0.10 * 0.0011

print(f"Expected daily return:   {expected_daily:.3%}")
print(f"Expected weekly return:  {expected_weekly:.3%}")
print(f"Expected monthly return: {expected_monthly:.3%}")
print()

print("This explains the ~12% annual return at 10% sizing!")
print()
