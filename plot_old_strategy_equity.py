"""
EQUITY CURVE: Original Strategy Across Multiple Years
======================================================
Shows how the original (losing) strategy performs over time.
"""
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

from optimize_staged import run_backtest_with_params

DATA_DIR = 'data_15m'
PAIRS = ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY']
INITIAL_CAPITAL = 500

# ORIGINAL PARAMETERS (the ones that showed -5.3% avg CAGR)
original_params = {
    'RISK_PER_TRADE': 0.004,
    'MIN_CONFIDENCE': 0.70,
    'EMERGENCY_STOP_PERIODS': 24,  # 6 hours
    'EMERGENCY_STOP_LOSS_PCT': -0.04,
    'TRAILING_STOP_TRIGGER': 0.001,
    'TRAILING_STOP_PCT': 0.75,
}

print("="*100)
print("EQUITY CURVE: Original Strategy Performance")
print("="*100)
print()
print("Testing original parameters across 2021-2025...")
print(f"  MIN_CONFIDENCE: {original_params['MIN_CONFIDENCE']}")
print(f"  RISK_PER_TRADE: {original_params['RISK_PER_TRADE']}")
print(f"  EMERGENCY_STOP: {original_params['EMERGENCY_STOP_PERIODS']} periods ({original_params['EMERGENCY_STOP_PERIODS']*0.25:.1f}h)")
print()

# Load raw data
print("Loading data...")
all_raw_data = {}
for pair in PAIRS:
    df = pd.read_csv(f'{DATA_DIR}/{pair}_15m.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    all_raw_data[pair] = df

# Load predictions
test_files = [
    ('2021', 'test_predictions_15m_2021_test.pkl'),
    ('2022', 'test_predictions_15m_2022_test.pkl'),
    ('2023', 'test_predictions_15m_2023_test.pkl'),
    ('2024', 'test_predictions_15m_2024_test.pkl'),
    ('2025', 'test_predictions_15m_2025_test.pkl'),
]

print("Running backtests for each year...")
print()

all_equity_data = []
cumulative_capital = INITIAL_CAPITAL

for year, filename in test_files:
    print(f"{year}:")
    with open(filename, 'rb') as f:
        preds = pickle.load(f)

    # Run backtest but capture equity curve
    # We need to modify run_backtest to return equity curve...
    # For now, let's just collect the results
    result = run_backtest_with_params(preds, all_raw_data, original_params)

    if result:
        print(f"  CAGR: {result:.1%}")
    else:
        print(f"  No trades")
    print()

print("="*100)
print("NOTE: To plot equity curve, we need to modify the backtest function")
print("to track equity at each trade. This is a simplified version.")
print("="*100)
