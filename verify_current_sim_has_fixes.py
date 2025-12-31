"""
Check if the current simulation includes both fixes by examining a specific prediction
"""
import pickle
import pandas as pd
from mock_broker_api import MockBrokerAPI
from production_simulation import calculate_features

# Load current production predictions
with open('model_predictions_production.pkl', 'rb') as f:
    prod_preds = pickle.load(f)

# Check one of the worst mismatches
pair = 'NZDUSD'
date_str = '2016-03-23'
test_date = pd.Timestamp(date_str)

print("="*100)
print(f"VERIFYING CURRENT SIMULATION FOR {pair} on {date_str}")
print("="*100)
print()

# Get what the simulation saved
if '2016Q1' in prod_preds and pair in prod_preds['2016Q1']:
    df = prod_preds['2016Q1'][pair]
    matching = df[df.index.normalize() == test_date.normalize()]
    if len(matching) > 0:
        row = matching.iloc[0]
        print(f"Production saved prediction:")
        print(f"  Date: {row.name}")
        print(f"  HIGH: {row['breakout_high_prob']:.6f}")
        print(f"  LOW:  {row['breakout_low_prob']:.6f}")
        print(f"  high_20d: {row['high_20d']:.6f}")
        print()
else:
    print("No production data for 2016Q1 NZDUSD")
    print()

# Test if the current API includes fix #2
print("Testing current API behavior:")
print("-"*100)
api = MockBrokerAPI(data_dir='data')
history = api.get_history(pair, count=999999, end_date=test_date)

print(f"Requested end_date: {test_date}")
print(f"Last date returned: {history.index[-1]}")
print(f"Dates match: {history.index[-1].normalize() == test_date.normalize()}")
print()

if history.index[-1].normalize() == test_date.normalize():
    print("[OK] Fix #2 (date normalization) is ACTIVE")
else:
    print("[ISSUE] Fix #2 (date normalization) is NOT active - simulation needs restart")
print()

# Calculate features and check
features = calculate_features(history)
features = features.dropna()
print(f"After features + dropna, last date: {features.index[-1]}")
print(f"Last date matches request: {features.index[-1].normalize() == test_date.normalize()}")
print()
