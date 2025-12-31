"""
DEBUG PREDICTIONS
=================
Quick check to see what's in both prediction files
"""
import pandas as pd
import pickle

print("="*100)
print("DEBUG: PREDICTION FILE CONTENTS")
print("="*100)
print()

# Load backtest predictions
with open('model_predictions_quarterly.pkl', 'rb') as f:
    backtest_preds = pickle.load(f)

# Load production predictions
with open('model_predictions_production.pkl', 'rb') as f:
    production_preds = pickle.load(f)

# Check first quarter
quarter = '2016Q1'

print(f"Checking quarter: {quarter}")
print()

print("BACKTEST:")
if quarter in backtest_preds:
    print(f"  Pairs: {list(backtest_preds[quarter].keys())[:5]}...")
    first_pair = list(backtest_preds[quarter].keys())[0]
    df = backtest_preds[quarter][first_pair]
    print(f"  Sample pair: {first_pair}")
    print(f"  Type: {type(df)}")
    print(f"  Shape: {df.shape}")
    print(f"  Index type: {type(df.index[0]) if len(df) > 0 else 'N/A'}")
    print(f"  First 3 dates: {df.index[:3].tolist()}")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Sample row:")
    print(df.head(1))
else:
    print(f"  Quarter not found!")

print()
print("PRODUCTION:")
if quarter in production_preds:
    print(f"  Pairs: {list(production_preds[quarter].keys())[:5]}...")
    first_pair = list(production_preds[quarter].keys())[0]
    df = production_preds[quarter][first_pair]
    print(f"  Sample pair: {first_pair}")
    print(f"  Type: {type(df)}")
    print(f"  Shape: {df.shape}")
    print(f"  Index type: {type(df.index[0]) if len(df) > 0 else 'N/A'}")
    print(f"  First 3 dates: {df.index[:3].tolist()}")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Sample row:")
    print(df.head(1))
else:
    print(f"  Quarter not found!")

print()
print("="*100)
