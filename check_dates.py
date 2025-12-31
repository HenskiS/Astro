import pickle
import pandas as pd

with open('model_predictions_quarterly.pkl', 'rb') as f:
    preds = pickle.load(f)

print("First 10 dates in 2016Q1 USDCAD backtest predictions:")
dates = preds['2016Q1']['USDCAD'].index[:10]
for d in dates:
    print(f"  {d}")

print("\nFirst 10 dates in 2016Q1 USDCAD production predictions:")
with open('model_predictions_production.pkl', 'rb') as f:
    prod_preds = pickle.load(f)

dates_prod = prod_preds['2016Q1']['USDCAD'].index[:10]
for d in dates_prod:
    print(f"  {d}")
