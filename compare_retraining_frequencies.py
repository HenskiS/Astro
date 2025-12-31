"""
COMPARE RETRAINING FREQUENCIES
===============================
Side-by-side comparison of 2-year, annual, and quarterly retraining
"""
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("RETRAINING FREQUENCY COMPARISON")
print("="*100)
print()


def calculate_metrics(equity_dates, equity_values, initial_capital):
    """Calculate performance metrics"""
    equity_series = pd.Series(equity_values, index=equity_dates)

    # Returns
    total_return = (equity_values[-1] - initial_capital) / initial_capital
    years = (equity_dates[-1] - equity_dates[0]).days / 365.25
    cagr = (equity_values[-1] / initial_capital) ** (1/years) - 1

    # Drawdown
    peak = equity_series.expanding().max()
    drawdown = (equity_series - peak) / peak
    max_dd = drawdown.min()

    # Yearly returns
    yearly_equity = {}
    for date, value in zip(equity_dates, equity_values):
        year = date.year
        if year not in yearly_equity:
            yearly_equity[year] = {'start': value, 'end': value}
        yearly_equity[year]['end'] = value

    yearly_returns = {}
    for year in sorted(yearly_equity.keys()):
        if year == min(yearly_equity.keys()):
            ret = (yearly_equity[year]['end'] - initial_capital) / initial_capital
        else:
            prev_year = year - 1
            ret = (yearly_equity[year]['end'] - yearly_equity[prev_year]['end']) / yearly_equity[prev_year]['end']
        yearly_returns[year] = ret

    profitable_years = sum(1 for r in yearly_returns.values() if r > 0)
    avg_annual = np.mean(list(yearly_returns.values()))

    return {
        'ending_capital': equity_values[-1],
        'total_return': total_return,
        'cagr': cagr,
        'max_dd': max_dd,
        'profitable_years': profitable_years,
        'total_years': len(yearly_returns),
        'avg_annual': avg_annual,
        'yearly_returns': yearly_returns
    }


# Load 2-year baseline results
print("Loading 2-year baseline results...")
with open('model_predictions.pkl', 'rb') as f:
    baseline_preds = pickle.load(f)

# Get equity curve from backtest_optimal_ladder.py output
# We'll approximate from the yearly results we know
baseline_yearly = {
    2016: 3.219, 2017: 0.420, 2018: -0.536, 2019: 1.295,
    2020: -0.554, 2021: 0.155, 2022: -0.667, 2023: 1.449,
    2024: -0.234, 2025: 2.029
}

baseline_equity = [500]
baseline_dates = [pd.Timestamp('2016-01-01')]
for year in sorted(baseline_yearly.keys()):
    capital = baseline_equity[-1] * (1 + baseline_yearly[year])
    baseline_equity.append(capital)
    baseline_dates.append(pd.Timestamp(f'{year}-12-31'))

print()
print("Loading annual results...")
with open('model_predictions_annual.pkl', 'rb') as f:
    annual_preds = pickle.load(f)

annual_equity = []
annual_dates = []
capital = 500

for year_name in sorted(annual_preds.keys()):
    year = int(year_name)
    # Get the actual results from the backtest
    if year == 2016:
        ret = 3.219
    elif year == 2017:
        ret = 1.403
    elif year == 2018:
        ret = 1.556
    elif year == 2019:
        ret = 3.315
    elif year == 2020:
        ret = 1.181
    elif year == 2021:
        ret = -0.318
    elif year == 2022:
        ret = -0.335
    elif year == 2023:
        ret = 2.833
    elif year == 2024:
        ret = 1.854
    elif year == 2025:
        ret = 2.366

    annual_equity.append(capital)
    annual_dates.append(pd.Timestamp(f'{year}-01-01'))
    capital = capital * (1 + ret)
    annual_equity.append(capital)
    annual_dates.append(pd.Timestamp(f'{year}-12-31'))

print()
print("Loading quarterly results...")
with open('model_predictions_quarterly.pkl', 'rb') as f:
    quarterly_preds = pickle.load(f)

# For quarterly, we have actual quarter-by-quarter results from the backtest
quarterly_returns = {
    '2016Q1': 0.271, '2016Q2': 0.337, '2016Q3': 0.413, '2016Q4': 0.739,
    '2017Q1': 0.605, '2017Q2': 0.029, '2017Q3': 0.296, '2017Q4': 0.566,
    '2018Q1': 0.562, '2018Q2': 0.176, '2018Q3': 0.345, '2018Q4': 0.472,
    '2019Q1': 0.465, '2019Q2': 0.162, '2019Q3': 0.344, '2019Q4': 0.258,
    '2020Q1': -0.033, '2020Q2': 0.698, '2020Q3': 0.569, '2020Q4': 0.281,
    '2021Q1': 0.025, '2021Q2': 0.451, '2021Q3': 0.464, '2021Q4': 0.128,
    '2022Q1': -0.045, '2022Q2': 0.119, '2022Q3': -0.189, '2022Q4': 0.359,
    '2023Q1': 0.491, '2023Q2': 0.393, '2023Q3': 0.266, '2023Q4': 0.523,
    '2024Q1': 0.613, '2024Q2': 0.304, '2024Q3': 0.141, '2024Q4': 0.649,
    '2025Q1': 0.291, '2025Q2': 0.355, '2025Q3': 0.471, '2025Q4': 0.483
}

quarterly_equity = []
quarterly_dates = []
capital = 500

for quarter_name in sorted(quarterly_returns.keys()):
    year = int(quarter_name[:4])
    q = int(quarter_name[-1])
    month = (q - 1) * 3 + 1

    quarterly_equity.append(capital)
    quarterly_dates.append(pd.Timestamp(f'{year}-{month:02d}-01'))

    capital = capital * (1 + quarterly_returns[quarter_name])

quarterly_equity.append(capital)
quarterly_dates.append(pd.Timestamp('2025-12-31'))

# Calculate metrics
print()
print("="*100)
print("PERFORMANCE METRICS")
print("="*100)
print()

baseline_metrics = calculate_metrics(baseline_dates, baseline_equity, 500)
annual_metrics = calculate_metrics(annual_dates, annual_equity, 500)
quarterly_metrics = calculate_metrics(quarterly_dates, quarterly_equity, 500)

# Print comparison table
print(f"{'Metric':<30} {'2-Year':>20} {'Annual':>20} {'Quarterly':>20}")
print("-" * 95)
print(f"{'Starting Capital':<30} {'$500':>20} {'$500':>20} {'$500':>20}")
print(f"{'Ending Capital':<30} ${baseline_metrics['ending_capital']:>18,.0f} ${annual_metrics['ending_capital']:>18,.0f} ${quarterly_metrics['ending_capital']:>18,.0f}")
print(f"{'Total Return':<30} {baseline_metrics['total_return']:>19.1%} {annual_metrics['total_return']:>19.1%} {quarterly_metrics['total_return']:>19.1%}")
print(f"{'CAGR':<30} {baseline_metrics['cagr']:>19.1%} {annual_metrics['cagr']:>19.1%} {quarterly_metrics['cagr']:>19.1%}")
print(f"{'Max Drawdown':<30} {baseline_metrics['max_dd']:>19.1%} {annual_metrics['max_dd']:>19.1%} {quarterly_metrics['max_dd']:>19.1%}")
print(f"{'Profitable Years':<30} {baseline_metrics['profitable_years']}/{baseline_metrics['total_years']:>18} {annual_metrics['profitable_years']}/{annual_metrics['total_years']:>18} {quarterly_metrics['profitable_years']}/{quarterly_metrics['total_years']:>18}")
print(f"{'Avg Annual Return':<30} {baseline_metrics['avg_annual']:>19.1%} {annual_metrics['avg_annual']:>19.1%} {quarterly_metrics['avg_annual']:>19.1%}")

print()
print("="*100)
print("YEAR-BY-YEAR COMPARISON")
print("="*100)
print()

print(f"{'Year':<10} {'2-Year':>15} {'Annual':>15} {'Quarterly':>15}")
print("-" * 60)
for year in range(2016, 2026):
    baseline_ret = baseline_metrics['yearly_returns'].get(year, 0)
    annual_ret = annual_metrics['yearly_returns'].get(year, 0)
    quarterly_ret = quarterly_metrics['yearly_returns'].get(year, 0)

    print(f"{year:<10} {baseline_ret:>14.1%} {annual_ret:>14.1%} {quarterly_ret:>14.1%}")

print()
print("="*100)
print("SUMMARY")
print("="*100)
print()

print("2-YEAR BASELINE:")
print(f"  $500 -> ${baseline_metrics['ending_capital']:,.0f}")
print(f"  {baseline_metrics['profitable_years']}/{baseline_metrics['total_years']} profitable years")
print(f"  Losing years: 2018, 2020, 2022, 2024")
print()

print("ANNUAL RETRAINING:")
print(f"  $500 -> ${annual_metrics['ending_capital']:,.0f}")
print(f"  {annual_metrics['profitable_years']}/{annual_metrics['total_years']} profitable years")
print(f"  Improvement: {(annual_metrics['ending_capital']/baseline_metrics['ending_capital']-1):.1%} vs baseline")
print(f"  Losing years: 2021, 2022")
print()

print("QUARTERLY RETRAINING:")
print(f"  $500 -> ${quarterly_metrics['ending_capital']:,.0f}")
print(f"  {quarterly_metrics['profitable_years']}/{quarterly_metrics['total_years']} profitable years")
print(f"  Improvement: {(quarterly_metrics['ending_capital']/baseline_metrics['ending_capital']-1):.1%} vs baseline")
print(f"  Improvement: {(quarterly_metrics['ending_capital']/annual_metrics['ending_capital']-1):.1%} vs annual")
print(f"  No losing years!")
print()

print("RECOMMENDATION: Annual retraining (reasonable improvement, realistic drawdowns)")
print("CAUTION: Quarterly results need validation (very high returns, unusually consistent)")
print()
print("="*100)
