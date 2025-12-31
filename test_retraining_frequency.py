"""
TEST RETRAINING FREQUENCY
==========================
Compare different model retraining frequencies to see if more frequent
retraining reduces drawdowns and improves consistency.

Test scenarios:
1. Current: Every 2 years (2016-2017, 2018-2019, etc.)
2. Annual: Every 1 year (2016, 2017, 2018, etc.)
3. 6-month: Every 6 months (more experimental)

Note: This requires retraining models, which we can't do without the original
data. Instead, we'll analyze the PATTERN of when losses occur and make
recommendations.
"""
import pandas as pd
import numpy as np

print("="*100)
print("RETRAINING FREQUENCY ANALYSIS")
print("="*100)
print()

# Yearly results from our optimal ladder backtest
yearly_results = [
    {'year': 2016, 'return': 3.219, 'period': '2016-2017', 'year_in_period': 1},
    {'year': 2017, 'return': 0.420, 'period': '2016-2017', 'year_in_period': 2},
    {'year': 2018, 'return': -0.536, 'period': '2018-2019', 'year_in_period': 1},
    {'year': 2019, 'return': 1.295, 'period': '2018-2019', 'year_in_period': 2},
    {'year': 2020, 'return': -0.554, 'period': '2020-2021', 'year_in_period': 1},
    {'year': 2021, 'return': 0.155, 'period': '2020-2021', 'year_in_period': 2},
    {'year': 2022, 'return': -0.667, 'period': '2022-2023', 'year_in_period': 1},
    {'year': 2023, 'return': 1.449, 'period': '2022-2023', 'year_in_period': 2},
    {'year': 2024, 'return': -0.234, 'period': '2024-2025', 'year_in_period': 1},
    {'year': 2025, 'return': 2.029, 'period': '2024-2025', 'year_in_period': 2},
]

df = pd.DataFrame(yearly_results)

print("CURRENT SETUP: 2-YEAR RETRAINING")
print("="*100)
print()

# Analyze by position in period
year1_returns = df[df['year_in_period'] == 1]['return']
year2_returns = df[df['year_in_period'] == 2]['return']

print(f"Year 1 of each period (right after retraining):")
print(f"  Years: {list(df[df['year_in_period'] == 1]['year'].values)}")
print(f"  Returns: {[f'{r:+.1%}' for r in year1_returns]}")
print(f"  Average: {year1_returns.mean():+.1%}")
print(f"  Profitable: {sum(year1_returns > 0)}/5")
print(f"  Win rate: {sum(year1_returns > 0)/len(year1_returns):.0%}")
print()

print(f"Year 2 of each period (model 1 year old):")
print(f"  Years: {list(df[df['year_in_period'] == 2]['year'].values)}")
print(f"  Returns: {[f'{r:+.1%}' for r in year2_returns]}")
print(f"  Average: {year2_returns.mean():+.1%}")
print(f"  Profitable: {sum(year2_returns > 0)}/5")
print(f"  Win rate: {sum(year2_returns > 0)/len(year2_returns):.0%}")
print()

# Statistical test
from scipy import stats
t_stat, p_value = stats.ttest_ind(year1_returns, year2_returns)
print(f"T-test for difference: t={t_stat:.2f}, p={p_value:.3f}")
if p_value < 0.05:
    print(f"  Result: SIGNIFICANT difference (p < 0.05)")
else:
    print(f"  Result: Not statistically significant (p > 0.05)")
print()

# Analyze losing years
print("="*100)
print("LOSING YEAR ANALYSIS")
print("="*100)
print()

losing_years = df[df['return'] < 0]
print(f"Losing years: {list(losing_years['year'].values)}")
print(f"Total: {len(losing_years)}/10 years")
print()

print("Pattern analysis:")
year1_losses = len(losing_years[losing_years['year_in_period'] == 1])
year2_losses = len(losing_years[losing_years['year_in_period'] == 2])

print(f"  Year 1 losses: {year1_losses}/4 losing years ({year1_losses/len(losing_years):.0%})")
print(f"  Year 2 losses: {year2_losses}/4 losing years ({year2_losses/len(losing_years):.0%})")
print()

if year1_losses > year2_losses:
    print("FINDING: Losses cluster in Year 1 (right after retraining)")
    print("  → This suggests NEW models struggle initially")
    print("  → More frequent retraining might make this WORSE")
    print("  → Consider: Model warm-up period or ensemble approaches")
elif year2_losses > year1_losses:
    print("FINDING: Losses cluster in Year 2 (stale models)")
    print("  → This suggests model staleness is the issue")
    print("  → More frequent retraining would likely HELP")
else:
    print("FINDING: Losses evenly distributed")
    print("  → Retraining frequency not the main issue")

print()

# Check if it's market conditions
print("="*100)
print("MARKET CONDITIONS HYPOTHESIS")
print("="*100)
print()

print("Were 2018, 2020, 2022, 2024 just bad years for this strategy?")
print()

# These would need to be checked against actual market data
bad_years_analysis = {
    2018: "Fed rate hikes, trade war tensions, late-year selloff",
    2020: "COVID crash (March), high volatility",
    2022: "Fed aggressive rate hikes, inflation fears, bear market",
    2024: "Election year uncertainty (?)"
}

for year, condition in bad_years_analysis.items():
    print(f"  {year}: {condition}")

print()
print("If these years had difficult market conditions (high volatility,")
print("trend reversals, etc.), then retraining frequency won't help much.")
print("Instead, focus on:")
print("  1. Regime filters (detect unfavorable conditions)")
print("  2. Position sizing adjustments")
print("  3. Accepting that some years will be tough")

print()

# Simulation: What if we trained annually?
print("="*100)
print("SIMULATION: ANNUAL RETRAINING")
print("="*100)
print()

print("Hypothetical scenario if we retrained every year:")
print()
print("Assumption: Year 1 performance becomes the norm (since every year is Year 1)")
print(f"  Average Year 1 return: {year1_returns.mean():+.1%}")
print(f"  Year 1 win rate: {sum(year1_returns > 0)/len(year1_returns):.0%}")
print()

# This is speculative
annual_avg = year1_returns.mean()
current_avg = df['return'].mean()

print(f"Current (2-year periods):  {current_avg:+.1%} avg, {sum(df['return'] > 0)}/10 years positive")
print(f"Annual (estimated):        {annual_avg:+.1%} avg, {int(sum(year1_returns > 0)/len(year1_returns)*10)}/10 years positive")
print()

if annual_avg > current_avg:
    print("CONCLUSION: Annual retraining might IMPROVE performance")
    print(f"  Potential gain: {annual_avg - current_avg:+.1%} per year")
else:
    print("CONCLUSION: Annual retraining likely WORSE")
    print(f"  Potential loss: {annual_avg - current_avg:+.1%} per year")

print()

# Recommendations
print("="*100)
print("RECOMMENDATIONS")
print("="*100)
print()

print("Based on the analysis:")
print()

if year1_returns.mean() < year2_returns.mean():
    print("1. DO NOT increase retraining frequency")
    print("   - New models underperform initially")
    print("   - Current 2-year cycle is likely optimal")
    print()
    print("2. Instead, consider:")
    print("   - Ensemble models (combine old + new)")
    print("   - Gradual transition (weight shift from old to new model)")
    print("   - Regime filters (reduce exposure in unfavorable conditions)")
    print("   - Accept the cyclical nature (4 losing years might be unavoidable)")
else:
    print("1. TEST annual retraining")
    print("   - Model staleness appears to be an issue")
    print("   - More frequent updates could help")
    print()
    print("2. Implementation:")
    print("   - Retrain models every January")
    print("   - Train on last 6-8 years of data")
    print("   - Keep computational overhead manageable")

print()
print("3. CRITICAL: Whatever you do, maintain walk-forward testing")
print("   - Never train on future data")
print("   - Always test on out-of-sample periods")
print("   - Keep the validation methodology robust")

print()
print("="*100)
print("ANALYSIS COMPLETE")
print("="*100)
