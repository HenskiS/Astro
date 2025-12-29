"""
Integrate planetary positions with forex data and train models with both features
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime, timezone
from geocentric_model import GeocentricModel
from export_data import PlanetaryDataExporter

print("="*100)
print("INTEGRATING PLANETARY POSITIONS WITH FOREX DATA")
print("="*100)
print()

DATA_DIR = 'data'
OUTPUT_DIR = 'data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY']


def load_forex_data(pair):
    """Load forex data for a pair"""
    filepath = os.path.join(DATA_DIR, f'{pair}_1day_oanda.csv')
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    # Remove timezone info for easier merging
    df['date'] = df['date'].dt.tz_localize(None)
    return df


def generate_planetary_features_for_date_range(start_date, end_date):
    """
    Generate planetary position data for forex date range

    Returns DataFrame with planetary features indexed by date
    """
    print("Generating planetary positions...")

    model = GeocentricModel()
    exporter = PlanetaryDataExporter(model)

    # Calculate number of days
    days = (end_date - start_date).days + 1

    # Generate time series with daily interval
    df_planets = exporter.generate_time_series(
        start_date=start_date,
        days=days,
        interval_hours=24.0
    )

    # Add planetary aspects
    df_planets = exporter.add_planetary_aspects(df_planets)

    # Add cyclical features
    df_planets = exporter.add_cyclical_features(df_planets)

    # Rename timestamp to date for merging
    df_planets = df_planets.rename(columns={'timestamp': 'date'})

    # Remove timezone if present
    if df_planets['date'].dt.tz is not None:
        df_planets['date'] = df_planets['date'].dt.tz_localize(None)

    # Drop the 'day' column (we'll use date)
    df_planets = df_planets.drop(columns=['day'])

    print(f"  Generated {len(df_planets)} days of planetary data")
    print(f"  Date range: {df_planets['date'].min()} to {df_planets['date'].max()}")
    print(f"  Total features: {len(df_planets.columns) - 1}")  # -1 for date column

    return df_planets


def get_planetary_feature_names(df_planets):
    """Extract planetary feature column names"""
    # All columns except 'date'
    return [col for col in df_planets.columns if col != 'date']


def merge_forex_with_planets(df_forex, df_planets):
    """Merge forex data with planetary positions on date"""
    df_merged = pd.merge(df_forex, df_planets, on='date', how='inner')

    print(f"  Merged: {len(df_forex)} forex rows + planetary features")
    print(f"  Result: {len(df_merged)} rows (after alignment)")

    return df_merged


# Main processing
print("Step 1: Loading forex data and determining date range...")
print()

# Load all forex data to find date range
all_forex = {}
min_date = None
max_date = None

for pair in PAIRS:
    try:
        df = load_forex_data(pair)
        all_forex[pair] = df

        pair_min = df['date'].min()
        pair_max = df['date'].max()

        if min_date is None or pair_min < min_date:
            min_date = pair_min
        if max_date is None or pair_max > max_date:
            max_date = pair_max

        print(f"  {pair}: {len(df)} days, {pair_min.date()} to {pair_max.date()}")

    except Exception as e:
        print(f"  {pair}: ERROR - {str(e)}")

print()
print(f"Overall date range: {min_date.date()} to {max_date.date()}")
print(f"Total days: {(max_date - min_date).days}")
print()

# Step 2: Generate planetary data
print("Step 2: Generating planetary positions for date range...")
print()

df_planets = generate_planetary_features_for_date_range(min_date, max_date)

# Save planetary data separately for reference
planets_file = os.path.join(OUTPUT_DIR, 'planetary_positions_forex_range.csv')
df_planets.to_csv(planets_file, index=False)
print(f"  Saved planetary data to: {planets_file}")
print()

# Get planetary feature names for later use
planetary_features = get_planetary_feature_names(df_planets)
print(f"Planetary features ({len(planetary_features)}):")

# Group features by type for display
feature_groups = {
    'Positions (x,y)': [f for f in planetary_features if f.endswith('_x') or f.endswith('_y')],
    'Angles': [f for f in planetary_features if f.endswith('_angle')],
    'Distances': [f for f in planetary_features if f.endswith('_distance')],
    'Aspects': [f for f in planetary_features if f.startswith('aspect_')],
    'Cyclical (sin/cos)': [f for f in planetary_features if f.endswith('_sin') or f.endswith('_cos')]
}

for group_name, features in feature_groups.items():
    print(f"  {group_name}: {len(features)} features")

print()

# Step 3: Merge and save integrated datasets
print("Step 3: Merging planetary data with forex pairs...")
print()

integrated_data = {}

for pair in PAIRS:
    if pair not in all_forex:
        continue

    print(f"{pair}:")

    df_forex = all_forex[pair]
    df_merged = merge_forex_with_planets(df_forex, df_planets)

    # Save integrated dataset
    output_file = os.path.join(OUTPUT_DIR, f'{pair}_1day_with_planets.csv')
    df_merged.to_csv(output_file, index=False)

    integrated_data[pair] = df_merged

    print(f"  Saved to: {output_file}")
    print()

print("="*100)
print("INTEGRATION COMPLETE")
print("="*100)
print()

print("Summary:")
print(f"  Forex pairs integrated: {len(integrated_data)}")
print(f"  Planetary features added: {len(planetary_features)}")
print(f"  Output directory: {OUTPUT_DIR}")
print()

print("Feature breakdown:")
for group_name, features in feature_groups.items():
    if len(features) > 0:
        print(f"  {group_name}:")
        # Show first few examples
        for feat in features[:3]:
            print(f"    - {feat}")
        if len(features) > 3:
            print(f"    ... and {len(features) - 3} more")

print()
print("Next steps:")
print("  1. Run: python train_with_planets.py")
print("  2. Compare models with/without planetary features")
print("  3. Analyze which planetary features are most predictive")
print()
