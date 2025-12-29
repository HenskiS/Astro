"""
Integrate REAL astronomical planetary positions with forex data
Uses NASA JPL ephemerides for actual planetary positions on historical dates
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime, timezone
from real_astronomy import RealAstronomyModel, RealAstronomyDataExporter

print("="*100)
print("INTEGRATING REAL ASTRONOMICAL POSITIONS WITH FOREX DATA")
print("Using NASA JPL Ephemerides (DE421)")
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
    # Keep timezone info for accurate planetary alignment
    # Forex data is timestamped to 6am PST (2pm UTC)
    if df['date'].dt.tz is None:
        # If no timezone, assume UTC
        df['date'] = df['date'].dt.tz_localize('UTC')
    return df


def generate_real_planetary_features(start_date, end_date):
    """
    Generate REAL planetary position data using NASA ephemerides

    Returns DataFrame with actual astronomical positions indexed by date
    """
    print("Initializing real astronomy model...")
    print()

    model = RealAstronomyModel()
    exporter = RealAstronomyDataExporter(model)

    # Calculate number of days
    days = (end_date - start_date).days + 1

    print()
    print(f"Generating real planetary positions...")
    print(f"  Date range: {start_date.date()} to {end_date.date()}")
    print(f"  Total days: {days}")
    print()

    # Generate time series with daily interval
    df_planets = exporter.generate_time_series(
        start_date=start_date,
        days=days,
        interval_hours=24.0
    )

    print()

    # Add planetary aspects
    df_planets = exporter.add_planetary_aspects(df_planets)

    # Add cyclical features
    df_planets = exporter.add_cyclical_features(df_planets)

    # Rename timestamp to date for merging
    df_planets = df_planets.rename(columns={'timestamp': 'date'})

    # Ensure timezone is UTC for merging
    if df_planets['date'].dt.tz is None:
        df_planets['date'] = df_planets['date'].dt.tz_localize('UTC')

    print()
    print(f"✓ Generated {len(df_planets)} days of REAL astronomical data")
    print(f"✓ Date range: {df_planets['date'].min()} to {df_planets['date'].max()}")
    print(f"✓ Total features: {len(df_planets.columns) - 1}")

    return df_planets


def get_planetary_feature_names(df_planets):
    """Extract planetary feature column names"""
    return [col for col in df_planets.columns if col != 'date']


def merge_forex_with_planets(df_forex, df_planets):
    """Merge forex data with planetary positions on date"""
    df_merged = pd.merge(df_forex, df_planets, on='date', how='inner')

    print(f"  Merged: {len(df_forex)} forex rows → {len(df_merged)} rows (after alignment)")

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
print(f"Overall forex date range: {min_date.date()} to {max_date.date()}")
print(f"Total days: {(max_date - min_date).days}")
print()

# Step 2: Generate REAL planetary data
print("="*100)
print("Step 2: Generating REAL planetary positions for date range...")
print("="*100)
print()

df_planets = generate_real_planetary_features(min_date, max_date)

# Save planetary data separately for reference
planets_file = os.path.join(OUTPUT_DIR, 'real_planetary_positions_forex_range.csv')
df_planets.to_csv(planets_file, index=False)
print()
print(f"✓ Saved real astronomical data to: {planets_file}")
print()

# Get planetary feature names
planetary_features = get_planetary_feature_names(df_planets)

# Group features by type
feature_groups = {
    'Positions (x,y in AU)': [f for f in planetary_features if f.endswith('_x') or f.endswith('_y')],
    'Ecliptic Longitude (degrees)': [f for f in planetary_features if f.endswith('_angle')],
    'Distances (AU from Earth)': [f for f in planetary_features if f.endswith('_distance')],
    'Planetary Aspects (degrees)': [f for f in planetary_features if f.startswith('aspect_')],
    'Cyclical Encoding (sin/cos)': [f for f in planetary_features if f.endswith('_sin') or f.endswith('_cos')],
    'Lunar Features': [f for f in planetary_features if 'lunar' in f.lower()]
}

print("="*100)
print(f"REAL ASTRONOMICAL FEATURES GENERATED: {len(planetary_features)} total")
print("="*100)

for group_name, features in feature_groups.items():
    if len(features) > 0:
        print(f"\n{group_name}: {len(features)} features")
        for feat in features[:5]:  # Show first 5
            print(f"  - {feat}")
        if len(features) > 5:
            print(f"  ... and {len(features) - 5} more")

print()

# Step 3: Merge and save
print("="*100)
print("Step 3: Merging real planetary data with forex pairs...")
print("="*100)
print()

integrated_data = {}

for pair in PAIRS:
    if pair not in all_forex:
        continue

    print(f"{pair}:")

    df_forex = all_forex[pair]
    df_merged = merge_forex_with_planets(df_forex, df_planets)

    # Save integrated dataset
    output_file = os.path.join(OUTPUT_DIR, f'{pair}_1day_with_real_planets.csv')
    df_merged.to_csv(output_file, index=False)

    integrated_data[pair] = df_merged

    print(f"  ✓ Saved to: {output_file}")
    print()

print("="*100)
print("INTEGRATION COMPLETE - USING REAL ASTRONOMICAL DATA")
print("="*100)
print()

print("Summary:")
print(f"  ✓ Forex pairs integrated: {len(integrated_data)}")
print(f"  ✓ Real astronomical features: {len(planetary_features)}")
print(f"  ✓ Data source: NASA JPL DE421 Ephemeris")
print(f"  ✓ Output directory: {OUTPUT_DIR}")
print()

print("Key Features:")
print("  • Ecliptic longitude: Actual celestial positions (0-360°)")
print("  • Distances: Real Earth-planet distances in AU")
print("  • Aspects: True angular separations between planets")
print("  • Lunar phase: Actual moon phase (new/full moon)")
print()

print("What makes this data REAL:")
print("  ✓ Uses NASA JPL ephemerides (same data NASA uses)")
print("  ✓ Positions match actual sky on those dates")
print("  ✓ Accounts for orbital mechanics, perturbations, etc.")
print("  ✓ Accurate to within arcminutes")
print()

print("Next steps:")
print("  1. Run: python train_with_real_planets.py")
print("  2. Test if REAL planetary positions correlate with forex")
print("  3. Compare with random/simulated data as control")
print()

print("Files created:")
for pair in integrated_data.keys():
    print(f"  • {pair}_1day_with_real_planets.csv")
print()
