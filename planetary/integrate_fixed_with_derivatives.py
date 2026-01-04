"""
FIXED integration: Handles DST properly and adds velocity/acceleration features
- Generates planetary data for EXACT forex timestamps
- Adds velocity (first derivative) and acceleration (second derivative) features
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime
from real_astronomy import RealAstronomyModel, RealAstronomyDataExporter

print("="*100)
print("FIXED INTEGRATION: Real Astronomy with Derivatives")
print("Handles DST properly + Adds velocity & acceleration")
print("="*100)
print()

DATA_DIR = 'data'
OUTPUT_DIR = 'data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY']


def load_all_forex_timestamps():
    """Load all unique timestamps from all forex pairs"""
    print("Step 1: Loading ALL forex data to get unique timestamps...")
    print()

    all_timestamps = set()
    all_forex_data = {}

    for pair in PAIRS:
        filepath = os.path.join(DATA_DIR, f'{pair}_1day_oanda.csv')
        try:
            df = pd.read_csv(filepath)
            df['date'] = pd.to_datetime(df['date'])

            # Collect all unique timestamps
            all_timestamps.update(df['date'])
            all_forex_data[pair] = df

            print(f"  {pair}: {len(df)} days")

        except Exception as e:
            print(f"  {pair}: ERROR - {str(e)}")

    # Convert to sorted list
    timestamps_list = sorted(list(all_timestamps))

    print()
    print(f"Total unique timestamps across all pairs: {len(timestamps_list)}")
    print(f"Date range: {timestamps_list[0]} to {timestamps_list[-1]}")
    print()

    # Show time distribution
    times = pd.Series([ts.time() for ts in timestamps_list])
    print("Time distribution (checking DST):")
    for time, count in times.value_counts().items():
        print(f"  {time}: {count} occurrences")
    print()

    return timestamps_list, all_forex_data


def generate_planetary_for_exact_timestamps(timestamps_list):
    """Generate planetary positions for exact forex timestamps"""
    print("Step 2: Generating planetary positions for EXACT forex timestamps...")
    print(f"(This will take ~5-10 minutes for {len(timestamps_list)} timestamps)")
    print()

    model = RealAstronomyModel()

    # Initialize data structure
    data = {'timestamp': []}

    for body_name in model.bodies.keys():
        data[f'{body_name}_x'] = []
        data[f'{body_name}_y'] = []
        data[f'{body_name}_angle'] = []
        data[f'{body_name}_distance'] = []

    data['lunar_phase'] = []

    # Generate for each timestamp
    for i, timestamp in enumerate(timestamps_list):
        if i % 500 == 0:
            print(f"  Progress: {i}/{len(timestamps_list)} ({i/len(timestamps_list)*100:.1f}%)")

        data['timestamp'].append(timestamp)

        positions = model.get_all_positions(timestamp)

        for body_name, (x, y) in positions.items():
            data[f'{body_name}_x'].append(x)
            data[f'{body_name}_y'].append(y)
            data[f'{body_name}_angle'].append(model.get_ecliptic_longitude(timestamp, body_name))
            data[f'{body_name}_distance'].append(model.get_distance(timestamp, body_name))

        data['lunar_phase'].append(model.get_lunar_phase(timestamp))

    print(f"  Complete: {len(timestamps_list)} timestamps")

    df = pd.DataFrame(data)
    df = df.rename(columns={'timestamp': 'date'})

    return df


def add_derivatives(df):
    """Add velocity and acceleration features (derivatives of position)"""
    print()
    print("Step 3: Calculating derivatives (velocity & acceleration)...")
    print()

    # Get list of celestial bodies
    bodies = ['Sun', 'Moon', 'Mercury', 'Venus', 'Mars', 'Jupiter', 'Saturn']

    for body in bodies:
        # VELOCITY (first derivative of position)
        # Angular velocity (degrees per day)
        df[f'{body}_angular_velocity'] = df[f'{body}_angle'].diff()

        # Handle 360-degree wrap-around (e.g., 359° -> 1° should be +2°, not -358°)
        df.loc[df[f'{body}_angular_velocity'] < -180, f'{body}_angular_velocity'] += 360
        df.loc[df[f'{body}_angular_velocity'] > 180, f'{body}_angular_velocity'] -= 360

        # Distance velocity (AU per day)
        df[f'{body}_distance_velocity'] = df[f'{body}_distance'].diff()

        # Cartesian velocity (AU per day)
        df[f'{body}_velocity_x'] = df[f'{body}_x'].diff()
        df[f'{body}_velocity_y'] = df[f'{body}_y'].diff()

        # ACCELERATION (second derivative of position)
        # Angular acceleration (degrees per day²)
        df[f'{body}_angular_acceleration'] = df[f'{body}_angular_velocity'].diff()

        # Distance acceleration (AU per day²)
        df[f'{body}_distance_acceleration'] = df[f'{body}_distance_velocity'].diff()

        # Cartesian acceleration
        df[f'{body}_acceleration_x'] = df[f'{body}_velocity_x'].diff()
        df[f'{body}_acceleration_y'] = df[f'{body}_velocity_y'].diff()

    # Lunar phase velocity
    df['lunar_phase_velocity'] = df['lunar_phase'].diff()

    print(f"Added derivative features:")
    print(f"  - Angular velocity/acceleration (7 bodies)")
    print(f"  - Distance velocity/acceleration (7 bodies)")
    print(f"  - Cartesian velocity/acceleration (7 bodies x 2D)")
    print(f"  - Lunar phase velocity")
    print(f"  Total new features: {7 * 8 + 1} = 57 derivative features")
    print()

    return df


def add_planetary_aspects(df):
    """Calculate angular separations between planets"""
    print("Step 4: Calculating planetary aspects...")

    planets = ['Sun', 'Moon', 'Mercury', 'Venus', 'Mars', 'Jupiter', 'Saturn']

    for i, planet1 in enumerate(planets):
        for planet2 in planets[i+1:]:
            angle1 = df[f'{planet1}_angle']
            angle2 = df[f'{planet2}_angle']

            # Angular separation (0-180 degrees)
            separation = np.abs(angle2 - angle1)
            separation = np.minimum(separation, 360 - separation)

            df[f'aspect_{planet1}_{planet2}'] = separation

    print(f"  Calculated {len(planets)*(len(planets)-1)//2} aspect pairs")
    print()

    return df


def add_cyclical_features(df):
    """Add sin/cos encoding for ML"""
    print("Step 5: Adding cyclical encoding (sin/cos)...")

    bodies = ['Sun', 'Moon', 'Mercury', 'Venus', 'Mars', 'Jupiter', 'Saturn']

    for body in bodies:
        angle_rad = np.radians(df[f'{body}_angle'])
        df[f'{body}_sin'] = np.sin(angle_rad)
        df[f'{body}_cos'] = np.cos(angle_rad)

    # Lunar phase cyclical
    df['lunar_phase_sin'] = np.sin(2 * np.pi * df['lunar_phase'])
    df['lunar_phase_cos'] = np.cos(2 * np.pi * df['lunar_phase'])

    print(f"  Added sin/cos for 7 bodies + lunar phase")
    print()

    return df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

timestamps_list, all_forex_data = load_all_forex_timestamps()

df_planets = generate_planetary_for_exact_timestamps(timestamps_list)

# Add derivative features
df_planets = add_derivatives(df_planets)

# Add aspects
df_planets = add_planetary_aspects(df_planets)

# Add cyclical encoding
df_planets = add_cyclical_features(df_planets)

# Save complete planetary dataset
planets_file = os.path.join(OUTPUT_DIR, 'real_planetary_FIXED_with_derivatives.csv')
df_planets.to_csv(planets_file, index=False)
print(f"Saved complete planetary dataset:")
print(f"  {planets_file}")
print(f"  {len(df_planets)} timestamps")
print(f"  {len(df_planets.columns)} total features")
print()

# Merge with each forex pair
print("="*100)
print("Step 6: Merging with each forex pair...")
print("="*100)
print()

for pair in PAIRS:
    if pair not in all_forex_data:
        continue

    print(f"{pair}:")

    df_forex = all_forex_data[pair]

    # Merge on date
    df_merged = pd.merge(df_forex, df_planets, on='date', how='inner')

    print(f"  Forex: {len(df_forex)} days")
    print(f"  Merged: {len(df_merged)} days")
    print(f"  Match rate: {len(df_merged)/len(df_forex)*100:.1f}%")

    if len(df_merged) < len(df_forex) * 0.95:
        print(f"  WARNING: Lost {len(df_forex) - len(df_merged)} days in merge!")

    # Save
    output_file = os.path.join(OUTPUT_DIR, f'{pair}_FIXED_with_derivatives.csv')
    df_merged.to_csv(output_file, index=False)
    print(f"  Saved: {output_file}")
    print()

print("="*100)
print("INTEGRATION COMPLETE - FIXED VERSION")
print("="*100)
print()

print("Summary:")
print(f"  Total timestamps processed: {len(timestamps_list)}")
print(f"  Total features per timestamp: {len(df_planets.columns) - 1}")
print(f"  Pairs integrated: {len([p for p in PAIRS if p in all_forex_data])}")
print()

print("Feature breakdown:")
print(f"  Basic positions: 28 (x, y, angle, distance × 7 bodies)")
print(f"  Derivatives: 57 (velocity & acceleration)")
print(f"  Aspects: 21 (angular separations)")
print(f"  Cyclical: 15 (sin/cos encoding)")
print(f"  Lunar: 4 (phase + derivatives + cyclical)")
print(f"  TOTAL: {28 + 57 + 21 + 15 + 4} = 125 features")
print()

print("New derivative features include:")
print("  - Angular velocity: How fast planet moves in degrees/day")
print("  - Angular acceleration: How fast velocity changes")
print("  - Distance velocity: How fast planet approaches/recedes")
print("  - Distance acceleration: Rate of change of distance velocity")
print("  - Cartesian velocity/acceleration: 2D motion in space")
print()

print("Next step:")
print("  python train_with_derivatives.py")
print()
