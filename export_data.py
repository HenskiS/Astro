"""
Export planetary position data for time series analysis and correlation with forex data
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from geocentric_model import GeocentricModel
from typing import Optional


class PlanetaryDataExporter:
    """Exports geocentric planetary position data to various formats."""

    def __init__(self, model: GeocentricModel):
        self.model = model

    def generate_time_series(
        self,
        start_date: datetime,
        days: int,
        interval_hours: float = 24.0
    ) -> pd.DataFrame:
        """
        Generate time series data of planetary positions.

        Args:
            start_date: Starting date for the time series
            days: Number of days to generate data for
            interval_hours: Hours between each data point

        Returns:
            DataFrame with columns: timestamp, day, and x_y for each planet,
            plus derived features like angles and distances
        """
        interval_days = interval_hours / 24.0
        num_points = int(days / interval_days)

        # Initialize data structure
        data = {
            'timestamp': [],
            'day': []
        }

        # Add columns for each celestial body
        for body_name in self.model.bodies.keys():
            data[f'{body_name}_x'] = []
            data[f'{body_name}_y'] = []
            data[f'{body_name}_angle'] = []  # Angle from Earth in radians
            data[f'{body_name}_distance'] = []  # Distance from Earth

        # Generate data points
        for i in range(num_points):
            day = i * interval_days
            timestamp = start_date + timedelta(days=day)

            data['timestamp'].append(timestamp)
            data['day'].append(day)

            positions = self.model.get_all_positions(day)

            for body_name, (x, y) in positions.items():
                data[f'{body_name}_x'].append(x)
                data[f'{body_name}_y'].append(y)

                # Calculate polar coordinates
                angle = np.arctan2(y, x)  # Angle in radians
                distance = np.sqrt(x**2 + y**2)

                data[f'{body_name}_angle'].append(angle)
                data[f'{body_name}_distance'].append(distance)

        df = pd.DataFrame(data)
        return df

    def add_planetary_aspects(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add astrological aspects (angular relationships between planets).

        Args:
            df: DataFrame with planetary positions

        Returns:
            DataFrame with additional aspect columns
        """
        planets = list(self.model.bodies.keys())

        # Calculate angular separations between all planet pairs
        for i, planet1 in enumerate(planets):
            for planet2 in planets[i+1:]:
                angle1 = df[f'{planet1}_angle']
                angle2 = df[f'{planet2}_angle']

                # Angular separation (normalized to -π to π)
                separation = angle2 - angle1
                separation = np.arctan2(np.sin(separation), np.cos(separation))

                # Store absolute separation in degrees
                df[f'aspect_{planet1}_{planet2}'] = np.abs(np.degrees(separation))

        return df

    def add_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add cyclical encoding of angles (sin/cos) for ML compatibility.

        Args:
            df: DataFrame with planetary positions

        Returns:
            DataFrame with cyclical features
        """
        for body_name in self.model.bodies.keys():
            angle = df[f'{body_name}_angle']
            df[f'{body_name}_sin'] = np.sin(angle)
            df[f'{body_name}_cos'] = np.cos(angle)

        return df

    def export_to_csv(
        self,
        filename: str,
        start_date: datetime,
        days: int,
        interval_hours: float = 24.0,
        include_aspects: bool = True,
        include_cyclical: bool = True
    ):
        """
        Export planetary data to CSV file.

        Args:
            filename: Output CSV filename
            start_date: Starting date for the time series
            days: Number of days to generate
            interval_hours: Hours between data points
            include_aspects: Whether to include angular aspects
            include_cyclical: Whether to include sin/cos encoding
        """
        print(f"Generating planetary data for {days} days...")
        df = self.generate_time_series(start_date, days, interval_hours)

        if include_aspects:
            print("Calculating planetary aspects...")
            df = self.add_planetary_aspects(df)

        if include_cyclical:
            print("Adding cyclical features...")
            df = self.add_cyclical_features(df)

        print(f"Exporting to {filename}...")
        df.to_csv(filename, index=False)
        print(f"Export complete! {len(df)} data points saved.")
        print(f"\nColumns: {', '.join(df.columns)}")

        return df

    def get_summary_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get summary statistics for the planetary data."""
        return df.describe()


def main():
    """Example usage of the data exporter."""
    model = GeocentricModel()
    exporter = PlanetaryDataExporter(model)

    # Export data for analysis
    # Using recent past date as example - you can change this
    start_date = datetime(2024, 1, 1)

    # Export daily data for 2 years
    df = exporter.export_to_csv(
        filename='planetary_positions_daily.csv',
        start_date=start_date,
        days=730,  # 2 years
        interval_hours=24.0,
        include_aspects=True,
        include_cyclical=True
    )

    print("\n" + "="*60)
    print("Data Summary:")
    print("="*60)
    print(df.head())
    print("\n")
    print(exporter.get_summary_statistics(df))

    # Export hourly data for shorter period (useful for forex correlation)
    print("\n" + "="*60)
    print("Generating hourly data...")
    print("="*60)
    df_hourly = exporter.export_to_csv(
        filename='planetary_positions_hourly.csv',
        start_date=start_date,
        days=90,  # 3 months
        interval_hours=1.0,
        include_aspects=True,
        include_cyclical=True
    )

    print("\nFiles generated:")
    print("  - planetary_positions_daily.csv (for long-term patterns)")
    print("  - planetary_positions_hourly.csv (for forex correlation)")


if __name__ == '__main__':
    main()
