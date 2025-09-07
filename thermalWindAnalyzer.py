#!/usr/bin/env python3
"""
Point Loma Wind Analysis with Desert Thermal Effects
Compares observed vs forecast wind data and analyzes thermal effects from inland desert
"""

import requests
import pandas as pd
import numpy as np
import csv
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import time

class ThermalWindAnalyzer:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'ThermalWindAnalyzer/1.0'
        })
        
        # Coordinates for analysis
        self.point_loma = (32.67, -117.24)  # Point Loma, San Diego
        self.desert_east = (32.75, -116.30)  # Borrego Desert (east of SD)
        
        # NOAA Buoy 46232 - Point Loma South
        self.buoy_station = "46232"

        self.start_date = datetime(2025, 8, 22, 0, 0, 0, tzinfo=timezone.utc)
        self.end_date = datetime(2025, 8, 24, 23, 59, 59, tzinfo=timezone.utc)

        
    def fetch_noaa_buoy_data(self, station_id, days_back=1):
        """
        Fetch observed data from NOAA buoy station
        Returns recent wind observations
        """
        # NOAA NDBC real-time data URL
        url = f"https://www.ndbc.noaa.gov/data/realtime2/{station_id}.txt"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            
            # Parse the fixed-width format data
            lines = response.text.strip().split('\n')
            
            if len(lines) < 3:
                return None
            
            # Extract header and units
            headers = lines[0].split()
            units = lines[1].split()
            
            # Parse data lines
            data_rows = []
            current_time = datetime.now(timezone.utc)
            cutoff_time = current_time - timedelta(days=days_back)
            
            for line in lines[2:]:
                if not line.strip():
                    continue
                    
                parts = line.split()
                if len(parts) < len(headers):
                    continue
                
                try:
                    # Parse timestamp
                    year = int(parts[0])
                    month = int(parts[1])
                    day = int(parts[2])
                    hour = int(parts[3])
                    minute = int(parts[4])
                    
                    obs_time = datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
                    
                    if obs_time < cutoff_time:
                        continue
                    
                    # Create data row
                    row = {'timestamp': obs_time}
                    for i, header in enumerate(headers[5:], start=5):
                        if i < len(parts):
                            try:
                                value = float(parts[i]) if parts[i] != 'MM' else None
                                row[header] = value
                            except ValueError:
                                row[header] = None
                    
                    data_rows.append(row)
                    
                except (ValueError, IndexError):
                    continue
            
            return data_rows
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching NOAA buoy data: {e}")
            return None
    
    def fetch_forecast_data(self, lat, lon, days=1):
        """
        Fetch forecast data from Open-Meteo for comparison
        """
        url = "https://api.open-meteo.com/v1/forecast"
        
        params = {
            'latitude': lat,
            'longitude': lon,
            'hourly': [
                'wind_speed_10m',
                'wind_direction_10m',
                'wind_gusts_10m',
                'temperature_2m'
            ],
            'timezone': 'UTC',
            'forecast_days': days,
            'past_days': 1  # Get yesterday's forecast for comparison
        }
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching forecast data: {e}")
            return None
    
    def fetch_desert_weather(self, lat, lon, days=1):
        """
        Fetch desert weather data including temperature and cloud cover
        """
        url = "https://api.open-meteo.com/v1/forecast"
        
        params = {
            'latitude': lat,
            'longitude': lon,
            'hourly': [
                'temperature_2m',
                'cloud_cover',
                'wind_speed_10m',
                'wind_direction_10m',
                'relative_humidity_2m',
                'surface_pressure'
            ],
            'timezone': 'UTC',
            'forecast_days': days,
            'past_days': 1
        }
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching desert weather: {e}")
            return None
    
    def process_buoy_data(self, buoy_data):
        """Process NOAA buoy data into DataFrame"""
        if not buoy_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(buoy_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Rename columns for clarity (NOAA buoy columns)
        column_mapping = {
            'WSPD': 'wind_speed_ms',
            'WDIR': 'wind_direction_deg',
            'GST': 'wind_gust_ms',
            'ATMP': 'air_temp_c',
            'WTMP': 'water_temp_c'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df[new_col] = df[old_col]
        
        df['data_source'] = 'observed'
        return df
    
    def process_forecast_data(self, forecast_data, location_name):
        """Process forecast data into DataFrame"""
        if not forecast_data or 'hourly' not in forecast_data:
            return pd.DataFrame()
        
        hourly = forecast_data['hourly']
        
        data_rows = []
        for i, timestamp in enumerate(hourly['time']):
            row = {
                'timestamp': pd.to_datetime(timestamp),
                'wind_speed_ms': hourly['wind_speed_10m'][i] if hourly['wind_speed_10m'][i] is not None else None,
                'wind_direction_deg': hourly['wind_direction_10m'][i] if hourly['wind_direction_10m'][i] is not None else None,
                'wind_gust_ms': hourly['wind_gusts_10m'][i] if hourly['wind_gusts_10m'][i] is not None else None,
                'temperature_c': hourly['temperature_2m'][i] if hourly['temperature_2m'][i] is not None else None,
                'location': location_name,
                'data_source': 'forecast'
            }
            data_rows.append(row)
        
        return pd.DataFrame(data_rows)
    
    def process_desert_data(self, desert_data):
        """Process desert weather data into DataFrame"""
        if not desert_data or 'hourly' not in desert_data:
            return pd.DataFrame()
        
        hourly = desert_data['hourly']
        
        data_rows = []
        for i, timestamp in enumerate(hourly['time']):
            row = {
                'timestamp': pd.to_datetime(timestamp),
                'desert_temp_c': hourly['temperature_2m'][i],
                'desert_cloud_cover': hourly['cloud_cover'][i],
                'desert_wind_speed_ms': hourly['wind_speed_10m'][i],
                'desert_wind_direction_deg': hourly['wind_direction_10m'][i],
                'desert_humidity': hourly['relative_humidity_2m'][i],
                'desert_pressure': hourly['surface_pressure'][i]
            }
            data_rows.append(row)
        
        return pd.DataFrame(data_rows)
    
    def calculate_thermal_index(self, desert_df, coastal_df):
        """
        Calculate thermal gradient and potential thermal wind effects
        """
        if desert_df.empty or coastal_df.empty:
            return pd.DataFrame()
        
        # Merge datasets on timestamp
        merged = pd.merge(desert_df, coastal_df, on='timestamp', how='inner')
        
        if merged.empty:
            return pd.DataFrame()
        
        # Calculate thermal metrics
        merged['temp_gradient'] = merged['desert_temp_c'] - merged['temperature_c']
        merged['cloud_effect'] = 100 - merged['desert_cloud_cover']  # Clear sky percentage
        merged['thermal_potential'] = merged['temp_gradient'] * merged['cloud_effect'] / 100
        
        # Estimate thermal wind strength (simplified model)
        # Higher temp gradient + clear skies = stronger thermal effect
        merged['thermal_wind_estimate'] = np.where(
            merged['thermal_potential'] > 5,  # Threshold for significant thermal effect
            merged['thermal_potential'] * 0.3,  # Scaling factor
            0
        )
        
        return merged
    
    def analyze_forecast_accuracy(self, observed_df, forecast_df):
        """
        Compare observed vs forecast wind data
        """
        if observed_df.empty or forecast_df.empty:
            return pd.DataFrame()
        
        # Align timestamps (round to nearest hour for comparison)
        observed_df['hour'] = observed_df['timestamp'].dt.floor('h').dt.tz_localize(None)
        forecast_df['hour'] = forecast_df['timestamp'].dt.floor('h').dt.tz_localize(None)
        
        # Merge on hourly basis
        comparison = pd.merge(
            observed_df.groupby('hour').agg({
                'wind_speed_ms': 'mean',
                'wind_direction_deg': 'mean',
                'wind_gust_ms': 'mean'
            }).reset_index(),
            forecast_df.groupby('hour').agg({
                'wind_speed_ms': 'mean',
                'wind_direction_deg': 'mean',
                'wind_gust_ms': 'mean'
            }).reset_index(),
            on='hour',
            suffixes=('_observed', '_forecast')
        )
        
        if comparison.empty:
            return pd.DataFrame()
        
        # Calculate differences
        comparison['wind_speed_diff'] = comparison['wind_speed_ms_forecast'] - comparison['wind_speed_ms_observed']
        comparison['wind_dir_diff'] = comparison['wind_direction_deg_forecast'] - comparison['wind_direction_deg_observed']
        
        # Handle wind direction circular difference
        comparison['wind_dir_diff'] = np.where(
            comparison['wind_dir_diff'] > 180,
            comparison['wind_dir_diff'] - 360,
            np.where(comparison['wind_dir_diff'] < -180,
                    comparison['wind_dir_diff'] + 360,
                    comparison['wind_dir_diff'])
        )
        
        return comparison
    
    def save_analysis_to_csv(self, observed_df, forecast_df, desert_df, thermal_df, comparison_df):
        """Save all analysis data to CSV files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        files_created = []
        
        # Save individual datasets
        if not observed_df.empty:
            filename = f"point_loma_observed_{timestamp}.csv"
            observed_df.to_csv(filename, index=False)
            files_created.append(filename)
        
        if not forecast_df.empty:
            filename = f"point_loma_forecast_{timestamp}.csv"
            forecast_df.to_csv(filename, index=False)
            files_created.append(filename)
        
        if not desert_df.empty:
            filename = f"desert_weather_{timestamp}.csv"
            desert_df.to_csv(filename, index=False)
            files_created.append(filename)
        
        if not thermal_df.empty:
            filename = f"thermal_analysis_{timestamp}.csv"
            thermal_df.to_csv(filename, index=False)
            files_created.append(filename)
        
        if not comparison_df.empty:
            filename = f"forecast_comparison_{timestamp}.csv"
            comparison_df.to_csv(filename, index=False)
            files_created.append(filename)
        
        return files_created
    
    def run_analysis(self):
        """Main analysis function"""
        print("Starting Point Loma wind and desert thermal analysis...")
        
        # Fetch observed data from NOAA buoy
        print(f"Fetching observed data from NOAA buoy {self.buoy_station}...")
        buoy_data = self.fetch_noaa_buoy_data(self.buoy_station, days_back=1)
        observed_df = self.process_buoy_data(buoy_data)
        
        # Fetch forecast data for Point Loma
        print("Fetching forecast data for Point Loma...")
        forecast_data = self.fetch_forecast_data(self.point_loma[0], self.point_loma[1])
        forecast_df = self.process_forecast_data(forecast_data, 'Point_Loma')
        
        # Fetch desert weather data
        print("Fetching desert weather data...")
        desert_data = self.fetch_desert_weather(self.desert_east[0], self.desert_east[1])
        desert_df = self.process_desert_data(desert_data)
        
        # Calculate thermal effects
        print("Calculating thermal effects...")
        thermal_df = self.calculate_thermal_index(desert_df, forecast_df)
        
        # Compare forecast accuracy
        print("Analyzing forecast accuracy...")
        comparison_df = self.analyze_forecast_accuracy(observed_df, forecast_df)
        
        # Save results to CSV
        print("Saving analysis to CSV files...")
        csv_files = self.save_analysis_to_csv(
            observed_df, forecast_df, desert_df, thermal_df, comparison_df
        )
        
        # Print summary
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        
        if not observed_df.empty:
            print(f"Observed data points: {len(observed_df)}")
            avg_wind = observed_df['wind_speed_ms'].mean()
            print(f"Average observed wind speed: {avg_wind:.1f} m/s")
        
        if not comparison_df.empty:
            avg_diff = comparison_df['wind_speed_diff'].mean()
            print(f"Average forecast error: {avg_diff:.1f} m/s")
        
        if not thermal_df.empty:
            max_thermal = thermal_df['thermal_potential'].max()
            print(f"Maximum thermal potential: {max_thermal:.1f}")
        
        print(f"\nCSV files created: {len(csv_files)}")
        for file in csv_files:
            print(f"  - {file}")
        
        return {
            'observed': observed_df,
            'forecast': forecast_df,
            'desert': desert_df,
            'thermal': thermal_df,
            'comparison': comparison_df,
            'csv_files': csv_files
        }

def main():
    """Main execution function"""
    analyzer = ThermalWindAnalyzer()
    
    try:
        results = analyzer.run_analysis()
        
        print("\n" + "="*60)
        print("KEY FINDINGS:")
        print("="*60)
        
        # Forecast accuracy insights
        if not results['comparison'].empty:
            comp_df = results['comparison']
            mean_error = comp_df['wind_speed_diff'].mean()
            rmse = np.sqrt((comp_df['wind_speed_diff'] ** 2).mean())
            
            print(f"Wind Speed Forecast Performance:")
            print(f"  Mean error: {mean_error:.2f} m/s")
            print(f"  RMSE: {rmse:.2f} m/s")
            
            if mean_error > 1:
                print("  → Forecast tends to OVERPREDICT wind speeds")
            elif mean_error < -1:
                print("  → Forecast tends to UNDERPREDICT wind speeds")
            else:
                print("  → Forecast shows good wind speed accuracy")
        
        # Thermal effect insights
        if not results['thermal'].empty:
            thermal_df = results['thermal']
            high_thermal = len(thermal_df[thermal_df['thermal_potential'] > 5])
            total_hours = len(thermal_df)
            
            print(f"\nThermal Effect Analysis:")
            print(f"  Hours with significant thermal potential: {high_thermal}/{total_hours}")
            print(f"  Maximum temperature gradient: {thermal_df['temp_gradient'].max():.1f}°C")
            print(f"  Average desert cloud cover: {thermal_df['desert_cloud_cover'].mean():.1f}%")
            
            if high_thermal > 0:
                print("  → Desert thermal effects may influence coastal winds")
            else:
                print("  → Minimal thermal effect observed")
        
    except Exception as e:
        print(f"Analysis failed: {e}")

if __name__ == "__main__":
    main()