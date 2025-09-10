#!/usr/bin/env python3
"""
Simple Wind Data Collector for Point Loma South (46232) and Imperial Beach (46235)
Fetches observed wind data from NOAA buoys and forecast data for comparison
"""

import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
import time

class SimpleWindCollector:
    def __init__(self, days_back=14):
        """
        Initialize collector for two specific stations
        
        Args:
            days_back (int): How many days back to collect data
        """
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'SimpleWindCollector/1.0'
        })
        
        # Our two target stations
        self.stations = {
            "46232": {
                "name": "Point Loma South", 
                "lat": 32.57, 
                "lon": -117.39
            },
            "46235": {
                "name": "Imperial Beach", 
                "lat": 32.53, 
                "lon": -117.28
            }
        }
        
        # Date range
        self.end_date = datetime.now(timezone.utc)
        self.start_date = self.end_date - timedelta(days=days_back)
        
        print(f"Collecting wind data for: Point Loma South (46232) & Imperial Beach (46235)")
        print(f"Date range: {self.start_date.strftime('%Y-%m-%d %H:%M')} to {self.end_date.strftime('%Y-%m-%d %H:%M')} UTC")

    def fetch_observed_wind(self, station_id):
        """
        Fetch observed wind data from NOAA buoy station
        """
        url = f"https://www.ndbc.noaa.gov/data/realtime2/{station_id}.txt"
        
        print(f"\n📡 Fetching observed data from {station_id} ({self.stations[station_id]['name']})...")
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            lines = response.text.strip().split('\n')
            if len(lines) < 3:
                print(f"❌ Not enough data from station {station_id}")
                return []
            
            # Parse header
            headers = lines[0].split()
            print(f"Available data: {headers[5:10]}...")  # Show first few data columns
            
            observations = []
            
            for line in lines[2:]:  # Skip header and units
                parts = line.split()
                if len(parts) < 8:  # Need at least timestamp + wind data
                    continue
                
                try:
                    # Parse timestamp: YYYY MM DD hh mm
                    year = int(parts[0])
                    month = int(parts[1])
                    day = int(parts[2])
                    hour = int(parts[3])
                    minute = int(parts[4])
                    
                    obs_time = datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
                    
                    # Filter by date range
                    if obs_time < self.start_date or obs_time > self.end_date:
                        continue
                    
                    # Extract wind data - typical NOAA order: WDIR WSPD GST
                    wind_dir_raw = parts[5] if len(parts) > 5 else 'MM'
                    wind_speed_raw = parts[6] if len(parts) > 6 else 'MM'
                    wind_gust_raw = parts[7] if len(parts) > 7 else 'MM'
                    
                    # Convert to numbers, handle missing data
                    def parse_value(raw_val):
                        if raw_val in ['MM', '999', '99.0', '999.0']:
                            return None
                        try:
                            val = float(raw_val)
                            return val if val >= 0 else None  # No negative values
                        except ValueError:
                            return None
                    
                    wind_direction = parse_value(wind_dir_raw)
                    wind_speed = parse_value(wind_speed_raw)
                    wind_gust = parse_value(wind_gust_raw)
                    
                    # Validate wind speed (reasonable range: 0-50 m/s)
                    if wind_speed is not None and (wind_speed > 50):
                        wind_speed = None
                    
                    observation = {
                        'timestamp': obs_time.isoformat(),
                        'station_id': station_id,
                        'station_name': self.stations[station_id]['name'],
                        'data_type': 'observed',
                        'wind_speed_ms': wind_speed,
                        'wind_direction_deg': wind_direction,
                        'wind_gust_ms': wind_gust,
                        'lat': self.stations[station_id]['lat'],
                        'lon': self.stations[station_id]['lon']
                    }
                    
                    observations.append(observation)
                    
                except (ValueError, IndexError):
                    continue  # Skip bad lines
            
            print(f"✅ Collected {len(observations)} wind observations from {station_id}")
            
            # Show sample of what we got
            if observations:
                valid_wind = sum(1 for obs in observations if obs['wind_speed_ms'] is not None)
                print(f"   Valid wind speed readings: {valid_wind}/{len(observations)} ({100*valid_wind/len(observations):.1f}%)")
            
            return observations
            
        except requests.RequestException as e:
            print(f"❌ Error fetching data from {station_id}: {e}")
            return []

    def fetch_forecast_wind(self, lat, lon, location_name):
        """
        Fetch forecast wind data using Open-Meteo API (free and reliable)
        """
        url = "https://api.open-meteo.com/v1/forecast"
        
        params = {
            'latitude': lat,
            'longitude': lon,
            'hourly': ['wind_speed_10m', 'wind_direction_10m', 'wind_gusts_10m'],
            'timezone': 'UTC',
            'start_date': self.start_date.strftime('%Y-%m-%d'),
            'end_date': self.end_date.strftime('%Y-%m-%d'),
        }
        
        print(f"\n📡 Fetching forecast data for {location_name}...")
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'hourly' not in data:
                print(f"❌ No hourly forecast data for {location_name}")
                return []
            
            hourly = data['hourly']
            forecasts = []
            
            for i, timestamp_str in enumerate(hourly['time']):
                forecast_time = datetime.fromisoformat(timestamp_str + '+00:00')
                
                forecast = {
                    'timestamp': forecast_time.isoformat(),
                    'location_name': location_name,
                    'data_type': 'forecast',
                    'wind_speed_ms': hourly['wind_speed_10m'][i],
                    'wind_direction_deg': hourly['wind_direction_10m'][i],
                    'wind_gust_ms': hourly['wind_gusts_10m'][i],
                    'lat': lat,
                    'lon': lon
                }
                forecasts.append(forecast)
            
            print(f"✅ Collected {len(forecasts)} forecast hours for {location_name}")
            return forecasts
            
        except requests.RequestException as e:
            print(f"❌ Error fetching forecast for {location_name}: {e}")
            return []

    def collect_all_data(self):
        """
        Collect both observed and forecast data for both stations
        """
        all_data = []
        
        print("=" * 60)
        print("COLLECTING OBSERVED WIND DATA")
        print("=" * 60)
        
        # Get observed data from both buoy stations
        for station_id in self.stations:
            observed_data = self.fetch_observed_wind(station_id)
            all_data.extend(observed_data)
            time.sleep(1)  # Be nice to NOAA servers
        
        print("\n" + "=" * 60)
        print("COLLECTING FORECAST WIND DATA")
        print("=" * 60)
        
        # Get forecast data for both station locations
        for station_id, info in self.stations.items():
            forecast_data = self.fetch_forecast_wind(
                info['lat'], 
                info['lon'], 
                f"{info['name']} Forecast"
            )
            all_data.extend(forecast_data)
            time.sleep(1)  # Rate limiting
        
        # Convert to DataFrame
        if all_data:
            df = pd.DataFrame(all_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            print(f"\n✅ Total data points collected: {len(df)}")
            
            # Show breakdown
            obs_count = len(df[df['data_type'] == 'observed'])
            forecast_count = len(df[df['data_type'] == 'forecast'])
            print(f"   Observed: {obs_count}")
            print(f"   Forecast: {forecast_count}")
            
            return df
        else:
            print("❌ No data collected")
            return pd.DataFrame()

    def create_comparison(self, df):
        """
        Create observed vs forecast comparison for the same locations
        """
        if df.empty:
            return pd.DataFrame()
        
        print("\n" + "=" * 60)
        print("CREATING OBSERVED vs FORECAST COMPARISON")
        print("=" * 60)
        
        observed = df[df['data_type'] == 'observed'].copy()
        forecast = df[df['data_type'] == 'forecast'].copy()
        
        if observed.empty or forecast.empty:
            print("❌ Missing observed or forecast data for comparison")
            return pd.DataFrame()
        
        comparisons = []
        
        for _, obs in observed.iterrows():
            obs_time = obs['timestamp']
            obs_lat, obs_lon = obs['lat'], obs['lon']
            
            # Find forecasts within 2 hours and 0.2 degrees (~20km) of observation
            time_window = pd.Timedelta(hours=2)
            location_tolerance = 0.2
            
            nearby_forecasts = forecast[
                (abs(forecast['timestamp'] - obs_time) <= time_window) &
                (abs(forecast['lat'] - obs_lat) <= location_tolerance) &
                (abs(forecast['lon'] - obs_lon) <= location_tolerance)
            ]
            
            if not nearby_forecasts.empty:
                # Find closest forecast by time
                time_diffs = abs(nearby_forecasts['timestamp'] - obs_time)
                closest_forecast = nearby_forecasts.iloc[time_diffs.argmin()]
                
                # Calculate wind speed error
                speed_error = None
                if (pd.notna(obs['wind_speed_ms']) and 
                    pd.notna(closest_forecast['wind_speed_ms'])):
                    speed_error = closest_forecast['wind_speed_ms'] - obs['wind_speed_ms']
                
                # Calculate wind direction error (handle circular nature)
                direction_error = None
                if (pd.notna(obs['wind_direction_deg']) and 
                    pd.notna(closest_forecast['wind_direction_deg'])):
                    dir_diff = closest_forecast['wind_direction_deg'] - obs['wind_direction_deg']
                    # Handle circular difference (e.g., 350° vs 10° = 20° difference, not 340°)
                    if dir_diff > 180:
                        dir_diff -= 360
                    elif dir_diff < -180:
                        dir_diff += 360
                    direction_error = dir_diff
                
                comparison = {
                    'timestamp': obs_time,
                    'station_id': obs['station_id'],
                    'station_name': obs['station_name'],
                    'time_diff_minutes': time_diffs.min().total_seconds() / 60,
                    
                    # Observed values
                    'observed_wind_speed_ms': obs['wind_speed_ms'],
                    'observed_wind_direction_deg': obs['wind_direction_deg'],
                    'observed_wind_gust_ms': obs['wind_gust_ms'],
                    
                    # Forecast values
                    'forecast_wind_speed_ms': closest_forecast['wind_speed_ms'],
                    'forecast_wind_direction_deg': closest_forecast['wind_direction_deg'],
                    'forecast_wind_gust_ms': closest_forecast['wind_gust_ms'],
                    
                    # Errors
                    'wind_speed_error_ms': speed_error,
                    'wind_direction_error_deg': direction_error,
                }
                
                comparisons.append(comparison)
        
        if comparisons:
            comparison_df = pd.DataFrame(comparisons)
            print(f"✅ Created {len(comparison_df)} observed vs forecast comparisons")
            
            # Show accuracy summary
            valid_speed_errors = comparison_df['wind_speed_error_ms'].dropna()
            if len(valid_speed_errors) > 0:
                mean_error = valid_speed_errors.mean()
                rmse = (valid_speed_errors ** 2).mean() ** 0.5
                print(f"   Average wind speed error: {mean_error:.2f} m/s")
                print(f"   Wind speed RMSE: {rmse:.2f} m/s")
            
            return comparison_df
        else:
            print("❌ No matching observations and forecasts found")
            return pd.DataFrame()

    def save_data(self, all_data_df, comparison_df):
        """
        Save data to CSV files
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = []
        
        print("\n" + "=" * 60)
        print("SAVING DATA TO CSV")
        print("=" * 60)
        
        # Save all data (observed + forecast)
        if not all_data_df.empty:
            all_data_file = f"wind_data_46232_46235_{timestamp}.csv"
            all_data_df.to_csv(all_data_file, index=False)
            saved_files.append(all_data_file)
            print(f"💾 Saved {all_data_file} ({len(all_data_df)} rows)")
        
        # Save comparison data
        if not comparison_df.empty:
            comparison_file = f"wind_comparison_46232_46235_{timestamp}.csv"
            comparison_df.to_csv(comparison_file, index=False)
            saved_files.append(comparison_file)
            print(f"💾 Saved {comparison_file} ({len(comparison_df)} rows)")
        
        return saved_files

    def run_analysis(self):
        """
        Run the complete analysis
        """
        print("🌪️  SIMPLE WIND ANALYSIS: Point Loma South (46232) & Imperial Beach (46235)")
        print("=" * 80)
        
        # Collect all data
        all_data = self.collect_all_data()
        
        if all_data.empty:
            print("❌ No data collected. Check station IDs and network connection.")
            return
        
        # Create comparison
        comparison = self.create_comparison(all_data)
        
        # Save results
        saved_files = self.save_data(all_data, comparison)
        
        # Final summary
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        
        # Data summary
        obs_data = all_data[all_data['data_type'] == 'observed']
        forecast_data = all_data[all_data['data_type'] == 'forecast']
        
        print(f"📊 Point Loma South (46232) observations: {len(obs_data[obs_data['station_id'] == '46232'])}")
        print(f"📊 Imperial Beach (46235) observations: {len(obs_data[obs_data['station_id'] == '46235'])}")
        print(f"📊 Total forecast points: {len(forecast_data)}")
        print(f"📊 Successful comparisons: {len(comparison)}")
        
        # Show data quality
        if not obs_data.empty:
            valid_wind_obs = obs_data['wind_speed_ms'].notna().sum()
            total_obs = len(obs_data)
            print(f"📊 Valid wind observations: {valid_wind_obs}/{total_obs} ({100*valid_wind_obs/total_obs:.1f}%)")
        
        print(f"\n📁 Files created:")
        for file in saved_files:
            print(f"   • {file}")
        
        return {
            'all_data': all_data,
            'comparison': comparison,
            'files': saved_files
        }

def main():
    """
    Main function
    """
    # Create collector - change days_back if you want more data
    collector = SimpleWindCollector(days_back=14)
    
    # Run analysis
    results = collector.run_analysis()
    
    print("\n🎉 Done! Check the CSV files for your wind data.")

if __name__ == "__main__":
    main()