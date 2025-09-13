#!/usr/bin/env python3
"""
Long Beach Wind Scraper (Observed + Historical Forecast)
- Observed: NOAA CO-OPS station 9410665 (PRJC1 – Long Beach Pier J)
- Historical forecast: Open-Meteo Historical Forecast API at station lat/lon
- Output: one tidy CSV with UTC timestamps, observed & forecast wind speed (m/s)

USAGE: python3 long_beach_wind_scraper.py --days-back 14 --out long_beach_wind.csv
"""
import argparse
import datetime as dt
from pathlib import Path
import sys

import pandas as pd
import requests

COOPS_BASE = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"

# Station constants (Long Beach Pier J / PRJC1)
STATION_ID = "9410665"
STATION_LAT = 33.733
STATION_LON = -118.186

UA = "LongBeachWindScraper/2.0 (contact: you@example.com)"

session = requests.Session()
session.headers.update({"User-Agent": UA})

def utc_now_floor_hour():
    now = dt.datetime.now(dt.timezone.utc)
    return now.replace(minute=0, second=0, microsecond=0)

def coops_observed_wind(days_back: int) -> pd.DataFrame:
    end_dt = utc_now_floor_hour()
    start_dt = end_dt - dt.timedelta(days=days_back)
    params = {
        "product": "wind",
        "station": STATION_ID,
        "begin_date": start_dt.strftime("%Y%m%d %H:%M"),
        "end_date": end_dt.strftime("%Y%m%d %H:%M"),
        "interval": "h",
        "time_zone": "gmt",
        "units": "metric",  # speeds in m/s
        "format": "json",
        "application": "LongBeachWindScraper",
    }
    r = session.get(COOPS_BASE, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()
    data = payload.get("data", [])
    if not data:
        raise RuntimeError(f"CO-OPS returned no data: {payload.get('error', {})}")

    rows = []
    for row in data:
        t = row.get("t")        # 'YYYY-MM-DD HH:MM'
        s = row.get("s")        # m/s
        ts = pd.to_datetime(t, utc=True)
        s_val = float(s) if s not in ("", None) else None
        rows.append({"timestamp": ts, "observed_wind_speed_ms": s_val})

    df = pd.DataFrame(rows).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df["timestamp"] = df["timestamp"].dt.floor("H")   # normalize to hour
    return df

def historical_forecast_open_meteo(lat: float, lon: float, start_dt: dt.datetime, end_dt: dt.datetime) -> pd.DataFrame:
    """
    Pull archived hourly forecasts for the requested window.
    Docs: https://open-meteo.com/en/docs/historical-forecast-api
    """
    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["wind_speed_10m"],
        "windspeed_unit": "ms",      # m/s to match CO-OPS
        "timezone": "UTC",
        "start_date": start_dt.strftime("%Y-%m-%d"),
        "end_date": end_dt.strftime("%Y-%m-%d"),
        # You can pin a model if you want (e.g., "hrrr" or "nbm"); otherwise best match:
        # "models": "hrrr"
    }
    r = session.get(url, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    if "hourly" not in data or "time" not in data["hourly"]:
        raise RuntimeError(f"Historical forecast missing hourly series: {data}")

    times = pd.to_datetime(data["hourly"]["time"], utc=True).floor("H")
    speeds = data["hourly"]["wind_speed_10m"]
    df = pd.DataFrame({"timestamp": times, "forecast_wind_speed_ms": speeds}).drop_duplicates("timestamp")
    # Clip to observed window exactly (hour-rounded)
    df = df[(df["timestamp"] >= start_dt.replace(minute=0, second=0, microsecond=0, tzinfo=dt.timezone.utc)) &
            (df["timestamp"] <= end_dt.replace(minute=0, second=0, microsecond=0, tzinfo=dt.timezone.utc))]
    return df.sort_values("timestamp")

def build_tidy_csv(days_back: int, out_path: Path) -> Path:
    obs = coops_observed_wind(days_back)
    if obs.empty:
        raise RuntimeError("No observed data returned")

    start_dt = obs["timestamp"].min().to_pydatetime()
    end_dt   = obs["timestamp"].max().to_pydatetime()

    fc = historical_forecast_open_meteo(STATION_LAT, STATION_LON, start_dt, end_dt)

    # quick sanity prints (remove if you like)
    print(f"Observed window: {start_dt} → {end_dt}  (rows={len(obs)})")
    if not fc.empty:
        print(f"Forecast window:  {fc['timestamp'].min()} → {fc['timestamp'].max()}  (rows={len(fc)})")
    else:
        print("Forecast window:  (no rows)")

    # LEFT JOIN on observed timestamps
    merged = (pd.merge(obs, fc, on="timestamp", how="left")
                .sort_values("timestamp")
                .reset_index(drop=True))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)

    # overlap count for debugging
    overlap = merged["forecast_wind_speed_ms"].notna().sum()
    print(f"Overlap hours (obs with forecast): {overlap}/{len(merged)}")
    return out_path

def main():
    parser = argparse.ArgumentParser(description="Long Beach wind scraper (observed + historical forecast)")
    parser.add_argument("--days-back", type=int, default=14, help="Observed days to retrieve (default: 14)")
    parser.add_argument("--out", type=Path, default=Path("long_beach_wind.csv"), help="Output CSV path")
    args = parser.parse_args()
    try:
        path = build_tidy_csv(args.days_back, args.out)
        print(f"✅ Wrote {path}")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
