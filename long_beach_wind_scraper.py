#!/usr/bin/env python3
"""
Long Beach Wind Scraper (Observed + Hourly Forecast)
- Observed: NOAA CO-OPS station 9410665 (PRJC1 – Long Beach Pier J)
- Forecast: NWS API hourly forecast at station lat/lon
- Output: one tidy CSV with UTC timestamps, observed & forecast wind speed (m/s)
"""
import argparse
import datetime as dt
import re
from pathlib import Path
import sys

import pandas as pd
import requests

COOPS_BASE = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
NWS_POINTS = "https://api.weather.gov/points/{lat},{lon}"

# Station constants (Long Beach Pier J / PRJC1)
STATION_ID = "9410665"
STATION_LAT = 33.733
STATION_LON = -118.186

UA = "LongBeachWindScraper/1.0 (contact: you@example.com)"
MPS_PER_MPH = 0.44704

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
        "units": "metric",
        "format": "json",
        "application": "LongBeachWindScraper",
    }
    r = requests.get(COOPS_BASE, params=params, headers={"User-Agent": UA}, timeout=30)
    r.raise_for_status()
    payload = r.json()
    data = payload.get("data", [])
    if not data:
        raise RuntimeError(f"CO-OPS returned no data: {payload.get('error', {})}")

    rows = []
    for row in data:
        t = row.get("t")  # 'YYYY-MM-DD HH:MM'
        s = row.get("s")  # m/s
        try:
            ts = pd.to_datetime(t, utc=True)
        except Exception:
            continue
        try:
            s_val = float(s) if s not in ("", None) else None
        except ValueError:
            s_val = None
        rows.append({"timestamp": ts, "observed_wind_speed_ms": s_val})

    return (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["timestamp"])
        .sort_values("timestamp")
    )

def nws_forecast_hourly(lat: float, lon: float) -> pd.DataFrame:
    r = requests.get(
        NWS_POINTS.format(lat=lat, lon=lon),
        headers={"User-Agent": UA, "Accept": "application/geo+json"},
        timeout=30,
    )
    r.raise_for_status()
    hourly_url = r.json()["properties"].get("forecastHourly")
    if not hourly_url:
        raise RuntimeError("NWS points response missing forecastHourly URL")

    r2 = requests.get(
        hourly_url,
        headers={"User-Agent": UA, "Accept": "application/geo+json"},
        timeout=30,
    )
    r2.raise_for_status()
    periods = r2.json().get("properties", {}).get("periods", [])

    rows = []
    for p in periods:
        start_time = p.get("startTime")
        ws = p.get("windSpeed")  # e.g., "7 mph" or "10 to 15 mph"
        m = re.search(r"(\d+)", ws or "")
        if not start_time or not m:
            continue
        ts = pd.to_datetime(start_time, utc=True)
        ms = float(m.group(1)) * MPS_PER_MPH
        rows.append({"timestamp": ts, "forecast_wind_speed_ms": ms})

    df = (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["timestamp"])
        .sort_values("timestamp")
    )
    # Keep only future (>= now-1h)
    now = utc_now_floor_hour() - dt.timedelta(hours=1)
    return df[df["timestamp"] >= now]

def build_tidy_csv(days_back: int, out_path: Path) -> Path:
    obs = coops_observed_wind(days_back)
    fc = nws_forecast_hourly(STATION_LAT, STATION_LON)

    # Normalize timestamps to exact hour
    obs["timestamp"] = obs["timestamp"].dt.floor("H")
    fc["timestamp"] = fc["timestamp"].dt.floor("H")

    merged = (
        pd.merge(obs, fc, on="timestamp", how="outer")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    return out_path

def main():
    parser = argparse.ArgumentParser(description="Long Beach wind scraper (observed + hourly forecast)")
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
