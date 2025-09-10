# NOAA → BigQuery Pipeline (VIBECODING PROJECT)
> A vibecoding project hoping to answer "why do weather forecasters always get seabreeze data wrong?"

## Background

As a professional yacht racing coach, my role is to be a weather strategist to the teams I coach. Often in seabreeze venues, the forecast largley underestimates the thermal strength of the wind. I have come to learn a few visual indicators that have allowed me to advise my clients more accuratley (ex: humidity/clouds over inland deserts), but want data to backup my assumptions. I am hoping to use this project to test my hypothesis:
## When there is a high humidity index inland and a big enough temperature delta, the observed wind will be a much higher velocity at peak thermal than forecast. What the index and delta are, I hope to find out. 

## Overview

This repo is a lightweight **data engineering pipeline** that collects NOAA observations/forecasts with simple Python scrapers, writes tidy CSVs, and **loads them into BigQuery** for analysis and visualization. Next step: expose selected, aggregated views to a frontend.

### What it does (today)

* Pulls observed/forecast data from NOAA sources (e.g., CO-OPS, NWS, NDBC).
* Normalizes into tidy DataFrames (UTC timestamps, clear units).
* Loads datasets to **BigQuery** with partitioned tables.
* Provides **starter SQL** for dashboards/viz.
* TODO: a minimal **API + frontend** to explore and compare metrics in the browser.

### Tech stack

* **Python 3.10+** (requests, pandas, google-cloud-bigquery)
* **BigQuery** ( SQL for transformations)


---

## Quickstart (TL;DR)

```bash
# 1) Create & activate a virtualenv
python3 -m venv .venv && source .venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) Auth to GCP (uses GOOGLE_APPLICATION_CREDENTIALS or gcloud auth)
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json

# 4) Run a scraper locally to generate .csv files

# 5) Load into BigQuery
python uploader.py

# 6) Run  query in GCP interface


## Repo structure (suggested)


