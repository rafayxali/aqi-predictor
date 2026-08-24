"""
Quick script to query AQICN's live feed for Islamabad and print the response.

Run this locally (not in the sandbox) with your real token:
    python check_aqicn_feed.py

Setup:
    pip install requests python-dotenv
    Create a .env file next to this script with:
        AQICN_TOKEN=your_token_here
"""

import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("AQICN_TOKEN")
CITY = "islamabad"

if not TOKEN:
    raise SystemExit("AQICN_TOKEN not found. Set it in a .env file or as an environment variable.")

url = f"https://api.waqi.info/feed/{CITY}/"
response = requests.get(url, params={"token": TOKEN})
response.raise_for_status()

data = response.json()

print("=== FULL RAW RESPONSE ===")
print(json.dumps(data, indent=2))

if data.get("status") == "ok":
    d = data["data"]
    print("\n=== KEY FIELDS ===")
    print(f"Station name : {d.get('city', {}).get('name')}")
    print(f"AQI          : {d.get('aqi')}")
    print(f"Dominant pol : {d.get('dominentpol')}")
    print(f"Timestamp    : {d.get('time', {}).get('s')}")
    print(f"Coordinates  : {d.get('city', {}).get('geo')}")

    iaqi = d.get("iaqi", {})
    print("\n=== INDIVIDUAL POLLUTANT / WEATHER SUB-INDICES (iaqi) ===")
    for key, val in iaqi.items():
        print(f"  {key}: {val.get('v')}")
else:
    print("\nAQICN returned an error status:")
    print(data)