"""
add_latlon_from_shapefile.py
════════════════════════════════════════════════════════════════
Adds latitude and longitude columns to your ERA5 district CSV
using district polygon centroids extracted from the UP shapefile.

HOW TO RUN:
    Step 1 — Install dependency (once):
        pip install geopandas pandas

    Step 2 — Put all these files in the SAME folder as this script:
        up_daily_weather_dataset.csv   ← your ERA5 dataset
        UP_districts.shp               ← shapefile
        UP_districts.dbf
        UP_districts.shx
        UP_districts.prj
        UP_districts.cpg               ← optional

    Step 3 — Run:
        python add_latlon_from_shapefile.py

OUTPUT:
    up_daily_weather_with_latlon.csv   ← new file with lat & lon added
════════════════════════════════════════════════════════════════
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import os

# ─────────────────────────────────────────────────────────────────
# CONFIG — change these paths if your files are in a different folder
# ─────────────────────────────────────────────────────────────────
CSV_PATH       = "data/raw/upshp/up_daily_weather_dataset.csv"   # your ERA5 CSV
SHAPEFILE_PATH = "data/raw/upshp/UP_districts.shp"               # your shapefile
OUTPUT_PATH    = "data/processed/up_daily_weather_with_latlon.csv"

# Name of the district column inside your shapefile
# Your shapefile uses NAME_2 — change only if yours is different
DISTRICT_COL_IN_SHP = "NAME_2"

# ─────────────────────────────────────────────────────────────────
# STEP 1 — Load the shapefile
# ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("  ADDING LAT/LON FROM SHAPEFILE TO ERA5 CSV")
print("=" * 60)

print("\n[1] Loading shapefile...")

if not os.path.exists(SHAPEFILE_PATH):
    raise FileNotFoundError(
        f"\n✗ Shapefile not found: {SHAPEFILE_PATH}"
        f"\n  Make sure UP_districts.shp is in the same folder as this script."
    )

gdf = gpd.read_file(SHAPEFILE_PATH)

print(f"    Shapefile loaded   : {len(gdf)} districts")
print(f"    CRS                : {gdf.crs}")
print(f"    District column    : {DISTRICT_COL_IN_SHP}")
print(f"    All columns in shp : {list(gdf.columns)}")

# ─────────────────────────────────────────────────────────────────
# STEP 2 — Reproject to metric CRS for accurate centroids
#           then convert centroids back to WGS84 lat/lon
# ─────────────────────────────────────────────────────────────────
print("\n[2] Computing accurate district centroids...")

# Reproject to UTM Zone 44N (best metric CRS for Uttar Pradesh)
# This makes centroid calculation accurate in metres, not degrees
gdf_utm = gdf.to_crs(epsg=32644)

# Compute centroids in UTM (accurate)
gdf_utm["centroid_utm"] = gdf_utm.geometry.centroid

# Convert centroids back to WGS84 (degrees lat/lon)
centroids_wgs84 = gdf_utm["centroid_utm"].to_crs(epsg=4326)
gdf["lat"] = centroids_wgs84.y.round(6)
gdf["lon"] = centroids_wgs84.x.round(6)

# Build a simple lookup: district name → (lat, lon)
centroid_lookup = gdf[[DISTRICT_COL_IN_SHP, "lat", "lon"]].copy()
centroid_lookup.columns = ["district", "lat", "lon"]

print(f"    Centroids computed for {len(centroid_lookup)} districts")
print()
print(f"    {'District':<25}  {'Lat':>9}  {'Lon':>9}")
print(f"    {'─'*25}  {'─'*9}  {'─'*9}")
for _, row in centroid_lookup.sort_values("district").iterrows():
    print(f"    {row['district']:<25}  {row['lat']:>9.4f}  {row['lon']:>9.4f}")

# ─────────────────────────────────────────────────────────────────
# STEP 3 — Load your ERA5 CSV
# ─────────────────────────────────────────────────────────────────
print(f"\n[3] Loading ERA5 CSV: {CSV_PATH} ...")

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"\n✗ CSV not found: {CSV_PATH}"
        f"\n  Make sure up_daily_weather_dataset.csv is in the same folder."
    )

df = pd.read_csv(CSV_PATH, parse_dates=["date"])

print(f"    Rows loaded        : {df.shape[0]:,}")
print(f"    Columns            : {list(df.columns)}")
print(f"    Districts in CSV   : {df['district'].nunique()}")
print(f"    Date range         : {df['date'].min().date()} → {df['date'].max().date()}")

# ─────────────────────────────────────────────────────────────────
# STEP 4 — Check district name matching
# ─────────────────────────────────────────────────────────────────
print("\n[4] Checking district name match between CSV and shapefile...")

csv_districts = set(df["district"].unique())
shp_districts = set(centroid_lookup["district"].unique())

matched    = csv_districts & shp_districts
only_csv   = csv_districts - shp_districts
only_shp   = shp_districts - csv_districts

print(f"    Matched            : {len(matched)} / {len(csv_districts)}")

if only_csv:
    print(f"\n    ⚠ In CSV but NOT in shapefile ({len(only_csv)}):")
    for d in sorted(only_csv):
        print(f"       '{d}'")
    print("\n    Fix: add these to the MANUAL_FIXES dict below")

if only_shp:
    print(f"\n    ℹ In shapefile but NOT in CSV ({len(only_shp)}):")
    for d in sorted(only_shp):
        print(f"       '{d}'")

# ── Manual name fixes (if needed) ─────────────────────────────
# Add entries here if district names differ between CSV and shapefile
# Format:  "name_in_csv" : "name_in_shapefile"
MANUAL_FIXES = {
    # Examples (not needed for your dataset — all 75 match perfectly):
    # "Allahabad"  : "Prayagraj",
    # "Faizabad"   : "Ayodhya",
}

if MANUAL_FIXES:
    print(f"\n    Applying {len(MANUAL_FIXES)} manual name fixes...")
    df["district_lookup"] = df["district"].replace(MANUAL_FIXES)
else:
    df["district_lookup"] = df["district"]

# ─────────────────────────────────────────────────────────────────
# STEP 5 — Merge lat/lon into the ERA5 dataframe
# ─────────────────────────────────────────────────────────────────
print("\n[5] Merging lat/lon into dataset...")

df = df.merge(
    centroid_lookup,
    left_on  = "district_lookup",
    right_on = "district",
    how      = "left",
    suffixes = ("", "_shp")
)

# Remove helper columns
df.drop(columns=["district_lookup"], inplace=True, errors="ignore")
df.drop(columns=["district_shp"],    inplace=True, errors="ignore")

# ─────────────────────────────────────────────────────────────────
# STEP 6 — Reorder columns so lat/lon appear right after date
# ─────────────────────────────────────────────────────────────────
# Final column order:
#   district | date | lat | lon | sp | tcc | u10 | v10 |
#   t2m | d2m | lcc | viwve | viwvn | rainfall_mm
FINAL_COLS = [
    "district", "date", "lat", "lon",
    "sp", "tcc", "u10", "v10",
    "t2m", "d2m", "lcc",
    "viwve", "viwvn",
    "rainfall_mm"
]

# Keep only columns that exist (in case your CSV has extra columns)
FINAL_COLS = [c for c in FINAL_COLS if c in df.columns]

# Also keep any extra columns not listed above
extra = [c for c in df.columns if c not in FINAL_COLS]
if extra:
    print(f"    Extra columns kept : {extra}")

df = df[FINAL_COLS + extra]

# ─────────────────────────────────────────────────────────────────
# STEP 7 — Verify and show results
# ─────────────────────────────────────────────────────────────────
print("\n[6] Verification...")

nan_lat = df["lat"].isna().sum()
nan_lon = df["lon"].isna().sum()

print(f"    Final shape        : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"    Final columns      : {list(df.columns)}")
print(f"    NaN in lat         : {nan_lat}")
print(f"    NaN in lon         : {nan_lon}")
print(f"    Lat range          : {df['lat'].min():.4f}°N → {df['lat'].max():.4f}°N")
print(f"    Lon range          : {df['lon'].min():.4f}°E → {df['lon'].max():.4f}°E")

if nan_lat > 0:
    unmatched = df[df["lat"].isna()]["district"].unique()
    print(f"\n    ⚠ {nan_lat} rows have no lat/lon — districts: {unmatched}")
    print(f"    Add these to MANUAL_FIXES and re-run.")
else:
    print(f"\n    ✓ All {df['district'].nunique()} districts have lat/lon — no NaN")

print(f"\n    Sample rows:")
print(df.head(5).to_string())

# ─────────────────────────────────────────────────────────────────
# STEP 8 — Save output CSV
# ─────────────────────────────────────────────────────────────────
print(f"\n[7] Saving output CSV...")

df.to_csv(OUTPUT_PATH, index=False)

size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
print(f"\n{'=' * 60}")
print(f"  ✓ DONE — File saved successfully")
print(f"{'=' * 60}")
print(f"  Output file : {OUTPUT_PATH}")
print(f"  File size   : {size_mb:.1f} MB")
print(f"  Rows        : {df.shape[0]:,}")
print(f"  Columns     : {df.shape[1]}  →  {list(df.columns)}")
print(f"{'=' * 60}")
print(f"""
  New columns added:
    lat  →  District centroid latitude  (degrees North)
    lon  →  District centroid longitude (degrees East)

  Source: Polygon centroids from UP_districts.shp
          Computed in UTM Zone 44N (EPSG:32644) for accuracy,
          then converted back to WGS84 (EPSG:4326)

  Next step: use {OUTPUT_PATH} to train your ML model
""")