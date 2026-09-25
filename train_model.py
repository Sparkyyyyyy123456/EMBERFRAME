# EMBERFRAME MODEL

import numpy as np
import pandas as pd
import requests
import time
import pickle
import os
from datetime import datetime, timedelta
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import config

def setup_grid():
    print(f"\n[1/5] REGION SETTINGS: BBOX={config.BBOX}, GRID_SIZE={config.GRID_SIZE}, LOOKBACK_DAYS={config.DAYS_DATA}")
    print(f"Using MAP_KEY: {config.MAP_KEY}")
    print(f"Targeting Bounding Box: {config.BBOX}")
    min_lon, min_lat, max_lon, max_lat = config.BBOX
    lons = np.linspace(min_lon, max_lon, config.GRID_SIZE + 1)
    lats = np.linspace(min_lat, max_lat, config.GRID_SIZE + 1)

    cells = []
    cell_id = 0
    for i in range(config.GRID_SIZE):
        for j in range(config.GRID_SIZE):
            cells.append({
                'cell_id': cell_id,
                'lon_min': lons[i], 'lon_max': lons[i+1],
                'lat_min': lats[j], 'lat_max': lats[j+1],
                'center_lon': (lons[i] + lons[i+1]) / 2,
                'center_lat': (lats[j] + lats[j+1]) / 2,
            })
            cell_id += 1

    print(f"Created {config.GRID_SIZE * config.GRID_SIZE} analysis cells.")
    return pd.DataFrame(cells)

def get_historical_fires(cells_df):
    print("\n[2/5] FETCHING NASA FIRMS FIRE DATA")
    print(f"Pulling fire data for the last {config.DAYS_DATA} days …")
    min_lon, min_lat, max_lon, max_lat = config.BBOX
    bbox_str = f"{min_lon},{min_lat},{max_lon},{max_lat}"
    
    all_fires_list = []
    today = datetime.now()
    last_url = ""
    chunk_days = 5
    num_chunks = config.DAYS_DATA//chunk_days
    all_fires_list = []

    for source in config.SATELLITE_SOURCES:

        for i in range(num_chunks):
            start_date = (today - timedelta(days=(i + 1) * chunk_days)).strftime('%Y-%m-%d')
            api_key = os.getenv('FIRMS_MAP_KEY', config.MAP_KEY)
            url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{api_key}/{source}/{bbox_str}/5/{start_date}"
            last_url = url

            try:
                response = requests.get(url, timeout=15)
                if response.status_code == 200:
                    from io import StringIO
                    df_chunk = pd.read_csv(StringIO(response.text))
                    if not df_chunk.empty:
                        all_fires_list.append(df_chunk)
                        print(f"OK | {url} | Found {len(df_chunk)} hotspots.")
                    else:
                        print(f"OK | {url} | No hotspots.")
                else:
                    print(f"ERROR {response.status_code} | {url}.")
            except Exception as e:
                print(f"EXCEPTION | {url} | {e}")
            time.sleep(0.4)

    fires = pd.concat(all_fires_list, ignore_index=True)
    fires = pd.concat(all_fires_list, ignore_index=True)
    fires['acq_date'] = pd.to_datetime(fires['acq_date'])
    
    def find_cell(row):
        match = cells_df[
            (cells_df.lon_min <= row.longitude) & (row.longitude < cells_df.lon_max) &
            (cells_df.lat_min <= row.latitude) & (row.latitude < cells_df.lat_max)
        ]
        return match.cell_id.values[0] if len(match) else None

    fires['cell_id'] = fires.apply(find_cell, axis=1)
    valid_fires = fires.dropna(subset=['cell_id'])
    print(f"Total valid fire events mapped to cells: {len(valid_fires)}")
    return valid_fires

def get_historical_weather(cells_df):
    print("\n[3/5] FETCHING HISTORICAL WEATHER")
    lookback = config.DAYS_DATA
    today = datetime.now()
    start_date = (today - timedelta(days=lookback)).strftime('%Y-%m-%d')
    end_date = (today - timedelta(days=1)).strftime('%Y-%m-%d')
    
    print(f"Pulling weather for {lookback} days: {start_date} to {end_date}...")
    weather_frames = []
    for idx, cell in cells_df.iterrows():
        url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={cell.center_lat}&longitude={cell.center_lon}"
            f"&start_date={start_date}&end_date={end_date}"
            f"&daily=temperature_2m_max,windspeed_10m_max,precipitation_sum"
            f"&timezone=auto"
        )
        try:
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                r = response.json()
                if 'daily' not in r:
                    continue
                df = pd.DataFrame(r['daily'])
                df['cell_id'] = cell.cell_id
                weather_frames.append(df)
                if (idx + 1) % 5 == 0:
                    print(f"Processed {idx + 1}/{len(cells_df)} cells")
            else:
                print(f"ERROR {response.status_code} | {url}")
        except Exception as e:
            print(f"EXCEPTION | {url} | {e}")
        
        time.sleep(0.2)

    weather_all = pd.concat(weather_frames)
    weather_all['time'] = pd.to_datetime(weather_all['time'])
    print(f"Downloaded {len(weather_all)} weather records matching fire data window.")
    return weather_all

def train():
    cells_df = setup_grid()
    fires = get_historical_fires(cells_df)

    if fires.empty:
        print("No fire data available – aborting training.")
        return
    weather_all = get_historical_weather(cells_df)

    fire_counts = fires.groupby(['cell_id', 'acq_date']).size().reset_index(name='fire_count')


    print("\n[4/5] BUILDING THE AI TRAINING TABLE")
    all_temps = weather_all['temperature_2m_max']
    region_hot_threshold = float(np.percentile(all_temps, 75))
    print(f"Computing region-specific HOT_THRESHOLD from data: {region_hot_threshold:.1f}°C")

    with open('config.py', 'r') as f:
        cfg = f.read()
    cfg = cfg.replace('HOT_THRESHOLD = 30.0', f'HOT_THRESHOLD = {region_hot_threshold}')
    with open('config.py', 'w') as f:
        f.write(cfg)
    print(f"Updated config.py: HOT_THRESHOLD = {region_hot_threshold:.1f}°C")

    density = fires.groupby('cell_id').size().reset_index(name='historical_fire_density')
    cells_df = cells_df.merge(density, on='cell_id', how='left').fillna(0)

    training = weather_all.merge(
        fire_counts, left_on=['cell_id', 'time'], right_on=['cell_id', 'acq_date'], how='left'
    ).fillna(0)

    training['fire_happened'] = (training['fire_count'] > 0).astype(int)
    training = training.merge(cells_df[['cell_id', 'historical_fire_density']], on='cell_id')
    print(f"Training table ready with {len(training)} samples.")

    print("\n[5/5] TRAINING THE XGBOOST MODEL")
    FEATURES = ['temperature_2m_max', 'windspeed_10m_max', 'precipitation_sum', 'historical_fire_density', 'is_hot', 'is_dry']
    training['is_hot'] = (training['temperature_2m_max'] > config.HOT_THRESHOLD).astype(int)
    training['is_dry'] = (training['precipitation_sum'] < config.DRY_THRESHOLD).astype(int)
    X = training[FEATURES]
    y = training['fire_happened']

    num_pos = y.sum()
    num_neg = len(y) - num_pos
    ratio = num_neg / num_pos if num_pos > 0 else 1
    print(f"Class imbalance ratio: {ratio:.2f}:1")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBClassifier(n_estimators=30, max_depth=4, eval_metric='logloss', scale_pos_weight=ratio)
    for epoch in range(1, 6):
        print(f"Epoch {epoch}/5")
        if epoch == 1:
            model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train, xgb_model=model.get_booster())

    print("\n" + "="*40)
    print("MODEL PERFORMANCE REPORT")
    print("="*40)
    from sklearn.metrics import classification_report
    print(classification_report(y_test, model.predict(X_test), zero_division=0))
    print("="*40)

    os.makedirs(config.LOC, exist_ok=True)
    with open(config.MODEL, 'wb') as f:
        pickle.dump(model, f)
    cells_df.to_csv(config.CELLS, index=False)
    print(f"\nSUCCESS! AI Brain saved to: {config.MODEL}")

if __name__ == "__main__":
    train()