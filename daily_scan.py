# EMBERFRAME HIGH RISK SCAN (To be run daily)

import pandas as pd
import pickle
import requests
import os
import folium
import config

def run_daily_scan():

    print("\n" + "="*50)
    print("DAILY HIGH-RISK SCAN")
    print("="*50)
    
    if not os.path.exists(config.MODEL) or not os.path.exists(config.PRONE):
        print("Error: Required files missing!")
        return

    try:
        with open(config.MODEL, 'rb') as f:
            model = pickle.load(f)
        prone_cells = pd.read_csv(config.PRONE)
    except Exception as e:
        print(f"Error loading files: {e}")
        return
    
    if prone_cells.empty:
        print("\n No prone cells found. Nothing to check today.")
        return

    print("\n[2/4] FETCHING WEATHER DATA")
    daily_rows = []

    for idx, cell in prone_cells.iterrows():
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={cell.center_lat}&longitude={cell.center_lon}"
            f"&daily=temperature_2m_max,windspeed_10m_max,precipitation_sum"
            f"&timezone=auto&forecast_days=1"
        )

        try:
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                r = response.json()
                if 'daily' in r:
                    daily_rows.append({
                        'cell_id': cell.cell_id,
                        'center_lat': cell.center_lat,
                        'center_lon': cell.center_lon,
                        'temperature_2m_max': r['daily']['temperature_2m_max'][0],
                        'windspeed_10m_max': r['daily']['windspeed_10m_max'][0],
                        'precipitation_sum': r['daily']['precipitation_sum'][0],
                        'historical_fire_density': cell.historical_fire_density,
                    })
            else:
                print(f"ERROR {response.status_code} | {url}")
        except Exception as e:
            print(f"EXCEPTION | {url} | {e}")
        
    if not daily_rows:
        print("No live data retrieved.")
        return

    today_df = pd.DataFrame(daily_rows)
    
    print("\n[3/4] CALCULATING RISK")
    today_df['is_hot'] = (today_df['temperature_2m_max'] > config.HOT_THRESHOLD).astype(int)
    today_df['is_dry'] = (today_df['precipitation_sum'] < config.DRY_THRESHOLD).astype(int)
    
    FEATURES = ['temperature_2m_max', 'windspeed_10m_max', 'precipitation_sum', 'historical_fire_density', 'is_hot', 'is_dry']
    probs = model.predict_proba(today_df[FEATURES])[:, 1]
    today_df['high_risk_prob'] = probs
    today_df['high_risk_flag'] = today_df['high_risk_prob'] >= config.HIGHRISK_THRESHOLD
    high_risk_cells = today_df[today_df['high_risk_flag']]
    
    print("\n[4/4] GENERATING RISK MAP")
    m = folium.Map(location=[(config.BBOX[1]+config.BBOX[3])/2, (config.BBOX[0]+config.BBOX[2])/2], zoom_start=8, tiles='OpenStreetMap')
    
    for _, row in prone_cells.iterrows():
        folium.CircleMarker([row.center_lat, row.center_lon], radius=15, color='#fa5102', fill=True, fill_colour='#fa8b02', tooltip="PRONE").add_to(m)
    
    for _, row in high_risk_cells.iterrows():
        folium.CircleMarker([row.center_lat, row.center_lon], radius=10, color='#ff0000', fill=True, fill_colour='#5c0101', tooltip=f"HIGH RISK: {row.high_risk_prob:.2f}").add_to(m)
    
    map_path = "LOC/MAP.html"
    m.save(map_path)
    
    print("="*50)
    print(f"DAILY SCAN COMPLETE")
    print(f"{len(high_risk_cells)} cells flagged as HIGH RISK today")
    print(f"Map saved to: {map_path}")
    print("="*50)

if __name__ == "__main__":
    run_daily_scan()
