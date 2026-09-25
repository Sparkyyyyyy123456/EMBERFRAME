# EMBERFRAME PRONE (To be run weekly)

import pandas as pd
import pickle
import requests
import time
import os
import config

def run_weekly_scan():
    print("\n" + "="*50)
    print("WEEKLY PRONE-AREA SCAN (VOLATILITY MODE)")
    print("="*50)
    
    # 1. Load assets
    if not os.path.exists(config.MODEL) or not os.path.exists(config.CELLS):
        print("Error: Run train_model.py first.")
        return

    try:
        with open(config.MODEL, 'rb') as f:
            model = pickle.load(f)
        cells_df = pd.read_csv(config.CELLS)
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    print("\n[2/4] ANALYZING SEASONAL WEATHER")
    weekly_rows = []
    for idx, cell in cells_df.iterrows():
        url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={cell.center_lat}&longitude={cell.center_lon}"
            f"&start_date=2024-01-01&end_date=2024-12-31"
            f"&daily=temperature_2m_max,windspeed_10m_max,precipitation_sum"
            f"&timezone=auto"
        )
        try:
            response = requests.get(url, timeout=15)
            if response.status_code != 200: continue
            r = response.json()
            if 'daily' not in r: continue
            df = pd.DataFrame(r['daily'])

            temp_std = df['temperature_2m_max'].std()
            
            if temp_std >= config.STD_DEV_LIMIT:
                hottest_20 = df.nlargest(20, 'temperature_2m_max')
                avg_temp = hottest_20['temperature_2m_max'].mean()
                avg_wind = hottest_20['windspeed_10m_max'].mean()
                
                driest_20 = df.nsmallest(20, 'precipitation_sum')
                avg_precip = driest_20['precipitation_sum'].mean()
                
                mode = "EXTREME"
            else:
                avg_temp = df['temperature_2m_max'].mean()
                avg_wind = df['windspeed_10m_max'].mean()
                avg_precip = df['precipitation_sum'].mean()
                mode = "AVERAGE"

            weekly_rows.append({
                'cell_id': cell.cell_id,
                'center_lat': cell.center_lat,
                'center_lon': cell.center_lon,
                'temperature_2m_max': avg_temp,
                'windspeed_10m_max': avg_wind,
                'precipitation_sum': avg_precip,
                'historical_fire_density': cell.historical_fire_density,
                'sampling_mode': mode,
                'temp_std': temp_std
            })
            if (idx + 1) % 5 == 0:
                print(f"Processed {idx + 1}/{len(cells_df)} cells")
        except Exception as e:
            print(f"Error for cell {cell.cell_id}: {e}")
        
        time.sleep(0.2)

    if not weekly_rows:
        print("CRITICAL ERROR: No weather data retrieved.")
        return

    weekly_df = pd.DataFrame(weekly_rows)
    weekly_df.to_csv(config.WEATHER_DATA, index=False)
    print(f"\nWeather means saved to: {config.WEATHER_DATA}")

    print("\n[3/4] CALCULATING PRONE PROBABILITIES")
    weekly_df['is_hot'] = (weekly_df['temperature_2m_max'] > config.HOT_THRESHOLD).astype(int)
    weekly_df['is_dry'] = (weekly_df['precipitation_sum'] < config.DRY_THRESHOLD).astype(int)
    
    FEATURES = ['temperature_2m_max', 'windspeed_10m_max', 'precipitation_sum', 'historical_fire_density', 'is_hot', 'is_dry']
    probs = model.predict_proba(weekly_df[FEATURES])[:, 1]
    weekly_df['prone_prob'] = probs

    prone_flag = (weekly_df['prone_prob'] >= config.PRONE_THRESHOLD)
    weekly_df['prone_flag'] = prone_flag

    print("\n" + "-"*60)
    print(f"{'Cell':<8} | {'Prob':<8} | {'Density':<10} | {'StdDev':<8} | {'Mode':<10} | {'Status'}")
    print("-" * 60)
    top_5 = weekly_df.sort_values(by='prone_prob', ascending=False).head(5)
    for i, row in top_5.iterrows():
        status = "PRONE" if row.prone_flag else "NOT PRONE"
        print(f"{int(row.cell_id):<8} | {row.prone_prob:<8.4f} | {int(row.historical_fire_density):<10} | {row.temp_std:<8.2f} | {row.sampling_mode:<10} | {status}")
    print("-" * 60)
    
    prone_cells = weekly_df[weekly_df['prone_flag']].copy()

    prone_cells.to_csv(config.PRONE, index=False)
    
    print("="*50)
    print(f"WEEKLY SCAN COMPLETE")
    print(f"{len(prone_cells)} out of {len(cells_df)} cells flagged as PRONE.")
    print("="*50)

if __name__ == "__main__":
    run_weekly_scan()