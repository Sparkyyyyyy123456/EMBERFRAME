# EMBERFRAME
A forest fire detection AI

## Overview:
EmbereFrame is an AI that predicts the risk of a fire occurring in forest regions based on weather data. It fetches recent forest fire data from NASA FIRMS, pulls matching historical weather data from OpenMeteo, then builds a training table and trains an XGBoost Model to map the features of the data to forest fires. After the model is trained, it produces weekly and daily risk maps by pulling recent weather data.

# Setup:

- Clone the project:
  `git clone https://github.com/Sparkyyyyyy123456/EMBERFRAME`

- Define the environment to run the code:
  ## Create a virtual environment:
  `python -m venv NAME_OF_VENV`
  `source NAME_OF_VENV/Scripts/activate`
  ## Install dependencies:
  `pip install -r requirements.txt`

- Configuration:
| Parameter          | Meaning                                                                                                 |
|--------------------|----------------------------------------------------------------------------------------------------------|
| **BBOX**           | Bounding box of the region (lon‑min, lat‑min, lon‑max, lat‑max).                                         |
| **SATELLITE_SOURCES** | List of FIRMS product IDs to query, e.g. `['VIIRS_NOAA20_NRT', 'MODIS_NRT', 'MODIS_NIGHT']`.          |
| **GRID_SIZE**      | Number of cells per side; total cells = `GRID_SIZE²`.                                                    |
| **DAYS_DATA**      | Number of days of fire and weather data to pull.                                                          |
| **PRONE_THRESHOLD**| AI probability cut‑off for the weekly “prone” flag.                                                     |
| **HIGHRISK_THRESHOLD** | Daily probability cut‑off for high‑risk cells.                                                       |
| **HOT_THRESHOLD / DRY_THRESHOLD** | Domain‑expert thresholds for temperature and precipitation.                              |
| **STD_DEV_LIMIT**  | Volatility trigger for extreme sampling.                                                                 |
| **MAP_KEY**        | Free API key – obtain from <https://firms.modaps.eosdis.nasa.gov/api/map_key/>.                          |
| **Paths (`LOC`, `MODEL`, …)** | Relative paths for data, model, and other output files.                                           |


## License & Credits
- This code is open-source and completely free to use, made using FLOSS tools.
- NASA FIRMS and Open‑Meteo provide the underlying data.
- The XGBoost model and folium map rendering are courtesy of their respective open‑source communities.


## Running the code:

- Order:
  config.py < train_model.py < weekly_scan.py < daily_scan.py
- Instructions:
  1. Create a relative folder for storing session data, e.g. `LOC`.
  2. Configure config.py variables.
  3. Run train_model.py to train the XGBoost Model.
  3. Run weekly_scan.py to get a list of prone cells in the defined bounding box.
  4. Run daily_scan.py to get a risk map of prone and high-risk cells displayed on a world map.
 
- Further steps:
  - Train a new model every time a new bounding box is defined
  - Run weekly_scan.py once a week
  - Run daily_scan.py once a day

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
