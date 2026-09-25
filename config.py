
# EMBERFRAME SETTINGS

#PREREQUISITES
BBOX = (11.601563,-8.928487,29.882813,8.407168) # Input bounding box as per this format (xmin, ymin, xmax, ymax)
SATELLITE_SOURCES = ['VIIRS_NOAA20_NRT', 'MODIS_NRT', 'VIIRS_SNPP_NRT'] # Input wanted satellite sources for fire data
MAP_KEY = "7bb86ca838c20a9947edfa4d13304b8a" # Input NASA FIIRMS API key
GRID_SIZE = 3 # Specify amount of rows and columns the BBOX should be divided into
DAYS_DATA = 50 # Specify how many days of data should be used

#SENSITIVITY
PRONE_THRESHOLD = 0.2 # >PRONE_THRESHOLD = Flagged as prone
HIGHRISK_THRESHOLD = 0.5 # >HIGHRISK_THRESHOLD = Flagged as high risk

#AI PARAMETERS
HOT_THRESHOLD = 31.8 # Defines above what temperature is considered hot
DRY_THRESHOLD = 2.0 # Defines below what precipitation is considered dry
STD_DEV_LIMIT = 5.0 # Defines the limit of standard deviation for which temperature varies

#FOLDER PATHS
LOC = "LOC" #Main folder for each AI model (Has to be created beforehand)
MODEL = "LOC/model.pkl" #File name of model
CELLS = "LOC/main_data.csv" #Main data for model
WEATHER_DATA = "LOC/weatherdata.csv" #Weather data for model
PRONE = "LOC/prone.csv" #Prone cells
