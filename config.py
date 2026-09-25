
# EMBERFRAME SETTINGS

#PREREQUISITES
BBOX = (00, 00, 00, 00) # Input bounding box as per this format (xmin, ymin, xmax, ymax)
SATELLITE_SOURCES = ['Source 1', 'Source 2', 'Source 3', '....'] # Input wanted satellite sources for fire data
MAP_KEY = "ENTER API KEY" # Input NASA FIIRMS API key
GRID_SIZE = 10 # Specify amount of rows and columns the BBOX should be divided into
DAYS_DATA = 180 # Specify how many days of data should be used

#SENSITIVITY
PRONE_THRESHOLD = 0.2 # >PRONE_THRESHOLD = Flagged as prone
HIGHRISK_THRESHOLD = 0.5 # >HIGHRISK_THRESHOLD = Flagged as high risk

#AI PARAMETERS
HOT_THRESHOLD = 30 # Defines above what temperature is considered hot
DRY_THRESHOLD = 2.0 # Defines below what precipitation is considered dry
STD_DEV_LIMIT = 5.0 # Defines the limit of standard deviation for which temperature varies

#FOLDER PATHS
LOC = "LOC" #Main folder for each AI model (Has to be created beforehand)
MODEL = "LOC/model.pkl" #File name of model
CELLS = "LOC/main_data.csv" #Main data for model
WEATHER_DATA = "LOC/weatherdata.csv" #Weather data for model
PRONE = "LOC/prone.csv" #Prone cells
