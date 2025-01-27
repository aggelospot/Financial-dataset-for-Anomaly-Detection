import os

# Get the root directory of the project by navigating up from the current file's directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # Navigate up one level

# Define other directories based on the root directory
DATA_DIR = os.path.join(BASE_DIR, 'data')              # Data directory
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')         # Output directory
SEC_DATA_DIR = os.path.join(DATA_DIR, 'companyfacts')  # SEC data directory

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# File paths
ECL_FILE_PATH = os.path.join(DATA_DIR, 'ECL_AA_subset.json')                          # ECL dataset
RAW_DATASET_FILEPATH = os.path.join(OUTPUT_DIR, 'ecl_companyfacts_raw.json')          # Combined ECL + companyfacts dataset
ALL_VARS_DATASET_FILEPATH = os.path.join(OUTPUT_DIR, 'ecl_companyfacts.json')                  # ECL + companyfacts (no nulls)
POST_PROCESSED_DATASET_FILEPATH = os.path.join(OUTPUT_DIR, 'ecl_companyfacts_processed.csv')  # Final processed dataset
COLUMN_STATS_FILEPATH = os.path.join(OUTPUT_DIR, 'column_statistics.csv')             # Column statistics file
