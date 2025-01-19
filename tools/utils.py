import json
import pandas as pd
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)


""" This file contains various shared functions for the project """
def clean_cik(cik_value):
    """
    Cleans and formats the CIK value to a 10-digit string with leading zeros.
    """
    cik_value = str(cik_value).split('.')[0]  # Remove decimal point if present
    cik_value = cik_value.strip()
    cik_value = cik_value.zfill(10)  # Pad with leading zeros to make it 10 digits
    return cik_value


def parse_date(date_str):
    """
    Parses a date string in 'YYYY-MM-DD' format and returns a datetime object.
    """
    try:
        return datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        return None


def parse_ECL_json(file_path):
    # To store unique keys across all rows
    key_set = set()
    first_row_key_count = None
    row_counter = 0
    inconsistent_rows = 0

    with open(file_path, 'r') as f:
        for line in f:
            try:
                row = json.loads(line.strip())
                print("currect cik: ", row["cik"])
                # Convert each line to a dictionary
                json_obj = json.loads(line.strip())

                # Extract keys from the current row
                current_keys = set(json_obj.keys())

                # Add current keys to the global key set
                key_set.update(current_keys)

                # Check consistency of the number of keys
                if first_row_key_count is None:
                    first_row_key_count = len(current_keys)
                elif len(current_keys) != first_row_key_count:
                    inconsistent_rows += 1

                row_counter += 1

            except json.JSONDecodeError:
                print(f"Error decoding JSON on row {row_counter + 1}")
                continue

    # Convert set to list and print
    key_list = list(key_set)
    print(f"Keys found in the JSON: {key_list}")
    print(f"Total rows processed: {row_counter}")
    print(f"Inconsistent rows (different key counts): {inconsistent_rows}")

    return key_list, row_counter, inconsistent_rows

def get_file_line_count(file_path):
    """
    Get the total number of rows in the input file to determine how many rows are remaining to be parsed.
    :param file_path: str
    :return: total_rows: str
    """
    total_rows = 0
    with open(file_path, 'r') as input_file:
        print(f"Reading the total number of rows of input file: {file_path}...")
        return sum(1 for _ in input_file)


def get_column_list_from_json(file_path):
    """
    Returns the column names of the first row of a json file.
    Used to retrieve the original column list of ECL without loading the whole file.
    """
    # Use pandas to read only a small chunk of the file
    df = pd.read_json(file_path, lines=True, chunksize=1)

    # Get the column names from the first chunk
    for chunk in df:
        column_names = chunk.columns.tolist()
        break

    print("Column Names:", column_names)
    return column_names



def analyze_combined_file(output_file_path):
    """
    *** DEPRECATED ***
    Analyzes the combined ECL dataset file.
    - Displays the total number of unique CIK numbers.
    - Checks if any row has different columns from the previous one.
    - Provides additional statistics.
    """
    unique_cik_set = set()
    previous_keys = None
    rows_with_different_columns = 0
    total_rows = 0
    total_columns = 0

    with open(output_file_path, 'r') as f:
        for line in f:
            try:
                row = json.loads(line.strip())
                total_rows += 1

                # Get the 'cik' field
                cik = row.get('cik', '')
                unique_cik_set.add(cik)

                # Get current row's keys
                current_keys = set(row.keys())
                total_columns += len(current_keys)

                if previous_keys is not None and current_keys != previous_keys:
                    rows_with_different_columns += 1

                previous_keys = current_keys

            except json.JSONDecodeError:
                logging.error(f"Error decoding JSON on line {total_rows}")
                continue

    print(f"Total unique CIK numbers: {len(unique_cik_set)}")
    print(f"Rows with different columns from previous row: {rows_with_different_columns}")
    print(f"Total rows analyzed: {total_rows}")
    average_columns_per_row = total_columns / total_rows if total_rows > 0 else 0
    print(f"Average number of columns per row: {average_columns_per_row:.2f}")

    rows_without_sec_data = 0
    # Define the original ECL keys to identify added SEC data keys
    original_keys = [
        'bankruptcy_date_1', 'label', 'bankruptcy_date_2', 'filing_date', 'datadate',
        'bankruptcy_date_3', 'opinion_text', 'item_7', 'bankruptcy_prediction_split',
        'cik', 'company', 'period_of_report', 'cik_year', 'qualified', 'gc_list',
        'can_label', 'filename', 'gvkey'
    ]

    with open(output_file_path, 'r') as f:
        for line in f:
            try:
                row = json.loads(line.strip())
                # Identify SEC data keys by subtracting original keys from all keys
                sec_keys = set(row.keys()) - set(original_keys)
                if not sec_keys:
                    rows_without_sec_data += 1
            except json.JSONDecodeError:
                continue

    print(f"Rows without any added SEC data: {rows_without_sec_data}")



# file_path = './data/ECL_AA_subset.json'
# parse_ECL_json(file_path)