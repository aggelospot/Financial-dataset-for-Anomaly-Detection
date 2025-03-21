import requests
import logging
import pandas as pd
from datetime import datetime
import logging
import re
import os
import json
from fuzzywuzzy import fuzz
from tools import config

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

def extract_year_from_cik_year(cik_year):
    cik_year = cik_year.split('__')[-1]
    return cik_year

def extract_year_from_filename(filename):
    match = re.search(r'-(\d{2})-', filename)

    # if match:
    #     print("returning ", match.group(1))
    return int("20" + match.group(1)) if match else None

def extract_accession_number_regex(filename):
    match = re.search(r'(\d{10}-\d{2}-\d{6})', filename)
    return match.group(1) if match else None

def extract_accession_number_index(filename: str) -> str:
    return filename[-25:-5]  # Extract last 20 characters before ".json"

def parse_date(date_str):
    """
    Parses a date string in 'YYYY-MM-DD' format and returns a datetime object.
    """
    try:
        return datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        return None


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

    # print("Column Names:", column_names)
    return column_names


def get_sec_companyfacts(cik_value, failed_requests_counter):
    """
    Retrieves data from the SEC API for a given CIK value.
    Increments failed_requests_counter if the request fails.
    No longer used, replaced by reading local json files instead.
    """
    logging.basicConfig(level=logging.INFO)
    HEADERS = {
        'User-Agent': 'MyAppName/1.0 (myemail@example.com)'
    }

    sec_api_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_value}.json"
    try:
        response = requests.get(sec_api_url, headers=HEADERS)
        if response.status_code == 200:
            return response.json(), failed_requests_counter
        else:
            # logging.warning(f"\nFailed to retrieve data for CIK {cik_value}: HTTP {response.status_code}")
            failed_requests_counter += 1
            return None, failed_requests_counter
    except requests.RequestException as e:
        logging.error(f"Request exception for CIK {cik_value}: {e}")
        failed_requests_counter += 1
        return None, failed_requests_counter


def match_concept_in_section(target_concepts, concept_list):
    """
    Matches a list of taxonomies with their possible synonyms, as defined in the 'xbrl_mapping.json' file.
    Returns a json of the matched concepts (does not include unmatched concepts)
    """
    from fuzzywuzzy import fuzz

    row = {}
    for concept in concept_list:
        concept_lower = concept.lower()

        # For each xbrl tag, see if there's a fuzzy match
        for col_name, taxonomy_object in target_concepts.items():
            best_score = 0
            # print(f"Current taxonomy: {col_name}")

            # Evaluate this concept against each synonym
            for synonym in taxonomy_object.get("concepts"):
                # print(f"matching synonym: {synonym}")

                score = fuzz.token_set_ratio(synonym.lower(), concept_lower) # Use partial_ratio to allow substring matches
                if score > best_score:
                    best_score = score

            if best_score >= taxonomy_object.get("match_threshold"):
                if col_name in row:
                    # print("row col ", row[col_name])
                    prev_score = fuzz.token_set_ratio(row[col_name].lower(), concept_lower)
                    if best_score > prev_score:
                        # print(f"REPLACED {row[col_name]} with {concept}, prevScore: {prev_score}, new score: {best_score}")
                        row[col_name] = concept
                else:
                    # print(f"Matched {col_name} with {concept}. Score: {best_score}")
                    row[col_name] = concept
            else:
                best_score = 0
                for synonym in taxonomy_object.get("concepts"):
                    score = fuzz.partial_ratio(synonym.lower(),concept_lower)  # Use partial_ratio to allow substring matches
                    if score > best_score:
                        best_score = score
                if best_score >= taxonomy_object.get("match_threshold"):
                    if col_name in row:
                        # print("row col ", row[col_name])
                        prev_score = fuzz.partial_ratio(row[col_name].lower(), concept_lower)
                        if best_score > prev_score:
                            row[col_name] = concept
                    else:
                        row[col_name] = concept
    return row



def analyze_missing_columns_by_cik(ecl_companyfacts_df, cik, years, columns_to_check):
    """
    Analyze missing columns for a given CIK and a list of years, and find related non-NaN columns.

    Args:
        ecl_companyfacts_df (pd.DataFrame): The DataFrame containing the data.
        cik (int): The CIK value to filter the data.
        years (list): A list of years to filter the data.
        columns_to_check (list): List of column names to check for missing values.

    Returns:
        dict: A dictionary with years as keys, mapping missing columns to related non-NaN columns and their values.
    """
    if (ecl_companyfacts_df[ecl_companyfacts_df['cik'] == cik].empty):
        print("CIK not found.")
        return

    print(
        f"Analyzing company \"{ecl_companyfacts_df[ecl_companyfacts_df['cik'] == cik]['company'].iloc[0]}\", with CIK = {cik}.")
    results = {}

    for year in years:
        print(f"--- Analyzing Year: {year} ---")

        # Filter the DataFrame for the specific cik and year
        filtered_row = ecl_companyfacts_df[
            (ecl_companyfacts_df['cik'] == cik) & (ecl_companyfacts_df['cik_year'] == year)
            ]
        if filtered_row.empty:
            print("No variables were matched for this year.")
            continue

        # Check which columns in columns_to_check are NaN in the filtered row
        missing_columns = [
            col for col in columns_to_check if pd.isna(filtered_row[col]).all()
        ]

        related_columns_dict = {}

        if missing_columns == []:
            print("All variables are present.")
            continue

        # print("Missing column list for this year: ", missing_columns)

        for missing_col in missing_columns:

            # Find columns with names similar to the missing column
            similar_columns = [
                col for col in ecl_companyfacts_df.columns if missing_col in col
            ]

            # Filter the similar columns to only include those with non-NaN values in the specific row
            non_nan_columns = [
                col for col in similar_columns if not pd.isna(filtered_row[col]).all()
            ]

            # Store the non-NaN columns and their values in the dictionary
            if non_nan_columns:
                related_columns_dict[missing_col] = {
                    "related_columns": non_nan_columns,
                    "values": filtered_row[non_nan_columns].to_dict(orient="records")[0],
                }
            else:
                print(f'No related variables found for missing column \"{missing_col}\".')

        # Store the results for this year
        results[year] = related_columns_dict

        # Print the results for this year
        for missing_col, details in related_columns_dict.items():
            print(f"Missing Column: {missing_col}")
            print("Related Columns:", details["related_columns"])
            print("Values in Related Columns:")
            print(details["values"])
            print("\n")

    return results














