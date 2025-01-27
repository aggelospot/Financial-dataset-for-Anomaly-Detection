# Python packages
import json
import time
import logging
import os
import pandas as pd
import numpy as np


# Our packages
from tools.utils import clean_cik, get_file_line_count, get_column_list_from_json
from tools import config


# Configure logging
logging.basicConfig(level=logging.INFO)


def process_ecl_with_local_sec_data(input_file_path, output_file_path, sec_data_dir):
    """
    Processes the ECL file, retrieves SEC data from local JSON files for each row,
    and adds 'us-gaap' and 'dei' variables to each row based on the matching date.
    Writes each updated row to the output file immediately.
    """
    row_counter = 0
    previous_cik = None
    sec_data = None

    # Ensure the error directory exists
    error_dir = './outputs/errors'
    os.makedirs(error_dir, exist_ok=True)
    error_file_path = os.path.join(error_dir, 'errors_log.txt')

    # Get the total number of rows in the input file to determine how many rows are remaining to be parsed.
    total_rows = get_file_line_count(input_file_path)

    # Open the error log and input files
    with open(error_file_path, 'w') as error_file:
        with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
            for line in input_file:
                try:
                    row_counter += 1
                    # if row_counter > 100: break  # Break condition for testing
                    row = json.loads(line.strip())

                    # Extract the year from the 'cik_year' field
                    cik_year_str = row.get('cik_year', '')
                    cik_year = cik_year_str.split('__')[-1]  # Extract the year part (e.g., "2015")

                    if cik_year is None:
                        logging.warning(f"Invalid date format in row {row_counter}: {cik_year_str}")
                        continue

                    current_cik = row.get('cik', '')
                    cik_cleaned = clean_cik(current_cik)

                    # Display the current CIK being processed
                    print(f"\rProcessing CIK: {cik_cleaned} | Row: {row_counter} / {total_rows}", end='')

                    if previous_cik != cik_cleaned:
                        # Load the SEC data from local JSON file
                        sec_file_name = f"CIK{cik_cleaned}.json"
                        sec_file_path = os.path.join(sec_data_dir, sec_file_name)
                        if os.path.exists(sec_file_path):
                            with open(sec_file_path, 'r') as sec_file:
                                sec_data = json.load(sec_file)
                                error_file.write(f"CIK{cik_cleaned}: \n")
                        else:
                            error_file.write(f"CIK{cik_cleaned}:\n  - SEC data file not found. \n")
                            sec_data = None

                    previous_cik = cik_cleaned

                    num_variables_matched = 0  # Initialize counter for variables matched

                    if sec_data:
                        # Extract variables under 'us-gaap' and 'dei' keys in 'facts'
                        facts_data = sec_data.get('facts', {})
                        us_gaap_data = facts_data.get('us-gaap', {})
                        dei_data = facts_data.get('dei', {})

                        # Combine 'us-gaap' and 'dei' data
                        all_facts = {**us_gaap_data, **dei_data}

                        total_variables = len(all_facts)

                        # For each variable, find the value matching the ecl_date
                        for key, value in all_facts.items():
                            units = value.get('units', {})
                            data_points = []
                            for unit_key, unit_values in units.items():
                                data_points.extend(unit_values)
                            # Now data_points is a list of dictionaries
                            # Find the data point that matches the ecl_date or closest date within threshold
                            matching_val = None
                            for data_point in data_points:
                                """ Note: I used the 'in' keyword instead of '==' to match columns with strings like '10-K/A'. """
                                if ('10-K' not in data_point.get('form')): continue # Only check the data_points from 10-K reports.

                                # New method: Only match the fiscal year.
                                fiscal_year = data_point.get('fy', '')

                                if fiscal_year:
                                    if str(cik_year) == str(fiscal_year):  # Exact year match
                                        matching_val = data_point.get('val')
                                        break  # Exact match found


                            if matching_val is not None:
                                # Add the variable and its value to the row
                                row[key] = matching_val
                                num_variables_matched += 1

                        # If no variables were matched, log the period of report
                        if num_variables_matched == 0:
                            error_file.write(
                                f"  - Period '{cik_year}': No data found.\n")
                        #     f"CIK {cik_cleaned} - Period '{ecl_date_str}': No data found.\n")
                        else:
                            percent_matched = (num_variables_matched / total_variables) * 100 if total_variables > 0 else 0
                            error_file.write(f"  - Period '{cik_year}': {num_variables_matched}/{total_variables} variables matched ({percent_matched:.2f}%)\n")
                    else:
                        # SEC data not found for this CIK
                        pass

                    # Write the updated row to the output file
                    json.dump(row, output_file)
                    output_file.write('\n')

                except json.JSONDecodeError:
                    logging.error(f"Error decoding JSON on row {row_counter}")
                    continue

    # Clear the progress bar line after completion
    print("\rProcessing complete.                      ")
    print(f"\nTotal rows processed: {row_counter}")
    print(f"Error log written to: {error_file_path}")


def post_process_ecl(input_file_path, output_file_path):
    """
    Reads the combined ECL dataset and filters out rows that do not contain any financial variables.
    Writes the filtered dataset to a new file.

    Parameters:
    - input_file_path: Path to the combined ECL dataset file (ecl_combined.json).
    - output_file_path: Path to the output file for the filtered dataset.
    """
    # Define the original ECL keys to identify added financial variables
    original_keys = set([
        'bankruptcy_date_1', 'label', 'bankruptcy_date_2', 'filing_date', 'datadate',
        'bankruptcy_date_3', 'opinion_text', 'item_7', 'bankruptcy_prediction_split',
        'cik', 'company', 'period_of_report', 'cik_year', 'qualified', 'gc_list',
        'can_label', 'filename', 'gvkey'
    ])

    current_row = 0
    rows_with_financial_vars = 0
    total_rows = get_file_line_count(input_file_path)

    with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
        for line in input_file:
            try:
                print(f"\rProcessing file... Progress: {current_row/total_rows:.0%}", end='')
                row = json.loads(line.strip())
                current_keys = set(row.keys())

                # Identify financial variables by subtracting original keys from all keys
                financial_vars = current_keys - original_keys

                if financial_vars:
                    # Row contains financial variables, write it to the output file
                    json.dump(row, output_file)
                    output_file.write('\n')
                    rows_with_financial_vars += 1
                else:
                    # Row does not contain financial variables, skip it
                    pass

                current_row += 1

            except json.JSONDecodeError:
                logging.error(f"Error decoding JSON on line {total_rows + 1}")
                continue

    print(f"Total rows processed: {total_rows}")
    print(f"Rows with financial variables: {rows_with_financial_vars}")
    print(f"Rows without financial variables (omitted): {total_rows - rows_with_financial_vars}")


def drop_sec_variables_by_null_percentage(columns_stats_df, ecl_companyfacts, output_filename=config.POST_PROCESSED_DATASET_FILEPATH, max_null_percentage=22.0):
    """
    Reads the column statistics file, and drops all SEC variables that don't meet a certain max_null_percentage threshold.
    Writes the resulting DF to a file.

    :param columns_stats_df: The 'column_statistics' dataframe
    :param ecl_companyfacts: The ECL dataset with added SEC variables
    :param max_null_percentage: The threshold.
    """

    selected_columns = columns_stats_df[columns_stats_df['percentage_nulls'] <= max_null_percentage]['column'].tolist()
    print("Selected columns: ", selected_columns)
    numeric_columns = ecl_companyfacts[selected_columns].select_dtypes(include=[np.number]).columns.tolist()
    print("Numeric columns: ", numeric_columns)


    # Combine numeric columns with original columns
    original_columns = ecl_companyfacts[get_column_list_from_json(config.ECL_FILE_PATH)]
    combined_dataset = pd.concat([original_columns, ecl_companyfacts[numeric_columns]], axis=1)

    # Drop rows where any of the selected numeric columns are null
    print("Shape before dropping NAs: ", combined_dataset.shape)
    cleaned_combined_dataset = combined_dataset.dropna(subset=numeric_columns)
    print("Shape after dropping NAs: ", cleaned_combined_dataset.shape)

    # Write the cleaned dataset to a CSV file
    cleaned_combined_dataset.to_csv(output_filename, index=False)

    print(f"Post processed dataset written to: {output_filename}")

