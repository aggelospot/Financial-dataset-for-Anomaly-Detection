# Python packages
import json
import time
import logging
import os

# Our packages
from tools.utils import clean_cik, get_file_line_count


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


    # with open(input_file_path, 'r') as input_file:
    #     print("Reading the total number of rows of input file: {input_file_path}...")
    #     total_rows = sum(1 for _ in input_file)

    # Open the error log and input files
    with open(error_file_path, 'w') as error_file:
        with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
            for line in input_file:
                try:
                    row_counter += 1
                    # if row_counter > 100: break  # Break condition for testing
                    row = json.loads(line.strip())

                    # (Old method) Get date and CIK key for this row
                    # ecl_date_str = row.get('period_of_report', '')
                    # ecl_date = parse_date(ecl_date_str)

                    # (New method) Extract the year from the 'cik_year' field
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
                            min_date_diff = 360  # Threshold in days
                            for data_point in data_points:
                                """ Note: I used the 'in' keyword instead of '==' to match columns with strings like '10-K/A'. """
                                if ('10-K' not in data_point.get('form')): continue # Only check the data_points from 10-K reports.


                                # (Old method)
                                # print("curr data point: ", data_point.get('form'), data_point.get('form') == '10-K')
                                # data_end_date_str = data_point.get('end', '')
                                # data_end_date = parse_date(data_end_date_str)

                                # New method: Only match the fiscal year.
                                fiscal_year = data_point.get('fy', '')

                                if fiscal_year:
                                    if str(cik_year) == str(fiscal_year):  # Exact year match
                                        matching_val = data_point.get('val')
                                        break  # Exact match found

                                    """ Old method -- Matches slightly more variables, but probably incorrectly. """
                                    # date_diff = abs((data_end_date - ecl_date).days)
                                    # if date_diff < min_date_diff:
                                    #     matching_val = data_point.get('val')
                                    #     min_date_diff = date_diff
                                    #     if date_diff == 0:
                                    #         break  # Exact match found


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

            # No additional error logging needed at the end

    # Clear the progress bar line after completion
    print("\rProcessing complete.                      ")
    print(f"Total rows processed: {row_counter}")
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



# def main():
#     input_file_path = './data/ECL_AA_subset.json'
#     sec_data_dir = './data/companyfacts'
#     output_dir = './outputs'
#
#
#
#     os.makedirs(output_dir, exist_ok=True)
#     output_file_path = os.path.join(output_dir, f'ecl_combined(10-K)_{int(time.time())}.json')
#
#     # Process the ECL file and write updated rows to the output file
#     process_ecl_with_local_sec_data(input_file_path, output_file_path, sec_data_dir)
#
#     # process_ecl_file(input_file_path, output_file_path, num_rows=50000)
#     # analyze_combined_file(output_file_path)
#
# if __name__ == '__main__':
#     main()
