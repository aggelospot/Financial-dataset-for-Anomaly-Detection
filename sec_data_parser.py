import json
import requests
import time
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)

# Define the User-Agent header as per SEC's requirements
HEADERS = {
    'User-Agent': 'MyAppName/1.0 (myemail@example.com)'
}

def clean_cik(cik_value):
    """
    Cleans and formats the CIK value to a 10-digit string with leading zeros.
    """
    cik_value = str(cik_value).split('.')[0]  # Remove decimal point if present
    cik_value = cik_value.strip()
    cik_value = cik_value.zfill(10)  # Pad with leading zeros to make it 10 digits
    return cik_value

def get_sec_data(cik_value, failed_requests_counter):
    """
    Retrieves data from the SEC API for a given CIK value.
    Increments failed_requests_counter if the request fails.
    No longer used, replaced by reading local json files instead.
    """
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

def parse_date(date_str):
    """
    Parses a date string in 'YYYY-MM-DD' format and returns a datetime object.
    """
    try:
        return datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        return None


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

    # Open the error log file
    with open(error_file_path, 'w') as error_file:
        with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
            for line in input_file:
                try:
                    row_counter += 1
                    # if row_counter > 100: break  # Break condition for testing
                    row = json.loads(line.strip())

                    # Get date and CIK key for this row
                    ecl_date_str = row.get('period_of_report', '')
                    ecl_date = parse_date(ecl_date_str)
                    if ecl_date is None:
                        logging.warning(f"Invalid date format in row {row_counter}: {ecl_date_str}")
                        continue

                    current_cik = row.get('cik', '')
                    cik_cleaned = clean_cik(current_cik)

                    # Display the current CIK being processed
                    print(f"\rProcessing CIK: {cik_cleaned} | Row: {row_counter}", end='')

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
                                if (data_point.get('form') != '10-K'): continue # Only check the data_points from 10-K reports.
                                # print("curr data point: ", data_point.get('form'), data_point.get('form') == '10-K')

                                data_end_date_str = data_point.get('end', '')
                                data_end_date = parse_date(data_end_date_str)
                                if data_end_date:
                                    date_diff = abs((data_end_date - ecl_date).days)
                                    if date_diff < min_date_diff:
                                        matching_val = data_point.get('val')
                                        min_date_diff = date_diff
                                        if date_diff == 0:
                                            break  # Exact match found


                            if matching_val is not None:
                                # Add the variable and its value to the row
                                row[key] = matching_val
                                num_variables_matched += 1

                        # If no variables were matched, log the period of report
                        if num_variables_matched == 0:
                            error_file.write(
                                f"  - Period '{ecl_date_str}': No data found.\n")
                        #     f"CIK {cik_cleaned} - Period '{ecl_date_str}': No data found.\n")
                        else:
                            # Optionally, log the number of variables matched
                            percent_matched = (num_variables_matched / total_variables) * 100 if total_variables > 0 else 0
                            # Uncomment the next line to log periods with matches
                            error_file.write(f"  - Period '{ecl_date_str}': {num_variables_matched}/{total_variables} variables matched ({percent_matched:.2f}%)\n")
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




def process_ecl_file(input_file_path, output_file_path, num_rows=2):
    """
    Processes the ECL file, retrieves SEC data for each row,
    and adds 'us-gaap' variables to each row based on the matching date.
    Writes each updated row to the output file immediately.
    """
    failed_requests_counter = 0
    row_counter = 0
    previous_cik = None
    sec_data = None

    with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
        for line in input_file:
            if row_counter >= num_rows:
                break
            try:
                row = json.loads(line.strip())

                # Get date and CIK key for this row
                ecl_date = row.get('period_of_report', '')
                current_cik = row.get('cik', '')

                if previous_cik != current_cik:
                    cik_cleaned = clean_cik(current_cik)
                    # Display the current CIK being processed
                    print(f"\rProcessing CIK: {cik_cleaned}. Number of failed requests: {failed_requests_counter}", end='')
                    sec_data, failed_requests_counter = get_sec_data(cik_cleaned, failed_requests_counter)
                    # Wait before the next request to comply with SEC's rate limiting
                    time.sleep(0.1)  # Sleep for 0.1 second to improve efficiency while being polite

                previous_cik = current_cik

                if sec_data:
                    # Extract all variables under the 'us-gaap' key in 'facts'
                    us_gaap_data = sec_data.get('facts', {}).get('us-gaap', {})

                    # Parse the ecl_date
                    ecl_date_dt = parse_date(ecl_date)
                    ecl_year = ecl_date_dt.year if ecl_date_dt else None

                    # For each variable, find the value matching the ecl_date
                    for key, value in us_gaap_data.items():
                        units = value.get('units', {})
                        data_points = []
                        for unit_key, unit_values in units.items():
                            data_points.extend(unit_values)
                        # Now data_points is a list of dictionaries
                        # Find the data point that matches ecl_year
                        matching_val = None
                        for data_point in data_points:
                            data_end_date = data_point.get('end', '')
                            data_end_dt = parse_date(data_end_date)
                            if data_end_dt and ecl_date_dt and data_end_dt.year == ecl_date_dt.year:
                                matching_val = data_point.get('val')
                                break  # Stop after finding the first match
                        if matching_val is not None:
                            # Add the variable and its value to the row
                            row[key] = matching_val

                # Write the updated row to the output file
                json.dump(row, output_file)
                output_file.write('\n')

                row_counter += 1

            except json.JSONDecodeError:
                logging.error(f"Error decoding JSON on row {row_counter + 1}")
                continue

    # Clear the progress bar line after completion
    print("\rProcessing complete.                      ")
    print(f"Total rows processed: {row_counter}")
    print(f"Total failed SEC data requests: {failed_requests_counter}")

def analyze_combined_file(output_file_path):
    """
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

def main():
    input_file_path = './data/ECL_AA_subset.json'
    sec_data_dir = './data/companyfacts'
    output_dir = './outputs'



    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, f'ecl_combined(10-K)_{int(time.time())}.json')

    # Process the ECL file and write updated rows to the output file
    process_ecl_with_local_sec_data(input_file_path, output_file_path, sec_data_dir)

    # process_ecl_file(input_file_path, output_file_path, num_rows=50000)
    # analyze_combined_file(output_file_path)

if __name__ == '__main__':
    main()
