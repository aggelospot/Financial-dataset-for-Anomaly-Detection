import requests
import json
import time
import logging

from tools.utils import clean_cik, parse_date

""" Initially, SEC data was retrieved from the API, but we instead chose to parse the zipped files instead. This file is kept for reference. """

# Define the User-Agent header as per SEC's requirements
HEADERS = {
    'User-Agent': 'MyAppName/1.0 (myemail@example.com)'
}

# Configure logging
logging.basicConfig(level=logging.INFO)


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
