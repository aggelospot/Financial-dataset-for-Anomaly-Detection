import json
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

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

    total_rows = 0
    rows_with_financial_vars = 0

    with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
        for line in input_file:
            try:
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

                total_rows += 1

            except json.JSONDecodeError:
                logging.error(f"Error decoding JSON on line {total_rows + 1}")
                continue

    print(f"Total rows processed: {total_rows}")
    print(f"Rows with financial variables: {rows_with_financial_vars}")
    print(f"Rows without financial variables (omitted): {total_rows - rows_with_financial_vars}")
    #
    # Total rows processed: 50000
    # Rows with financial variables: 27863
    # Rows without financial variables (omitted): 22137




def main():
    # Set the paths to your input and output files
    input_file_path = './outputs/ecl_combined_1733174753.json'  # Path to your combined ECL dataset
    output_file_path = './outputs/ecl_filtered.json'  # Path for the filtered dataset

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file_path)
    os.makedirs(output_dir, exist_ok=True)

    # Post-process the ECL dataset to filter out rows without financial variables
    post_process_ecl(input_file_path, output_file_path)

if __name__ == '__main__':
    main()
