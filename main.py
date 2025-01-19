import os

from sec_data_parser import process_ecl_with_local_sec_data, post_process_ecl

def main():
    print("Main function start.")
    # Step 1: Combine ECL data with financial variables from 10-K filings.
    ecl_file_path = './data/ECL_AA_subset.json'
    sec_data_dir = './data/companyfacts'
    output_dir = './outputs'
    os.makedirs(output_dir, exist_ok=True)

    output_file_path = os.path.join(output_dir, 'ecl_companyfacts_raw.json') # to add a timestamp: {int(time.time())}
    process_ecl_with_local_sec_data(ecl_file_path, output_file_path, sec_data_dir)



    # Step 2: Parse the combined dataset and remove rows with no added data. Write the results to another file.
    ecl_companyfacts_path = output_file_path
    output_file_path = os.path.join(output_dir, 'ecl_companyfacts.json')

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file_path)
    os.makedirs(output_dir, exist_ok=True)

    # Post-process the ECL dataset to filter out rows without financial variables
    post_process_ecl(ecl_companyfacts_path, output_file_path)
    print("Finished. ")


    # analyze_combined_file(output_file_path)

if __name__ == '__main__':
    main()
