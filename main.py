# Project packages
from sec_data_parser import process_ecl_with_local_sec_data, post_process_ecl, drop_sec_variables_by_null_percentage
from tools.utils import get_column_list_from_json
from tools.data_loader import DataLoader
from tools import config
import os

data_loader = DataLoader()

def main():
    print("Main function start.")

    """ Combine ECL data with financial variables from 10-K filings. """
    process_ecl_with_local_sec_data(
        input_file_path=config.ECL_FILE_PATH,
        output_file_path=config.RAW_DATASET_FILEPATH,
        sec_data_dir=config.SEC_DATA_DIR
    )

    """ Post Processing """
    # Drop years with no data
    post_process_ecl(config.RAW_DATASET_FILEPATH, config.ALL_VARS_DATASET_FILEPATH)

    # Load the dataset generated from the first step
    ecl_companyfacts_df = data_loader.load_dataset(config.ALL_VARS_DATASET_FILEPATH, alias="ecl_companyfacts", lines=True)

    # Generate statistics for all variables matched, and save as csv
    original_keys = get_column_list_from_json(config.ECL_FILE_PATH)
    column_stats_key = data_loader.analyze_numeric_columns(key="ecl_companyfacts", original_keys=original_keys, alias='column_stats')
    data_loader.save_dataset(key=column_stats_key, out_path=config.COLUMN_STATS_FILEPATH)

    # Load the statistics file
    column_stats_df = data_loader.get_dataset(column_stats_key)

    drop_sec_variables_by_null_percentage(
        columns_stats_df=column_stats_df,
        ecl_companyfacts=ecl_companyfacts_df,
        output_filename=config.POST_PROCESSED_DATASET_FILEPATH,
        max_null_percentage=10
    )

    # print("Loaded datasets:", data_loader.list_datasets())
    print("Finished.")

if __name__ == '__main__':
    main()
