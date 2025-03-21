# Python packages
import json
import time
import logging
import os
from os import path
import pandas as pd
import numpy as np
from datetime import datetime
import requests

# Our packages
from tools.utils import clean_cik, get_file_line_count, get_column_list_from_json, extract_year_from_filename
from tools import config


# Configure logging
logging.basicConfig(level=logging.INFO)


import os
import glob
import json


def match_accession_in_data(acc_number, data_dict, df, idx):
    """
    Attempt to find acc_number in data_dict['accessionNumber'].
    If found, populate df fields for the given row index and return True.
    Otherwise return False.

    data_dict is either data['filings']['recent'] (for main files)
    or data (for split files), depending on structure.
    """
    # Because the structure differs, we just assume 'accessionNumber', 'form', etc.
    # are all at the same level in data_dict.
    acc_numbers = data_dict.get('accessionNumber', [])
    for i, existing_acc in enumerate(acc_numbers):
        if existing_acc == acc_number:
            # Populate DF with the matched data
            df.at[idx, 'form'] = data_dict['form'][i]
            df.at[idx, 'primaryDocument'] = data_dict['primaryDocument'][i]
            df.at[idx, 'isXBRL'] = data_dict['isXBRL'][i]
            df.at[idx, 'reportDateIndex'] = i # store the i-th index as 'reportDateIndex' for debugging
            return True
    return False


def load_data_for_cik(cik_str):
    """
    Load the main JSON (if present) and any split JSONs (if present),
    returning a tuple: (main_data, list_of_split_data).

    main_data will be a dict with structure:
        {
            'filings': {
                'recent': { ... }
            }
        }
    or None if file not found or error reading.

    list_of_split_data will be a list of dicts, each containing top-level arrays
    for accessionNumber, form, isXBRL, etc.
    """
    main_data = None
    split_data_list = []

    main_file = f"CIK{cik_str}.json"
    main_path = os.path.join(config.SEC_SUBMISSIONS_DIR, main_file)
    if os.path.isfile(main_path):
        try:
            with open(main_path, 'r') as f:
                main_data = json.load(f)
        except Exception as e:
            logging.warning(f"[load_data_for_cik] Error reading main file {main_file}: {e}")
            main_data = None

    # Collect all the split files
    split_file_pattern = os.path.join(config.SEC_SUBMISSIONS_DIR, f"CIK{cik_str}-submissions-*")
    all_splits = glob.glob(split_file_pattern)

    for sf_path in all_splits:
        try:
            with open(sf_path, 'r') as sf:
                split_data = json.load(sf)
            split_data_list.append(split_data)
        except Exception as e:
            print(f"[load_data_for_cik] Error reading split file {sf_path}: {e}")

    return main_data, split_data_list



def add_submissions_metadata(df):
    """
    For each row in df, uses 'cik' and 'accessionNumber' to look up
    the position (index) of that accessionNumber in the JSON data,
    and stores relevant fields (form, primaryDocument, isXBRL) in df.

    If a match is not found in the main file CIKxxxxx.json,
    it checks any additional split files (CIKxxxxx-submissions-001, etc.) for a match.

    A small cache is used so that for multiple rows sharing the same CIK,
    the JSON data is only loaded once.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset, which must have columns 'cik', 'accessionNumber', etc.
    submissions_folder_path : str
        (Implied usage of config.SEC_SUBMISSIONS_DIR)
        Path to the folder containing the SEC CIK JSON files, e.g. 'CIK0000012345.json'.

    Returns
    -------
    pd.DataFrame
        The same dataframe with (potentially) added or updated columns:
        'reportDateIndex', 'form', 'primaryDocument', 'isXBRL'.
    """

    # Make sure we have the column 'reportDateIndex'
    if 'reportDateIndex' not in df.columns:
        df['reportDateIndex'] = None



    print("Retrieving metadata for all rows...")

    # Simple cache to avoid re-loading the same CIK's data repeatedly
    cached_cik = None
    main_data = None
    split_data_list = []

    for idx, row in df.iterrows():
        print(f"\rCurrent row {idx}. ", end='')
        cik_str = clean_cik(row['cik'])
        # If we are on a new CIK, load the data fresh and cache it
        if cik_str != cached_cik:
            main_data, split_data_list = load_data_for_cik(cik_str)
            cached_cik = cik_str

        acc_number = row['accessionNumber']
        match_found = False

        # Attempt to match in the main data
        if main_data is not None:
            recent_section = main_data.get('filings', {}).get('recent', {})
            match_found = match_accession_in_data(acc_number, recent_section, df, idx)

        # If not matched in main, check the split data
        if not match_found:
            for split_data in split_data_list:
                match_found = match_accession_in_data(acc_number, split_data, df, idx)
                if match_found:
                    break

        if not match_found:
            logging.warning(f"No match for {row.get('company', 'Unknown company')} "
                  f"({row.get('year', 'Unknown year')}) in main or split files for CIK {cik_str}.")

    return df