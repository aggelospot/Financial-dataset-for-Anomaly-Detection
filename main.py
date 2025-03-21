# Project packages
import datetime

import numpy as np
import pandas as pd
import os

from sec_data_parser import process_ecl_with_local_sec_data,\
    post_process_ecl, drop_sec_variables_by_null_percentage, add_accession_and_form,\
    fetch_calculation_linkbases, process_ecl_with_local_sec_data_new,\
    find_custom_revenue_concepts, retrieve_sec_tags_and_values
from tools.utils import get_column_list_from_json, \
     extract_year_from_filename, extract_accession_number_index
import submissions_parser
from tools.data_loader import DataLoader
from tools import config

data_loader = DataLoader()
debug = True
preprocess = True

def main():
    print("Main function start.")
    pd.options.display.max_colwidth = 1500
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    # out_path = os.path.join(config.OUTPUT_DIR, 'financials_ic_tags.csv')
    # df = data_loader.load_dataset(out_path)
    # print(df.head(100))
    # return

    # out_path = os.path.join(config.OUTPUT_DIR, 'ecl_with_matched_concepts.csv')
    # ecl = data_loader.load_dataset(out_path, alias='ecl')#, chunksize=1055)
    # excluded_cols = ["cik","company","period_of_report","gvkey","datadate","filename","label","filing_date","cik_year","form","accessionNumber","primaryDocument","calc_linkbase"]
    # df_filtered = ecl.drop(columns=excluded_cols)
    # null_counts = df_filtered.isnull().sum()
    # column_to_check = "earnings_per_share_basic"
    # # 3. Calculate unique value counts and percentages per column
    # total_rows = len(df_filtered)
    # unique_value_stats = {
    #     col: df_filtered[col].value_counts(dropna=False).to_frame(name="Count").assign(
    #         Percentage=lambda x: (x["Count"] / total_rows) * 100
    #     )
    #     for col in df_filtered.columns
    # }
    #
    # # Print results
    # print("Filtered DataFrame:")
    # print(df_filtered)
    # print("\nNull counts per column:")
    # print(null_counts)
    # print("\nUnique values with counts and percentages per column:")
    #
    # for col, stats in unique_value_stats.items():
    #     print(f"\nColumn: {col}")
    #     print(stats)
    # return

    # ecl = data_loader.load_dataset(config.ECL_METADATA_NOTEXT_PATH, alias='ecl', lines=True)#, chunksize=1055)
    # ecl = process_ecl_with_local_sec_data_test(ecl)
    # print(ecl.info)
    # print(ecl.head(10))
    # out_path = os.path.join(config.OUTPUT_DIR, 'ecl_with_matched_concepts.csv')
    #
    # data_loader.save_dataset(key_or_df=ecl, out_path=out_path)
    # return
    #
    # df_income = parse_income_statement_matches2(config.FINANCIALS_DIR)
    # print(df_income.head(25))
    # print(df_income.isnull().sum())  # counts how many missing matches we had
    # return
    #
    #
    # ecl = data_loader.load_dataset(config.ECL_METADATA_NOTEXT_PATH, alias='ecl', lines=True)#, chunksize=1055)
    # # ecl = next(ecl)
    #
    # ecl["year"] = ecl["filename"].apply(extract_year_from_filename)
    # ecl = ecl.loc[ecl["year"] >= 2013]  # Years before the threshold are dropped
    # # ecl.drop(['year'], axis=1, inplace=True)
    #
    # ic = parse_financial_section(ecl, config.FINANCIALS_DIR, 'ic')
    # return
    #
    # create_financials_dataframes(config.FINANCIALS_DIR)
    # return

    # df_income = parse_income_statement_matches2(config.FINANCIALS_DIR)
    # print(df_income.head(155))
    # print(df_income.isnull().sum())  # counts how many missing matches we had
    # return
    #
    #
    # unique_labels = get_unique_concepts_from_financials(config.FINANCIALS_DIR)
    # print("Unique labels found: \n")
    # print(unique_labels)
    # return
    # Notes :
    # https://www.sec.gov/Archives/edgar/data/46129/000155837024002487/index.json
    # https://www.sec.gov/Archives/edgar/data/1750/000110465924080890/0001104659-24-080890-index.html
    # https://www.sec.gov/Archives/edgar/data/1750/000110465924080890/air-20240531_cal.xml
    # https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data

    # submissions example
    # https://data.sec.gov/submissions/CIK0000001750.json
    # https://stackoverflow.com/questions/36087829/xbrl-dimensions-linkbase-parsing



    # Latest research:
    # http://www.xbrlsite.com/2015/Demos/ComparingReportingStyles/ComparingDifferentReportingStyles.html
    # http://www.xbrlsite.com/2015/fro/us-gaap/html/ReportFrames/ReportFrames.html
    # Ongoing thread about our topic on the xbrl forums: https://xbrl.us/forums/topic/how-to-find-a-complete-list-of-similar-concept/


    ecl = data_loader.load_dataset(config.ECL_METADATA_NOTEXT_PATH, alias='ecl', lines=True)
    retrieve_sec_tags_and_values(ecl, data_loader)
    return

    """ Refractor start """
    print("original columns", ecl.columns)

    # columns_to_keep = ['cik', 'company', 'period_of_report', 'gvkey', 'label', 'filing_date', 'year', 'accessionNumber']
    # ecl = ecl[columns_to_keep]
    ecl = ecl.drop('opinion_text', axis=1)
    ecl = ecl.drop('item_7', axis=1)
    print("after drop columns ", ecl.columns)

    original_shape = ecl.shape
    ecl["year"] = pd.to_numeric(ecl["filename"].apply(extract_year_from_filename))
    ecl["accessionNumber"] = ecl["filename"].apply(extract_accession_number_index)
    ecl = ecl.loc[ecl["year"] >= 2010]  # Years before the threshold are dropped

    print(f"Original Shape: {original_shape}. After filtering for >2010: {ecl.shape}")

    """ Adding /submissions/ metadata to each row """
    ecl_metadata = submissions_parser.add_submissions_metadata(ecl)
    ecl_metadata["accessionNumber"] = ecl_metadata['accessionNumber'].astype(str)
    data_loader.save_dataset(key_or_df=ecl_metadata, out_path=config.ECL_METADATA_NOTEXT_PATH)

    print("Done")
    return
    """ Refractor end """


    preprocess = False
    """ Basic Preprocessing on ECL for faster load times """
    if preprocess:

        ecl = data_loader.load_dataset(config.ECL_FILE_PATH, alias='ecl', lines=True) # chunksize=25
        # ecl = next(ecl)

        print("original columns", ecl.columns)

        columns_to_keep = ['cik', 'company', 'period_of_report', 'gvkey', 'label', 'filing_date', 'year', 'accessionNumber']

        original_shape = ecl.shape
        ecl["year"] = pd.to_numeric(ecl["filename"].apply(extract_year_from_filename))
        ecl["accessionNumber"] = ecl["filename"].apply(extract_accession_number_index)
        # ecl["cik"] = ecl.apply()
        # ecl["period_of_report_to_datetime"] = pd.to_datetime(ecl["period_of_report"])
        ecl = ecl.loc[ecl["year"] >= 2010] # Years before the threshold are dropped
        # ecl.drop(['period_of_report_to_datetime'], axis=1, inplace=True)
        ecl = ecl[columns_to_keep]
        print(f"Original Shape: {original_shape}. After filtering for >2010: {ecl.shape}")


        """ Adding /submissions/ metadata to each row """
        ecl_metadata = submissions_parser.add_submissions_metadata(ecl) # data_loader.save_dataset(key_or_df='ecl', out_path=config.ECL_SUBMISSIONS_METADATA_PATH) ----- ecl_metadata = data_loader.load_dataset(config.ECL_SUBMISSIONS_METADATA_PATH, alias='ecl_metadata', lines=True) #, chunksize=25)
        ecl_metadata["accessionNumber"] =  ecl_metadata['accessionNumber'].astype(str)
        # ecL_metadata = fetch_calculation_linkbases(ecl_metadata)
        data_loader.save_dataset(key_or_df=ecl_metadata, out_path=config.ECL_METADATA_NOTEXT_PATH)

    ecl = data_loader.load_dataset(config.ECL_METADATA_NOTEXT_PATH, alias='ecl', lines=True)
    # ecl = data_loader.load_dataset(config.ECL_METADATA_NOTEXT_PATH, alias='ecl', lines=True, chunksize=25)  #
    # ecl = process_ecl_with_local_sec_data_new(next(ecl))

    print(ecl.head())
    # TODO: Change extract year function to retrieve the year from filing date?? or a special case for the 20th century dates
    # 36 of our rows with bankruptcy labels are not in xbrl format
    ecl["filing_date"] = pd.to_datetime(ecl["filing_date"]).dt.date
    ecl["period_of_report"] = pd.to_datetime(ecl["period_of_report"]).dt.date

    # ecl['que'] = np.where(ecl["filing_date"] < ecl["period_of_report"], ecl["filing_date"], ecl["period_of_report"])
    # ecl['que'] = ecl.apply(lambda x: True if x['filing_date'].year == x['year'] else False, axis=1)
    #
    # print(ecl[['que', 'filing_date', 'period_of_report', 'year']].head(5))
    # print(ecl[ecl['que'] == False][['que', 'filing_date', 'period_of_report', 'year', 'cik']].value_counts())
    # return


    print(ecl[ecl["filing_date"] < datetime.date(2000,1,1)]['year'].value_counts())
    ecl = ecl[ecl['label'] == False]
    print(ecl[ecl['isXBRL'] == 0]['year'].value_counts())
    # TODO: income statement fuzzy matches


    return


    ecl = data_loader.load_dataset(config.ECL_METADATA_NOTEXT_PATH, alias='ecl', lines=True, chunksize=25)
    test_df = process_ecl_with_local_sec_data_new(next(ecl))
    print (test_df.columns)

    print("colwidth: ", pd.options.display.max_colwidth)
    print(test_df[['company', 'cik_year', 'TotalRevenues', 'OperatingExpenses', 'OperatingIncomeLoss', 'NetIncomeLoss', 'EarningsPerShareDiluted', 'calc_linkbase', 'accessionNumber']].head(25))

    """
    "TotalRevenues": [
      "TotalRevenues",
      "Revenues",
      "SalesRevenueNet",
      "SalesOrRevenue",
      "RevenueFromContractWithCustomerExcludingAssessedTax"
    ],
    
    """
    matches = find_custom_revenue_concepts(os.path.join(config.CALC_LINKBASES_DIR, 'air-20190531_cal.xml'), 'us-gaap_RevenueFromContractWithCustomerExcludingAssessedTax')
    print("Matches: ", matches)



    if not debug:

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
    print("Main function end.")

if __name__ == '__main__':
    main()

"""
Variables: 

1. Income Statement (Statement of Operations)
    Total Revenues (a.k.a. Net Sales)
    Operating Expenses (sometimes broken down into Cost of Goods Sold, SG&A, R&D, etc.)
    Operating Income (Loss)
    Net Income (Loss)
    Earnings per Share (Basic & Diluted)
    
2. Balance Sheet (Statement of Financial Position)
    Total Current Assets (including Cash & Equivalents, Accounts Receivable, Inventories, etc.)
    Total Noncurrent Assets (including Property, Plant & Equipment, Goodwill/Intangibles if applicable)
    Total Assets
    Total Current Liabilities (including Accounts Payable, etc.)
    Total Noncurrent Liabilities (including Long-Term Debt, etc.)
    Total Liabilities
    Stockholders’ Equity (common stock, additional paid-in capital, retained earnings, treasury stock, etc.)
    Total Liabilities & Stockholders’ Equity

3. Cash Flow Statement
    Net Cash from Operating Activities
    Net Cash from Investing Activities
    Net Cash from Financing Activities
    Net Increase (Decrease) in Cash
    Cash at Beginning of Period / Cash at End of Period
    
4. Statement of Stockholders’ Equity
    Beginning Equity
    Net Income (transferred in from the income statement)
    Dividends (if declared)
    Other Comprehensive Income (foreign currency translation, unrealized gains/losses, etc.)
    Ending Equity


"""