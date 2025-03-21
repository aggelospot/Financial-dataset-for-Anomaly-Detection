import psycopg2
import dotenv
from tools import config, db_schema
import pandas as pd
import os
from tools.utils import match_concept_in_section
from dotenv import load_dotenv
from decimal import Decimal

load_dotenv()

def create_connection():
    """
    Creates and returns a psycopg2 connection to the PostgreSQL database.
    Replace the placeholders with your own connection parameters.
    """
    try:
        conn = psycopg2.connect(
            host=os.getenv("HOST"),
            database=os.getenv("DATABASE_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("PASSWORD")
        )
        return conn
    except psycopg2.Error as e:
        print("Error connecting to the database:", e)
        raise

def close_connection(conn):
    if conn:
        conn.close()

def import_csv_to_sec_num(conn, file_path):
    """
    Imports CSV data into the 'sec_num' table using the COPY command.
    Assumes the CSV file has a header row matching the columns of sec_num.
    """
    try:
        with conn.cursor() as cur:
            copy_sql = """
                COPY sub
                FROM STDIN
                WITH (
                    FORMAT CSV,
                    DELIMITER \'\t\',
                    HEADER TRUE
                )
            """
            with open(file_path, 'r') as f:
                cur.copy_expert(copy_sql, f)
        # commit so changes are persisted
        conn.commit()
    except psycopg2.Error as e:
        print("Error importing CSV to sec_num:", e)
        raise


def import_all_num_csv(conn, db_imports_dir):
    """
    Recursively traverses the db_imports_dir and its subdirectories,
    looking for 'num.csv' files and importing them to the sec_num table.

    :param conn: psycopg2 connection object
    :param db_imports_dir: The path to the directory containing subdirectories (e.g., db_imports/2020q1, db_imports/2020q2, etc.)
    """
    for root, dirs, files in os.walk(db_imports_dir):
        print(f"root={root}, dir={dirs}, files={files}")
        for file_name in files:
            if file_name.lower() == "pre.txt":
                file_path = os.path.join(root, file_name)
                print(f"Found sub.csv: {file_path}. Importing to database...")

                # uncomment to import sub, then pre. Add more import_sub_related_table calls for other files
                # import_sub_table(conn,file_path)
                import_sub_related_table(conn,file_path,table_name="pre")
                print("CSV successfully imported!")


def import_sub_table(conn, file_path):
    """
    Imports rows from a CSV file into the 'sub' table, filtering only
    those rows whose 'form' contains '10-K'.
    """
    # Example staging schema. Adjust columns/types as needed to match your CSV.
    staging_table_ddl = db_schema.STAGING_TABLE_SUB_SCHEMA

    with conn.cursor() as cur:
        # 1. Drop staging table if it exists.
        cur.execute("DROP TABLE IF EXISTS staging_sub;")

        # 2. Create staging table with the schema that matches your CSV structure.
        cur.execute(staging_table_ddl)

        # 3. Copy CSV into staging table.
        copy_sql = """
            COPY staging_sub
            FROM STDIN
            WITH (
                FORMAT CSV,
                DELIMITER '\t',
                HEADER TRUE
            );
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            cur.copy_expert(copy_sql, f)

        # 4. Insert only the rows that have '10-K' in the 'form' column into the final 'sub' table.
        # Make sure 'sub' has the same or compatible columns. Adjust column names as needed.
        insert_sql = """
            INSERT INTO sub 
            SELECT *
            FROM staging_sub
            WHERE form LIKE '%10-K%';
        """
        cur.execute(insert_sql)

        # 5. Drop the staging table now that the data is transferred and filtered.
        cur.execute("DROP TABLE staging_sub;")

    # Commit the changes to make them permanent.
    conn.commit()

def get_table_schema(table_name):
    if table_name == 'num':
        return db_schema.STAGING_TABLE_NUM_SCHEMA
    if table_name == 'pre':
        return db_schema.STAGING_TABLE_PRE_SCHEMA

def import_sub_related_table(conn, file_path, table_name):
    """
    Imports rows from a CSV file into a target table,
    but only for those rows where 'adsh' is already present in the 'sub' table.
    """

    drop_staging_sql = f"DROP TABLE IF EXISTS staging_{table_name};"
    create_staging_sql = get_table_schema(table_name)

    #  load CSV into the staging table
    copy_sql = f"""
        COPY staging_{table_name}
        FROM STDIN
        WITH (
            FORMAT CSV,
            DELIMITER '\t',
            HEADER TRUE
        );
    """
    print("copy sql : ", copy_sql)

    insert_sql = f"""
            WITH filtered_sub AS (
            SELECT s.*
            FROM staging_{table_name} AS s
            JOIN sub ON s.adsh = sub.adsh
        )
        INSERT INTO {table_name}
        SELECT * FROM filtered_sub;
        """

    # 5. Drop the staging table when done
    drop_staging_after_sql = f"DROP TABLE staging_{table_name};"

    # Execute the SQL commands in one transaction block
    with conn.cursor() as cur:
        # Drop staging table if it exists
        cur.execute(drop_staging_sql)

        # Create new staging table
        cur.execute(create_staging_sql)

        # Bulk load CSV into staging table
        with open(file_path, 'r', encoding='utf-8') as f:
            cur.copy_expert(copy_sql, f)

        # Insert only matching records into the real table
        cur.execute(insert_sql)

        # Drop the staging table to clean up
        cur.execute(drop_staging_after_sql)

    # Commit the transaction
    conn.commit()


def get_tag_values_by_accession_number(connection, accession_number, tags, num_quarters=4):
    """
    Fetches the values from a list of tags. example: tags=['NetIncomeLoss','EarningsPerShareBasic']
    """

    select_sql = f"""
    SELECT DISTINCT ON (tag) tag,value
    FROM public.num
    WHERE adsh = %s
    AND qtrs = %s
    AND tag = ANY(%s)
    ORDER BY tag, ddate DESC;
    """

    try:
        with connection.cursor() as cursor:
            cursor.execute(select_sql, (accession_number, num_quarters, tags))
            results = cursor.fetchall()
            return results
    except Exception as e:
        print("Error executing query:", e)
        return None

def get_tags_value_by_accession_number(connection, accession_number, tag, num_quarters=[0,4]):
    # Ensure tags are unique and properly formatted for SQL
    select_sql = f"""
    SELECT DISTINCT ON (tag) value
    FROM public.num
    WHERE adsh = %s
    AND qtrs = ANY(%s)
    AND tag = %s
    ORDER BY tag, ddate DESC;
    """

    try:
        with connection.cursor() as cursor:
            cursor.execute(select_sql, (accession_number, num_quarters, tag))
            result = cursor.fetchone()  # Fetch a single row instead of fetchall()

            if result:  # Check if there's a result
                return float(result[0]) if isinstance(result[0], Decimal) else result[0]
            else:
                return None  # If no result found, return None
    except Exception as e:
        print("Error executing query:", e)
        return None


def retrieve_statement_taxonomies_by_accession_number(accession_number, financial_statement, num_quarters):
    """
    Retrieves a financial statement's unique taxonomies found in the report using the accession_number.

    Options for financial statements:
        BS = BalanceSheet, IS = IncomeStatement, CF = CashFlow,
        EQ = Equity, CI = ComprehensiveIncome, SI = Schedule of Investments,
        UN = Unclassifiable Statement

    Use num_quarters = 0 for the balance sheet, and num_quarters = [0,4] for IS, CF, and EQ.
    """

    # Ensure num_quarters is iterable
    if isinstance(num_quarters, list):
        num_quarters_clause = " OR ".join(f"num.qtrs = {q}" for q in num_quarters)
        where_qtrs_clause = f"({num_quarters_clause})"
    else:
        where_qtrs_clause = f"num.qtrs = {num_quarters}"

    query_sql = f"""
        SELECT DISTINCT ON (num.tag) num.tag, num.qtrs
        FROM public.num
        LEFT OUTER JOIN public.pre 
        ON num.adsh = pre.adsh
        AND num.tag = pre.tag 
        WHERE num.adsh = '{accession_number}'
        AND pre.stmt = '{financial_statement}'
        AND {where_qtrs_clause}
        ORDER BY num.tag, num.ddate DESC;
    """

    try:
        with conn.cursor() as cur:
            cur.execute(query_sql)
            result = cur.fetchall()

            return [row[0] for row in result]

    except psycopg2.Error as e:
        print("Error executing query on sec_num:", e)
        raise






conn = create_connection()
pd.options.display.max_colwidth = 1500
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

try:

    # file_path = os.path.join(os.path.join(config.IMPORTS_DIR, "2020q4(1)"), "num.csv")
    # import_all_num_csv(conn, config.IMPORTS_DIR)
    from tools.data_loader import DataLoader
    import json
    data_loader = DataLoader()

    ecl = data_loader.load_dataset(config.ECL_METADATA_NOTEXT_PATH, alias='ecl', lines=True)
    with open(config.XBRL_MAPPING_PATH, 'r') as file:
        xbrl_mapping = json.load(file)


    tag_list = []
    for section in ["IncomeStatement", "BalanceSheet", "CashFlow", "StatementOfStockholdersEquity"]:
        for key, value in xbrl_mapping[section].items():
            if key not in tag_list:  # Avoid duplicates
                tag_list.append(key)
    for tag in tag_list:
        ecl[tag] = None  # You can replace None with a default value if needed


    results = []

    for idx, row in ecl.iterrows():
        print(f"\rCurrent row: {idx}", end='')
        # if idx > 100: break
        metadata_row = {
            "accession_number": row['accessionNumber'],
            "isXBRL": row['isXBRL']
        }
        if row['isXBRL'] == 0: continue
        result = retrieve_statement_taxonomies_by_accession_number(row['accessionNumber'], 'IS', [0,4])
        target_concepts = xbrl_mapping.get("IncomeStatement")
        matched_is_items = match_concept_in_section(target_concepts, result)

        result = retrieve_statement_taxonomies_by_accession_number(row['accessionNumber'], 'BS', 0)
        target_concepts = xbrl_mapping.get("BalanceSheet")
        matched_bs_items = match_concept_in_section(target_concepts, result)

        result = retrieve_statement_taxonomies_by_accession_number(row['accessionNumber'], 'CF', [0, 4])
        target_concepts = xbrl_mapping.get("CashFlow")
        matched_cf_items = match_concept_in_section(target_concepts, result)

        result = retrieve_statement_taxonomies_by_accession_number(row['accessionNumber'], 'EQ', [0, 4])
        target_concepts = xbrl_mapping.get("StatementOfStockholdersEquity")
        matched_eq_items = match_concept_in_section(target_concepts, result)


        metadata_row = {**metadata_row, **matched_is_items, **matched_bs_items, **matched_cf_items, **matched_eq_items}
        all_matches = {**matched_is_items, **matched_bs_items, **matched_cf_items, **matched_eq_items}

        # if all_matches:
        for tag in tag_list:
            tag_synonym = all_matches.get(tag)
            if tag_synonym is not None:
                ecl.at[idx, tag] = get_tags_value_by_accession_number(connection=conn, accession_number=row['accessionNumber'], tag=tag_synonym)

        results.append(metadata_row)

    result_df = pd.DataFrame(results)
    print("Resulting df: \n", result_df.head(1000))
    print("Resulting ecl df \n", ecl.head(100))
    data_loader.save_dataset(result_df, os.path.join(config.OUTPUT_DIR, "tags.csv"))
    data_loader.save_dataset(ecl, os.path.join(config.OUTPUT_DIR, "ecl_with_financial_tags.csv"))








    # print("Query results : ", result)
    # print("Target concepts: \n", target_concepts)
    # print("End!\n", matched_ic_items)



finally:
    # 4. Close the connection regardless of success/failure
    close_connection(conn)