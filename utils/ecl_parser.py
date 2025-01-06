import json


def parse_ECL_json(file_path):
    # To store unique keys across all rows
    key_set = set()
    first_row_key_count = None
    row_counter = 0
    inconsistent_rows = 0

    with open(file_path, 'r') as f:
        for line in f:
            try:
                row = json.loads(line.strip())
                print("currect cik: ", row["cik"])
                # Convert each line to a dictionary
                json_obj = json.loads(line.strip())

                # Extract keys from the current row
                current_keys = set(json_obj.keys())

                # Add current keys to the global key set
                key_set.update(current_keys)

                # Check consistency of the number of keys
                if first_row_key_count is None:
                    first_row_key_count = len(current_keys)
                elif len(current_keys) != first_row_key_count:
                    inconsistent_rows += 1

                row_counter += 1

            except json.JSONDecodeError:
                print(f"Error decoding JSON on row {row_counter + 1}")
                continue

    # Convert set to list and print
    key_list = list(key_set)
    print(f"Keys found in the JSON: {key_list}")
    print(f"Total rows processed: {row_counter}")
    print(f"Inconsistent rows (different key counts): {inconsistent_rows}")

    return key_list, row_counter, inconsistent_rows



file_path = './data/ECL_AA_subset.json'
parse_ECL_json(file_path)