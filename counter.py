import pandas as pd

#
# def get_top_dense_columns(filename, top_n=50):
#     # Load the CSV file into a DataFrame
#     df = pd.read_csv(filename, index_col=0)
#
#     # Sort by 'Null Percentage' in ascending order to find the most dense columns
#     sorted_df = df.sort_values(by='Null Percentage')
#
#     # Select the top N columns based on density
#     top_dense_columns = sorted_df.head(top_n)
#
#     return top_dense_columns


# Usage
filename = './outputs/ecl_filtered.json'
cols_to_remove = ['item_7', "opinion_text"]

csv_file = pd.read_json(filename, lines=True)
# cols = csv_file.columns
# print(cols.to_list())
# cols = cols.to_list()
#
#
#
# cols.remove("item_7")
# cols.remove("opinion_text")
# print("columns are : ", cols)
# print(csv_file.describe())
print("total nulls", csv_file.isna().sum())



# top_dense_columns = get_top_dense_columns(filename)
# print(top_dense_columns)
