import dask.dataframe as dd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def perform_eda(file_path, sample_frac=0.01, output_dir='./eda_results'):
    """
    Performs exploratory data analysis on the given JSON file.
    - Counts nulls in each column (as counts and percentages).
    - Computes the correlation matrix on sampled data.
    - Writes the results to files.
    """
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the data using Dask for efficient processing of large files
    print("Loading data...")
    df = dd.read_json(file_path, lines=True)

    # Sample the data
    if sample_frac < 1.0:
        print(f"Sampling {sample_frac * 100}% of the data...")
        df = df.sample(frac=sample_frac, random_state=42)
    else:
        print("Using the full dataset...")

    # Compute the number of nulls and percentages in each column
    print("Computing null counts and percentages per column...")
    null_counts = df.isnull().sum().compute()
    total_counts = len(df)
    null_percentages = (null_counts / total_counts) * 100
    null_summary = pd.DataFrame({
        'Null Count': null_counts,
        'Null Percentage': null_percentages
    })

    # Write null summary to a CSV file
    null_summary_file = os.path.join(output_dir, 'null_summary.csv')
    null_summary.to_csv(null_summary_file)
    print(f"Null counts and percentages per column written to {null_summary_file}")

    # Optionally, filter numerical columns for correlation matrix
    print("\nSelecting numerical columns for correlation matrix...")
    numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
    print(f"Total numerical columns: {len(numerical_columns)}")

    # Filter out columns with high null percentages
    threshold = 99.99  # Exclude columns with more than 50% nulls
    cols_to_use = null_summary[null_summary['Null Percentage'] <= threshold].index.tolist()
    numerical_cols_to_use = [col for col in numerical_columns if col in cols_to_use]
    print(f"Numerical columns with <= {threshold}% nulls: {len(numerical_cols_to_use)}")

    # Limit the number of columns for correlation matrix to avoid huge matrices
    max_columns = 30
    if len(numerical_cols_to_use) > max_columns:
        print(f"Limiting numerical columns to first {max_columns} columns for correlation matrix.")
        numerical_cols_to_use = numerical_cols_to_use[:max_columns]

    # Check if there are enough numerical columns to compute correlation
    if len(numerical_cols_to_use) < 2:
        print("Not enough numerical columns to compute a correlation matrix.")
    else:
        # Compute the correlation matrix
        print("\nComputing correlation matrix...")
        # We need to compute the Dask DataFrame to a Pandas DataFrame for correlation
        df_sample = df[numerical_cols_to_use].compute()
        corr_matrix = df_sample.corr()
        print("Correlation matrix computed.")

        # Write the correlation matrix to a CSV file
        corr_matrix_file = os.path.join(output_dir, 'correlation_matrix.csv')
        corr_matrix.to_csv(corr_matrix_file)
        print(f"Correlation matrix written to {corr_matrix_file}")

        # Plot the correlation matrix heatmap and save it
        print("\nPlotting correlation matrix heatmap...")
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
        plt.title('Correlation Matrix Heatmap')
        heatmap_file = os.path.join(output_dir, 'correlation_heatmap.png')
        plt.savefig(heatmap_file)
        plt.close()
        print(f"Correlation matrix heatmap saved to {heatmap_file}")

    # FUTURE TODO:
    # mean, median, std for numerical cols
    # outliers
    # value counts for categorical columns
    # distributions
    # TSA?
    # VSCODE + JYPITER
    # could remove the text for statistics
    # convert to csv ?
    # find the densest submatrix (< x% of nulls)
    # double check validity of data by sampling ecl dataset and cross checking with the sec
    # check sec documentation on why no data exists after 2010
    # !!! use pandas instead


    # Compute basic statistics and write to a file
    print("\nComputing basic statistics for numerical columns...")
    stats = df[numerical_cols_to_use].describe().compute()
    stats_file = os.path.join(output_dir, 'numerical_stats.csv')
    stats.to_csv(stats_file)
    print(f"Basic statistics for numerical columns written to {stats_file}")


if __name__ == "__main__":
    file_path = './outputs/ecl_filtered.json'
    # Adjust the sample_frac parameter to control the sample size
    sample_frac = 0.50  # % sample
    perform_eda(file_path, sample_frac=sample_frac)
