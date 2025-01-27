import matplotlib.pyplot as plt
import random


def plot_value_counts_per_year(df, year_column, target_column):
    """
    Plots value counts of a target column per year.

    Parameters:
    - df: DataFrame containing the data
    - year_column: Column name representing the year
    - target_column: Column name whose value counts will be plotted

    Returns:
    - A bar plot showing value counts of the target column per year
    """
    # Get value counts per year
    counts_per_year = df.groupby(year_column)[target_column].value_counts().rename("counts").reset_index()

    # Pivot for easier plotting
    pivot_table = counts_per_year.pivot(index=year_column, columns=target_column, values="counts").fillna(0)

    # Plot the data
    ax = pivot_table.plot(kind="bar", stacked=True, figsize=(12, 6), alpha=0.8)
    plt.xlabel('Year')
    plt.ylabel('Counts')
    plt.title(f"Value Counts of '{target_column}' per Year, where label is True")
    plt.legend(title=target_column, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()



def find_and_visualize_dropped_years(full_df, smaller_df, cik_column='cik', year_column='year', selected_ciks=None, random_sample_size=5):
    """
    Find and visualize dropped years for each CIK, with options to filter or randomly sample CIKs.

    Parameters:
    - full_df: Full dataset
    - smaller_df: Smaller dataset with some rows dropped
    - cik_column: Column containing CIK identifiers
    - year_column: Column containing year values (in ecl: 'cik_year', but needs preprocessing)
    - selected_ciks: List of CIKs to plot (optional)
    - random_sample_size: Number of random CIKs to plot if no list is provided

    Returns:
    - A DataFrame showing missing years for each CIK
    """
    # Group years by CIK for both datasets
    full_years = full_df.groupby(cik_column)[year_column].apply(set)
    smaller_years = smaller_df.groupby(cik_column)[year_column].apply(set)
    

    # Find missing years for each CIK
    missing_years = full_years.subtract(smaller_years, fill_value=set())

    missing_years_df = missing_years.reset_index(name='dropped_years')

    # Filter CIKs to plot
    if selected_ciks:
        filtered_missing = missing_years_df[missing_years_df[cik_column].isin(selected_ciks)]
    else:
        # Randomly sample CIKs if no list is provided
        all_ciks = missing_years_df[cik_column].unique()
        sampled_ciks = random.sample(list(all_ciks), min(random_sample_size, len(all_ciks)))
        filtered_missing = missing_years_df[missing_years_df[cik_column].isin(sampled_ciks)]

    # Visualize missing years
    plt.figure(figsize=(12, 6))
    for _, row in filtered_missing.iterrows():
        # Convert set of missing years to a sorted list for plotting
        missing_years_list = sorted(list(row['dropped_years']))
        plt.scatter([row[cik_column]] * len(missing_years_list), missing_years_list, label=f"CIK {row[cik_column]}")

    # Customize the plot
    plt.xlabel('CIK')
    plt.ylabel('Dropped Years')
    plt.title('Dropped Years for Selected CIKs')
    plt.xticks(ticks=filtered_missing[cik_column], labels=filtered_missing[cik_column], rotation=45)   
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='CIK', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    return filtered_missing
