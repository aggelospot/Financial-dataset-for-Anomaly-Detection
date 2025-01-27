import pandas as pd
import os
from typing import Dict, List


class DataLoader:
    """
    A data loader class to manage multiple data files, transformations, and comparisons in a structured way.

    Attributes
    ----------
    loaded_data : dict
        Dictionary to store file name -> DataFrame mappings for easy access.
    """

    def __init__(self):
        self.loaded_data: Dict[str, pd.DataFrame] = {}

    def load_dataset(self, file_path: str, alias: str = None, **kwargs) -> pd.DataFrame:
        """
        Detects CSV or JSON based on file extension, loads the dataset,
        and stores it in the loaded_data dictionary.

        Parameters
        ----------
        file_path : str Path to the file to load.
        **kwargs : dict Additional arguments to pass to the Pandas read functions.

        Returns
        -------
        pd.DataFrame
            Loaded data as a Pandas DataFrame.
        """
        # Detect file extension
        _, extension = os.path.splitext(file_path)
        extension = extension.lower()

        print(f"Loading dataset: {os.path.basename(file_path)}")
        if extension == '.csv':
            df = pd.read_csv(file_path, **kwargs)
        elif extension == '.json':
            df = pd.read_json(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file extension: {extension}")

        # Use alias if provided; otherwise, default to file path
        key = alias if alias else file_path
        self.loaded_data[key] = df
        return df

    def get_dataset(self, key: str) -> pd.DataFrame:
        if key in self.loaded_data:
            return self.loaded_data[key]
        else:
            raise KeyError(f"No dataset loaded with key: {key}")

    def save_dataset(self, key: str, out_path: str, **kwargs) -> None:
        """
        Saves the DataFrame associated with the given key to an output file as
        either CSV or JSON, inferred by the file extension.

        Parameters
        ----------
        key : str
            The key in self.loaded_data corresponding to the DataFrame to save.
        out_path : str
            The path (including filename) to which the DataFrame will be saved.
            The file extension (.csv or .json) determines the format.
        **kwargs : dict
            Additional arguments to pass to the Pandas writer methods (e.g., index=False).
        """
        df = self.get_dataset(key)

        _, extension = os.path.splitext(out_path)
        extension = extension.lower()

        if extension == '.csv':
            df.to_csv(out_path, index=False, **kwargs)
        elif extension == '.json':
            df.to_json(out_path, **kwargs)
        else:
            raise ValueError("Unsupported file extension. Please use .csv or .json.")


    def drop_na(self, key: str, how: str = 'any'):
        df = self.get_dataset(key)
        self.loaded_data[key] = df.dropna(how=how)

    def column_distribution(self, key: str, column_name: str) -> pd.Series:
        """
        Returns the frequency (value counts) of each unique entry in the specified
        column for the given dataset key.

        Parameters
        ----------
        key : str
            The key in self.loaded_data for the desired DataFrame.
        column_name : str
            The name of the column in the DataFrame to analyze.

        Returns
        -------
        pd.Series
            A Pandas Series with the counts of each unique value (including NaN),
            indexed by the unique values in `column_name`.
        """
        df = self.get_dataset(key)
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in DataFrame '{key}'.")

        return df[column_name].value_counts(dropna=False)

    def list_datasets(self):
        return list(self.loaded_data.keys())


    def analyze_numeric_columns(self, key: str, original_keys: List[str], alias: str = 'column_stats') -> str:
        """
        Exclude a predefined list of 'original' keys,
        then compute null-count and basic statistics (mean, median, std) for numeric columns.

        Parameters
        ----------
        key : str
            Key in self.loaded_data for the desired DataFrame.
        original_keys : list
            A list of columns to be excluded from the analysis.
        alias : str
            The alias of the dataframe that will be created.

        Returns
        -------
        key: str
            The key of the newly created DataFrame.
        """
        df = self.get_dataset(key)
        total_rows = len(df)

        # Identify columns to analyze by excluding `original_keys`
        columns_to_analyze = [col for col in df.columns if col not in original_keys]

        columns_stats = []
        for col in columns_to_analyze:
            null_count = df[col].isnull().sum()
            percentage_nulls = (null_count / total_rows) * 100

            # Initialize mean, median, std to None
            mean_value = None
            median_value = None
            std_value = None

            # Check if column is numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                mean_value = df[col].mean()
                median_value = df[col].median()
                std_value = df[col].std()

            columns_stats.append({
                'column': col,
                'null_count': null_count,
                'percentage_nulls': percentage_nulls,
                'mean': mean_value,
                'median': median_value,
                'std': std_value
            })

        # Convert list of dicts to DataFrame
        columns_stats_df = pd.DataFrame(columns_stats)

        self.loaded_data[alias] = columns_stats_df

        return alias
