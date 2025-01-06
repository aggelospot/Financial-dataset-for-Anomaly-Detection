import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Alpha Vantage API key
API_KEY = 'A8SX7A1XCBLT3W2O'


def get_stock_data(api_key, symbol, start_date=None, end_date=None):
    """
    Fetches historical stock data for a given symbol between start_date and end_date.

    Parameters:
        api_key (str): Your Alpha Vantage API key.
        symbol (str): Stock ticker symbol.
        start_date (str): Start date in 'YYYY-MM-DD' format (optional).
        end_date (str): End date in 'YYYY-MM-DD' format (optional).

    Returns:
        pandas.DataFrame or None: Historical stock data or None if an error occurs.
    """
    url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'TIME_SERIES_DAILY',  # Using the free-tier endpoint
        'symbol': symbol,
        'outputsize': 'full',
        'datatype': 'json',
        'apikey': api_key
    }

    response = requests.get(url, params=params)
    data = response.json()

    if 'Error Message' in data:
        print(data['Error Message'])
        return None

    if 'Time Series (Daily)' not in data:
        print("No data found for symbol:", symbol)
        print("AlphaVantage response was: ", data)
        return None

    df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')
    df = df.astype(float)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    # Filter by date if start_date or end_date is specified
    if start_date:
        df = df[df.index >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df.index <= pd.to_datetime(end_date)]

    return df

def plot_stock_data(df, symbol):
    """
    Plots the closing prices of the stock data.

    Parameters:
        df (pandas.DataFrame): The stock data DataFrame.
        symbol (str): Stock ticker symbol.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['4. close'], label=f'{symbol} Closing Price')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.title(f'{symbol} Stock Closing Price Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage: Fetch GOOG's 1-year historical data
if __name__ == '__main__':
    # API_KEY = 'YOUR_API_KEY_HERE'  # Replace with your actual API key
    symbol = 'GOOG'
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')

    goog_data = get_stock_data(API_KEY, symbol, start_date=start_date, end_date=end_date)
    if goog_data is not None:
        print(goog_data)
        plot_stock_data(goog_data, symbol)

