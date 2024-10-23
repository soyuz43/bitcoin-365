# fetch_bitcoin_data.py

import requests
import pandas as pd
from datetime import datetime

def fetch_bitcoin_data(vs_currency='usd', days=365):
    """
    Fetches Bitcoin market data from CoinGecko API.

    Parameters:
        vs_currency (str): The target currency (e.g., 'usd').
        days (int): Number of days to retrieve data for.

    Returns:
        dict: JSON response from the API containing price data.
    """
    url = f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {
        'vs_currency': vs_currency,
        'days': days
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an error for bad status codes
        data = response.json()
        return data
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")  # Handle HTTP errors
    except Exception as err:
        print(f"An error occurred: {err}")  # Handle other errors

def parse_prices(prices):
    """
    Parses the price data into a DataFrame.

    Parameters:
        prices (list): List of [timestamp, price] pairs.

    Returns:
        DataFrame: Pandas DataFrame with 'Date' and 'Price' columns.
    """
    # Convert timestamps from milliseconds to datetime
    df = pd.DataFrame(prices, columns=['Timestamp', 'Price'])
    df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df = df[['Date', 'Price']]
    return df

def save_to_csv(df, filename='bitcoin_prices.csv'):
    """
    Saves the DataFrame to a CSV file.

    Parameters:
        df (DataFrame): Pandas DataFrame to save.
        filename (str): Name of the output CSV file.
    """
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

def main():
    # Fetch data
    data = fetch_bitcoin_data()
    if data and 'prices' in data:
        # Parse data
        df = parse_prices(data['prices'])
        # Save to CSV
        save_to_csv(df)
    else:
        print("No price data found.")

if __name__ == "__main__":
    main()
