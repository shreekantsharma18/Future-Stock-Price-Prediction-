import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_random_stock_data(num_rows=5000, filename="your_stock_data.csv"):
    """
    Generates a CSV file with random stock data for actual and predicted prices.

    Args:
        num_rows (int): The number of data points (rows) to generate.
        filename (str): The name of the CSV file to save the data to.
                        Defaults to "your_stock_data.csv".
    """
    print(f"Generating {num_rows} rows of random stock data...")

    # 1. Generate Dates
    # Start date for the data (e.g., 5000 days ago from today)
    start_date = datetime.now() - timedelta(days=num_rows + 365) # Add extra days to ensure enough dates
    dates = [start_date + timedelta(days=i) for i in range(num_rows)]

    # 2. Generate Actual Prices
    # Start with a base price and simulate daily fluctuations
    base_price = 100.0
    actual_prices = []
    current_price = base_price
    for _ in range(num_rows):
        # Simulate a daily change (small random walk)
        change = np.random.normal(0, 1.5) # Mean 0, std dev 1.5 for daily change
        current_price += change
        # Ensure price doesn't go too low (e.g., below 10)
        if current_price < 10:
            current_price = 10 + np.random.rand() * 5 # Reset to a slightly higher base
        actual_prices.append(round(current_price, 2)) # Round to 2 decimal places

    # 3. Generate Predicted Prices
    # Predicted prices will be close to actual prices with some noise
    predicted_prices = []
    for actual_p in actual_prices:
        # Add random noise (e.g., +/- 2% of the actual price)
        noise = np.random.uniform(-0.02, 0.02) * actual_p
        predicted_p = actual_p + noise
        predicted_prices.append(round(predicted_p, 2))

    # 4. Create DataFrame
    data = {
        'Date': dates,
        'Actual Price': actual_prices,
        'Predicted Price': predicted_prices
    }
    df = pd.DataFrame(data)

    # 5. Save to CSV
    try:
        df.to_csv(filename, index=False)
        print(f"Successfully generated '{filename}' with {num_rows} rows.")
        print(f"File saved at: {os.path.abspath(filename)}")
    except Exception as e:
        print(f"Error saving file: {e}")

if __name__ == "__main__":
    # Call the function to generate the CSV file
    generate_random_stock_data(num_rows=5000, filename="your_stock_data.csv")
