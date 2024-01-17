import pandas as pd
import os

def parquet_to_csv(parquet_folder, csv_filename):
    # Get a list of all parquet files in the specified folder
    parquet_files = [f for f in os.listdir(parquet_folder) if f.endswith('.parquet')]

    # Check if there are any parquet files in the folder
    if not parquet_files:
        print("No Parquet files found in the specified folder.")
        return

    # Create a list comprehension to read each parquet file into a DataFrame
    dfs = [pd.read_parquet(os.path.join(parquet_folder, parquet_file)) for parquet_file in parquet_files]

    # Concatenate all DataFrames in the list
    combined_data = pd.concat(dfs, ignore_index=True)

    # Save the combined data as a CSV file
    combined_data.to_csv(csv_filename, index=False)

    print(f"Combined data saved to {csv_filename}")

if __name__ == '__main__':
    parquet_folder = 'data'
    csv_filename = 'dataset.csv'
    parquet_to_csv(parquet_folder, csv_filename)
