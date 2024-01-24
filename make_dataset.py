import os
import pyarrow as pa
import pyarrow.parquet as pq

def parquet_to_parquet(parquet_folder, parquet_filename):
    # Get a list of all parquet files in the specified folder
    parquet_files = [f for f in os.listdir(parquet_folder) if f.endswith('.parquet')]

    # Check if there are any parquet files in the folder
    if not parquet_files:
        print("No Parquet files found in the specified folder.")
        return

    # Create a list comprehension to read each parquet file into a PyArrow Table
    tables = [pq.read_table(os.path.join(parquet_folder, parquet_file)) for parquet_file in parquet_files]

    # Concatenate all Tables in the list
    combined_table = pa.concat_tables(tables)

    # Flatten any nested structures in the table
    flattened_table = combined_table.flatten()

    # Save the flattened table as a Parquet file
    pq.write_table(flattened_table, parquet_filename)

    print(f"Combined and flattened data saved to {parquet_filename}")

if __name__ == '__main__':
    parquet_folder = 'data'
    parquet_filename = 'dataset.parquet'
    parquet_to_parquet(parquet_folder, parquet_filename)