# main.py
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
import sys

def create_sample_parquet_file(file_name="sample.parquet"):
    """
    Creates a sample Parquet file with some data for demonstration purposes.
    """
    # Create a pandas DataFrame.
    # A DataFrame is a 2D labeled data structure with columns of potentially different types.
    # It is the most common way to work with tabular data in Python.
    data = {
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
        'age': [25, 30, 35, 40, 45],
        'city': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'],
        'is_student': [True, False, False, True, False]
    }
    df = pd.DataFrame(data)

    # Convert the pandas DataFrame to a PyArrow Table.
    # PyArrow is a library for in-memory columnar data, which is highly efficient.
    table = pa.Table.from_pandas(df)

    # Write the PyArrow Table to a Parquet file.
    # The Parquet format is a columnar storage file format that is highly efficient
    # for storing and querying large datasets.
    print(f"Creating a sample Parquet file named '{file_name}'...")
    pq.write_table(table, file_name)
    print("Sample file created successfully.")
    return file_name

def view_parquet_file(file_name):
    """
    Reads a Parquet file and displays its contents.
    """
    # Before running this script, you need to install the required libraries.
    # You can install them using pip:
    # pip install pandas pyarrow
    
    if not os.path.exists(file_name):
        print(f"Error: The file '{file_name}' was not found.")
        return

    try:
        # Read the Parquet file into a PyArrow Table.
        # The read_table() function reads the entire file into memory.
        print(f"\nReading data from '{file_name}'...")
        table = pq.read_table(file_name)

        # Convert the PyArrow Table to a pandas DataFrame for easy viewing.
        # The to_pandas() method provides a familiar and powerful way to inspect the data.
        df_from_parquet = table.to_pandas()

        # Print the contents of the DataFrame.
        print("\nContents of the Parquet file:")
        print(df_from_parquet)

        # You can also view the schema (the structure of the data).
        print("\nSchema of the Parquet file:")
        print(table.schema)

    except Exception as e:
        print(f"An error occurred while reading the Parquet file: {e}")

if __name__ == "__main__":
    # Check if the user provided a filename as a command-line argument.
    if len(sys.argv) != 2:
        print("Usage: python main.py <path_to_your_parquet_file>")
        # Example of how to create a file to test with:
        # print("\nCreating a sample file 'sample.parquet' for you to test with.")
        # create_sample_parquet_file("sample.parquet")
        # print("You can now run: python main.py sample.parquet")
        sys.exit(1) # Exit the script if no file is provided.

    # The first command-line argument is the filename.
    parquet_file_name = sys.argv[1]

    # View the contents of the specified Parquet file.
    view_parquet_file(parquet_file_name)
