# MySQL connection and queries

import mysql.connector
import pandas as pd
import os, sys

def create_connection():
    connection = mysql.connector.connect(
        host="localhost",         
        user="root",              
        password="root",      
        database="mdm"   
    )
    return connection

# Let's Fetch all data from a given table and return it as a Pandas DataFrame.
def fetch_table_data(table_name):
    query = f"SELECT * FROM {table_name};"
    conn = create_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(query)
    result = cursor.fetchall()
    conn.close()
    return pd.DataFrame(result)

# Fetch data from a table and save it as a CSV file in the specified directory.
def save_table_as_csv(table_name, output_dir):
    # Fetch table data
    df = fetch_table_data(table_name)
    
    # We also need to ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the DataFrame to a CSV file
    file_path = os.path.join(output_dir, f"{table_name}.csv")
    df.to_csv(file_path, index=False)
    print(f"Table '{table_name}' saved to {file_path}")

# Example usage
if __name__ == "__main__":
    print("Done")
