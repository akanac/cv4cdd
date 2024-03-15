import os
import pandas as pd

def aggregate_csv_files(folder_path, output_file):
    # Initialize an empty DataFrame to store aggregated data
    aggregated_data = pd.DataFrame()

    # Iterate through each file in the folder
    for file_name in os.listdir(folder_path):
        # Check if the file is a CSV file and ends with "_agg.csv"
        if file_name.endswith("evaluated_1.csv"):
            # Read the CSV file into a DataFrame
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path)
            
            # Add a new column with the file name
            if file_name.split("_")[2] == "j" or  file_name.split("_")[2] == "graphs":
                approach_name = file_name.split("_")[1] + "_" + file_name.split("_")[2]
            else:
                approach_name = file_name.split("_")[1]
            df['approach'] = approach_name
            
            # Append the DataFrame to the aggregated data
            aggregated_data = aggregated_data.append(df, ignore_index=True)
    
    # Write the aggregated data to a single CSV file
    aggregated_data.to_csv(output_file, index=False)

# Example usage:
folder_path = os.getcwd()
output_file = "ALL_results_aggregated_step1.csv"
aggregate_csv_files(folder_path, output_file)
