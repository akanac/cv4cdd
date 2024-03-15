import os
import pandas as pd

# Get the current working directory
folder_path = os.getcwd()

# Get a list of all files in the folder
files = os.listdir(folder_path)

# Initialize an empty DataFrame to store aggregated data
aggregated_data = pd.DataFrame()

# Iterate through each file in the folder
for file in files:
    if file.startswith('algorithm_results_evaluation_measure_') and file.endswith('.csv'):
        # Read the CSV file into a DataFrame
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        
        # Add a column to indicate the measure value from the file name
        measure_value = float(file.split('_')[-1][:-4])  # Extract measure value from file name
        df['Measure'] = measure_value
        
        # Concatenate the DataFrame to the aggregated DataFrame
        aggregated_data = pd.concat([aggregated_data, df])

# Reset index of aggregated DataFrame
aggregated_data.reset_index(drop=True, inplace=True)

# Save the aggregated data to a CSV file
output_file_path = os.path.join(folder_path, 'aggregated_evaluation_measure.csv')
aggregated_data.to_csv(output_file_path, index=False)

print("Aggregated data saved to:", output_file_path)

best_rows = aggregated_data.loc[aggregated_data.groupby(['Category', 'Measure'])['F1'].idxmax()]
output_file_path = os.path.join(folder_path, 'aggregated_evaluation_measure_best.csv')
best_rows.to_csv(output_file_path, index=False)
print("Data with the best results for each approach is saved to:", output_file_path)
