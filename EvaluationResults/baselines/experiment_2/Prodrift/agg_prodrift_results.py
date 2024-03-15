import os
import pandas as pd

def combine_csv_files():
    # Get the current directory
    current_directory = os.getcwd()

    # Create an empty DataFrame to store data from all CSV files
    combined_data = pd.DataFrame()

    # Walk through all subdirectories of the current directory
    for root, dirs, files in os.walk(current_directory):
        for file in files:
            # Check if the file name starts with "evaluation_results_prodrift_"
            if file.startswith("evaluation_results_prodrift_") and file.endswith(".csv"):
                file_path = os.path.join(root, file)
                # Read the CSV file and append its content to the combined_data DataFrame
                df = pd.read_csv(file_path)
                prodrift_window_size = file.split("_")[-1].split(".")[0]
                df['parameter'] = prodrift_window_size
                combined_data = pd.concat([combined_data, df], ignore_index=True)

    # Write the combined data to a new CSV file
    combined_file_path = os.path.join(current_directory, "evaluation_results_all_parameters.csv")
    #combined_data.to_csv(combined_file_path, index=False)

    print("Combined CSV file created successfully.")

    best_rows = combined_data.groupby(['lag', 'parameter']).agg({'TP':'sum', 'FP':'sum', 'FN_TP':'sum'})
    precision = best_rows['TP'] / (best_rows['TP'] + best_rows['FP'])
    recall = best_rows['TP'] / (best_rows['FN_TP'])
    f1_score = 2 * (precision * recall) / (precision + recall)

    # Add precision, recall, and F1 score to the grouped data
    best_rows['Precision'] = precision
    best_rows['Recall'] = recall
    best_rows['F1 Score'] = f1_score

    # Reset index to make 'lag' and 'parameter' columns accessible
    best_rows_reset = best_rows.reset_index()

    # Group by 'lag' and find the index of the row with the highest F1 score
    best_f1_indices = best_rows_reset.groupby('lag')['F1 Score'].idxmax()

    # Select rows with the highest F1 score for each lag
    best_parameters = best_rows_reset.loc[best_f1_indices]


    output_file_path = os.path.join(current_directory, 'evaluation_results_all_parameters_best_results.csv')
    best_parameters.to_csv(output_file_path, index=False)
    print("Data with the best results for each approach is saved to:", output_file_path)



# Call the function to combine CSV files
combine_csv_files()
