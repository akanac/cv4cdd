import os
import pandas as pd

def extract_model_name_from_path(root):
    depth = 3  # Set the depth to the number of times you want to split the path
    root_temp = str(root)
    for _ in range(depth):
        root_temp, folder_name = os.path.split(root_temp)
    model_id = folder_name.split("_sgd_")[-1]

    return model_id



# Define the function to combine evaluation_results.csv files
def combine_csv_files():
    # Get the current directory
    current_dir = os.getcwd()
    # Initialize an empty DataFrame to store the combined data
    combined_data = pd.DataFrame()

    # Walk through all subdirectories of the current directory
    for root, dirs, files in os.walk(current_dir):
        # Check if evaluation_results.csv exists in the current directory
        if 'evaluation_results.csv' in files:
            print(root)
            # Construct the path to the CSV file
            csv_path = os.path.join(root, 'evaluation_results.csv')
            # Load the CSV file into a DataFrame
            df = pd.read_csv(csv_path)

            model_id = extract_model_name_from_path(root)
            df['model_id'] = model_id
            # Append the DataFrame to the combined_data DataFrame
            combined_data = pd.concat([combined_data, df], ignore_index=True)

    best_rows = combined_data.groupby(['model_id', 'lag']).agg({'TP':'sum', 'FP':'sum', 'FN_TP':'sum'})
    precision = best_rows['TP'] / (best_rows['TP'] + best_rows['FP'])
    recall = best_rows['TP'] / (best_rows['FN_TP'])
    f1_score = 2 * (precision * recall) / (precision + recall)

    # Add precision, recall, and F1 score to the grouped data
    best_rows['Precision'] = precision
    best_rows['Recall'] = recall
    best_rows['F1 Score'] = f1_score

    # Group the data by 'lag' and calculate mean and standard deviation
    average_values = best_rows.groupby('lag').agg({'Precision': 'mean', 'Recall': 'mean', 'F1 Score': 'mean'})
    standard_values = best_rows.groupby('lag').agg({'Precision': 'std', 'Recall': 'std', 'F1 Score': 'std'})

    print(average_values)
    print(standard_values)

    # Check if any evaluation_results.csv files were found
    if combined_data.empty:
        print("No evaluation_results.csv files found in subdirectories.")
    else:
        # Write the combined data to a new CSV file
        combined_data.to_csv('combined_evaluation_results.csv', index=True)
        print("Combined evaluation results saved to combined_evaluation_results.csv")

        best_rows.to_csv('combined_evaluation_results_per_lag_and_model.csv', index=True)
        print("Grouped combined evaluation results saved to combined_evaluation_results_per_lag_and_model.csv")

# Call the function to combine the CSV files
combine_csv_files()
