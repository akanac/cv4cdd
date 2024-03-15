import os
import csv
from tqdm import tqdm
from pm4py.objects.log.importer.xes import importer as xes_importer
from multiprocessing import Pool, cpu_count

def count_traces_in_log(log_file):
    variant = xes_importer.Variants.ITERPARSE
    parameters = {variant.value.Parameters.SHOW_PROGRESS_BAR: False}
    log = xes_importer.apply(log_file, variant=variant, parameters=parameters)
    num_traces = len(log)
    directory, filename = os.path.split(log_file)
    folder_name = os.path.basename(directory)
    if folder_name == "Bose":
        change_points = [1199, 2399, 3599, 4799]
    elif folder_name == "Ostovar":
        change_points = [999, 1999]
    elif folder_name == "Ceravolo":
        change_points = str([(int(filename.split("_")[3])//2) - 1])
    else:
        change_points = []


    return (folder_name, filename.split(".")[0], num_traces, change_points)

def count_traces_in_logs(directory):
    log_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.xes'):
                log_files.append(os.path.join(root, file))

    with Pool(num_cores) as pool:
        log_info = list(tqdm(pool.imap(count_traces_in_log, log_files), total=len(log_files), desc="Processing logs", unit="file"))

    return log_info

def export_to_csv(log_info, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['dataset', 'log_name', 'n_traces', "change_points"])
        for log_data in log_info:
            csv_writer.writerow(log_data)

if __name__ == "__main__":
    num_cores = 50
    logs_directory = '/work/alexkrau/projects/cdrift-evaluation/EvaluationLogs'  # Change this to the directory containing your XES logs
    output_csv_file = 'log_size_info.csv'

    log_info = count_traces_in_logs(logs_directory)
    export_to_csv(log_info, output_csv_file)
    print(f"CSV file '{output_csv_file}' has been created.")
