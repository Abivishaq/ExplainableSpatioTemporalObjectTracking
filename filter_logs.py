import torch
import os

def filter_logs(log_folder):
    """Iterates through all log files in a folder and prints those with only one key in predicted_movements and influential_movements."""
    log_files = [f for f in os.listdir(log_folder) if f.endswith(".pt")]
    sorted_log_files = []
    for i in range(1,len(log_files)+1):
        sorted_log_files.append(f"log_{i}.pt")
    log_files = sorted_log_files
    
    matching_logs = []
    
    for log_file in log_files:
        log_path = os.path.join(log_folder, log_file)
        log_data = torch.load(log_path, map_location=torch.device('cpu'))
        
        if isinstance(log_data, list) and len(log_data) > 2:
            predicted_movements = log_data[1]
            influential_movements = log_data[2]
            
            if len(predicted_movements) == 1 and len(influential_movements) == 1:
                matching_logs.append(log_file)
    
    print("Log files with only one key in predicted_movements and influential_movements:")
    for log in matching_logs:
        print(log)
    print("matching_logs: ", len(matching_logs))
    print(matching_logs)

if __name__ == "__main__":
    log_folder = "logs/run_7"  # Change this to the appropriate log directory
    filter_logs(log_folder)
