import os
import csv

def create_metadata_csv(base_dir, split):
    """
    Creates a CSV file (train_metadata.csv or test_metadata.csv) that maps each
    audio filename to its corresponding speaker ethnicity label. The directory 
    structure for the training data is assumed as:
      base_dir/Train/Asian, base_dir/Train/White
    and for testing data:
      base_dir/Test/Asian, base_dir/Test/White
    
    Args:
        base_dir (str): Path to the base directory (e.g., 'Data/raw').
        split (str): Either 'Train' or 'Test'.
    """
    # Define the subfolders that represent ethnicities
    ethnicities = ['Asian', 'White']
    
    # Prepare the output CSV name based on the split
    csv_filename = f"{split.lower()}_metadata.csv"
    
    # Collect (filename, ethnicity) pairs
    data_rows = []
    
    for ethnicity in ethnicities:
        folder_path = os.path.join(base_dir, split, ethnicity)
        
        # List all files in the subfolder
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                # Check if the file is an audio file (e.g., ends with .wav)
                if file_name.lower().endswith('.wav'):
                    data_rows.append((file_name, ethnicity))
    
    # Write the results to CSV
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['filename', 'ethnicity'])
        # Write data rows
        writer.writerows(data_rows)
    
    print(f"Metadata CSV created: {csv_filename}, total entries: {len(data_rows)}")


if __name__ == "__main__":
    base_directory = r"../Data/raw"
    
    # Create metadata for training split
    create_metadata_csv(base_directory, "Train")
    
    # Create metadata for testing split
    create_metadata_csv(base_directory, "Test")
