import os

def delete_non_mp3_files(folder_path):
    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            # Check if the file is not an MP3 file
            if not file_name.lower().endswith('.mp3'):
                try:
                    # Attempt to delete the file
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except PermissionError:
                    print(f"Permission denied: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")


# Example usage
folder_path = './Sound'
delete_non_mp3_files(folder_path)
