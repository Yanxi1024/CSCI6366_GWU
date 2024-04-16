import os


def check_npy_file_size(folder_path, target_size):
    different_files = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.npy'):
            file_path = os.path.join(folder_path, file_name)
            file_size = os.path.getsize(file_path)
            if file_size != target_size:
                different_files.append(file_name)

    if different_files:
        print("The following files have a different size:")
        for file_name in different_files:
            print(file_name)
    else:
        print("All .npy files have the correct size.")


folder_path = "/Users/ruiyangchen/Desktop/Course/6366/CSCI6366_GWU/CatSoundClassification/Data/scatter/npy_data/Happy"
target_size = 882128  # bytes

check_npy_file_size(folder_path, target_size)


