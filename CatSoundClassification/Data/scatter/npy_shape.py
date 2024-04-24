import numpy as np

def check_npy_shape(file_path):
    try:
        # Load the .npy file
        data = np.load(file_path)

        # Check the shape of the loaded data
        shape = data.shape

        print("Shape of the .npy file:", shape)
    except FileNotFoundError:
        print("File not found.")
    except Exception as e:
        print("An error occurred:", e)

# Example usage:
if __name__ == "__main__":
    file_path = "./npy_data/Angry/cat04_aug1(1).npy"  # Replace this with the path to your .npy file
    check_npy_shape(file_path)
