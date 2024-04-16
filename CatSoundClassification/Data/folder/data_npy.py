import os
import numpy as np
from pandas import DataFrame
import librosa
from tqdm import tqdm

def get_class_list(sound_dir):
    return os.listdir(sound_dir)

def generator(csv_path, sound_dir, npy_dir):
    """
    data items: filename, filepath, label
    """
    class_list = get_class_list(sound_dir)
    data = {'filename': [], 'filepath': [], 'label': []}
    npy_data = []
    with open(os.path.join(os.path.dirname(csv_path), 'classes.txt'), 'w', encoding='utf-8') as class_file:
        for idx, class_name in enumerate(class_list):
            class_file.write(f"{idx} {class_name}\n")
            class_path = os.path.join(sound_dir, class_name)
            npy_class_path = os.path.join(npy_dir, class_name)
            os.makedirs(npy_class_path, exist_ok=True)  # Create directory if it doesn't exist
            for filename in tqdm(os.listdir(class_path), desc=f"Processing {class_name}"):
                filepath = os.path.join(class_path, filename)
                data['filename'].append(filename)
                # Change the filepath to the corresponding .npy file path
                npy_filename = os.path.splitext(filename)[0] + '.npy'
                npy_filepath = os.path.join(npy_class_path, npy_filename)
                data['filepath'].append(npy_filepath)
                data['label'].append(idx)
                # Load audio file and convert it to numpy array
                audio_data, _ = librosa.load(filepath, sr=None)  # Load audio with its original sample rate
                np.save(npy_filepath, audio_data)  # Save audio data as .npy file
                npy_data.append(npy_filepath)
    dataframe = DataFrame(data=data)
    dataframe.to_csv(csv_path, encoding='utf-8')
    return npy_data

csv_path = r'/Users/ruiyangchen/Desktop/Course/6366/CSCI6366_GWU/CatSoundClassification/Data/scatter/refer.csv'
sound_dir = r'/Users/ruiyangchen/Desktop/Course/6366/CSCI6366_GWU/CatSoundClassification/Data/folder/Sound'
npy_dir = r'/Users/ruiyangchen/Desktop/Course/6366/CSCI6366_GWU/CatSoundClassification/Data/scatter/npy_data'
npy_data = generator(csv_path, sound_dir, npy_dir)
