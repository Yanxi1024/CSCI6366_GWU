import os
import numpy as np
from pandas import DataFrame
from librosa.core import load
from tqdm import tqdm

def get_class_list(sound_dir):
    return os.listdir(sound_dir)

# Unify the length of the audio to make the tensor size equal
def process_audio(filepath, target_duration, target_sr):
    # Load audio file
    audio_data, sr = load(filepath, sr=None)
    # Trim or pad audio to target duration
    target_length = int(target_duration * target_sr)
    if len(audio_data) > target_length:
        audio_data = audio_data[:target_length]
    elif len(audio_data) < target_length:
        audio_data = np.pad(audio_data, (0, target_length - len(audio_data)), mode='constant')
    return audio_data, sr


def generator(csv_path, sound_dir, npy_dir, target_duration=5.0, target_sr=44100):
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
                if "aug1(1)" in filename:
                    continue  # Skip files with "aug1(1)" in the filename
                filepath = os.path.join(class_path, filename)
                data['filename'].append(filename)
                # Change the filepath to the corresponding .npy file path
                npy_filename = os.path.splitext(filename)[0] + '.npy'
                npy_filepath = os.path.join(npy_class_path, npy_filename)
                data['filepath'].append(npy_filepath)
                data['label'].append(idx)
                # Process audio file to ensure fixed duration
                audio_data, sr = process_audio(filepath, target_duration, target_sr)
                # Reshape audio data to the appropriate shape
                audio_data = np.expand_dims(audio_data, axis=0)  # Add batch dimension
                audio_data = np.expand_dims(audio_data, axis=0)  # Add channel dimension
                np.save(npy_filepath, audio_data)  # Save audio data as .npy file
                npy_data.append(npy_filepath)
    dataframe = DataFrame(data=data)
    dataframe.to_csv(csv_path, encoding='utf-8')
    return npy_data

csv_path = r'/Users/ruiyangchen/Desktop/Course/6366/CSCI6366_GWU/CatSoundClassification/Data/scatter/refer.csv'
sound_dir = r'/Users/ruiyangchen/Desktop/Course/6366/CSCI6366_GWU/CatSoundClassification/Data/folder/Sound'
npy_dir = r'/Users/ruiyangchen/Desktop/Course/6366/CSCI6366_GWU/CatSoundClassification/Data/scatter/npy_data'
target_duration = 5.0  # Set the target duration in seconds
target_sr = 44100  # Set the target sample rate to 44.1kHz
npy_data = generator(csv_path, sound_dir, npy_dir, target_duration)
