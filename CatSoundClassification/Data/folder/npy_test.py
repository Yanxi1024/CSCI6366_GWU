import numpy as np
from CatSoundClassification.Data.folder.audio_preprocess import audio_preprocess

if __name__ == '__main__':
    filepath = 'cat_single.mp3'
    npy_filepath = './car_single.npy'
    # Process audio
    fbank_db = audio_preprocess(filepath)
    # Reshape audio data to the appropriate shape
    # audio_data = np.expand_dims(audio_data, axis=0)  # Add batch dimension
    # audio_data = np.expand_dims(audio_data, axis=0)  # Add channel dimension

    np.save(npy_filepath, fbank_db)  # Save audio data as .npy file
