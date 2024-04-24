import numpy as np
import librosa

def audio_extraction(filepath, target_duration=5.0, target_sr=44100):
    # Load original audio file
    y, fs = librosa.load(filepath, sr=None, mono=False)
    yt = y

    # Calculate target length
    target_length = int(target_duration * target_sr)

    # Check if the audio length is less than the target length
    if len(y) < target_length:
        # Calculate the amount of mute audio to add at the beginning and end
        padding_length_l = (target_length - len(y)) // 2
        padding_length_r = target_length - len(y) - padding_length_l
        # Create mute audio with the same length as the padding
        padding_audio_l = np.zeros(padding_length_l)
        padding_audio_r = np.zeros(padding_length_r)

        # Add the mute audio at the beginning and end of the original audio
        y_padded = np.concatenate((padding_audio_l, y, padding_audio_r))

        # Update the original audio and sample rate
        yt = y_padded
        fs = target_sr

    # Check if the audio length is greater than the target length
    elif len(y) > target_length:
        # Eliminate mute at the beginning and the end
        yt, _ = librosa.effects.trim(y, top_db=20)

        # If the trimmed audio is still longer than the target length, trim from both ends
        if len(yt) > target_length:
            trim_length = (len(yt) - target_length) // 2
            yt = yt[trim_length:target_length + trim_length]

        # If the trimmed audio is shorter than the target length, pad to the target duration
        elif len(yt) < target_length:
            padding_length_l = (target_length - len(yt)) // 2
            padding_length_r = target_length - len(yt) - padding_length_l
            padding_audio_l = np.zeros(padding_length_l)
            padding_audio_r = np.zeros(padding_length_r)
            yt = np.concatenate((padding_audio_l, yt, padding_audio_r))

    # check the output audio shape
    if yt.shape != (220500,):
        print(yt.shape)

    # Filter Banks
    # Use Mel-scaled filter banks
    win_length_fb = 512
    hop_length_fb = 160
    n_fft_fb = 512
    n_mels_fb = 128

    fbank = librosa.feature.melspectrogram(y=yt,
                                           sr=fs,
                                           n_fft=n_fft_fb,
                                           win_length=win_length_fb,
                                           hop_length=hop_length_fb,
                                           n_mels=n_mels_fb)

    fbank_db = librosa.power_to_db(fbank, ref=np.max)

    return fbank_db

if __name__ == '__main__':
    file_path = 'cat_long.mp3'
    arr = audio_extraction(file_path)
    print(arr.shape)