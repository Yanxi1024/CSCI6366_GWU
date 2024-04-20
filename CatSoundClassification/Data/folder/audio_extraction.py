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
        padding_length = (target_length - len(y)) // 2
        # Create mute audio with the same length as the padding
        padding_audio = np.zeros(padding_length)

        # Add the mute audio at the beginning and end of the original audio
        y_padded = np.concatenate((padding_audio, y, padding_audio))

        # Update the original audio and sample rate
        yt = y_padded
        fs = target_sr

    # Check if the audio length is greater than the target length
    elif len(y) > target_length:
        # Trim the audio using librosa.effects.trim to match the target length
        yt, _ = librosa.effects.trim(y, top_db=30, frame_length=target_length, hop_length=target_length)

    # Feature Banks
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