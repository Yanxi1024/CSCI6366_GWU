import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

def audio_preprocess(filepath):

    # Load original audio file
    y, fs = librosa.load(filepath, sr=None, mono=False)
    print(y.shape)
    print(fs)

    # Eliminate mute at the beginning and the end
    yt, index = librosa.effects.trim(y, top_db=30)
    print(index)

    # Display the waveform
    fig, axs = plt.subplots(nrows=2, ncols=1)
    librosa.display.waveshow(y, sr=fs, ax=axs[0], color="blue")
    axs[0].vlines(index[0] / fs, -0.5, 0.5, colors='r')
    axs[0].vlines(index[1] / fs, -0.5, 0.5, colors='r')
    librosa.display.waveshow(yt, sr=fs, ax=axs[1], color="blue")

    plt.show()

    # Split the audio by mute
    intervals = librosa.effects.split(y, top_db=20)
    print(intervals)
    y_remix = librosa.effects.remix(y, intervals)
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
    librosa.display.waveshow(y, sr=fs, ax=axs[0], color="blue")
    librosa.display.waveshow(y_remix, sr=fs, ax=axs[1], offset=intervals[0][0] / fs, color="blue")

    for interval in intervals:
        axs[0].vlines(interval[0] / fs, -0.5, 0.5, colors='r')
        axs[0].vlines(interval[1] / fs, -0.5, 0.5, colors='r')
    plt.show()


    # STFT
    frame_t = 25  # 25ms window length
    hop_length_t = 10  # 10ms step size

    win_length = int(frame_t * fs / 1000)
    hop_length = int(hop_length_t * fs / 1000)
    n_fft = int(2 ** np.ceil(np.log2(win_length)))

    S = np.abs(librosa.stft(yt, n_fft=n_fft, hop_length=hop_length, win_length=win_length))

    plt.imshow(S, origin='lower', cmap='hot')
    plt.show()

    # Take the value of dB as the amplitude
    # Calculate the horizontal and vertical coordinates
    S = librosa.amplitude_to_db(S, ref=np.max)
    D, N = S.shape
    range_D = np.arange(0, D, 20)
    range_N = np.arange(0, N, 20)
    range_t = range_N * (hop_length / fs)
    range_f = range_D * (fs / n_fft / 1000)

    librosa.display.specshow(S, y_axis='linear', x_axis='time', hop_length=hop_length, sr=fs)
    plt.colorbar()
    plt.show()

    # Pre Emphasis (use split audio)
    n_fft_em = 512
    win_length_em = 512
    hop_length_em = 160

    y_filt = librosa.effects.preemphasis(y_remix)
    S_preemp = librosa.stft(y_filt, n_fft=n_fft_em, hop_length=hop_length_em, win_length=win_length_em)
    S_preemp = librosa.amplitude_to_db(np.abs(S_preemp))

    S_split = librosa.stft(y_remix, n_fft=n_fft_em, hop_length=hop_length_em, win_length=win_length_em)
    S_split = librosa.amplitude_to_db(np.abs(S_split))

    fig_em, axs_em = plt.subplots(2, 1, sharex=True, sharey=True)

    librosa.display.specshow(S_split, sr=fs, hop_length=hop_length_em, y_axis='linear', x_axis='time', ax=axs_em[0])
    axs[0].set(title='Original signal')

    img_em = librosa.display.specshow(S_preemp, sr=fs, hop_length=hop_length_em, y_axis='linear', x_axis='time', ax=axs_em[1])
    axs_em[1].set(title='pre-emphasis signal')
    fig_em.colorbar(img_em, ax=axs_em, format="%+2.f dB")
    plt.show()

    # Filter Banks
    # Use Mel-scaled filter banks
    win_length_fb = 512
    hop_length_fb = 160
    n_fft_fb = 512
    n_mels_fb = 128
    melfb = librosa.filters.mel(sr=fs, n_fft=n_fft_fb, n_mels=n_mels_fb, htk=True)
    print(melfb.shape)
    # x_fb = np.arange(melfb.shape[1]) * fs / n_fft_fb
    # fig_fb = plt.figure()
    # plt.plot(x_fb, melfb.T)
    # plt.show()
    fig_fb = plt.figure()
    fbank = librosa.feature.melspectrogram(y=y_remix,
                                           sr=fs,
                                           n_fft=n_fft_fb,
                                           win_length=win_length_fb,
                                           hop_length=hop_length_fb,
                                           n_mels=n_mels_fb)
    print(fbank.shape)
    fbank_db = librosa.power_to_db(fbank, ref=np.max)
    img_fb = librosa.display.specshow(fbank_db, x_axis='time', y_axis='mel', sr=fs, fmax=fs / 2, )
    fig_fb.colorbar(img_fb, format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.show()

    return fbank_db



if __name__ == '__main__':
    file_path = 'cat_single.mp3'
    audio_preprocess(file_path)