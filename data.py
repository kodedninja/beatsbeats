import torch as th
import torchaudio
from tqdm import tqdm
import numpy as np

FRAME_LENGTH = 15
FRAME_HALF = 7
HOP_LENGTH = 512
WIN_LENGTHS = [1024, 2048, 4096]
SAMPLE_RATE = 44100

_transforms =[torchaudio.transforms.MelSpectrogram(SAMPLE_RATE, n_fft=wl, hop_length=HOP_LENGTH, n_mels=80, f_min=27.5, f_max=16000) for wl in WIN_LENGTHS]

def make_frames(X, y, onsets, sample_rate):
    X_frames, y_frames = [], []
    
    for onset_time in onsets:
        onset_idx = int(seconds_to_bins(onset_time, sample_rate))
        
        start = max(0, onset_idx - FRAME_LENGTH//2)
        end = min(onset_idx + FRAME_LENGTH//2 + 1, X.shape[2] - FRAME_LENGTH)
        
        idx = start
        while idx < end:
            X_frames.append(X[:, :, idx:idx+FRAME_LENGTH])
            y_frames.append(y[idx:idx+FRAME_LENGTH])
            idx += 1

    return X_frames, y_frames

def make_target(onsets, length, sample_rate):
    y = th.zeros(length)

    for x in onsets:
        x_t = int(seconds_to_bins(x, sample_rate))
        y[x_t] = 1
        
        if x_t - 1 >= 0 and y[x_t - 1] != 1:
            y[x_t - 1] = 0.25
            
        if x_t + 1 < length and y[x_t + 1] != 1:
            y[x_t + 1] = 0.25

    return y

def seconds_to_bins(a, sample_rate):
    return a * sample_rate / HOP_LENGTH

def mel(waveform):
    mel_specs = [transform(waveform) for transform in _transforms]    
    return th.log10(th.stack(mel_specs) + 1e-08)

def load_onsets(file_path):
    with open(file_path, 'r') as f:
        onsets = list(map(float, f.read().split()))
    return np.array(onsets)

def preprocess_audio(files):
    spectograms = []
    sample_rates = []

    for file_path in tqdm(files):
        waveform, sample_rate = torchaudio.load(file_path, normalize=True)
        mel_specgram = mel(waveform[0])
        spectograms.append(mel_specgram)
        sample_rates.append(sample_rate)

    return spectograms, sample_rates

class AudioOnsetDataset(th.utils.data.Dataset):
    def __init__(self, spectograms, sample_rates, targets, sample_onsets):
        self.X = []
        self.y = []

        for X, sample_rate, y, onsets in zip(spectograms, sample_rates, targets, sample_onsets):            
            X_frames, y_frames = make_frames(X, y, onsets, sample_rate)
            
            self.X += X_frames
            self.y += y_frames
    
        tmp = th.cat(self.X)
        self.mean = th.mean(tmp, dim=(0, 2)).unsqueeze(1)
        self.std = th.std(tmp, dim=(0, 2)).unsqueeze(1)
        del tmp
        
        self.X = [(x - self.mean)/self.std for x in self.X]
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, i):
        return self.X[i], self.y[i]