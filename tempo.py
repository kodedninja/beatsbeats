import torch as th
import numpy as np
from scipy.signal import find_peaks
import json
import glob
from os import path

import models
import data
import onsets

def autocorrelate(signal, tao):
    r = np.zeros(len(signal) - tao)
    for t in range(len(signal) - tao):
        r[t] = signal[t + tao] * signal[t]
    return np.sum(r)

def to_bpm(max_r):
    return 60 * data.SAMPLE_RATE / data.HOP_LENGTH / (max_r + 25)

def autocorrelate_tao(signal, min_tao=25, max_tao=87):
    return np.array([autocorrelate(signal, tao) for tao in range(min_tao, max_tao)])

def tempo(model, x, onset_signal=None, top_n=2):
    onset_signal = onset_signal if onset_signal is not None else onsets.onset_signal(model, x)

    taos = autocorrelate_tao(onset_signal)
    peaks = find_peaks(taos)[0]
    highest_peaks = np.argsort(-taos[peaks])[:top_n]

    return list(reversed([to_bpm(r) for r in peaks[highest_peaks]]))

if __name__ == '__main__':
    model = models.Resi(3)
    model.load_state_dict(th.load('resi.pt'))

    X_files_submission = glob.glob(path.join('data/test', '*.wav'))
    X_submission, sample_rates_submission = data.preprocess_audio(X_files_submission)

    try:
        with open('final_predictions.json', 'r') as f:
            solution = json.load(f)
    except IOError:
        solution = {}

    tempos = [tempo(model, x) for x in X_submission]

    for idx, t in enumerate(tempos):
        name = f'test{idx + 1}'

        if name in solution:
            solution[name]['tempo'] = t
        else:
            solution[name] = {'tempo': t}

    with open('final_predictions.json', 'w') as f:
        json.dump(solution, f)

