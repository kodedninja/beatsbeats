import torch as th
import models
import numpy as np
from scipy.signal import find_peaks
import data
import mir_eval
import pickle
import glob
from os import path
import json

device = 'cuda' if th.cuda.is_available() else 'cpu'

with open('dataset-stats.pkl', 'rb') as f:
    stats = pickle.load(f)
    mean = th.Tensor(stats['mean']).to(device)
    std = th.Tensor(stats['std']).to(device)

@th.no_grad()
def onset_signal(model, x):
    model = model.to(device)
    model.eval()

    x = x.to(device)
    x = (x - mean)/std
    x = x.unsqueeze(0)
    
    out = th.sigmoid(model(x)).detach().cpu()
    out = np.convolve(out[0], np.hamming(5))

    return out

def onsets(onset_signal):
    res = []
    for idx in find_peaks(onset_signal)[0]:
        if onset_signal[idx] >= 0.95:
            res.append(idx * data.HOP_LENGTH / data.SAMPLE_RATE)

    return np.array(res)

def evaluate_onsets(model, X, y):
    f_scores = []
    for idx, x in enumerate(X):
        pred = onsets(onset_signal(model, x))
        f, _, _ = mir_eval.onset.f_measure(y[idx], pred, window=0.05)
        f_scores.append(f)
        
    return f_scores

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

    preds = [onsets(onset_signal(model, X)) for X in X_submission]

    for idx, onsets in enumerate(preds):
        name = f'test{idx + 1}'
        
        if name in solution:
            solution[name]['onsets'] = list(onsets)
        else:
            solution[name] = {'onsets': list(onsets)}

    with open('final_predictions.json', 'w') as f:
        json.dump(solution, f)