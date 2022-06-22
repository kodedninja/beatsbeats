# Based on https://www.ee.columbia.edu/~dpwe/pubs/Ellis07-beattrack.pdf

import torch as th
import numpy as np
import scipy.signal as signal
import glob
from os import path
import json

import models
import data
import onsets
import tempo

def _normalize(onsets):
    norm = onsets.std(ddof=1)
    if norm > 0:
        onsets = onsets / norm
    return onsets

def _beat_signal(onset_signal, period):
    """Convolve the onset signal with a given period"""
    window = np.exp(-0.5 * (np.arange(-period, period + 1) * 32.0 / period) ** 2)
    return signal.convolve(_normalize(onset_signal), window, "same")

def _beats_dp(onset_signal, period, tightness=680):
    """Based on the paper Beat Tracking by Dynamic Programming by Daniel P.W. Ellis"""

    signal = _beat_signal(onset_signal, period)

    backlink = -np.ones_like(signal, dtype=int)
    cumulative_score = signal.copy()

    # search range for previous beat
    prange = np.arange(-2 * period, -np.round(period / 2), dtype=int)
    # log-gaussian window over search range
    txcost = -tightness * (np.log(-prange / period) ** 2)

    for i in range(max(-prange + 1), len(signal)):
        timerange = i + prange

        candidates = txcost + cumulative_score[timerange]
        beat_location = np.argmax(candidates)

        cumulative_score[i] = candidates[beat_location] + signal[i]
        backlink[i] = timerange[beat_location]

    # backtrace
    beats = [np.argmax(cumulative_score)]

    while backlink[beats[0]] > 0:
        beats = [backlink[beats[0]]] + beats

    return beats

def beats(X):
    onset_signal = onsets.onset_signal(model, X)

    bpm = tempo.tempo(None, None, onset_signal=onset_signal, top_n=1)[0]
    period = round(60.0 * data.SAMPLE_RATE / data.HOP_LENGTH / bpm)
    beats_bins = _beats_dp(onset_signal, period)

    return [(bin * data.HOP_LENGTH / data.SAMPLE_RATE) for bin in beats_bins]


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

    preds = [beats(x) for x in X_submission]

    for idx, b in enumerate(preds):
        name = f'test{idx + 1}'

        if name in solution:
            solution[name]['beats'] = b
        else:
            solution[name] = {'beats': b}

    with open('final_predictions.json', 'w') as f:
        json.dump(solution, f)

