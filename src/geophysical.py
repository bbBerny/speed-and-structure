import numpy as np
from scipy.signal import hilbert

def compute_metrics(gather):
    m = gather.mean()
    s = gather.std()
    r = np.sqrt((gather**2).mean())
    traces = gather.reshape(-1, gather.shape[1])
    env = np.abs(hilbert(traces, axis=1))
    e_mean, e_max = env.mean(), env.max()
    n_time = gather.shape[1]
    freqs = np.fft.rfftfreq(n_time, d=1.0)
    doms = []
    for shot in range(gather.shape[0]):
        spec = np.abs(np.fft.rfft(gather[shot], axis=0))
        doms.append(freqs[np.argmax(spec.mean(axis=1))])
    dfreq = np.mean(doms)
    return m, s, r, e_mean, e_max, dfreq

def pick_first_breaks(gather, threshold_frac=0.1, method = 'mean'):
    picks = []
    for shot in range(gather.shape[0]):
        mean_trace = gather[shot].mean(axis=1)
        env = np.abs(hilbert(mean_trace))
        thresh = threshold_frac * env.max()
        picks.append(np.argmax(env > thresh))
    return np.mean(picks)

def pick_first_breaks_per_shot(gather, threshold_frac=0.1, method = 'mean'):
    picks = []
    for shot in range(gather.shape[0]):
        mean_trace = gather[shot].mean(axis=1)
        env = np.abs(hilbert(mean_trace))
        thresh = threshold_frac * env.max()
        picks.append(np.argmax(env > thresh))
    return np.array(picks)

def compute_semblance_panel(gather, window=20):
    n_shots, n_time, n_recv = gather.shape
    sem = np.zeros((n_shots, n_time))
    for sh in range(n_shots):
        for t in range(n_time):
            st = max(0, t - window//2)
            en = min(n_time, t + window//2)
            w = gather[sh, st:en, :]
            num = w.sum()**2
            den = w.size * (w**2).sum()
            sem[sh, t] = num/den if den>0 else 0
    return sem