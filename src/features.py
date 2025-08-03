import numpy as np
from scipy.signal import hilbert

def compute_time_stats(gather: np.ndarray):
    """Mean, std, RMS over all shots and receivers."""
    m = gather.mean()
    s = gather.std()
    r = np.sqrt((gather**2).mean())
    return m, s, r

def compute_envelope_stats(gather: np.ndarray):
    """Mean & max of the analytic‐signal envelope."""
    traces = gather.reshape(-1, gather.shape[2])   # (shots*receivers, time)
    env = np.abs(hilbert(traces, axis=1))
    return env.mean(), env.max()

def compute_spectral_stats(gather: np.ndarray, fs=1000):
    """Dominant frequency & spectral centroid."""
    n_time = gather.shape[1]
    freqs = np.fft.rfftfreq(n_time, d=1/fs)
    specs = np.abs(np.fft.rfft(gather, axis=1))     # (shots, freqs, receivers)
    spec_mean = specs.mean(axis=(2))                # avg over receivers → (shots, freqs)
    dom_freqs    = freqs[np.argmax(spec_mean, axis=1)]
    centroid     = (spec_mean * freqs).sum(axis=1) / spec_mean.sum(axis=1)
    return dom_freqs.mean(), centroid.mean()

def compute_spatial_stats(vp: np.ndarray):
    """Mean gradient & curvature (Laplacian)."""
    gx, gy = np.gradient(vp)
    grad_mean = np.sqrt(gx**2 + gy**2).mean()
    lap = (np.gradient(gx, axis=0) + np.gradient(gy, axis=1))
    return grad_mean, np.abs(lap).mean()

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

