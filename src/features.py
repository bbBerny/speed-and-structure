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
