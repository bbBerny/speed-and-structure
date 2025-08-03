import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def plot_silhouette(X, labels, k):
    import matplotlib.cm as cm
    from sklearn.metrics import silhouette_samples
    sil_vals = silhouette_samples(X, labels)
    fig, ax = plt.subplots(figsize=(6,4))
    y_lower = 10
    for i in range(k):
        ith_vals = np.sort(sil_vals[labels==i])
        y_upper = y_lower + len(ith_vals)
        color = cm.nipy_spectral(float(i) / k)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_vals, facecolor=color, alpha=0.7)
        ax.text(-0.02, y_lower + 0.5*len(ith_vals), str(i))
        y_lower = y_upper + 10
    sil_avg = silhouette_score(X, labels)
    ax.axvline(sil_avg, color='k', linestyle='--')
    ax.set_xlabel("Silhouette Coefficient")
    ax.set_ylabel("Cluster")
    ax.set_title(f"Silhouette Plot (k={k}, avg={sil_avg:.2f})")
    plt.show()
    return sil_avg

def compute_gap_statistic(X, ks=range(1,11), n_refs=10):
    """
    Compute the Gap statistic for an array of cluster counts.
    """
    gaps = np.zeros(len(ks))
    sk = np.zeros(len(ks))
    n_samples, n_dims = X.shape
    # Bounding box of original data
    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)
    
    for i, k in enumerate(ks):
        # Reference dispersions
        ref_disps = np.zeros(n_refs)
        for j in range(n_refs):
            # Uniform reference
            X_ref = np.random.uniform(mins, maxs, size=(n_samples, n_dims))
            km = KMeans(n_clusters=k, random_state=j).fit(X_ref)
            ref_disps[j] = km.inertia_
        # Log dispersions
        log_ref = np.log(ref_disps)
        # Dispersion for original data
        km = KMeans(n_clusters=k, random_state=0).fit(X)
        orig_disp = km.inertia_
        
        gaps[i] = np.mean(log_ref) - np.log(orig_disp)
        sk[i] = np.sqrt(np.var(log_ref) * (1 + 1.0/n_refs))
    return gaps, sk