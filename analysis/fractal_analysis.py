
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend
from scipy import stats

# ------------------------------------------
# Higuchi Fractal Dimension (HFD)
# ------------------------------------------
def higuchi_fd(signal, k_max=10):
    N = len(signal)
    Lk = []

    for k in range(1, k_max + 1):
        Lmk = []
        for m in range(k):
            Lm = 0
            n_max = int(np.floor((N - m - 1) / k))
            for i in range(1, n_max):
                Lm += abs(signal[m + i * k] - signal[m + (i - 1) * k])
            Lm = (Lm * (N - 1)) / (k * n_max * k)
            Lmk.append(Lm)
        Lk.append(np.mean(Lmk))

    lnLk = np.log(Lk)
    lnk = np.log(1.0 / np.arange(1, k_max + 1))

    # Linear regression to estimate slope = fractal dimension
    slope, intercept, r_value, p_value, std_err = stats.linregress(lnk, lnLk)
    return slope

# ------------------------------------------
# Katz Fractal Dimension (KFD)
# ------------------------------------------
def katz_fd(signal):
    L = np.sum(np.abs(np.diff(signal)))  # Total length
    d = np.max(np.abs(signal - signal[0]))  # Diameter
    n = len(signal)
    if d == 0 or n <= 1:
        return 0
    return np.log10(n) / (np.log10(n) + np.log10(d / L))

# ------------------------------------------
# Detrended Fluctuation Analysis (DFA)
# ------------------------------------------
def dfa_fd(signal, scale_min=4, scale_max=16):
    signal = np.array(signal)
    X = np.cumsum(signal - np.mean(signal))  # Integrated profile
    N = len(X)
    scales = np.arange(scale_min, min(scale_max + 1, N // 4))
    
    if len(scales) == 0:
        return 1.0  # Default value for very short signals
    
    F = []

    for s in scales:
        shape = (N // s, s)
        split = np.reshape(X[:shape[0] * s], shape)
        local_rms = []
        for segment in split:
            poly = np.polyfit(np.arange(s), segment, 1)  # Linear trend
            detrended = segment - np.polyval(poly, np.arange(s))
            local_rms.append(np.sqrt(np.mean(detrended ** 2)))
        F.append(np.mean(local_rms))

    if len(F) < 2:
        return 1.0
        
    log_scales = np.log(scales)
    log_F = np.log(F)

    # Linear regression to estimate slope = alpha (DFA exponent)
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_scales, log_F)
    return slope

def dfa(signal, scale_min=4, scale_max=16):
    """Alias for backward compatibility"""
    return dfa_fd(signal, scale_min, scale_max)
