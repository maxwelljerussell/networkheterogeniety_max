import numpy as np

def build_discrete_mode_matrix(lasing_modes_list):
    """
    Build matrix using raw discrete modes.
    """

    all_freqs = []
    for lm in lasing_modes_list:
        spectrum = lm.get_spectrum(only_lasing_modes=True)
        all_freqs.extend(spectrum.frequencies)
    
    global_freqs = np.unique(np.round(all_freqs, 6))
    n_patterns = len(lasing_modes_list)
    n_modes = len(global_freqs)

    X = np.zeros((n_patterns, n_modes))

    for i, lm in enumerate(lasing_modes_list):
        spectrum = lm.get_spectrum(only_lasing_modes=True)
        for f, I in zip(spectrum.frequencies, spectrum.intensities):
            idx = np.argmin(np.abs(global_freqs - f))
            X[i, idx] = I

    return global_freqs, X