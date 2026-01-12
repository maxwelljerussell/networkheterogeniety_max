import numpy as np

def is_linear_mode(P, I, r2_min=0.98):
    """
    Determine whether a mode behaves as a linear-increasing mode,
    based only on slope > 0 and high R².
    
    Parameters:
        P       : array-like, pump powers (monotonic increasing)
        I       : array-like, intensities for this mode
        r2_min  : minimum R² for declaring the mode linear

    Returns:
        True / False
        diagnostics : dict(slope, r2, etc.)
    """

    P = np.asarray(P)
    I = np.asarray(I)

    # Prevent division errors for zero-variance I
    if np.allclose(I, I[0]):  # completely flat
        return False, {"reason": "flat"}

    # --- Fit a straight line ---
    A = np.vstack([P, np.ones_like(P)]).T
    slope, intercept = np.linalg.lstsq(A, I, rcond=None)[0]

    I_pred = slope * P + intercept

    ss_res = np.sum((I - I_pred)**2)
    ss_tot = np.sum((I - np.mean(I))**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0

    # --- Decision ---
    is_linear = (slope > 0) and (r2 >= r2_min)

    diagnostics = {
        "slope": slope,
        "r2": r2,
        "I_min": float(np.min(I)),
        "I_max": float(np.max(I)),
    }

    return is_linear, diagnostics