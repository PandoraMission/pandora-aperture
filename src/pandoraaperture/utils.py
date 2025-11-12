"""Utility functions"""

# Third-party
import numpy as np


def interpfunc(l, lp, PSF0):
    """Interpolation function.
    Given a grid of points l and a desired point lp will interpolate n dimensional PSF0.
    Grid is always assumed to be the last dimension."""
    if l in lp:
        PSF1 = PSF0[:, :, np.where(lp == l)[0][0]]
    elif l < lp[0]:
        PSF1 = PSF0[:, :, 0]
    elif l > lp[-1]:
        PSF1 = PSF0[:, :, -1]
    else:
        # Find the two closest frames
        d = np.argsort(np.abs(lp - l))[:2]
        d = d[np.argsort(lp[d])]
        # Linearly interpolate
        slope = (PSF0[:, :, d[0]] - PSF0[:, :, d[1]]) / (lp[d[0]] - lp[d[1]])
        PSF1 = PSF0[:, :, d[1]] + (slope * (l - lp[d[1]]))
    return PSF1
