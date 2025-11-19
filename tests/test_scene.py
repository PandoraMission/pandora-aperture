# First-party/Local

# Standard library
import os

# Third-party
import numpy as np
import pytest
from astropy.time import Time

# First-party/Local
import pandoraaperture as pa
from pandoraaperture import DOCSDIR


def test_prf():
    if os.getenv("GITHUB_ACTIONS") == "true":
        pytest.skip("Skipping this test on GitHub Actions.")
    ra, dec, theta = (285.6794224553767, 50.24130600481639, 85.1230985)
    for cls in [pa.SkyScene, pa.ROISkyScene, pa.DispersedSkyScene]:
        if cls is pa.ROISkyScene:
            rR, rC = np.mgrid[-500:500:3j, -500:500:3j]
            ROI_corners = [
                (r + 1024 - 25, c + 1024 - 25)
                for r, c in zip(rR.ravel(), rC.ravel())
            ]
            nROIs = len(ROI_corners)
            ROI_size = (50, 50)
            prf = cls.from_pointing(
                ra,
                dec,
                theta,
                time=Time.now(),
                nROIs=nROIs,
                ROI_size=ROI_size,
                ROI_corners=ROI_corners,
            )
        else:
            prf = cls.from_pointing(ra, dec, theta)
        fig = prf.plot()
        fig.savefig(
            DOCSDIR + f"images/{cls.__name__}.png",
            dpi=150,
            bbox_inches="tight",
        )
        assert len(prf.evaluate()) == 3
