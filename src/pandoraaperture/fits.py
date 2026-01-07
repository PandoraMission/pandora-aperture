"""Mixins for FITS functions. Defines any fits formatting for this package."""

# Standard library
import sys

# Third-party
import astropy.units as u
import numpy as np
import pandas as pd
import pandoraref as pr
import sparse3d
from astropy import __version__ as astropyversion
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from gaiaoffline import __version__ as gaiaofflineversion

from . import __version__, logger
from .docstrings import add_docstring


class FITSMixins:
    @property
    def header_cards(self):
        """Header cards that are necessary to show state of package."""
        cards = [
            (
                "PY_VER",
                sys.version.split("|")[0].strip(),
                "Python version number",
            ),
            ("AP_VER", astropyversion, "astropy version number"),
            ("PA_VER", __version__, "pandoraaperture version number"),
            ("PR_VER", pr.__version__, "pandoraref version number"),
            (
                "S3D_VER",
                sparse3d.__version__,
                "sparse3d version number",
            ),
            ("GO_VER", gaiaofflineversion, "gaiaoffline version"),
        ]
        return [fits.Card(*c) for c in cards]

    @add_docstring(
        parameters=[
            "target",
            "delta_pos",
            "relative_threshold",
            "absolute_threshold",
        ]
    )
    def get_aperture_hdu(
        self,
        target,
        delta_pos=None,
        relative_threshold=0.005,
        absolute_threshold=50,
    ):
        """
        Obtain the aperture HDU.

        Returns
        -------
        hdu: fits.hdu
            FITS HDU containing the aperture, with aperture metrics in header.
        """
        if isinstance(target, SkyCoord):
            c = SkyCoord(self.cat.RA.values, self.cat.Dec.values, unit="deg")
            sep = c.separation(target).min().to(u.deg) / (
                np.mean(np.abs(self.wcs_trimmed.wcs.cdelt)) * u.deg / u.pixel
            )
            if sep > (3 * u.pixel):
                logger.warning(
                    "No matching target in catalog, finding the closest target, try updating the catalog or checking your RA/Dec."
                )
            idx = int(c.separation(target).argmin())
            # target = SkyCoord(self.cat.RA.values[idx], self.cat.Dec.values[idx], unit='deg')
        elif isinstance(target, str):
            loc = self.cat.source_id.isin([target]).values
            if not loc.any():
                logger.warning(
                    "No matching target in catalog, choosing the most central target, try updating the catalog or checking your RA/Dec."
                )
                idx = 0
            else:
                idx = int(np.where(loc)[0][0])
            # target = SkyCoord(self.cat.RA.values[idx], self.cat.Dec.values[idx], unit='deg')
        elif isinstance(target, (int, np.int64)):
            idx = int(target)
        aper, contamination, completeness, total_in_aperture = (
            self.get_aperture(
                target=idx,
                delta_pos=delta_pos,
                relative_threshold=relative_threshold,
                absolute_threshold=absolute_threshold,
            )
        )
        hdu = fits.CompImageHDU(data=aper.astype(np.int16), name="APERTURE")
        hdu.header["CONTAM"] = (
            contamination,
            "Flux not from the target in aperture in e/s",
        )
        hdu.header["COMPLTE"] = (
            completeness,
            "Fraction of flux from the target in aperture",
        )
        hdu.header["TOTAP"] = (
            total_in_aperture,
            "Total flux expected in aperture in e/s",
        )
        for attr in ["RA", "Dec", "row", "column"]:
            hdu.header[attr] = self.cat.iloc[idx][attr]
        hdu.header["GAIA_ID"] = self.cat.iloc[idx]["source_id"]
        hdu.header["IMSIZE0"] = (
            self.prf.imshape[0],
            "Size of the full detector image in ROW",
        )
        hdu.header["IMCRNR0"] = (
            self.prf.imcorner[0],
            "Corner of the image in ROW.",
        )
        hdu.header["IMSIZE1"] = (
            self.prf.imshape[1],
            "Size of the full detector image in COLUMN",
        )
        hdu.header["IMCRNR1"] = (
            self.prf.imcorner[1],
            "Corner of the image in COLUMN.",
        )

        # [hdu.header.append(c) for c in self.wcs.to_header(relax=True).cards]
        return hdu

    @add_docstring(parameters=["delta_pos"])
    def get_catalog_hdu(self, delta_pos=None):
        """
        Obtain the catalog HDU.

        Returns
        -------
        hdu: fits.hdu
            FITS HDU containing the catalog of all sources that will land on pixels.
        """
        with pd.option_context("future.no_silent_downcasting", True):
            cat = self.cat.copy().infer_objects(copy=True)
        if delta_pos is None:
            delta_pos = (0, 0)
        cat["row"] += delta_pos[0]
        cat["column"] += delta_pos[1]
        cat["row0"] = self._get_row0(delta_pos=delta_pos)
        cat["column0"] = self._get_column0(delta_pos=delta_pos)
        tab_hdu = fits.convenience.table_to_hdu(Table.from_pandas(cat))
        tab_hdu.header["EXTNAME"] = "CATALOG"
        return tab_hdu

    @add_docstring(parameters=["delta_pos"])
    def get_prf_hdu(self, delta_pos=None):
        """
        Obtain the PRF HDU.

        Returns
        -------
        hdu: fits.hdu
            FITS HDU containing the  of all sources that will land on pixels.
        """
        A = self.A(delta_pos=delta_pos)
        prf_hdu = fits.CompImageHDU(
            A.subdata.transpose([2, 0, 1]),
            name="PRF_MODEL",
            tile_shape=(1, *A.subshape[:2]),
        )
        if delta_pos is None:
            delta_pos = (0, 0)
        prf_hdu.header["DPOS0"] = (
            delta_pos[0],
            "Change in position in Row from original WCS.",
        )
        prf_hdu.header["DPOS1"] = (
            delta_pos[1],
            "Change in position in Column from original WCS.",
        )
        return prf_hdu

    @property
    def wcs_cards(self):
        """Returns the WCS header as cards."""
        return self.wcs_trimmed.to_header(relax=True).cards

    @add_docstring(parameters=["delta_pos"])
    def get_model_hdu(self, delta_pos=None):
        """
        Obtain the model HDU.

        Returns
        -------
        hdu: fits.hdu
            FITS HDU containing the model of all sources, including their brightness.
        """
        A = self.A(delta_pos=delta_pos)
        image = A.dot(self.flux.value)
        hdu = fits.CompImageHDU(
            image,
            header=self.wcs_trimmed.to_header(relax=True),
            name="MODEL_IMAGE",
        )
        hdu.header["IMSIZE0"] = (
            self.prf.imshape[0],
            "Size of the full detector image in ROW",
        )
        hdu.header["IMCRNR0"] = (
            self.prf.imcorner[0],
            "Corner of the image in ROW.",
        )
        hdu.header["IMSIZE1"] = (
            self.prf.imshape[1],
            "Size of the full detector image in COLUMN",
        )
        hdu.header["IMCRNR1"] = (
            self.prf.imcorner[1],
            "Corner of the image in COLUMN.",
        )

        return hdu
