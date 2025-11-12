# Standard library
import warnings
from dataclasses import dataclass
from typing import List, Tuple

# Third-party
import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import Distance, SkyCoord
from astropy.time import Time
from astropy.wcs import WCS
from gaiaoffline import Gaia
from sparse3d import Sparse3D, stack

from . import NIRDAReference, VISDAReference, config
from .docstrings import add_docstring
from .fits import FITSMixins
from .prf import PRF, DispersedPRF, SpatialPRF


@add_docstring(parameters=["prf", "wcs", "time", "user_cat"])
@dataclass
class SkyScene(FITSMixins):
    """Helper that takes astronomy catalogs and makes them a scene"""

    prf: PRF
    wcs: WCS
    time: Time
    user_cat: pd.DataFrame = None

    def __repr__(self):
        return "SkyScene"

    @add_docstring(parameters=["cat"])
    def _clean_catalog(self, cat):
        """Hidden method that returns a cleaned version of a catalog."""
        k = (cat.row.values > (self.prf.imcorner[0] - self.pixel_buffer)) & (
            cat.row.values
            < (self.prf.imcorner[0] + self.prf.imshape[0] + self.pixel_buffer)
        )
        k &= (
            cat.column.values > (self.prf.imcorner[1] - self.pixel_buffer)
        ) & (
            cat.column.values
            < (self.prf.imcorner[1] + self.prf.imshape[1] + self.pixel_buffer)
        )
        return cat[k].reset_index(drop=True)

    @add_docstring(parameters=["coord", "radius"], returns=["cat"])
    def _get_catalog_from_radec(self, coord, radius: float = 1):
        """Function to obtain a catalog of sources relevant to this SkyScene"""
        if isinstance(coord, SkyCoord):
            coord = coord
        elif isinstance(coord, (tuple, list)) and len(coord) == 2:
            ra, dec = coord
            coord = SkyCoord(ra * u.deg, dec * u.deg)
        else:
            raise TypeError("`coord` must be SkyCoord or (ra, dec) tuple")
        radius = u.Quantity(radius, u.deg)
        with Gaia(photometry_output="flux", tmass_crossmatch=True) as gaia:
            df = gaia.conesearch(coord.ra.deg, coord.dec.deg, radius.value)
            # The integers are too hard to coerce everywhere
            df["source_id"] = df.source_id.astype(str)
        if self.user_cat is not None:
            df = pd.concat([df, self.user_cat])
        if len(df) == 0:
            return pd.DataFrame(
                columns=["RA", "Dec", *self.cols, "row", "column"]
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cat_coord = SkyCoord(
                ra=df["ra"].values * u.deg,
                dec=df["dec"].values * u.deg,
                pm_ra_cosdec=df.pmra.fillna(0).values * u.mas / u.year,
                pm_dec=df.pmdec.fillna(0).values * u.mas / u.year,
                obstime=Time.strptime("2016", "%Y"),
                distance=Distance(
                    parallax=df.parallax.fillna(0).values * u.mas,
                    allow_negative=True,
                ),
                radial_velocity=df.radial_velocity.fillna(0).values
                * u.km
                / u.s,
            ).apply_space_motion(self.time)

        col, row = self.wcs.world_to_pixel(cat_coord)
        cat = pd.DataFrame(
            np.asarray(
                [
                    cat_coord.ra.deg,
                    cat_coord.dec.deg,
                    *[df[col].values for col in self.cols],
                    row,
                    col,
                ]
            ).T,
            columns=["RA", "Dec", *self.cols, "row", "column"],
        )

        return self._clean_catalog(cat)

    @add_docstring(parameters=["imcorner", "imshape"], returns=["cat"])
    def _get_catalog_from_pixelbox(self, imcorner, imshape):
        center = (imcorner[0] + imshape[0] / 2, imcorner[1] + imshape[1] / 2)
        c = self.wcs.pixel_to_world(*center[::-1])
        r1, r2 = (
            imcorner[0] - self.pixel_buffer,
            imcorner[0] + imshape[0] + self.pixel_buffer,
        )
        c1, c2 = (
            imcorner[1] - self.pixel_buffer,
            imcorner[1] + imshape[1] + self.pixel_buffer,
        )

        # add buffer here for DispersedPRF
        if isinstance(self.prf, DispersedPRF):
            r1 += self.prf.y.min()
            r2 += self.prf.y.max()
            c1 += self.prf.x.min()
            c2 += self.prf.x.max()
        radius = np.max(
            self.wcs.pixel_to_world(
                [c1, c1, c2, c2], [r1, r2, r1, r2]
            ).separation(c)
        )

        return self._get_catalog_from_radec(c, radius=radius.deg)

    @add_docstring(parameters=["location", "gradients"])
    def _get_sparse_matrix(self, location, gradients=True):
        """Hidden method to get a sparse matrix for a particular location."""
        return self.prf.to_sparse3d(location, gradients=gradients)

    def _get_X(self):
        """Hidden method to obtain the PRF matrices."""
        cat = self._get_catalog_from_pixelbox(
            self.prf.imcorner, self.prf.imshape
        )
        X, dX0, dX1 = [], [], []
        for r, c in cat[["row", "column"]].values:
            x, dx0, dx1 = self._get_sparse_matrix((r, c))
            X.append(x)
            dX0.append(dx0)
            dX1.append(dx1)
        if len(X) == 0:
            return None, None, None, cat
        X, dX0, dX1 = stack(X), stack(dX0), stack(dX1)
        return X, dX0, dX1, cat

    def _check_user_cat(self):
        """Hidden function to verify the `user_cat` has the right columns."""
        if self.user_cat is not None:
            if not isinstance(self.user_cat, pd.DataFrame):
                raise ValueError("`user_cat` must be a `pandas.DataFrame`.")
            for attr in ["ra", "dec", *self.cols]:
                if attr not in self.user_cat.columns:
                    raise ValueError(
                        f"`user_cat` must have the column `{attr}`"
                    )
        return

    def __post_init__(self):
        self.pixel_buffer = int(config["SETTINGS"]["pixel_buffer"])
        self.cols = config["SETTINGS"]["catalog_columns"].split(", ")
        self._check_user_cat()
        self.X, self.dX0, self.dX1, self.cat = self._get_X()

    @add_docstring(parameters=["delta_pos"], returns=["A"])
    def A(self, delta_pos=None):
        """Returns the design matrix of the SkyScene."""
        if self.X is None:
            return None
        if delta_pos is None:
            return self.X
        jitterint = tuple(np.round(delta_pos).astype(int))
        jitterdec = np.asarray(delta_pos) - np.asarray(jitterint)
        return self.X._new_s3d(
            new_data=self.X.subdata
            + self.dX0.subdata * -jitterdec[0]
            + self.dX1.subdata * -jitterdec[1],
            new_row=self.X.subrow + jitterint[0],
            new_col=self.X.subcol + jitterint[1],
        )

    def _get_VDAflux(self, cat):
        """Gives the flux on the VDA. This can be updated with a reference product in the future...!"""
        # This is approximately the right flux for the VDA in electrons per second
        return cat.phot_bp_mean_flux.values * 0.9 * u.electron / u.second

    def _get_NIRDAflux(self, cat):
        """Gives the flux on the NIRDA. This can be updated with a reference product in the future...!"""
        # This is approximately the right flux for the NIRDA in electrons per second
        return cat.j_flux.values * 1 * u.electron / u.second

    @property
    def VDAflux(self):
        """Gives the flux on the VDA. This can be updated with a reference product in the future...!"""
        # This is approximately the right flux for the VDA in electrons per second
        return self._get_VDAflux(self.cat)

    @property
    def NIRDAflux(self):
        """Gives the flux on the NIRDA. This can be updated with a reference product in the future...!"""
        # This is approximately the right flux for the NIRDA in electrons per second
        return self._get_NIRDAflux(self.cat)

    @property
    def flux(self):
        """Here we set the flux that is assumed to be the default for this object. For regular sky scenes it's VDA."""
        return self.VDAflux

    @classmethod
    @add_docstring(parameters=["ra", "dec", "theta", "time"])
    def from_pointing(cls, ra, dec, theta, time=Time.now()):
        wcs = VISDAReference().get_wcs(
            target_ra=ra, target_dec=dec, theta=theta
        )
        prf = SpatialPRF.from_reference("VISDA")
        return cls(prf=prf, wcs=wcs, time=time)


@add_docstring(parameters=["prf", "wcs", "time", "user_cat"])
@dataclass()
class DispersedSkyScene(SkyScene):
    """Special version of a SkyScene that works with dispersed PRFs"""

    @add_docstring(parameters=["cat"])
    def _clean_catalog(self, cat):
        """Hidden method that returns a cleaned version of a catalog."""
        length = self.prf.y.max() - self.prf.y.min()
        k = (
            cat.row.values
            > (self.prf.imcorner[0] - length - self.pixel_buffer)
        ) & (
            cat.row.values
            < (
                self.prf.imcorner[0]
                + self.prf.imshape[0]
                + length
                + self.pixel_buffer
            )
        )
        length = self.prf.x.max() - self.prf.x.min()
        k &= (
            cat.column.values
            > (self.prf.imcorner[1] - length - self.pixel_buffer)
        ) & (
            cat.column.values
            < (
                self.prf.imcorner[1]
                + self.prf.imshape[1]
                + length
                + self.pixel_buffer
            )
        )
        k &= self._get_NIRDAflux(cat) > (1000 * u.electron / u.second)
        return cat[k].reset_index(drop=True)

    def _get_X(self):
        """Hidden method to obtain the PRF matrices."""
        cat = self._get_catalog_from_pixelbox(
            self.prf.imcorner, self.prf.imshape
        )
        R, C = np.mgrid[
            self.prf.imcorner[0] : self.prf.imcorner[0] + self.prf.imshape[0],
            self.prf.imcorner[1] : self.prf.imcorner[1] + self.prf.imshape[1],
        ]
        X, dX0, dX1 = [], [], []
        for r, c in cat[["row", "column"]].values:
            x, dx0, dx1 = self._get_sparse_matrix((r, c))
            x, dx0, dx1 = [
                Sparse3D(
                    data=a.dot(self._spectrum_norm)[:, :, None],
                    row=R[:, :, None],
                    col=C[:, :, None],
                    imshape=self.prf.imshape,
                    imcorner=self.prf.imcorner,
                )
                for a in [x, dx0, dx1]
            ]
            X.append(x)
            dX0.append(dx0)
            dX1.append(dx1)
        if len(X) == 0:
            return None, None, None, cat
        X, dX0, dX1 = stack(X), stack(dX0), stack(dX1)
        return X, dX0, dX1, cat

    def __post_init__(self):
        if not isinstance(self.prf, DispersedPRF):
            raise ValueError("Must pass `DispersedPRF`.")
        self.pixel_buffer = int(config["SETTINGS"]["pixel_buffer"])
        self.cols = config["SETTINGS"]["catalog_columns"].split(", ")
        self._spectrum_norm = (
            NIRDAReference.get_spectrum_normalization_per_pixel(self.prf.y)
        )
        self._spectrum_norm /= np.trapz(self._spectrum_norm, self.prf.y)
        self._check_user_cat()
        self.X, self.dX0, self.dX1, self.cat = self._get_X()

    @property
    def flux(self):
        """Here we set the flux that is assumed to be the default for this object. For dispersed sky scenes it's NIRDA."""
        return self.NIRDAflux


@add_docstring(
    parameters=[
        "prf",
        "wcs",
        "time",
        "user_cat",
        "nROIs",
        "ROI_size",
        "ROI_corners",
    ]
)
@dataclass()
class ROISkyScene(SkyScene):
    """Special version of a SkyScene that works with a ROI sparse matrix"""

    nROIs: int = 1
    ROI_size: Tuple = (50, 50)
    ROI_corners: List[Tuple[int, int]] = (1024 - 25, 1024 - 25)

    def __repr__(self):
        return "ROISkyScene"

    @add_docstring(parameters=["cat"])
    def _clean_catalog(self, cat):
        """Hidden method that returns a cleaned version of a catalog."""
        k = np.zeros(len(cat), bool)
        for idx in range(self.nROIs):
            k |= (
                (
                    cat.row.values
                    > (self.ROI_corners[idx][0] - self.pixel_buffer)
                )
                & (
                    cat.row.values
                    < (
                        self.ROI_corners[idx][0]
                        + self.ROI_size[0]
                        + self.pixel_buffer
                    )
                )
                & (
                    cat.column.values
                    > (self.ROI_corners[idx][1] - self.pixel_buffer)
                )
                & (
                    cat.column.values
                    < (
                        self.ROI_corners[idx][1]
                        + self.ROI_size[1]
                        + self.pixel_buffer
                    )
                )
            )
        return cat[k].reset_index(drop=True)

    @add_docstring(parameters=["location", "gradients"])
    def _get_sparse_matrix(self, location, gradients=True):
        """Hidden method to get a sparse matrix for a particular location.

        Returns
        -------
        X: Sparse3D
            A matrix of the trace for each target, with the correct expected sensitivity function.
        """
        if gradients:
            X, dX0, dX1 = self.prf.to_sparse3d(location, gradients=gradients)
            return [
                a.to_ROISparse3D(
                    nROIs=self.nROIs,
                    ROI_size=self.ROI_size,
                    ROI_corners=self.ROI_corners,
                )
                for a in [X, dX0, dX1]
            ]

        else:
            return self.prf.to_sparse3d(
                location, gradients=gradients
            ).to_ROISparse3D(
                nROIs=self.nROIs,
                ROI_size=self.ROI_size,
                ROI_corners=self.ROI_corners,
            )

    @classmethod
    @add_docstring(parameters=["ra", "dec", "theta", "time"])
    def from_pointing(cls, ra, dec, theta, time=Time.now()):
        wcs = VISDAReference().get_wcs(
            target_ra=ra, target_dec=dec, theta=theta
        )
        prf = DispersedPRF.from_reference("NIRDA")
        return cls(prf=prf, wcs=wcs, time=time)
