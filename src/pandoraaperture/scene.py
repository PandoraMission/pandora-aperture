# Standard library
import warnings
from dataclasses import dataclass
from typing import List, Tuple

# Third-party
import astropy.units as u
import matplotlib.pyplot as plt
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
    time: Time = Time.now()
    user_cat: pd.DataFrame = None

    @property
    def wcs_trimmed(self):
        return self.wcs[
            self.prf.imcorner[0] : self.prf.imshape[0] + self.prf.imcorner[0],
            self.prf.imcorner[1] : self.prf.imshape[1] + self.prf.imcorner[1],
        ]

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
        new_cat = cat[k].reset_index(drop=True)
        center = np.asarray(self.imcorner) + np.asarray(self.imshape) / 2
        dist = (
            (new_cat.row.values - center[0]) ** 2
            + (new_cat.column.values - center[1]) ** 2
        ) ** 0.5
        return new_cat.iloc[np.argsort(dist)].reset_index(drop=True)

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
                pm_ra_cosdec=np.nan_to_num(df.pmra.values, 0) * u.mas / u.year,
                pm_dec=np.nan_to_num(df.pmdec.values, 0) * u.mas / u.year,
                obstime=Time.strptime("2016", "%Y"),
                distance=Distance(
                    parallax=np.nan_to_num(df.parallax.values, 0) * u.mas,
                    allow_negative=True,
                ),
                radial_velocity=np.nan_to_num(df.radial_velocity.values, 0)
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
            r1 += self.prf.trace_column.value.min()
            r2 += self.prf.trace_column.value.max()
            c1 += self.prf.trace_row.value.min()
            c2 += self.prf.trace_row.value.max()
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
        self.time = Time(self.time)
        self.X, self.dX0, self.dX1, self.cat = self._get_X()

    @property
    def imcorner(self):
        return self.prf.imcorner

    @property
    def imshape(self):
        return self.prf.imshape

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

    @add_docstring(parameters=["delta_pos"])
    def evaluate(self, delta_pos=None):
        """Returns a dense image of the SkyScene"""
        r, c = (
            np.arange(self.imcorner[0], self.imcorner[0] + self.imshape[0]),
            np.arange(self.imcorner[1], self.imcorner[1] + self.imshape[1]),
        )
        if self.X is None:
            return r, c, np.zeros(self.imshape)
        return r, c, self.A(delta_pos=delta_pos).dot(self.flux.value)

    def _get_VDAflux(self, cat):
        """Gives the flux on the VDA. This can be updated with a reference product in the future...!"""
        # This is approximately the right flux for the VDA in electrons per second
        return (
            np.nan_to_num(cat.phot_bp_mean_flux.values, 0)
            * 0.9
            * u.electron
            / u.second
        )

    def _get_NIRDAflux(self, cat):
        """Gives the flux on the NIRDA. This can be updated with a reference product in the future...!"""
        # This is approximately the right flux for the NIRDA in electrons per second
        return np.nan_to_num(cat.j_flux.values, 0) * 1 * u.electron / u.second

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
        wcs = VISDAReference.get_wcs(target_ra=ra, target_dec=dec, theta=theta)
        prf = SpatialPRF.from_reference("VISDA")
        return cls(prf=prf, wcs=wcs, time=time)

    def plot(self, **kwargs):
        """Plots the SkyScene. Use this functon to visually inspect the SkyScene."""
        r, c, image = self.evaluate()
        fig, ax = plt.subplots(
            figsize=(7, 7),
            dpi=kwargs.pop("dpi", 100),
        )
        cmap = kwargs.pop("cmap", "viridis")
        vmin = kwargs.pop("vmin", 0)
        vmax = kwargs.pop("vmax", 100)
        im = ax.pcolormesh(
            c,
            r,
            image,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set(
            xlabel="Pixel Column",
            ylabel="Pixel Row",
            title="SkyScene",
            aspect="equal",
        )
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Flux [e$^-$/s]")
        return fig

    def _get_row0(self, delta_pos=None):
        if delta_pos is None:
            delta_pos = (0, 0)
        return self.X.subrow[0, 0] + np.round(delta_pos[0]).astype(int)

    def _get_column0(self, delta_pos=None):
        if delta_pos is None:
            delta_pos = (0, 0)
        return self.X.subcol[0, 0] + np.round(delta_pos[1]).astype(int)

    @add_docstring(
        parameters=[
            "target",
            "delta_pos",
            "relative_threshold",
            "absolute_threshold",
        ],
        returns=[
            "aperture",
            "contamination",
            "completeness",
            "total_in_aperture",
        ],
    )
    def get_aperture(
        self,
        target,
        delta_pos=None,
        relative_threshold=0.005,
        absolute_threshold=50,
    ):
        """
        Obtain the aperture and aperture statistics for a particular target.
        """
        if isinstance(target, SkyCoord):
            c = SkyCoord(self.cat.RA.values, self.cat.Dec.values, unit="deg")
            sep = c.separation(target).min().to(u.deg) / (
                np.mean(np.abs(self.wcs_trimmed.wcs.cdelt)) * u.deg / u.pixel
            )
            if sep > (3 * u.pixel):
                raise ValueError(
                    "No matching target in catalog, try updating the catalog."
                )
            idx = int(c.separation(target).argmin())
            # target = SkyCoord(self.cat.RA.values[idx], self.cat.Dec.values[idx], unit='deg')
        elif isinstance(target, str):
            loc = self.cat.source_id.isin([target]).values
            if not loc.any():
                raise ValueError(
                    "No matching target in catalog, try updating the catalog."
                )
            idx = int(np.where(loc)[0][0])
            # target = SkyCoord(self.cat.RA.values[idx], self.cat.Dec.values[idx], unit='deg')
        elif isinstance(target, (int, np.int64)):
            idx = int(target)
        if self.X is None:
            aper = np.zeros(self.imshape, bool)
            contamination = 0.0
            completeness = 0.0
            total_in_aperture = 0.0
        else:
            A = self.A(delta_pos=delta_pos)

            im1 = A[:, :, idx].dot(np.ones(1) * self.flux[idx].value)
            im2 = A.dot(self.flux.value)
            aper = (im1 > absolute_threshold) & (
                (im1 / self.flux[idx].value) > relative_threshold
            )

            aper = aper.astype(bool)
            contamination = (im2 - im1)[aper].sum() / im1.sum()
            completeness = (im1)[aper].sum() / (self.flux[idx].value)
            total_in_aperture = (im1)[aper].sum()
        return aper, contamination, completeness, total_in_aperture

    @add_docstring(
        parameters=["delta_pos", "relative_threshold", "absolute_threshold"],
        returns=[
            "aperture",
            "contamination",
            "completeness",
            "total_in_aperture",
        ],
    )
    def get_all_apertures(
        self, delta_pos=None, relative_threshold=0.005, absolute_threshold=50
    ):
        """
        Obtain the aperture and aperture statistics for all targets.
        """
        if self.X is None:
            apers = np.zeros((1, *self.imshape), bool)
            contamination = np.asarray([0.0])
            completeness = np.asarray([0.0])
            total_in_aperture = np.asarray([0.0])
            return (
                apers,
                contamination,
                completeness,
                total_in_aperture,
            )
        apers = []
        contamination, completeness, total_in_aperture = np.zeros(
            (3, len(self.cat))
        )
        for idx in range(len(self.cat)):
            A = self.A(delta_pos=delta_pos)

            im1 = A[:, :, idx].dot(np.ones(1) * self.flux[idx].value)
            im2 = A.dot(self.flux.value)
            aper = (im1 > absolute_threshold) & (
                (im1 / self.flux[idx].value) > relative_threshold
            )

            apers.append(aper.astype(bool))
            contamination[idx] = (im2 - im1)[aper].sum() / im1.sum()
            completeness[idx] = (im1)[aper].sum() / (self.flux[idx].value)
            total_in_aperture[idx] = (im1)[aper].sum()
        return (
            np.asarray(apers),
            contamination,
            completeness,
            total_in_aperture,
        )


@add_docstring(parameters=["prf", "wcs", "time", "user_cat"])
@dataclass()
class DispersedSkyScene(SkyScene):
    """Special version of a SkyScene that works with dispersed PRFs"""

    @add_docstring(parameters=["cat"])
    def _clean_catalog(self, cat):
        """Hidden method that returns a cleaned version of a catalog."""
        length = (
            self.prf.trace_row.value.max() - self.prf.trace_row.value.min()
        )
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
        # Pandora NIR side has a physical block on certain regions so we'll remove any part of the catalog that has sources in those regions
        k &= cat.row.values > (512 - length - self.pixel_buffer)
        k &= cat.row.values < ((1024 + 512) + length + self.pixel_buffer)

        length = (
            self.prf.trace_column.value.max()
            - self.prf.trace_column.value.min()
        )
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

        # Pandora NIR side has a physical block on certain regions so we'll remove any part of the catalog that has sources in those regions
        k &= cat.column.values > ((1024 + 256) - length - self.pixel_buffer)

        # Faint sources are a waste of compute
        k &= self._get_NIRDAflux(cat) > (500 * u.electron / u.second)
        new_cat = cat[k].reset_index(drop=True)
        center = np.asarray(self.imcorner) + np.asarray(self.imshape) / 2
        dist = (
            (new_cat.row.values - center[0]) ** 2
            + (new_cat.column.values - center[1]) ** 2
        ) ** 0.5
        return new_cat.iloc[np.argsort(dist)].reset_index(drop=True)

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
                    data=a.dot(self._spectrum_norm)[:, :, None].value,
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
            NIRDAReference.get_spectrum_normalization_per_pixel(
                self.prf.trace_row.value
            )
        )
        self._spectrum_norm /= np.trapz(
            self._spectrum_norm, self.prf.trace_row.value
        )
        self._check_user_cat()
        self.X, self.dX0, self.dX1, self.cat = self._get_X()

    @property
    def flux(self):
        """Here we set the flux that is assumed to be the default for this object. For dispersed sky scenes it's NIRDA."""
        return self.NIRDAflux

    @classmethod
    @add_docstring(parameters=["ra", "dec", "theta", "time"])
    def from_pointing(cls, ra, dec, theta, time=Time.now()):
        wcs = NIRDAReference.get_wcs(target_ra=ra, target_dec=dec, theta=theta)
        prf = DispersedPRF.from_reference("NIRDA")
        return cls(prf=prf, wcs=wcs, time=time)


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
        new_cat = cat[k].reset_index(drop=True)
        center = np.asarray(self.imcorner) + np.asarray(self.imshape) / 2
        dist = (
            (new_cat.row.values - center[0]) ** 2
            + (new_cat.column.values - center[1]) ** 2
        ) ** 0.5
        return new_cat.iloc[np.argsort(dist)].reset_index(drop=True)

    @add_docstring(parameters=["delta_pos"])
    def evaluate(self, delta_pos=None):
        """Returns a dense image of the SkyScene

        Returns
        -------
        r: npt.NDArray
            Array of row positions of the image
        c: npt.NDArray
            Array of column positions of the image
        im: npt.NDArray
            Image of the scene
        """
        if delta_pos is None:
            delta_pos = (0, 0)
        r = (
            np.mgrid[: self.nROIs, : self.ROI_size[0]][1]
            + np.asarray(self.ROI_corners)[:, 0][:, None].astype(int)
            + np.round(delta_pos[0]).astype(int)
        )
        c = (
            np.mgrid[: self.nROIs, : self.ROI_size[1]][1]
            + np.asarray(self.ROI_corners)[:, 1][:, None].astype(int)
            + np.round(delta_pos[1]).astype(int)
        )
        if self.X is None:
            return r, c, np.zeros(self.imshape)
        return r, c, self.A(delta_pos=delta_pos).dot(self.flux.value)

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
    @add_docstring(
        parameters=[
            "ra",
            "dec",
            "theta",
            "time",
            "nROIs",
            "ROI_corners",
            "ROI_size",
        ]
    )
    def from_pointing(
        cls, ra, dec, theta, nROIs, ROI_corners, ROI_size, time=Time.now()
    ):
        wcs = VISDAReference.get_wcs(target_ra=ra, target_dec=dec, theta=theta)
        prf = SpatialPRF.from_reference("VISDA")
        return cls(
            prf=prf,
            wcs=wcs,
            nROIs=nROIs,
            ROI_corners=ROI_corners,
            ROI_size=ROI_size,
            time=time,
        )

    def plot(self, **kwargs):
        """Plots the SkyScene. Use this functon to visually inspect the SkyScene."""
        r, c, image = self.evaluate()
        fig, ax = plt.subplots(
            figsize=(7, 7),
            dpi=kwargs.pop("dpi", 100),
        )
        cmap = kwargs.pop("cmap", "viridis")
        vmin = kwargs.pop("vmin", 0)
        vmax = kwargs.pop("vmax", 100)
        im = ax.pcolormesh(
            np.hstack(image),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        [
            ax.axvline((i + 1) * self.ROI_size[0], c="white", ls="--")
            for i in range(self.nROIs)
        ]
        ax.set(
            xlabel="ROI Pixel Column",
            ylabel="ROI Pixel Row",
            title="ROISkyScene",
            aspect="equal",
            xlim=(0, self.ROI_size[0] * self.nROIs),
        )
        cbar = plt.colorbar(im, ax=ax, orientation="horizontal")
        cbar.set_label("Flux [e$-$/s]")
        return fig
