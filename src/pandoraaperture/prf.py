"""Basic PRF Functions"""

# Third-party
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.interpolate import RectBivariateSpline
from sparse3d import Sparse3D

from . import NIRDAReference, VISDAReference
from .docstrings import add_docstring
from .utils import interpfunc


class PRF(object):
    """Pixel Response Function class"""

    @add_docstring(
        parameters=[
            "flux",
            "pixel_size",
            "sub_pixel_size",
            "scale",
            "imshape",
            "imcorner",
        ]
    )
    def __init__(
        self,
        flux,
        pixel_size,
        sub_pixel_size,
        scale=1,
        imshape=(2048, 2048),
        imcorner=(0, 0),
    ):
        """
        PRF class for making Pixel Response Functions.
        """
        self.flux = flux
        self.pixel_size = pixel_size
        self.sub_pixel_size = sub_pixel_size
        self.scale = scale
        self.imshape = imshape
        self.imcorner = imcorner
        self.column = (
            self.scale
            * (
                (np.arange(self.shape[1]) - self.shape[1] // 2)
                * u.pixel
                * self.sub_pixel_size
            )
            / self.pixel_size
        )
        self.column -= np.median(np.diff(self.column)) / 2
        self.row = (
            self.scale
            * (
                (np.arange(self.shape[0]) - self.shape[0] // 2)
                * u.pixel
                * self.sub_pixel_size
            )
            / self.pixel_size
        )
        self.row -= np.median(np.diff(self.row)) / 2
        self.flux_grad = np.asarray(
            np.gradient(
                self.flux,
                np.median(np.diff(self.row)).value / self.scale,
                axis=(0, 1),
            )
        )
        self._r, self._c = (
            np.arange(
                np.floor(self.row[0].to(u.pixel).value).astype(int),
                np.ceil(self.row[-1].to(u.pixel).value).astype(int) + 1,
                1,
            ),
            np.arange(
                np.floor(self.column[0].to(u.pixel).value).astype(int),
                np.ceil(self.column[-1].to(u.pixel).value).astype(int) + 1,
                1,
            ),
        )
        self.spline = None
        self._default_location = (
            self.imcorner[0] + self.imshape[0] / 2,
            self.imcorner[1] + self.imshape[1] / 2,
        )

    @add_docstring(parameters=["location"])
    def _make_spline(self, location=None):
        """
        Hidden function to create the spline function for interpolating the PRF.
        """
        if self.spline is None:
            self.spline = RectBivariateSpline(
                self.row.to(u.pixel).value.astype(float),
                self.column.to(u.pixel).value.astype(float),
                self.flux,
            )

    @property
    def shape(self):
        return self.flux.shape

    @classmethod
    @add_docstring(parameters=["file"])
    def from_file(cls, file):
        """
        Load a PRF object from a file.
        """
        if isinstance(file, fits.HDUList):
            hdulist = file
        else:
            hdulist = fits.open(file)
        if not len(hdulist) == 4:
            raise ValueError("Expected 4 HDUList extensions in PRF fits file.")
        flux = hdulist[1].data.transpose([2, 3, 0, 1])
        row = hdulist[2].data[:, 0]
        column = hdulist[3].data[0]
        flux = interpfunc(
            column.mean(),
            column,
            interpfunc(row.mean(), row, flux),
        )
        pixel_size = hdulist[0].header["PIXSIZE"] * u.micron / u.pix
        sub_pixel_size = hdulist[0].header["SUBPIXSZ"] * u.micron / u.pix
        imshape = (hdulist[0].header["IMSIZE0"], hdulist[0].header["IMSIZE1"])
        imcorner = (
            hdulist[0].header["IMCRNR0"],
            hdulist[0].header["IMCRNR1"],
        )
        return cls(
            flux=flux,
            pixel_size=pixel_size,
            sub_pixel_size=sub_pixel_size,
            imshape=imshape,
            imcorner=imcorner,
        )

    @classmethod
    @add_docstring(parameters=["name"])
    def from_reference(cls, name: str = "visda"):
        """
        Load a PRF from `pandoraref`.
        """
        if name.lower() in ["v", "vis", "vda", "visda"]:
            file = VISDAReference.prf_file
        elif name.lower() in ["n", "nir", "nirda", "ir"]:
            raise ValueError(
                f"Can not open NIRDA PRF with class `{cls.__name__}`. Try a `DispersedPRF`."
            )
        else:
            raise ValueError(
                f"Can not parse PRF name '{name}', please select a different name."
            )
        return cls.from_file(file)

    def __repr__(self):
        return "PRF"

    @add_docstring(
        parameters=["spline", "location", "normalize", "gradients"],
        returns=["row_im", "column_im", "prf_im", "dprf_im"],
    )
    def _evaluate(
        self, spline, location=None, normalize=True, gradients=False
    ):
        """
        Hidden method to evaluate the PRF model at a given location.
        This method is hidden because it enables us to alter the evaluation
        procedure for subclasses.
        """
        if location is None:
            location = self._default_location
        prf_im = spline(
            self._r - (location[0] % 1), self._c - (location[1] % 1), grid=True
        )
        if gradients:
            dprf_im = np.asarray(
                [
                    spline(
                        self._r - (location[0] % 1),
                        self._c - (location[1] % 1),
                        grid=True,
                        dx=1,
                        dy=0,
                    ),
                    spline(
                        self._r - (location[0] % 1),
                        self._c - (location[1] % 1),
                        grid=True,
                        dx=0,
                        dy=1,
                    ),
                ]
            )
        if normalize:
            norm = prf_im.sum()
        if gradients:
            if normalize:
                return (
                    self._r + np.floor(location[0]).astype(int),
                    self._c + np.floor(location[1]).astype(int),
                    prf_im / norm,
                    dprf_im / norm,
                )
            return (
                self._r + np.floor(location[0]).astype(int),
                self._c + np.floor(location[1]).astype(int),
                prf_im,
                dprf_im,
            )
        if normalize:
            return (
                self._r + np.floor(location[0]).astype(int),
                self._c + np.floor(location[1]).astype(int),
                prf_im / norm,
            )
        return (
            self._r + np.floor(location[0]).astype(int),
            self._c + np.floor(location[1]).astype(int),
            prf_im,
        )

    @add_docstring(
        parameters=["location", "normalize", "gradients"],
        returns=["row_im", "column_im", "prf_im", "dprf_im"],
    )
    def evaluate(self, location=None, normalize=True, gradients=False):
        """
        Evaluate the PRF model for a target at a given location on the detector.
        """
        self._make_spline(location=location)
        return self._evaluate(
            spline=self.spline,
            location=location,
            normalize=normalize,
            gradients=gradients,
        )

    @add_docstring(
        parameters=["location", "normalize", "gradients"],
        returns=["X", "dX0", "dX1"],
    )
    def to_sparse3d(
        self,
        location=None,
        normalize=True,
        gradients=False,
    ):
        """
        Converts this `PRF` object to a `Sparse3D` object.
        """
        if gradients:
            r, c, prf_im, dprf_im = self.evaluate(
                location=location, normalize=normalize, gradients=gradients
            )
            r, c = np.meshgrid(r, c, indexing="ij")
            r, c, prf_im, dprf_im = (
                r[None, :, :],
                c[None, :, :],
                prf_im[None, :, :],
                dprf_im[:, None, :, :],
            )
            dX0 = Sparse3D(
                data=dprf_im[0].transpose([1, 2, 0]),
                row=r.transpose([1, 2, 0]),
                col=c.transpose([1, 2, 0]),
                imshape=self.imshape,
                imcorner=self.imcorner,
            )
            dX1 = Sparse3D(
                data=dprf_im[1].transpose([1, 2, 0]),
                row=r.transpose([1, 2, 0]),
                col=c.transpose([1, 2, 0]),
                imshape=self.imshape,
                imcorner=self.imcorner,
            )
            X = Sparse3D(
                data=prf_im.transpose([1, 2, 0]),
                row=r.transpose([1, 2, 0]),
                col=c.transpose([1, 2, 0]),
                imshape=self.imshape,
                imcorner=self.imcorner,
            )
            return X, dX0, dX1
        else:
            r, c, prf_im = self.evaluate(
                location=location, normalize=normalize
            )
            r, c = np.meshgrid(r, c, indexing="ij")
            r, c, prf_im = r[None, :, :], c[None, :, :], prf_im[None, :, :]
            X = Sparse3D(
                data=prf_im.transpose([1, 2, 0]),
                row=r.transpose([1, 2, 0]),
                col=c.transpose([1, 2, 0]),
                imshape=self.imshape,
                imcorner=self.imcorner,
            )
            return X

    def plot(self, **kwargs):
        """Plots the PRF. Use this functon to visually inspect the PRF."""
        fig, ax = plt.subplots(
            figsize=(6.5, 5),
            dpi=kwargs.pop("dpi", 100),
        )
        cmap = kwargs.pop("cmap", "viridis")
        vmin = kwargs.pop("vmin", 0)
        vmax = kwargs.pop("vmax", 0.01)
        im = ax.pcolormesh(
            self.column.value,
            self.row.value,
            self.flux[:, :],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set(xlabel="Pixel Column", ylabel="Pixel Row", title="PRF")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("PRF")
        return fig


class SpatialPRF(PRF):
    """Special case where the PRF is a function of focal plane position."""

    @add_docstring(
        parameters=[
            "focal_row",
            "focal_column",
            "flux",
            "pixel_size",
            "sub_pixel_size",
            "scale",
            "imshape",
            "imcorner",
        ]
    )
    def __init__(
        self,
        focal_row,
        focal_column,
        flux,
        pixel_size,
        sub_pixel_size,
        scale=1,
        imshape=(2048, 2048),
        imcorner=(0, 0),
    ):
        """
        Special PRF class for making Pixel Response Functions that change as a function of focal plane position.
        """
        if not flux.ndim == 4:
            raise ValueError("`flux` must be 4D.")
        if not focal_row.shape[0] == flux.shape[2]:
            raise ValueError(
                "`flux` does not have the same input dimension as `row`."
            )
        if not focal_column.shape[0] == flux.shape[3]:
            raise ValueError(
                "`flux` does not have the same input dimension as `column`."
            )
        super().__init__(
            flux=flux,
            pixel_size=pixel_size,
            sub_pixel_size=sub_pixel_size,
            scale=scale,
            imshape=imshape,
            imcorner=imcorner,
        )
        self.focal_row = u.Quantity(focal_row)
        self.focal_column = u.Quantity(focal_column)
        return

    @add_docstring(parameters=["location"])
    def _make_spline(self, location=None):
        """
        Hidden function to create the spline function for interpolating the PRF.
        """
        if location is None:
            location = self._default_location
        flux = interpfunc(
            location[1],
            self.focal_column.value,
            interpfunc(location[0], self.focal_row.value, self.flux),
        )
        self.spline = RectBivariateSpline(
            self.row.to(u.pixel).value.astype(float),
            self.column.to(u.pixel).value.astype(float),
            flux,
        )

    def __repr__(self):
        return "SpatialPRF"

    @add_docstring(parameters=["location"])
    def to_PRF(self, location=None):
        """
        Converts a spatial PRF into a PRF evaluated at a single location, making it faster to compute multiple PRFs, if necessary.
        """
        if location is None:
            location = self._default_location
        self._make_spline(location=location)
        return PRF(
            flux=self.spline(self.row, self.column),
            pixel_size=self.pixel_size,
            sub_pixel_size=self.sub_pixel_size,
            imshape=self.imshape,
            imcorner=(
                int(location[0] - self.imshape[0] / 2),
                int(location[1] - self.imshape[1] / 2),
            ),
        )

    @classmethod
    @add_docstring(parameters=["file"])
    def from_file(cls, file):
        """
        Load a PRF object from a file.
        """
        if isinstance(file, fits.HDUList):
            hdulist = file
        else:
            hdulist = fits.open(file)
        if not len(hdulist) == 4:
            raise ValueError("Expected 4 HDUList extensions in PRF fits file.")
        flux = hdulist[1].data.transpose([2, 3, 0, 1])
        focal_row = hdulist[2].data[:, 0]
        focal_column = hdulist[3].data[0]
        pixel_size = hdulist[0].header["PIXSIZE"] * u.micron / u.pix
        sub_pixel_size = hdulist[0].header["SUBPIXSZ"] * u.micron / u.pix
        imshape = (hdulist[0].header["IMSIZE0"], hdulist[0].header["IMSIZE1"])
        imcorner = (
            hdulist[0].header["IMCRNR0"],
            hdulist[0].header["IMCRNR1"],
        )
        return cls(
            focal_row=u.Quantity(focal_row, "pixel"),
            focal_column=u.Quantity(focal_column, "pixel"),
            flux=flux,
            pixel_size=pixel_size,
            sub_pixel_size=sub_pixel_size,
            imshape=imshape,
            imcorner=imcorner,
        )

    def plot(self, **kwargs):
        """Plots the PRF. Use this functon to visually inspect the PRF."""
        fig, ax = plt.subplots(
            self.flux.shape[2],
            self.flux.shape[3],
            figsize=(10, 10),
            sharex=True,
            sharey=True,
        )
        cmap = kwargs.pop("cmap", "viridis")
        vmin = kwargs.pop("vmin", 0)
        vmax = kwargs.pop("vmax", 0.01)
        plt.subplots_adjust(hspace=0, wspace=0)
        for idx in np.arange(0, self.flux.shape[2]):
            for jdx in np.arange(0, self.flux.shape[3]):
                ax[self.flux.shape[2] - idx - 1, jdx].pcolormesh(
                    self.column.value,
                    self.row.value,
                    self.flux[:, :, idx, jdx],
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                )
                ax[self.flux.shape[2] - idx - 1, 0].set(
                    ylabel=int(self.focal_row[idx].value)
                )
                ax[self.flux.shape[2] - 1, jdx].set(
                    xlabel=int(self.focal_column[jdx].value)
                )
        return fig


class DispersedPRF(PRF):
    """Special case where PRF is dispersed in a line"""

    @add_docstring(
        parameters=[
            "trace_row",
            "trace_column",
            "flux",
            "pixel_size",
            "sub_pixel_size",
            "scale",
            "imshape",
            "imcorner",
        ]
    )
    def __init__(
        self,
        trace_row,
        trace_column,
        flux,
        pixel_size,
        sub_pixel_size,
        scale=1,
        norm=1,
        imshape=(400, 80),
        imcorner=(0, 0),
    ):
        """
        Special PRF class for making Pixel Response Functions that is dispersed across pixels.
        """
        if not flux.ndim == 3:
            raise ValueError("`flux` must be 3D.")
        if not trace_row.shape[0] == flux.shape[0]:
            raise ValueError(
                "`flux` does not have the same input dimension as `trace_row`."
            )
        if not trace_column.shape[0] == flux.shape[0]:
            raise ValueError(
                "`flux` does not have the same input dimension as `trace_column`."
            )
        super().__init__(
            flux=flux,
            pixel_size=pixel_size,
            sub_pixel_size=sub_pixel_size,
            scale=scale,
            imshape=imshape,
            imcorner=imcorner,
        )
        self.norm = norm
        self.trace_column = u.Quantity(trace_column, "pixel")
        self.trace_row = u.Quantity(trace_row, "pixel")

        return

    @property
    def shape(self):
        return self.flux.shape[1:]

    def __len__(self):
        return self.flux.shape[0]

    def __repr__(self):
        return f"DispersedPRF [{len(self)} elements]"

    @add_docstring(parameters=["location"])
    def _make_spline(self, location=None):
        """
        Hidden function to create the spline function for interpolating the PRF.
        """
        if self.spline is None:
            self.spline = [
                RectBivariateSpline(
                    self.column.to(u.pixel).value.astype(float),
                    self.row.to(u.pixel).value.astype(float),
                    self.flux[idx],
                )
                for idx in range(len(self))
            ]

    @add_docstring(
        parameters=["location", "normalize", "gradients"],
        returns=["row_im", "column_im", "prf_im", "dprf_im"],
    )
    def evaluate(self, location=None, normalize=True, gradients=False):
        if location is None:
            location = self._default_location
        self._make_spline(location=location)
        rs, cs, prf_ims = [], [], []
        if gradients:
            dprf_ims = []
        for idx in range(len(self)):
            if gradients:
                r, c, prf_im, dprf_im = self._evaluate(
                    spline=self.spline[idx],
                    location=(
                        location[0] + self.trace_row[idx].value,
                        location[1] + self.trace_column[idx].value,
                    ),
                    normalize=normalize,
                    gradients=gradients,
                )
            else:
                r, c, prf_im = self._evaluate(
                    spline=self.spline[idx],
                    location=(
                        location[0] + self.trace_row[idx].value,
                        location[1] + self.trace_column[idx].value,
                    ),
                    normalize=normalize,
                    gradients=gradients,
                )
            if normalize:
                prf_im *= self.norm
                if gradients:
                    dprf_im *= self.norm
            rs.append(r)
            cs.append(c)
            prf_ims.append(prf_im)
            if gradients:
                dprf_ims.append(dprf_im)
        rs, cs = np.asarray(rs), np.asarray(cs)
        rs = rs[:, :, None] * np.ones((cs.shape[0], 1, cs.shape[1]), int)
        cs = cs[:, None, :] * np.ones((rs.shape[0], rs.shape[1], 1), int)
        if gradients:
            return (
                rs,
                cs,
                np.asarray(prf_ims),
                np.asarray(dprf_ims).transpose([1, 0, 2, 3]),
            )
        return rs, cs, np.asarray(prf_ims)

    @add_docstring(
        parameters=["location", "normalize", "gradients"],
        returns=["X", "dX0", "dX1"],
    )
    def to_sparse3d(
        self,
        location=None,
        normalize=True,
        gradients=False,
    ):
        if gradients:
            r, c, prf_im, dprf_im = self.evaluate(
                location=location, normalize=normalize, gradients=gradients
            )
            dX0 = Sparse3D(
                data=dprf_im[0].transpose([1, 2, 0]),
                row=r.transpose([1, 2, 0]),
                col=c.transpose([1, 2, 0]),
                imshape=self.imshape,
                imcorner=self.imcorner,
            )
            dX1 = Sparse3D(
                data=dprf_im[1].transpose([1, 2, 0]),
                row=r.transpose([1, 2, 0]),
                col=c.transpose([1, 2, 0]),
                imshape=self.imshape,
                imcorner=self.imcorner,
            )
            X = Sparse3D(
                data=prf_im.transpose([1, 2, 0]),
                row=r.transpose([1, 2, 0]),
                col=c.transpose([1, 2, 0]),
                imshape=self.imshape,
                imcorner=self.imcorner,
            )
            return X, dX0, dX1
        else:
            r, c, prf_im = self.evaluate(
                location=location, normalize=normalize
            )
            X = Sparse3D(
                data=prf_im.transpose([1, 2, 0]),
                row=r.transpose([1, 2, 0]),
                col=c.transpose([1, 2, 0]),
                imshape=self.imshape,
                imcorner=self.imcorner,
            )
            return X

    @classmethod
    @add_docstring(parameters=["file", "pixel_resolution"])
    def from_file(cls, file, pixel_resolution=0.25):
        """
        Load a PRF object from a file.
        """
        if isinstance(file, fits.HDUList):
            hdulist = file
        else:
            hdulist = fits.open(file)
        # if not len(hdulist) == 4:
        #     raise ValueError("Expected 4 HDUList extensions in PRF fits file.")
        flux = hdulist[1].data
        trace_column = u.Quantity(hdulist[2].data, "pixel")
        trace_row = u.Quantity(hdulist[3].data, "pixel")

        def _interpolate_prf(x, y, flux, pixel_resolution):
            y2 = np.arange(
                np.floor(y.value.min()),
                np.ceil(y.value.max()),
                pixel_resolution,
            )
            x2 = np.interp(y2, y.value, x.value)
            flux2 = np.asarray(
                [
                    interpfunc(yi, y.value, flux.transpose([1, 2, 0]))
                    for yi in y2
                ]
            )
            return x2 * u.pixel, y2 * u.pixel, flux2

        trace_column, trace_row, flux = _interpolate_prf(
            trace_column, trace_row, flux, pixel_resolution=pixel_resolution
        )

        pixel_size = hdulist[0].header["PIXSIZE"] * u.micron / u.pix
        sub_pixel_size = hdulist[0].header["SUBPIXSZ"] * u.micron / u.pix
        # This should come from the file...
        imshape = (hdulist[0].header["IMSIZE0"], hdulist[0].header["IMSIZE1"])
        imcorner = (
            hdulist[0].header["IMCRNR0"],
            hdulist[0].header["IMCRNR1"],
        )
        norm = hdulist[0].header["NORM"]
        return cls(
            trace_row=trace_row,
            trace_column=trace_column,
            flux=flux,
            pixel_size=pixel_size,
            sub_pixel_size=sub_pixel_size,
            norm=norm,
            imshape=imshape,
            imcorner=imcorner,
        )

    @classmethod
    @add_docstring(parameters=["name", "pixel_resolution"])
    def from_reference(cls, name: str = "nirda", pixel_resolution=0.25):
        """
        Load a PRF from `pandoraref`.
        """
        if name.lower() in ["v", "vis", "vda", "visda"]:
            raise ValueError(
                f"Can not open VISDA PRF with class `{cls.__name__}`. Try a `PRF` or `SpatialPRF`."
            )
        elif name.lower() in ["n", "nir", "nirda", "ir"]:
            file = NIRDAReference.prf_file
        else:
            raise ValueError(
                f"Can not parse PRF name '{name}', please select a different name."
            )
        return cls.from_file(file, pixel_resolution=pixel_resolution)

    def plot(self, **kwargs):
        """Plots the PRF. Use this functon to visually inspect the PRF."""
        X = self.to_sparse3d(
            (
                self.imcorner[0] + self.imshape[0] / 2,
                self.imcorner[1] + self.imshape[1] / 2,
            )
        )
        fig, ax = plt.subplots(
            figsize=(7, 7),
            dpi=kwargs.pop("dpi", 100),
        )
        cmap = kwargs.pop("cmap", "viridis")
        vmin = kwargs.pop("vmin", 0)
        vmax = kwargs.pop("vmax", 0.1)
        im = ax.pcolormesh(
            X.dot(np.ones(len(X))),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set(
            xlabel="Pixel Column",
            ylabel="Pixel Row",
            title="PRF",
            aspect="equal",
        )
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("PRF")
        return fig
