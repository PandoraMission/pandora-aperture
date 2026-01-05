# Standard library
import functools

# Third-party
import astropy.units as u
import numpy.typing as npt
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.wcs import WCS
from scipy.interpolate import RectBivariateSpline
from sparse3d import Sparse3D

DOCSTRINGS = {
    "name": (str, "Name of detector, choose from VISDA or NIRDA."),
    "delta_pos": (
        tuple,
        "Change in position in pixels. Use format (row, column).",
    ),
    "file": (
        str,
        "Input file to use. Choose either a string path to the file or an `astropy.fits.HDUList` object",
    ),
    "flux": (
        npt.NDArray,
        "Array of the flux of the PRF as a function of position. Usually normalized such that the total flux is 1.",
    ),
    "gradients": (
        bool,
        "Whether to return gradients. If True, will return an additional 2 arrays that contain the gradients in each axis.",
    ),
    "pixel_size": (
        u.Quantity,
        "True detector pixel size in dimensions of length/pixel",
    ),
    "sub_pixel_size": (
        u.Quantity,
        "PSF file pixel size in dimensions of length/pixel",
    ),
    "scale": (
        float,
        "How much to scale the PRF by. Scale of 2 makes the PSF 2x broader. Default is 1.",
    ),
    "imshape": (
        tuple,
        "Tuple of the shape of the true image. "
        "If using ROIs, use the shape of the image that "
        "each ROI is cut out from. Use format (row, column).",
    ),
    "imcorner": (
        tuple,
        "Tuple of the lower left corner of the image, i.e. it's origin. "
        "Use this to move the image around on the grid. If using a window mode,"
        " make sure to set this to the right corner. Use format (row, column).",
    ),
    "location": (
        tuple,
        "Location of the source on the detector. Use format (row, column). "
        "If not set will default to `self._default_location` which is in the "
        "middle of the image as set by `imshape` and `imcorner`.",
    ),
    "spline": (RectBivariateSpline, "A spline model describing the PRF."),
    "normalize": (
        bool,
        "Whether to normalize the input data so that the total flux is 1.",
    ),
    "row_im": (
        npt.NDArray,
        "1D Array of integer row values at which the PRF is evaluated.",
    ),
    "column_im": (
        npt.NDArray,
        "1D Array of integer column values at which the PRF is evaluated.",
    ),
    "prf_im": (
        npt.NDArray,
        "2D Array of PRF values.",
    ),
    "dprf_im": (
        npt.NDArray,
        "2, 2D arrays of the gradient of the PRF values. Only returned if `gradients`=True.",
    ),
    "X": (Sparse3D, "Sparse3D object containing the PRF at a given location."),
    "dX0": (
        Sparse3D,
        "Sparse3D object containing the gradient of the PRF in axis 0"
        " at a given location. Only returned if `gradients`=True.",
    ),
    "dX1": (
        Sparse3D,
        "Sparse3D object containing the gradient of the PRF in axis 1"
        " at a given location. Only returned if `gradients`=True.",
    ),
    "focal_row": (
        u.Quantity,
        "Row position on the focal plane that corresponds to each element in the flux array. Units of pixels.",
    ),
    "focal_column": (
        u.Quantity,
        "Column position on the focal plane that corresponds to each element in the flux array. Units of pixels.",
    ),
    "trace_row": (
        u.Quantity,
        "Row position within the spectral trace that corresponds to each element in the flux array. Units of pixels.",
    ),
    "trace_column": (
        u.Quantity,
        "Column position within the spectral trace that corresponds to each element in the flux array. Units of pixels.",
    ),
    "prf": ("PRF", "pandoraaperture PRF class."),
    "wcs": (WCS, "astropy World Coordinate System"),
    "time": (Time, "astropy Time"),
    "user_cat": (
        pd.DataFrame,
        "Optional catalog from the user. Use this to pass a catalog of objects expected in this"
        " data that are not part of the Gaia catalog. You must include all the columns "
        "specified in your config file under `catalog_columns`.",
    ),
    "nROIs": (int, "The number of regions of interest in the larger image"),
    "ROI_size": (
        tuple,
        "The size the regions of interest in (row, column) pixels. All ROIs must be the same size.",
    ),
    "ROI_corners": (
        list,
        "The origin (lower left) corner positon for each of the ROIs. Must have length nROIs. List of tuples.",
    ),
    "cat": (
        pd.DataFrame,
        "Catalog of sources that will land within the image.",
    ),
    "coord": (SkyCoord, "Coordinate in the sky."),
    "radius": (u.Quantity, "A radius in degrees for a cone search."),
    "A": (
        Sparse3D,
        "Matrix containing the PRFs of all targets in the scene. ",
    ),
    "target": (
        int,
        "Indicates a target in the catalog. Use either in integer to express "
        "an index in the catalog, or a SkyCoord to find the closest target",
    ),
    "relative_threshold": (
        float,
        "Threshold to cut aperture off at in units of total source flux.",
    ),
    "absolute_threshold": (
        float,
        "Threshold to cut aperture off at, in units of electrons/s.",
    ),
    "ra": (u.Quantity, "Right Ascention"),
    "dec": (u.Quantity, "Declination"),
    "theta": (u.Quantity, "Roll angle in degrees"),
    "pixel_resolution": (
        float,
        "The separation between different elements in the PRF model in pixels. For example 0.25"
        " means there will be approximately 0.25 pixels between each element.",
    ),
    "aperture": (
        npt.NDArray,
        "Array of bools that shows which pixels are considered to be part of the target.",
    ),
    "contamination": (
        float,
        "Value showing how much of the flux in the aperture is due to other sources/the amount of flux inside the aperture.",
    ),
    "completeness": (
        float,
        "Value showing how much of the target flux is inside the aperture/total expected flux.",
    ),
    "total_in_aperture": (
        float,
        "Value showing how much flux is inside the aperture.",
    ),
}


def extract_docstring_type(dtype, desc):
    if isinstance(dtype, tuple):
        dtype_str = " or ".join(
            [t._name if hasattr(t, "_name") else t.__name__][0]
            for t in dtype
            if t is not None
        )
        dtype_str += " or None" if None in dtype else ""
    elif isinstance(dtype, str):
        dtype_str = dtype
    else:
        dtype_str = dtype.__name__
    return dtype_str, desc


def clean_docstring(func, additional_docstring, indent_str, heading):
    existing_docstring = func.__doc__ or ""
    if heading in existing_docstring:
        func.__doc__ = (
            existing_docstring.split("---\n")[0]
            + "---\n"
            + additional_docstring
            + "---\n".join(existing_docstring.split("---\n")[1:])
        )
    else:
        func.__doc__ = (
            existing_docstring
            + f"\n\n{indent_str}{heading}\n{indent_str}----------\n"
            + additional_docstring
        )


# Decorator to add common parameters to docstring
def add_docstring(func=None, *, parameters=None, returns=None):
    def decorator(func, parameters=parameters, returns=returns):
        param_docstring, return_docstring = "", ""
        if func.__doc__:
            # Determine the current indentation level
            lines = func.__doc__.splitlines()
            if len(lines[0]) == 0:
                indent = len(lines[1]) - len(lines[1].lstrip())
            else:
                indent = len(lines[0]) - len(lines[0].lstrip())
        else:
            indent = 0
        indent_str = " " * indent
        if isinstance(parameters, str):
            parameters = [parameters]
        if isinstance(returns, str):
            returns = [returns]

        if parameters is not None:
            for name in parameters:
                if name in DOCSTRINGS:
                    dtype_str, desc = extract_docstring_type(*DOCSTRINGS[name])
                    param_docstring += f"{indent_str}{name}: {dtype_str}\n{indent_str}    {desc}\n"
            clean_docstring(func, param_docstring, indent_str, "Parameters")

        if returns is not None:
            for name in returns:
                if name in DOCSTRINGS:
                    dtype_str, desc = extract_docstring_type(*DOCSTRINGS[name])
                    return_docstring += f"{indent_str}{name}: {dtype_str}\n{indent_str}    {desc}\n"
            clean_docstring(func, return_docstring, indent_str, "Returns")
        return func

    return decorator


# Decorator to inherit docstring from base class
def inherit_docstring(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    if func.__doc__ is None:
        for base in func.__qualname__.split(".")[0].__bases__:
            base_func = getattr(base, func.__name__, None)
            if base_func and base_func.__doc__:
                func.__doc__ = base_func.__doc__
                break
    return wrapper
