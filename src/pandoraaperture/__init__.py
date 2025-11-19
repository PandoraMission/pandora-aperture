# Standard library
import logging  # noqa: E402
import os  # noqa
from glob import glob

# Third-party
from rich.console import Console  # noqa: E402
from rich.logging import RichHandler  # noqa: E402

PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))
DOCSDIR = "/".join(PACKAGEDIR.split("/")[:-2]) + "/docs/"
TESTDIR = "/".join(PACKAGEDIR.split("/")[:-2]) + "/tests/"
PANDORASTYLE = glob(f"{PACKAGEDIR}/data/pandora.mplstyle")

# Standard library
import configparser  # noqa: E402
from importlib.metadata import PackageNotFoundError, version  # noqa

# Third-party
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pandoraref as pr  # noqa: E402
from appdirs import user_config_dir, user_data_dir  # noqa: E402


def get_version():
    try:
        return version("pandoraaperture")
    except PackageNotFoundError:
        return "unknown"


__version__ = get_version()


# Custom Logger with Rich
class PandoraLogger(logging.Logger):
    def __init__(self, name, level=logging.INFO):
        super().__init__(name, level)
        console = Console()
        self.handler = RichHandler(
            show_time=False, show_level=False, show_path=False, console=console
        )
        self.handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        self.addHandler(self.handler)


def get_logger(name="pandoraaperture"):
    """Configure and return a logger with RichHandler."""
    return PandoraLogger(name)


CONFIGDIR = user_config_dir("pandoraaperture")
os.makedirs(CONFIGDIR, exist_ok=True)
CONFIGPATH = os.path.join(CONFIGDIR, "config.ini")

logger = get_logger("pandoraaperture")


def reset_config():
    """Set the config to defaults."""
    # use this function to set your default configuration parameters.
    config = configparser.ConfigParser()
    config["SETTINGS"] = {
        "log_level": "WARNING",
        "data_dir": user_data_dir("pandoraaperture"),
        "pixel_buffer": 15,
        "catalog_columns": "source_id, phot_g_mean_flux, phot_bp_mean_flux, phot_rp_mean_flux, j_flux, h_flux, k_flux, teff_gspphot",
    }
    with open(CONFIGPATH, "w") as configfile:
        config.write(configfile)


def load_config() -> configparser.ConfigParser:
    """
    Loads the configuration file, creating it with defaults if it doesn't exist.

    Returns
    -------
    configparser.ConfigParser
        The loaded configuration.
    """

    config = configparser.ConfigParser()

    if not os.path.exists(CONFIGPATH):
        # Create default configuration
        reset_config()
    config.read(CONFIGPATH)
    return config


def save_config(config: configparser.ConfigParser) -> None:
    """
    Saves the configuration to the file.

    Parameters
    ----------
    config : configparser.ConfigParser
        The configuration to save.
    app_name : str
        Name of the application.
    """
    with open(CONFIGPATH, "w") as configfile:
        config.write(configfile)


config = load_config()

# Use this to check that keys you expect are in the config file.
# If you update the config file and think users may be out of date
# add the config parameters to this loop to check and reset the config.
for key in ["data_dir", "log_level"]:
    if key not in config["SETTINGS"]:
        logger.error(
            f"`{key}` missing from the `pandoraaperture` config file. Your configuration is being reset."
        )
        reset_config()
        config = load_config()

DATADIR = config["SETTINGS"]["data_dir"]
logger.setLevel(config["SETTINGS"]["log_level"])


def display_config() -> pd.DataFrame:
    dfs = []
    for section in config.sections():
        df = pd.DataFrame(
            np.asarray(
                [(key, value) for key, value in dict(config[section]).items()]
            )
        )
        df["section"] = section
        df.columns = ["key", "value", "section"]
        df = df.set_index(["section", "key"])
        dfs.append(df)
    return pd.concat(dfs)


NIRDAReference = pr.NIRDAReference()
VISDAReference = pr.VISDAReference()

from .prf import PRF, DispersedPRF, SpatialPRF  # noqa: E402,F401
from .scene import DispersedSkyScene, ROISkyScene, SkyScene  # noqa: E402,F401
