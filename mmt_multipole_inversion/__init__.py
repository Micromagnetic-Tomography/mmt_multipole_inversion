from .multipole_inversion import MultipoleInversion
from .magnetic_sample import MagneticSample
import logging
import sys

from .__about__ import (
    __author__,
    __copyright__,
    __email__,
    __license__,
    __summary__,
    __title__,
    __uri__,
    __version__,
)

__all__ = [
    "MultipoleInversion",
    "MagneticSample",
    "__title__",
    "__summary__",
    "__uri__",
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "__copyright__",
]

# Adapter from: https://alexandra-zaharia.github.io/posts/make-your-own-custom-color-formatter-with-python-logging/
class CustomFormatter(logging.Formatter):
    """Logging colored formatter, adapted from https://stackoverflow.com/a/56944256/3638629"""

    grey = "\x1b[37;40m"
    yellow = "\x1b[33;40m"
    blue = "\x1b[34;40m"
    red = "\x1b[31;40m"
    bold_red = "\x1b[1;31;40m"
    reset = "\x1b[0m"

    def __init__(self, fmt):
        super().__init__()
        self.fmt = fmt
        self.FORMATS = {
            logging.DEBUG: self.grey + self.fmt + self.reset,
            logging.INFO: self.blue + self.fmt + self.reset,
            logging.WARNING: self.yellow + self.fmt + self.reset,
            logging.ERROR: self.red + self.fmt + self.reset,
            logging.CRITICAL: self.bold_red + self.fmt + self.reset
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


# Setup root logger! Default basicConfig has only a StreamHandler
# The root logger should not be modified. Submodules should have their own unique name. They inherit root logger config
# Access to root logger to add additional Handlers should be done with: looging.getLogger('')
# TODO: add a file handler to save simulation messages to logfile
FORMAT = '%(asctime)s | %(levelname)8s | %(module)s :: %(message)s'
handler_st = logging.StreamHandler(sys.stdout)
handler_st.setFormatter(CustomFormatter(FORMAT))
logging.basicConfig(format=FORMAT, datefmt='%d-%m-%Y %H:%M:%S', handlers=[handler_st])

# Set default level of this module's logger and children loggers
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
