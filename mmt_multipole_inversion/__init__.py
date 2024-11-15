from .multipole_inversion import MultipoleInversion
from .magnetic_sample import MagneticSample
import logging

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

# Setup root logger! Default basicConfig has only a StreamHandler
# The root logger should not be modified. Submodules should have their own unique name. They inherit root logger config
# Access to root logger to add additional Handlers should be done with: looging.getLogger('')
# TODO: add a file handler to save simulation messages to logfile
FORMAT = '%(asctime)s - %(levelname)s - %(module)s :: %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO, datefmt='%d-%m-%Y %H:%M:%S')
