"""
chromnn
~~~~~~

ChromNN project.

:author: Jacques Serizay
:license: CC BY-NC 4.0

"""

from .version import __format_version__, __version__
from .models import ChromNNModel
from .data_prep import one_hot_encode, load_bigwig_data, split_data

__all__ = [
    "__version__",
    "__format_version__",
    "Momics",
]


