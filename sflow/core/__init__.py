import tensorflow as tf

g = globals()
for k, v in tf.__dict__.items():
    if k not in g:
        g[k] = v

# __version__ = '1.2.1'

try:
    # noinspection PyUnresolvedReferences
    __tfversion__ = tf.__version__
    # noinspection PyUnresolvedReferences
    __builtins__ = tf.__builtins__
    # noinspection PyUnresolvedReferences
    # __git_version__ = tf.__git_version__
    # # noinspection PyUnresolvedReferences
    # __compiler_version__ = tf.__compiler_version__
except AttributeError:
    pass

from . import logg

from .iconst import *
from .icore import *
from .defaults import *
from .icontext import *

from .initializer import *
from .ioptimizer import *
from .ifeeding import *
from .isummary import *
from .layertool import *
from .irandom import *
from .itrain import *
from .ibind import (_, bind, bind_scope)
from .ifunctional import *

from .ipatch import (concat, lookup, lookup_2d, flat, sum, mean, min, max, all, any, prod,
                     top_k, sort, argsort, shiftdim,
                     to_tensor, astype, repeat, repeats, select, pad_to_shape,
                     crop)
from .iextra import *

from . import optim
from . import decay

__all__ = list(g.keys())

