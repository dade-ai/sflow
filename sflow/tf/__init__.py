# -*- coding: utf-8 -*-
# from __future__ import absolute_import

from .iflag import *
import sys as _sys

# region check cuda option and set CUDA_VISIBLE_DEVICES

add_flag('cuda', '', 'the value of CUDA_VISIBLE_DEVICES')


help_flag = tuple(a for a in _sys.argv[1:]
                  if a.startswith('--help') or a.startswith('-h'))
if not help_flag:
    parse_flag(verbose=False)
    if flag.cuda:
        from sflow import gpu
        # string..
        flag.cuda = gpu.visible_devices(flag.cuda)

# endregion

# from sflow.core import *
# from sflow.core import (_)
from .icnn import *
from .ireshape import *
from .loss import *
from .measure import *

# from .train import *
# import sflow.feed as feed
# import sflow.img as img
# import sflow.io as io
from .. import feed
from .. import img
from .. import io
from .. import sample
from . import cmap

from .run_grid import run_grid
# import run_grid

from ..core import *

