#!/usr/bin/env python

"""
"""

__author__ = 'Anna Kukleva'
__date__ = 'June 2019'


from ute.corpus import Corpus
from ute.utils.arg_pars import opt
from ute.utils.logging_setup import logger
from ute.utils.util_functions import timing, update_opt_str, join_return_stat, parse_return_stat
import data_utils.dummy_utils.update_argpars as dummy_utils

dummy_utils.update()