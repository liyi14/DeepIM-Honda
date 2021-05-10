# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Yi Li
# --------------------------------------------------------
from __future__ import print_function, division
from lib.fcn.config import cfg
import pprint

def print_and_log(string):
    print(string)
    if not cfg.NO_LOG:
        if cfg.logger:
            cfg.logger.info(string)
        else:
            print("NO LOGGER")
