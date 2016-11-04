"""
  **************************************
  Created by Romano Foti - rfoti
  On 10/22/2016
  *************************************
"""

#******************************************************************************
# Importing packages
#******************************************************************************
#-----------------------------
# Standard libraries
#-----------------------------
import os
import sys
import numpy as np
import pandas as pd
import random
import logging
#-----------------------------
# User defined modules and functions
#-----------------------------



#-----------------------------
# Logging
#-----------------------------

class Logging():
    '''
    '''
    def __init__(self):
        pass
    #end

    def configure_logger(self, logname, logfile):
        """ Configures a logger object, and adds a stream handler as well as a 
        file handler. """
        logger = logging.getLogger(logname)
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        fh = logging.FileHandler(logfile)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        return logger
    #end

#end

#-----------------------------
# SWAP ROWS and COLUMNS
#-----------------------------

def swap_rows(nparray, frm, to, return_array=False):
    nparray[[frm, to],:] = nparray[[to, frm],:] #swaps rows using advanced slicing
    if return_array:
      return nparray
    else:
      return
    #end
#end
    
def swap_cols(nparray, frm, to, return_array=False):
    nparray[:,[frm, to]] = nparray[:,[to, frm]] #swaps columns using advanced slicing
    if return_array:
      return nparray
    else:
      return
    #end
#end

#-----------------------------