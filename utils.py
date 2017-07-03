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

class Timing():
    '''
    '''
    def __init__(self):
        pass
    #end

    def timer(self, start_time=None, logger=None):
        '''
        '''
        if not start_time:
            start_time = datetime.now()
            return start_time
        elif start_time:
            tmin, tsec = divmod((datetime.now() - start_time).total_seconds(), 60)
            if not logger:
                print(' Time taken: %i minutes and %s seconds.' % (tmin, round(tsec, 2)))
                sys.stdout.flush()
            else:
                logger.info(' Time taken: %i minutes and %s seconds.' % (tmin, round(tsec, 2)))
            #end
        #end
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