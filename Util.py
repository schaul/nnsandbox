from time import time as now
from BigMat import empty
import cPickle as cp
import os

###################################################

_last_tic = 0.0

def tic():
    global _last_tic
    _last_tic = now()
    return _last_tic

def toc(tic_time=0.0):
    global _last_tic
    toc_time = now()
    if tic_time == 0.0:
        tic_time = _last_tic
    return toc_time - tic_time


##################################################

class TempMatrix(object):
    '''
    A temporary matrix that can be easily resized as-needed.
    '''
    def __init__(self,m=1,n=1):
        self._A = empty((m,n))
        self._m = m

    def get(self): return self._A[:self._m,:]

    def get_capacity(self,m,n=-1):
        '''
        Ensures that the matrix has enough capacity for m rows when there are exactly n columns.
        Returns an ndarray (a view) into the first m rows of the memory.
        '''
        # Ensure columns match n exactly, and rows are at least m
        if n < 0: 
            n = self._A.shape[1]
        if m > self._A.shape[0] or n != self._A.shape[1]:
            self._A = empty((m,n))

        self._m = m

        # Return view to the first m rows
        return self._A[:m,:]

#########################################################################


def quickdump(filename,object):
    f = open(filename,'wb')
    cp.dump(object,f,-1)
    f.close()

def quickload(filename):
    f = open(filename,'rb')
    object = cp.load(f)
    f.close()
    return object

def ensure_dir(path):
    try:    os.makedirs(path)
    except: pass
    return path
