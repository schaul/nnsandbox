import numpy as np
import sys

_gnumpy_loaded = False
try:
    import gnumpy as gp
    cudamat = gp.cmat
    _gnumpy_loaded = True
except ImportError:
    pass

default_dtype = 'float32'
#default_dtype = 'float64'

backend = None   # NumpyBackend or GnumpyBackend

class NumpyBackend(object):

    @staticmethod
    def empty(shape):    return np.empty(shape,default_dtype)

    @staticmethod
    def zeros(shape):    return np.zeros(shape,default_dtype)

    @staticmethod
    def ones(shape):     return np.ones(shape,default_dtype)

    @staticmethod
    def rand(*shape):    return np.array(np.random.rand(*shape),default_dtype)

    @staticmethod
    def array(A):        return np.array(A,default_dtype)

    @staticmethod
    def asarray(A):      return np.asarray(A,default_dtype)

    @staticmethod
    def as_numpy(A):     return A

    @staticmethod
    def dot(A,B,out):    return np.dot(A,B,out=out)

    @staticmethod
    def dot_tn(A,B,out): return np.dot(A.T,B,out=out)

    @staticmethod
    def dot_nt(A,B,out): return np.dot(A,B.T,out=out)

    @staticmethod
    def square(A,out):   return np.square(A,out=out)

    @staticmethod
    def logistic(A,out):
        if out == None: out = A.copy()
        else:           out[:] = A[:]
        out *= -1
        np.exp(out,out=out)
        out += 1
        NumpyBackend.reciprocal(out,out=out)
        return out

    @staticmethod
    def tanh(A,out):     return np.tanh(A,out=out)

    @staticmethod
    def sqrt(A,out):     return np.sqrt(A,out=out)

    @staticmethod
    def exp(A,out):      return np.exp(A,out=out)

    @staticmethod
    def log(A,out):      return np.log(A,out=out)

    @staticmethod
    def abs(A,out):      return np.abs(A,out=out)

    @staticmethod
    def sign(A,out):     return np.sign(A,out=out)

    @staticmethod
    def sum(A,axis,out): return np.sum(A,axis=axis,out=out.ravel() if out != None else None)

    @staticmethod
    def mean(A,axis,out):return np.mean(A,axis=axis,out=out.ravel() if out != None else None)

    @staticmethod
    def add(A,B,out):       return np.add(A,B,out=out)

    @staticmethod
    def iadd(A,B):          A += B

    @staticmethod
    def iaddmul(A,B,alpha): B *= alpha; A += B

    @staticmethod
    def subtract(A,B,out):  return np.subtract(A,B,out=out)

    @staticmethod
    def isub(A,B):          A -= B

    @staticmethod
    def multiply(A,B,out):  return np.multiply(A,B,out=out)

    @staticmethod
    def imul(A,B):          A *= B

    @staticmethod
    def divide(A,B,out):    return np.divide(A,B,out=out)

    @staticmethod
    def idiv(A,B):          A /= B

    @staticmethod
    def reciprocal(A,out):  return np.divide(1.,A,out=out)

    @staticmethod
    def maximum(A,B,out):   return np.maximum(A,B,out=out)

#############################################

class GnumpyBackend(object):

    @staticmethod
    def empty(shape):    return gp.empty(shape)

    @staticmethod
    def zeros(shape):    return gp.zeros(shape)

    @staticmethod
    def ones(shape):     return gp.ones(shape)

    @staticmethod
    def rand(*shape):    return gp.garray(np.random.rand(*shape)) # gp.rand(*shape)

    @staticmethod
    def array(A):        return gp.garray(A)

    @staticmethod
    def asarray(A):      return gp.garray(A,copy=False)

    @staticmethod
    def as_numpy(A):     return A.as_numpy_array(default_dtype)

    @staticmethod
    def dot(A,B,out):
        if out == None:
            out = gp.empty((A.shape[0],B.shape[1]))
        cudamat.dot(B._base_as_2d(),A._base_as_2d(),target=out._base_as_2d())
        return out

    @staticmethod
    def dot_tn(A,B,out):
        if out == None:
            out = gp.empty((A.shape[1],B.shape[1]))
        cudamat.dot(B._base_as_2d(),A._base_as_2d().T,target=out._base_as_2d())
        return out

    @staticmethod
    def dot_nt(A,B,out):
        # Using B._base_as_2d().T does not work; cudamat returns dimensionality error
        B._base.mat.is_trans = not B._base.mat.is_trans
        if out == None:
            out = gp.empty((A.shape[1],B.shape[1]))
        cudamat.dot(B._base_as_2d(),A._base_as_2d(),target=out._base_as_2d())
        B._base.mat.is_trans = not B._base.mat.is_trans
        return out
    
    @staticmethod
    def square(A,out):
        if out == None:
            out = gp.empty(A.shape)
        cudamat.pow(A._base_as_row(),2,target=out._base_as_row())
        return out

    @staticmethod
    def _unary(func,A,out):
        if out == None:
            out = gp.empty(A.shape)
        func(A._base_as_row(),target=out._base_as_row())
        return out

    @staticmethod
    def logistic(A,out): return GnumpyBackend._unary(cudamat.sigmoid,A,out)

    @staticmethod
    def tanh(A,out):     return GnumpyBackend._unary(cudamat.tanh,A,out)

    @staticmethod
    def sqrt(A,out):     return GnumpyBackend._unary(cudamat.sqrt,A,out)

    @staticmethod
    def exp(A,out):      return GnumpyBackend._unary(cudamat.exp,A,out)

    @staticmethod
    def log(A,out):      return GnumpyBackend._unary(cudamat.log,A,out)

    @staticmethod
    def abs(A,out):      return GnumpyBackend._unary(cudamat.abs,A,out)

    @staticmethod
    def sign(A,out):     return GnumpyBackend._unary(cudamat.CUDAMatrix.sign,A,out)

    @staticmethod
    def sum(A,axis,out):
        if A.ndim == 2: 
            if out == None:
                out = gp.empty((A.shape[0],1) if axis == 1 else (1,A.shape[1]))
            cudamat.sum(A._base_shaped(1),1-axis,target=out._base_shaped(1))
            return out
        else:
            r = gp.sum(A,axis)  # gnumpy has optimized sum over 1D vectors, so use it
            if out != None:
                assert(out.size == 1)
                out[:] = r[:]
            return r

    @staticmethod
    def mean(A,axis,out):
        out = GnumpyBackend.sum(A,axis,out)
        GnumpyBackend.imul(out,1./A.shape[axis])
        return out

    @staticmethod
    def _add(A,B,out):
        if out == None:
            out = gp.empty(A.shape)
        if np.isscalar(B): 
            A._base_shaped(1).add(B,target=out._base_shaped(1))
        elif (B.ndim == 1 or B.shape[0] == 1) and B.size == A.shape[1]:
            A._base_shaped(1).add_col_vec(B._base_shaped(1),target=out._base_shaped(1))
        elif (B.ndim == 1 or B.shape[1] == 1) and B.size == A.shape[0]:
            A._base_shaped(1).add_row_vec(B._base_shaped(1),target=out._base_shaped(1))
        else:
            A._base_shaped(1).add(B._base_shaped(1),target=out._base_shaped(1))
        return out

    @staticmethod
    def add(A,B,out):
        # turn vec + matrix into matrix + vec
        if not np.isscalar(B) and (A.ndim < B.ndim or A.shape[0] < B.shape[0] or A.shape[1] < B.shape[1]):
            A,B = B,A
        return GnumpyBackend._add(A,B,out)

    @staticmethod
    def iadd(A,B):          GnumpyBackend._add(A,B,A)

    @staticmethod
    def iaddmul(A,B,alpha): A._base_shaped(1).add_mult(B._base_shaped(1),alpha)

    @staticmethod
    def subtract(A,B,out):
        if out == None:
            out = gp.empty(A.shape)
        if np.isscalar(B):         A._base_shaped(1).subtract(B,target=out._base_shaped(1))
        elif A.shape == B.shape:   A._base_shaped(1).subtract(B._base_shaped(1),target=out._base_shaped(1))
        else: raise NotImplementedError("broadcasted subtraction not implemented by cudamat")
        return out

    @staticmethod
    def isub(A,B):          GnumpyBackend.subtract(A,B,A)

    @staticmethod
    def _multiply(A,B,out):
        if out == None:
            out = gp.empty(A.shape)
        if np.isscalar(B): 
            A._base_shaped(1).mult(B,target=out._base_shaped(1))
        elif (B.ndim == 1 or B.shape[0] == 1) and B.size == A.shape[1]:
            A._base_shaped(1).mult_by_col(B._base_shaped(1),target=out._base_shaped(1))
        elif (B.ndim == 1 or B.shape[1] == 1) and B.size == A.shape[0]:
            A._base_shaped(1).mult_by_row(B._base_shaped(1),target=out._base_shaped(1))
        else:
            A._base_shaped(1).mult(B._base_shaped(1),target=out._base_shaped(1))
        return out

    @staticmethod
    def multiply(A,B,out):
        # turn vec * matrix into matrix * vec
        if not np.isscalar(B) and (A.ndim < B.ndim or A.shape[0] < B.shape[0] or A.shape[1] < B.shape[1]):
            A,B = B,A
        return GnumpyBackend._multiply(A,B,out)

    @staticmethod
    def imul(A,B):         GnumpyBackend._multiply(A,B,A)

    @staticmethod
    def divide(A,B,out):
        if out == None:
            out = gp.empty(A.shape)
        if np.isscalar(B):         A._base_shaped(1).divide(B,target=out._base_shaped(1))
        elif A.shape == B.shape:   A._base_shaped(1).divide(B._base_shaped(1),target=out._base_shaped(1))
        else: raise NotImplementedError("broadcasted division not implemented by cudamat")
        return out

    @staticmethod
    def idiv(A,B):          GnumpyBackend.divide(A,B,A)


    @staticmethod
    def reciprocal(A,out):
        if out == None:
            out = gp.empty(A.shape)
        A._base_as_row().reciprocal(out._base_as_row())
        return out

    @staticmethod
    def maximum(A,B,out):
        raise NotImplementedError("maximum not implemented in cudamat")

###############################################################
# Provide versions of numpy/gnumpy functions with "out" arguments
# since current version of gnumpy doesn't support 'out' functions
# (even though I hacked a few of them to support it)
# 
#
# These seemingly trivial mulx/addx functions exist because
# using A *= scalar with a gnumpy matrix creates extra copy_kernel instances
# on the GPU and seems slightly slower.
#


def empty(shape):          return backend.empty(shape)
def zeros(shape):          return backend.zeros(shape)
def ones(shape):           return backend.ones(shape)
def rand(*shape):          return backend.rand(*shape)
def array(A):              return backend.array(A)         # new copy of A
def asarray(A):            return backend.asarray(A)       # new *view* of A
def as_numpy(A):           return backend.as_numpy(A)
def dot(A,B,out=None):     return backend.dot(A,B,out)
def dot_tn(A,B,out=None):  return backend.dot_tn(A,B,out)
def dot_nt(A,B,out=None):  return backend.dot_nt(A,B,out)
def square(A,out=None):    return backend.square(A,out)   if not np.isscalar(A) else A*A
def logistic(A,out=None):  return backend.logistic(A,out) if not np.isscalar(A) else 1./(1+np.exp(-A))
def tanh(A,out=None):      return backend.tanh(A,out)     if not np.isscalar(A) else np.tanh(A)
def sqrt(A,out=None):      return backend.sqrt(A,out)     if not np.isscalar(A) else np.sqrt(A)
def exp(A,out=None):       return backend.exp(A,out)      if not np.isscalar(A) else np.exp(A)
def log(A,out=None):       return backend.log(A,out)      if not np.isscalar(A) else np.log(A)
def abs(A,out=None):       return backend.abs(A,out)      if not np.isscalar(A) else np.abs(A)
def sign(A,out=None):      return backend.sign(A,out)     if not np.isscalar(A) else np.sign(A)
def sum(A,axis=0,out=None):return backend.sum(A,axis,out)
def mean(A,axis=0,out=None):return backend.mean(A,axis,out)
def add(A,B,out=None):     return backend.add(A,B,out)       # A + B
def iadd(A,B):             return backend.iadd(A,B)          # A += B
def iaddmul(A,B,alpha):    return backend.iaddmul(A,B,alpha) # A += B*alpha (WARNING: value stored in B is undefined after this)
def subtract(A,B,out=None):return backend.subtract(A,B,out)  # A - B
def isub(A,B):             return backend.isub(A,B)          # A -= B
def multiply(A,B,out=None):return backend.multiply(A,B,out)  # A * B
def imul(A,B):             return backend.imul(A,B)          # A *= B
def divide(A,B,out=None):  return backend.divide(A,B,out)    # A / B
def idiv(A,B):             return backend.idiv(A,B)          # A /= B
def reciprocal(A,out=None):return backend.reciprocal(A,out)  # 1. / A 
def maximum(A,B,out=None): return backend.maximum(A,B,out)



###################################################

def set_backend(name,dtype='float32'):
    global backend
    global default_dtype
    if name == 'gnumpy':
        assert(dtype == 'float32')
        if not _gnumpy_loaded:
            raise RuntimeError("cannot set backend to gnumpy; module 'gnumpy' failed to import")
        backend = GnumpyBackend
        default_dtype = 'float32'
    elif name == 'numpy':
        backend = NumpyBackend
        default_dtype = dtype
    else:
        raise ValueError("unrecognized backend '%s'" % name)

def sync_backend(): 
    '''Manually calls cudaSynchronizeThreads() if using cuda, else does nothing'''
    if _gnumpy_loaded:
        cudamat.cuda_sync_threads()

set_backend('numpy')