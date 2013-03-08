from BigMat import *
from Util import TempMatrix

class Activation(object):
    '''
    An activation function.
    An instance f can be evaluated element-wise on input matrix A three ways:
       H = f(A)             # transform A by f(.)
       f(A,out=H)           # same, but matrix H pre-allocated
       f(A,out=H,dout=df)   # compute both H=f(A), and df=f'(A)
    '''
    def ideal_loss(self): return 'mse'

class ActivationLogistic(Activation):
    '''Activation function sigmoid(A), i.e. logisitic function'''
    def name(self):         return "logistic"
    def ideal_domain(self): return [-0.1,1.1]
    def ideal_range(self):  return [ 0.1,0.9]

    def __call__(self,A,out=None,dout=None):
        if out == None:
            return 1./(1+expx(-A))
        logistic(A,out=out)
        if dout != None:
            square(out,out=dout)
            imul(dout,-1)
            iadd(dout,out)


class ActivationTanh(Activation):
    '''Activation function tanh(A)'''
    def name(self):         return "tanh"
    def ideal_domain(self): return [-1.2,1.2]
    def ideal_range(self):  return [-0.9,0.9]

    def __call__(self,A,out=None,dout=None):
        if out == None:
            return tanhx(A)
        tanh(A,out=out)
        if dout != None:
            square(out,out=dout)
            imul(dout,-1)
            iadd(dout,1)


class ActivationRelu(Activation):
    '''Activation function max(0,A), i.e. rectified linear'''
    def name(self):         return "relu"
    def ideal_domain(self): return [-1.2,1.2]
    def ideal_range(self):  return [ 0.0,1.0]

    def __call__(self,A,out=None,dout=None):
        if out == None:
            return maximum(0,A)
        maximum(0,A,out=out)
        if dout != None:
            sign(out,out=dout)


class ActivationSoftmax(Activation):
    '''Activation function softmax(A)'''
    def __init__(self):
        self._tmp_denom = TempMatrix()

    def name(self):         return "softmax"
    def ideal_domain(self): return [0.0,1.0]
    def ideal_range(self):  return [0.0,1.0]
    def ideal_loss(self):   return 'nll'

    def __call__(self,A,out=None,dout=None):
        # First pre-allocate enough memory to accumulate denominator of each sample
        denom = self._tmp_denom.get_capacity(A.shape[0],1)

        # Then compute softmax
        if out == None:
            expA = exp(A)
            sum(expA,axis=1,out=denom)
            return (1./denom) * expA
        exp(A,out=out)
        sum(out,axis=1,out=denom)
        reciprocal(denom,out=denom)
        multiply(out,denom,out=out)
        if dout != None:
            pass # for Softmax+NLL, 'df' is not used to compute Cost.dloss, so don't bother setting the dout in that case


##########################################################

def make_activation(typename):
    if   typename == "logistic": return ActivationLogistic()
    elif typename == "tanh":     return ActivationTanh()
    elif typename == "relu":     return ActivationRelu()
    elif typename == "softmax":  return ActivationSoftmax()
    elif typename == None:       return None
    raise ValueError("unrecognized activation function '%s'" % typename)
