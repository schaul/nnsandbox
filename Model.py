from BigMat import *
from Util import TempMatrix
import numpy as np
import numpy.random as random

##############################################################

class Model(object):
    '''
    A trainable model. parameters,
    inputs X, and targets Y.

    Let Z = M(X) be the final outputs, and let H be the hidden activities incurred by X.
    A cost is defined as the sum of three additive components:
       - loss(Z,Y) depends on the final outputs and targets
       - regularizer(H) depends only on the hidden activations
       - penalty(M) depends ony on the model and its own parameters
    '''
    def __init__(self,loss=None):
        if isinstance(loss,str):
            self._loss_type = loss
            self._loss_fn = None
            self._loss_delta_fn = None
            if   loss == 'mse': self._loss_fn = self._loss_mse; self._loss_delta_fn = self._loss_delta_mse
            elif loss == 'nll': self._loss_fn = self._loss_nll; self._loss_delta_fn = self._loss_delta_nll
            elif loss: raise ValueError("unrecognized loss '%s' requested" % loss_type)
        else:
            self._loss_type = 'callback'
            self._loss_fn = loss[0]
            self._loss_delta_fn = loss[1]

        self._tmp_E = TempMatrix()  # memory for computing loss
        self._tmp_e = TempMatrix()  # memory for computing loss

    def cost(self,data):
        '''
        Computes the cost = loss(output(X),Y) + regularizer(hidden(X)) + penalty(model)
        '''
        X,Y = data
        H = self.eval(X,want_hidden=True)
        l = self.loss(H[-1],Y)
        r = self.regularizer(H)
        p = self.penalty()
        c = l+r+p
        return c,l,r,p

    def apply_constraints(self):
        pass

    ############### LOSS ###############

    def loss(self,Z,Y):
        '''Loss of outputs Z with respect to targets Y.'''
        return float(self._loss_fn(Z,Y))

    def _loss_delta(self,Z,Y,df,out=None):
        '''Computes the gradient error signal to backpropagate up the network.
           Here df is the derivative f'(A) of the output activation f(A).'''
        return self._loss_delta_fn(Z,Y,df,out)

    def _loss_mse(self,Z,Y):
        """Mean squared error (mse) of ouputs Z with respect to targets Y."""
        E = self._tmp_E.get_capacity(*Z.shape)
        e = self._tmp_e.get_capacity(Z.shape[0],1)
        subtract(Z,Y,out=E)
        square(E,out=E)
        sum(E,axis=1,out=e)
        return 0.5*as_numpy(mean(e))               # = mean(sum(square(Z-Y),axis=1))

    def _loss_delta_mse(self,Z,Y,df,out=None):
        """Computes the MSE gradient error signal to backpropagate up the network.
           Here df is the derivative f'(A) of the output activation f(A)."""
        self._loss_delta_nll(Z,Y,df,out)
        imul(out,df)                       # = 1/m * (Z-Y) * df    for mse
        return out

    def _loss_nll(self,Z,Y):
        '''Negative log-likelihood of outputs Z with respect to targets Y.'''
        E = self._tmp_E.get_capacity(*Z.shape)
        e = self._tmp_e.get_capacity(Z.shape[0],1)
        multiply(Z,Y,out=E)
        sum(E,axis=1,out=e)
        log(e,out=e)
        return -as_numpy(mean(e))              # = -mean(log(sum(Z*Y,axis=1)))

    def _loss_delta_nll(self,Z,Y,df,out=None):
        """Computes the NLL gradient error signal to backpropagate up the network."""
        subtract(Z,Y,out=out)
        imul(out,1./Z.shape[0])            # = 1/m * (Z-Y)         for nll
        return out


    ############### REGULARIZER ###############

    def regularizer(self,H):
        return 0.0

    ############### PENALTY ###############

    def penalty(self):
        return 0.0

    ################################# UTILITY FUNCTIONS #########################

    def relative_error(self,A,B,abs_eps):
        absA = np.abs(A)
        absB = np.abs(B)
        I = np.logical_not(np.logical_or(A==B,np.logical_or(absA < abs_eps, absB < abs_eps)))
        E = np.zeros(A.shape,dtype=A.dtype)
        E[I] = np.abs(A[I]-B[I]) / min(absA[I] + absB[I])
        return E

    def gradcheck(self,data):
        # Only use a tiny subset of the data, both for speed and to avoid 
        # averaging gradient of many inputs.
        data_subset = data[:min(4,data.size)]

        # Swap a new weights object into the model, so that we can purturb the model's weights from outside
        weights0 = self.weights
        self.weights = weights1 = weights0.copy()

        # Compute gradient by forward-difference, looping over each individual weight
        neps  = 1e-7
        ngrad = self.make_weights()
        for k in range(len(weights1)):
            w1 = weights1[k]; wg = ngrad[k]
            for i in range(len(w1)):
                # temporarily perturb parameter w[i] by 'neps' and evaluate the new loss
                temp = w1[i]; 
                w1[i] -= neps; c0,l0,r0,p0 = self.cost(data_subset); w1[i] = temp
                w1[i] += neps; c1,l1,r1,p1 = self.cost(data_subset); w1[i] = temp
                wg[i] = (c1-c0)/(2*neps)

        self.weights = weights0  # restore the original weights object for the model

        # Compute backprop's gradient (bgrad), keeping loss/regularizer/penalty separate
        bgrad = self.grad(data_subset)

        A = ngrad.ravel()
        B = bgrad.ravel()
        aerr = np.abs(A-B)
        rerr = self.relative_error(A,B,1e-20)
        print 'absolute_error in gradient: min =', np.min(aerr), 'max =', np.max(aerr)
        print 'relative_error in gradient: min =', np.min(rerr), 'max =', np.max(rerr)
        

