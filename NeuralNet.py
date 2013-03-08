from itertools import izip
from operator import mul
from Activation import make_activation
from BigMat import *
from Model import *
import numpy as np

##############################################################

class NeuralNetCfg(object):
    '''
    A list of layer-specifiers, where each layer specifies 
    the type, size, and behaviour of a single network layer.
    Activation is a string: 'logistic','tanh','relu' or 'softmax'.
    Maxnorm is the maximum norm of the weights *entering* 
    each hidden unit of the layer, or None if no limit.
    '''
    def __init__(self,L1=0.0,L2=0.0,maxnorm=None,dropout=None,sparsity=None):
        # These values (L1,L2,maxnorm,dropout) are the defaults for
        # all layers of this neural net, but can be overridden
        # for a specific layer via add_layer
        self.L1 = L1
        self.L2 = L2
        self.maxnorm = maxnorm
        self.dropout = dropout
        self.sparsity = sparsity
        
        self._layers = [None,None]

    def input(self,size,dropout=None):
        layer = NeuralNetLayerCfg(size,None)
        layer.dropout = dropout
        self._layers[0] = layer

    def hidden(self,size,activation,L1=None,L2=None,maxnorm=None,dropout=None,sparsity=None):
        layer = NeuralNetLayerCfg(size,activation)
        layer.L1 = L1
        layer.L2 = L2
        layer.maxnorm = maxnorm
        layer.dropout = dropout
        layer.sparsity = sparsity
        self._layers.insert(-1,layer)

    def output(self,size,activation,L1=None,L2=None,maxnorm=None):
        layer = NeuralNetLayerCfg(size,activation)
        layer.L1 = L1
        layer.L2 = L2
        layer.maxnorm = maxnorm
        layer.dropout = None
        layer.sparsity = None
        self._layers[-1] = layer

    def __len__(self):       return len(self._layers)
    def __iter__(self):      return self._layers.__iter__()
    def __getitem__(self,i): return self._layers[i]

    def finalize(self):
        assert(self._layers[0]  != None) # input layer must be specified
        assert(self._layers[-1] != None) # output layer must be specified
        K = len(self._layers)
        for k in range(K):
            layer = self._layers[k]
            layer.f = make_activation(layer.activation)
            if k > 0:
                if layer.L1 == None:       layer.L1 = self.L1
                if layer.L2 == None:       layer.L2 = self.L2
                if layer.maxnorm == None:  layer.maxnorm = self.maxnorm
                if layer.sparsity == None: layer.sparsity = self.sparsity
                if k < K:
                    if layer.dropout == None:  layer.dropout = self.dropout
        return self

    def __repr__(self):  # Useful for printing 
        str = ''
        for name,value in self.__dict__.iteritems():
            if name == '_layers':
                continue
            if (isinstance(value,NeuralNetLayerCfg)):
                str += '{0}=...\n'.format(name)
            elif (isinstance(value,basestring)):
                str += '{0}=\'{1}\'\n'.format(name,value)
            else:
                str += '{0}={1}\n'.format(name,value)
        for k in range(len(self._layers)):
            str += ('layer[%d] = {\n' % k) + self._layers[k].__repr__() + '}\n'
        return str

##############################################################

class NeuralNetLayerCfg(object):
    def __init__(self,size,activation):
        self.shape      = (size,) if isinstance(size,(long,int)) else size  # Shape can be a tuple, like (28,28) for MNIST, for example.
        self.size       = reduce(mul,list(self.shape))         # Size should always be the raw number of units in the layer.
        self.activation = activation
        self.f          = None # will be set to make_activation
        self.L1         = None
        self.L2         = None
        self.maxnorm    = None
        self.dropout    = None
        self.sparsity   = None

    def __repr__(self):  # Useful for printing 
        str = ''
        for name,value in self.__dict__.iteritems():
            if (name in ('shape','f')) or value == None:
                continue
            if (isinstance(value,basestring)):
                str += '  {0}=\'{1}\'\n'.format(name,value)
            else:
                str += '  {0}={1}\n'.format(name,value)
        return str


##############################################################

class DenseWeights(object):
    '''
    A set of dense weights, going from srclayer to dstlayer.
    init_scale is the scale of the random initial weights, centered about zero.
    Operations:
        += DenseWeights
        -= DenseWeights
        *= scalar
        W,b = DenseWeights   (unpacks into ref to weights 'W' and ref to biases 'b')
    '''
    def __init__(self,inlayer,outlayer,init_scale=0.0):
        self.inlayer  = inlayer
        self.outlayer = outlayer

        # Initialize to small random values uniformly centered around 0.0
        n,m = inlayer.size,outlayer.size
        self.W = 2*init_scale*(rand(n,m)-0.5)
        self.b = 2*init_scale*(rand(1,m)-0.5)

        self._tmp_W = None

    def copy(self):
        cp = DenseWeights(self.inlayer,self.outlayer)
        cp.W[:] = self.W[:]
        cp.b[:] = self.b[:]
        return cp

    def ravel(self):
        return np.hstack([as_numpy(self.W).ravel(),as_numpy(self.b).ravel()])

    def get_tmp_W(self):
        if self._tmp_W == None:
            self._tmp_W = (empty(self.W.shape),empty((1,self.W.shape[1])))
        return self._tmp_W

    def __iter__(self):
        return [self.W,self.b].__iter__() # Used so that "W,b = weights" works, for convenience ... unpack W, then b

    def __len__(self):       return self.W.size + self.b.size
    def __getitem__(self,i): return self.W.flat[i] if i < self.W.size else self.b.flat[i - self.W.size]
    def __setitem__(self,i,value):
        if i < self.W.size: self.W.flat[i] = value
        else:               self.b.flat[i - self.W.size] = value

    def __iadd__(self,other):
        iadd(self.W,other.W)
        iadd(self.b,other.b)
        return self

    def __isub__(self,other):
        isub(self.W,other.W)
        isub(self.b,other.b)
        return self

    def __imul__(self,alpha):
        imul(self.W,alpha)
        imul(self.b,alpha)
        return self

##############################################################

class WeightSet(object):
    '''
    A set of all weights defining a NeuralNet.
    Operations:
        += WeightSet
        -= WeightSet
        *= scalar
        len(WeightSet)
    '''
    def __init__(self,cfg,init_scale=0.0):
        # For each pair of consecutive layers, create a dense set of weights between them
        self._layers = [DenseWeights(cfg[k],cfg[k+1],init_scale)     for k in range(len(cfg)-1) ] if cfg else None

    def copy(self):
        ws = WeightSet(None)
        ws._layers = [layer.copy() for layer in self._layers]
        return ws

    def ravel(self):
        return np.hstack([layer.ravel() for layer in self._layers])

    def step_by(self,delta,alpha=1.0):
        for layer,dlayer in zip(self._layers,delta._layers):
            iaddmul(layer.W,dlayer.W,alpha)
            iaddmul(layer.b,dlayer.b,alpha)

    def __getitem__(self,i):
        return self._layers[i]

    def __iter__(self):
        return self._layers.__iter__()

    def __len__(self):
        return len(self._layers)

    def __iadd__(self,other):
        for w,v in izip(self._layers,other._layers):
            w += v
        return self

    def __isub__(self,other):
        for w,v in izip(self._layers,other._layers):
            w -= v 
        return self

    def __imul__(self,alpha):
        for w in self._layers:
            w *= alpha
        return self


##############################################################

class NeuralNet(Model):
    '''
    A NeuralNet is defined by a sequence of layers.
    Each layer has its own size, and its own activation function.
    Each pair of consecutive layers has its own weights,
    and the set of all weights is contained in the member
    function 'weights'.
    '''
    def __init__(self,cfg):
        self._cfg = cfg.finalize()
        Model.__init__(self,cfg[-1].f.ideal_loss())

        init_weight_scale = 0.005
        self.weights = WeightSet(cfg,init_weight_scale)

        # Each _tmp_H[k] and _tmp_df[k] contains pre-allocated buffers for storing
        # information during forwardprop that is useful during backprop.
        # It is intended to be used on minibatches only, since the
        # gradient (backprop) is never needed on the full training set,
        self._tmp_H  = [TempMatrix(1,w.outlayer.size) for w in self.weights]  # H  = f(A)
        self._tmp_df = [TempMatrix(1,w.outlayer.size) for w in self.weights]  # df = f'(A)
        self._tmp_D  = [TempMatrix(1,w.outlayer.size) for w in self.weights]  # Delta for backprop
        self._tmp_R  = [TempMatrix(1,w.outlayer.size) for w in self.weights]  # temp for regularizer

    def numlayers(self):
        '''The number of layers, including output but excluding input.'''
        return len(self.weights) # everything but the input layer is a 'layer'

    def ideal_domain(self): return self._cfg[ 1].f.ideal_domain()   # first hidden layer's ideal domain
    def ideal_range(self):  return self._cfg[-1].f.ideal_range()    # output layer's ideal range

    def make_weights(self,init_scale=0.0): return WeightSet(self._cfg,init_scale)

    def __call__(self,X):
        return self.eval(X)

    def eval(self,X,want_hidden=False,want_grad=False):
        '''
        Given (m x n_0) matrix X, evaluate all m inputs on the neural network.
        The result is an (m x n_K) matrix of final outputs.
        '''
        m,n = X.shape
        assert(m >= 1)
        assert(n == self._cfg[0].size)
        H  = self._get_tmp(self._tmp_H ,m) if want_grad else [empty((m,layer.size)) for layer in self._cfg[1:]]
        df = self._get_tmp(self._tmp_df,m) if want_grad else [None                  for layer in self._cfg[1:]]

        # Forward pass, starting from earliest layer, storing all intermediate computations
        for k in range(self.numlayers()):
            Hj = H[k-1] if k > 0 else X # Hj has well-defined value
            Hk = H[k]                   # Hk has undefined value, at this point

            # Compute activation inputs A and store them in memory Hk
            W,b = self.weights[k]
            dot(Hj,W,out=Hk);
            iadd(Hk,b)

            # Apply the activation function f(.) and its derivative f'(.) to A 
            f = self.weights[k].outlayer.f
            f(Hk,out=Hk,dout=df[k]) # compute output f(A) and, if for_backprop, also keep f'(A) in the eval_mem        

        if not want_hidden:
            H,df = H[-1],df[-1] # return only the final output, unless all layers were explicitly requested
        return (H,df) if want_grad else H

    def backprop(self,X,Y,out=None):
        '''
        Compute the gradient of a cost function.
        Warning: assumes that the eval() forward pass was called previously, which caches
                 all values needed to evaluate the corresponding backward pass.
        '''
        dweights = out or self.make_weights()

        # H[k] and df[k] are assumed to have been computed in a call to _eval(), i.e. a forward pass.
        H  = self._get_tmp(self._tmp_H)
        df = self._get_tmp(self._tmp_df)
        D  = self._get_tmp(self._tmp_D,Y.shape[0])  # D[k] is temporary storage for delta
        R = self._get_tmp(self._tmp_R,Y.shape[0])   # R[k] is temporary storage for computing regularizer

        # Calculate initial Delta based on loss function, outputs Z=H[-1] and targets Y
        self._loss_delta(Y,H[-1],df[-1],out=D[-1])

        # Backward pass
        for k in reversed(range(self.numlayers())):
            Wk,bk = self.weights[k]
            dW,db = dweights[k]

            # Compute gradient contribution of loss function
            j = k-1
            Hj = H[j] if k > 0 else X
            dot_tn(Hj,D[k],out=dW)
            sum(D[k],axis=0,out=db)

            # Add gradient contribution of penalty 
            self._penalty_grad(self.weights[k],dW)

            # compute the Delta value for the next iteration k-1
            if k > 0:
                dot_nt(D[k],Wk,out=D[j])
              
                # Add gradient contribution of hidden-unit regularizer
                self._regularizer_delta(j,H[j],D[j],R[j])
                
                # Multiply Delta by f'(A) from the corresponding layer
                imul(D[j],df[j])

        return dweights


    def apply_constraints(self):
        self._normalize_weights()

    ############### REGULARIZER ###############

    def regularizer(self,H):
        '''Returns the sum of all regularization costs on hidden units (sparsity cost)'''
        R = self._get_tmp(self._tmp_R,H[0].shape[0])
        cost = 0.0
        for k in range(self.numlayers()):
            lambd,alpha = self._cfg[k+1].sparsity or (0.0,1.0)
            if lambd == 0.0:
                continue
            square(H[k],out=R[k])
            add(R[k],alpha**2,out=R[k])
            log(R[k],out=R[k])           # log(h^2 + alpha^2)
            cost += lambd*as_numpy(sum(R[k].ravel()))/H[k].shape[0]
            cost -= lambd*H[k].shape[1]*log(alpha**2)  # subtract off a constant to make perfect sparsity have zero cost
        return cost

    def _regularizer_delta(self,k,Hk,Dk,Rk):
        '''
        Adds the hidden-unit regularizer contribution to the delta matrix Dk
        for layer k, based on hidden activations Hk
        '''
        lambd,alpha = self._cfg[k+1].sparsity or (0.0,1.0)
        if lambd == 0.0:
            return
        square(Hk,out=Rk)
        iadd(Rk,alpha**2)
        divide(Hk,Rk,out=Rk)
        iaddmul(Dk,Rk,lambd * 2. / Hk.shape[0])   # D += lambda * 2/m * H ./ (H.^2 + alpha^2)

    ############### PENALTY ###############

    def penalty(self):
        L1 = L2 = 0.0
        for layer in self.weights:
            if layer.outlayer.L1 > 0.0:
                absW,_ = layer.get_tmp_W()
                abs(layer.W,out=absW)
                L1 += layer.outlayer.L1*as_numpy(sum(absW.ravel()))    # L1 * sum(abs(W))
            if layer.outlayer.L2 > 0.0:
                sqrW,_ = layer.get_tmp_W()
                square(layer.W,out=sqrW)
                L2 += layer.outlayer.L2*0.5*as_numpy(sum(sqrW.ravel()))   # L2 * 0.5 * sum(W.^2)
        return L1 + L2

    def _penalty_grad(self,layer,dW):
        if layer.outlayer.L1 > 0.0:
            W,_ = layer.get_tmp_W()
            sign(layer.W,out=W)
            iaddmul(dW,W,layer.outlayer.L1)      # dW += L1 * sign(W)
        if layer.outlayer.L2 > 0.0:
            W,_ = layer.get_tmp_W()
            multiply(layer.W,layer.outlayer.L2,out=W)
            iadd(dW,W)                           # dW += L2 * W

    ###################### UTILITY FUNCTIONS ######################

    def _normalize_weights(self):
        for layer in self.weights:
            if layer.outlayer.maxnorm:
                # Get tmp matrix the same size as this layer's incoming weight matrix
                W,w = layer.get_tmp_W()
                
                # Compute the square of the norm of weights entering each destination unit (norm along rows)
                square(weights.W,out=W)
                sum(W,axis=0,out=w)

                # Make sure all norms <= maxnorm have no effect
                maximum(w,outlayer.maxnorm**2,out=w)

                # Divide each W[i,j] > maxnorm by its actual norm
                sqrt(w,out=w)
                weights.W /= w


    def _get_tmp(self,temp_list,m=-1):
        if m == -1:
            return [ A.get() for A in temp_list ]
        else:
            return [ A.get_capacity(m) for A in temp_list ]
