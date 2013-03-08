from numpy import *
import BigMat as bm
import cPickle as cp
from time import time as now

class BatchSet(object):
    def __init__(self,X,Y,batchsize):
        self._size = m = X.shape[0]
        self._X = X
        self._Y = Y
        self._index = 0
        self._batchsize = batchsize
        self._blocksize = batchsize * max(1,2048//batchsize)
        self._batches = vstack([arange(0        ,m          ,batchsize),
                                arange(batchsize,m+batchsize,batchsize)]).transpose()
        self._batches[-1,-1] = m

    def __iter__(self):
        return self

    def next(self):
        if self._index >= len(self._batches):
            raise StopIteration
        s = slice(*(self._batches[self._index]))
        batch = DataFold(self._X[s,:],self._Y[s,:])
        self._index += 1
        return batch

    def shuffle(self):
        random.shuffle(self._batches)
        self._index = 0

class DataFold(object):
    '''
    A simple structure containing a subset of inputs X,
    and the corresponding target outputs Y
    '''
    def __init__(self,X,Y):
        assert(X.shape[0] == Y.shape[0])
        self.X = X
        self.Y = Y
        self.size = X.shape[0]

    def __iter__(self):
        return [self.X,self.Y].__iter__()   # let X,Y = data unpack

    def __getitem__(self,i):
        return DataFold(self.X[i,:],self.Y[i,:])

    def make_batches(self,batchsize):
        return BatchSet(self.X,self.Y,batchsize)


class DataSet(object):
    '''
    A simple structure containing three DataFold instances:
    a 'train', 'valid', and 'test.
    '''
    def __init__(self,X,Y,Xshape=None,Yshape=None,Xrange=None,Yrange=None):
        self._X = X
        self._Y = Y
        self._size  = X.shape[0]
        self.Xshape = Xshape or (1,X.shape[1])
        self.Yshape = Yshape or (1,Y.shape[1])
        self.Xdim   = X.shape[1]
        self.Ydim   = Y.shape[1]
        self.Xrange = Xrange or (X.ravel().min(),X.ravel().max())
        self.Yrange = Yrange or (Y.ravel().min(),Y.ravel().max())
        self.train = DataFold(X[:,:],Y[:,:])
        self.valid = DataFold(X[0:0,:],Y[0:0,:])
        self.test  = DataFold(X[0:0,:],Y[0:0,:])

    def keys(self):   return ['train','valid','test']
    def values(self): return [self.train,self.valid,self.test]
    def items(self):  return zip(self.keys(),self.values())

    def __getitem__(self,key):
        if   key == 'train': return self.train
        elif key == 'valid': return self.valid
        elif key == 'test':  return self.test
        raise KeyError("invalid key for DataSet fold")

    def shuffle(self):
        perm = random.permutation(self._size)
        self._X[:] = self._X[perm].copy()
        self._Y[:] = self._Y[perm].copy()

    def split(self,trainsplit,validsplit=0,testsplit=0):
        assert(trainsplit + validsplit + testsplit <= 100)
        trainsize = int(trainsplit * self._size // 100)
        validsize = int(validsplit * self._size // 100)
        testsize  = int(testsplit  * self._size // 100)
        self.train.X    = self._X[:trainsize]
        self.train.Y    = self._Y[:trainsize]
        self.train.size = trainsize
        self.valid.X    = self._X[trainsize:trainsize+validsize]
        self.valid.Y    = self._Y[trainsize:trainsize+validsize]
        self.valid.size = validsize
        self.test.X     = self._X[trainsize+validsize:trainsize+validsize+testsize]
        self.test.Y     = self._Y[trainsize+validsize:trainsize+validsize+testsize]
        self.test.size  = testsize

    def rescale(self,Xrange,Yrange):
        '''
        Rescales the entire dataset so that all inputs X lie within (Xrange[0],Xrange[1])
        and all targets Y lie within (Yrange[0],Yrange[1]).
        The same scaling factor is applied to all folds.
        '''
        if Xrange != self.Xrange:
            for fold in self.values():
                fold.X -= self.Xrange[0]
                fold.X *= (Xrange[1]-Xrange[0])/(self.Xrange[1]-self.Xrange[0])
                fold.X += Xrange[0]
            self.Xrange = Xrange
        if Yrange != self.Yrange:
            for fold in self.values():
                fold.Y -= self.Yrange[0]
                fold.Y *= (Yrange[1]-Yrange[0])/(self.Yrange[1]-self.Yrange[0])
                fold.Y += Yrange[0]
            self.Yrange = Yrange

################################################

def pickle(filename,object):
    f = open(filename,'wb')
    cp.dump(object,f,-1)
    f.close()

def unpickle(filename):
    f = open(filename,'rb')
    object = cp.load(f)
    f.close()
    return object

def load_mnist(digits=range(10),split=[50,15,35],xform=[]):
    X,Y = [],[]
    for d in digits:
        for set in ('train','test'):
            # Load all N instances of digit 'd' as a Nx768 row vector of inputs, 
            # and an Nx10 target vector. 
            Xd,Yd = unpickle("data/mnist/mnist_%s_%i.pkl" % (set,d))
            X.append(Xd)
            Y.append(Yd[:,digits])  # make the output dimensionality match the number of actual targets, for faster training on subsets of digits

            # Apply transformations to the digitis
            for f in xform:
                Ximg = Xd.reshape((-1,28,28)).copy()
                if   f[0] > 0: Ximg[:,:,f[0]:] = Ximg[:,:,:-f[0]]; Ximg[:,:,:f[0]] = 0
                elif f[0] < 0: Ximg[:,:,:f[0]] = Ximg[:,:,-f[0]:]; Ximg[:,:,f[0]:] = 0
                if   f[1] > 0: Ximg[:,f[1]:,:] = Ximg[:,:-f[1],:]; Ximg[:,:f[1],:] = 0
                elif f[1] < 0: Ximg[:,:f[1],:] = Ximg[:,-f[1]:,:]; Ximg[:,f[1]:,:] = 0
                X.append(Ximg.reshape((-1,28*28)))
                Y.append(Yd[:,digits])




    X = bm.asarray(vstack(X))
    Y = bm.asarray(vstack(Y))
    
    data = DataSet(X,Y,Xshape=(28,28),Xrange=[0.0,255.0],Yrange=[0.0,1.0])
    data.shuffle()
    data.split(*split)
    return data
    
