""" First attempt at parsing the log files

Tom """

from Util import quickload, quickdump
from gen_trainees import Setting
from scipy import zeros, log10
import os 
from pybrain.datasets.supervised import SupervisedDataSet

logdir = '../../Dropbox/metanets/data/trainees/mnist'


def getAllFilesIn(dir, tag='', extension='.pkl'):
    """ return a list of all filenames in the specified directory
    (with the given tag and/or extension). """
    allfiles = os.listdir(dir)
    res = []
    for f in allfiles:
        if f[-len(extension):] == extension and f[:len(tag)] == tag:
            res.append(dir + '/' + f)#[:-len(extension)])
    return res


def inspect1():
    for f in sorted(getAllFilesIn(logdir)[:60]):
        print f
        x = quickload(f)
        print len(x), type(x)
        for i, (k, v) in enumerate(x.items()):
            print i, k, type(v)
        for k, v in x['setting'].items():
            print k
            for k, vv in v.items():
                print '  ', k, vv
        print
        for i, sn in enumerate(x['snapshots']):
            print i
            for k, v in sn.items():
                if k != 'weights':
                    print '\t', k, v
        break
    
set_feats = 13
snap_feats = 3 + 2 * 4
num_targ = 1 


def parseSnapshot(snapshot, verbose=False):
    """"""
    res = zeros(snap_feats)
    ri = 0
    for k, v in sorted(snapshot.items()):
        if k in ['time', 'test', 'weights']:
            continue
        if verbose:
            print k
        if k in ['train', 'valid']:
            res[ri:ri + 4] = [(v[kk] + 1e-10) for kk in sorted(v.keys())] 
            ri += 4
        else:
            if v is not None:
                res[ri] = log10(v) 
            ri += 1
            
    return res

def parseFeatures(settings, snapshots, verbose=False):
    """ Transform a dictionary of settings, and a variable number of snapshot data into a single 
    feature vector """
    res = zeros(set_feats + len(snapshots) * snap_feats)
    for i, snapshot in enumerate(snapshots):
        res[set_feats + i * snap_feats:set_feats + (i + 1) * snap_feats] = parseSnapshot(snapshot, verbose=verbose)
    
    ri = 0
    for k, v in sorted(settings['model'].items()) + sorted(settings['train'].items()):
        if k in ['activation', 'epochs']:
            continue
        if verbose:
            print k
        if k == 'sparsity':
            if v is None:
                ri += 2
            else:
                res[ri] = log10(v[0])
                ri += 1
                res[ri] = log10(v[1])
                ri += 1
            continue
        if v is None:
            ri += 1
            continue
        if k in ['momentum', 'learn_rate_decay']:
            v = 1 - v
        elif k in ['dropout', 'size']:
            v = sum(v)
        if v > 0 and not k in ['dropout']:
            v = log10(v)
        res[ri] = v
        ri += 1
        if ri > set_feats:
            print 'Oh-oh', ri, set_feats, settings
            break
    if verbose:
        print res
    return res
    
    
def parseTarget(snapshot):
    """ Extract the target predictions (single vector) """
    return snapshot['valid']['error rate']
    
def buildDataset(filenames,
                 history=2, # how many snapshots into the past?
                 ):
    D = SupervisedDataSet(set_feats + history * snap_feats, num_targ)
    for fname in filenames:
        rundata = quickload(fname)
        snapshots = rundata['snapshots']
        settings = rundata['setting']
        for i in range(len(snapshots) - history - 1):
            inp = parseFeatures(settings, snapshots[i:i + history])
            prevtarget = parseTarget(snapshots[i + history-1])
            nexttarget = parseTarget(snapshots[i + history])
            # percentage gain
            target = (-nexttarget+prevtarget)/(nexttarget+prevtarget)/2.
            D.addSample(inp, [target])        
    return D
    
ds_file = 'sup_dataset.pkl'

def readAndStore():
    D = None
    try:
        D = quickload(ds_file)        
    except Exception, e:
        print 'Oh-oh', e
        
    if D is None:
        D = buildDataset(sorted(getAllFilesIn(logdir))[:50])
    else:
        print 'already found'    
        
    quickdump(ds_file, D)
    return D
    
def testTraining(D):
    print len(D), 'samples'
    from core.datainterface import ModuleWrapper
    from algorithms import SGD, vSGDfd
    import pylab
    from pybrain.datasets import SupervisedDataSet
    from pybrain import LinearLayer, FullConnection, FeedForwardNetwork, TanhLayer, BiasUnit
    from pybrain.utilities import dense_orth
    net = FeedForwardNetwork()
    net.addInputModule(LinearLayer(D.indim, name='in'))
    net.addModule(BiasUnit(name='bias'))
    net.addModule(TanhLayer(14, name='h'))
    net.addOutputModule(LinearLayer(1, name='out'))
    net.addConnection(FullConnection(net['in'], net['h']))
    net.addConnection(FullConnection(net['bias'], net['h']))
    net.addConnection(FullConnection(net['h'], net['out']))
    net.addConnection(FullConnection(net['bias'], net['out']))
    net.sortModules()
        
    
    # tracking progress by callback
    ltrace = []
    def storer(a):
        if a._num_updates % 10 == 0:
            a.provider.nextSamples(250)
            ltrace.append(pylab.mean(a.provider.currentLosses(a.bestParameters)))
    
    x = net.params
    x *= 0.001
    
    f = ModuleWrapper(D, net)
    #algo = SGD(f, net.params.copy(), callback=storer, learning_rate=0.0001)
    algo = vSGDfd(f, net.params.copy(), callback=storer)
    algo.run(10000)
    pylab.plot(ltrace, 'r-')
    pylab.xlabel('epochs x10')
    pylab.ylabel('MSE')
    pylab.show()
    
    
if __name__ == '__main__':
    #inspect1()
    D = readAndStore()
    testTraining(D)


