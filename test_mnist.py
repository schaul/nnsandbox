from NeuralNet import *
from DataSet import *
from Util import *
from TrainingRun import *

grad_check_mode = False

def main():

    ######################################################
    # Load MNIST dataset
    reset_rand_seed(942123)
    tic()
    data = load_mnist(digits=range(10),
                      split=[70,15,15],
                      #split=[70,15,15]
                      )
    print ("Data loaded in %.1fs" % toc())

    ######################################################
    # Create a neural network with matching input/output dimensions
    cfg = NeuralNetCfg()#L1=1e-4,L2=1e-5)
    cfg.input(data.Xshape)
    cfg.hidden(1000,"logistic",sparsity=[0.0005,0.05])
    cfg.output(data.Yshape,"softmax")

    reset_rand_seed(942123)
    nn = NeuralNet(cfg)

    ######################################################
    # Rescale the data to match the network's domain/range
    data.rescale(nn.ideal_domain(),nn.ideal_range())

    ######################################################
    # Train the network
    trainer = TrainingRun(nn,data,
                          verbose=True,verbose_interval=10,visualize=True,
                          learn_rate=0.5,
                          learn_rate_decay=.95,
                          #momentum=0.0,momentum_range=range(10,1000),
                          batchsize=64)

    tic()
    trainer.train(50)
    print ("Training took %.1fs" % toc())

    #####################################################
    
    if grad_check_mode:
        nn.grad_check(data.train)

    raw_input()





######################################################

import random
import numpy.random

def reset_rand_seed(seed):
    random.seed(seed)
    numpy.random.seed(seed*7)


if grad_check_mode:
    set_backend("numpy","float64")
else:
    set_backend("gnumpy")

main()


