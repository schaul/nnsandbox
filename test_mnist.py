from NeuralNet import *
from DataSet import *
from Util import *
from TrainingRun import *

def main():

    set_backend("gnumpy")
    #set_gradcheck_mode(True)

    ######################################################
    # Load MNIST dataset
    tic()
    data = load_mnist(digits=range(10),
                      split=[70,15,15],
                      #split=[30,0,0]   # for faster training when debugging
                      )
    print ("Data loaded in %.1fs" % toc())

    ######################################################
    # Create a neural network with matching input/output dimensions
    cfg = NeuralNetCfg(L1=1e-7,init_scale=0.01)
    cfg.input(data.Xshape)
    cfg.hidden(1000,"logistic",maxnorm=2.0)
    cfg.output(data.Yshape,"softmax")

    model = NeuralNet(cfg)

    ######################################################
    # Rescale the data to match the network's domain/range
    data.rescale(model.ideal_domain(),model.ideal_range())

    ######################################################
    # Train the network
    report_args = { 'verbose'   : True,
                    'interval'  : 10,       # how many epochs between progress reports (larger is faster)
                    'visualize' : True}

    trainer = TrainingRun(model,data,report_args,
                          learn_rate=.02,
                          learn_rate_decay=.995,
                          momentum=0.9,momentum_range=[0,1000],
                          batchsize=128)

    tic()
    trainer.train(1000)
    print ("Training took %.1fs" % toc())

    #####################################################
    
    if get_gradcheck_mode():
        model.gradcheck(data.train)

    raw_input()


main()


