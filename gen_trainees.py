from NeuralNet import *
from DataSet import *
from Util import *
from TrainingRun import *
from Report import open_logfile,close_logfile
from BigMat import garbage_collect
import itertools

outdir = ensure_dir("data/trainees/mnist")
#outdir = ensure_dir("C:/Users/Andrew/Dropbox/share/tom/data/trainees/mnist")

def main():
    global outdir

    set_backend("gnumpy")
    
    report_args = { 'verbose'   : True,
                    'interval'  : 3,       # how many epochs between progress reports (larger is faster)
                    'interval_rate' : 1.35,
                    'visualize' : True,
                    'log_mode'  : 'html_anim' }

    data = load_mnist(digits=[5,6,8,9],
                      split=[30,10,0])

    settings_name = "basic2"

    ######################################################
    # Generate a list of training configurations to try, 
    # and loop over them.
    #
    settings,prefix = make_train_settings(settings_name)
    settings = enumerate_settings(settings)
    index = 801
    index_end = 1000
    settings = settings[index-1:index_end]
    num_restarts = 1  # How many random restarts do we try with the same parameters

    open_logfile("gen_trainees-%s" % prefix,"%d total variants; %d restarts each" % (len(settings),num_restarts))
    settingsfile = '%s/%s-settings.pkl' % (outdir,prefix)
    quickdump(settingsfile,settings)
    for setting in settings:
        print ('\n\n-------------------- %s-%d --------------------' % (prefix,index))
        print setting['model']
        print setting['train']

        for rep in range(num_restarts):
            # Create the model and scale the data if necessary
            model = make_model(setting['model'],data.Xshape,data.Yshape)
            data.rescale(model.ideal_domain(),model.ideal_range())

            # Set up a callback to collect snapshots of the training run
            snapshots = []
            report_args['callback'] = lambda event,status: report_callback(setting,snapshots,model,event,status)

            # Train the model
            trainer = TrainingRun(model,data,report_args,**setting['train'])
            trainer.train()
            trainer = None # force gc on any Tk window objects

            # Save the training run to disk
            save_trainee(snapshots,setting,prefix,index); 
            index += 1

    #####################################################
    
    close_logfile()
    raw_input()


######################################################

def report_callback(setting,snapshots,model,event,stats):
    if event == 'start':
        # If event is start of new training run, print settings
        return "<table><tr><th>model</th><th>training</th></tr><tr><td style='margin-right:20px;vertical-align:top'><pre>%s</pre></td><td style='margin-left:20px;vertical-align:top'><pre>%s</pre></td></tr></table><br/>\n" % (setting['model'].__repr__(),setting['train'].__repr__())  # this is how the settings get logged at start of each new training run

    if event == 'stop' or stats['epoch'] < 1:
        return 

    # If event is epoch, save current test performance along with a snapshot of the weights
    snapshot = {}
    snapshot.update(stats)
    for fold in ('train','valid','test'):  # remove the full-blown set of hidden activations because it makes
        if stats[fold] != None:            # the dataset too huge on disk
            snapshot[fold] = {}
            snapshot[fold].update(stats[fold])
            snapshot[fold].pop("H")
    snapshot["weights"] = [(as_numpy(W).copy(),as_numpy(b).copy()) for W,b in model.weights]
    snapshots.append(snapshot)

def save_trainee(snapshots,setting,prefix,index):
    global outdir
    filename = '%s/%s-%05i.pkl' % (outdir,prefix,index)
    trainee = { 'setting'  : setting,
                'snapshots': snapshots }
    quickdump(filename,trainee)


######################################################

def make_train_settings(name):
    return eval("make_train_settings_%s()" % name)

def make_train_settings_basic():

    # model parameters to try
    model = Setting()
    model.activation = [["logistic","softmax"]]
    model.size     = [[400]]
    model.dropout  = [None]
    model.maxnorm  = [None]
    model.sparsity = [None,(1e-6,1e-2),(1e-5,1e-2),(1e-4,1e-2),
                      None,(1e-6,1e-3),(1e-5,1e-3),(1e-4,1e-2),
                      None,(1e-6,1e-4),(1e-5,1e-4),(1e-4,1e-2)]
    model.L1       = [None,1e-7,1e-5,5e-5,1e-4]
    model.L2       = [None,1e-7,1e-5,5e-5,1e-4]
    model.init_scale=[0.01,0.1]

    # training parameters to try
    train = Setting()
    train.learn_rate       = [0.03,0.08,0.2,0.5]
    train.learn_rate_decay = [0.99]
    train.momentum         = [0.75]
    train.batchsize        = [20,50,100]
    train.epochs           = [85]

    return { 'model' : model, 'train' : train }, "basic"

def make_train_settings_basic2():

    # model parameters to try
    model = Setting()
    model.activation = [["logistic","logistic","softmax"]]
    model.size     = [[100,100]]
    model.dropout  = [None,[0.0,0.2,0.0],[0.2,0.5,0.5],[0.2,0.5,0.0]]
    model.maxnorm  = [None,4.0]
    model.sparsity = [None,(1e-5,1e-2),(1e-4,1e-2),
                      None,(1e-5,1e-3),(1e-4,1e-3)]
    model.L1       = [None,1e-7,1e-6,5e-5]
    model.L2       = [None,1e-7,1e-6,5e-5]
    model.init_scale=[0.01,0.05,0.10]

    # training parameters to try
    train = Setting()
    train.learn_rate       = [0.2,0.5,2.0,5.0]
    train.learn_rate_decay = [.995,0.985]
    train.momentum         = [0.5]
    train.batchsize        = [128]
    train.epochs           = [113]


    return { 'model' : model, 'train' : train }, "basic-s100-s100"



######################################################

class Setting:  # empty class to attach dynamic attributes 
    def __setitem__(self,name,value): self.__dict__[name] = value
    def __getitem__(self,name): return self.__dict__[name]
    def has_key(self,name):     return self.__dict__.has_key(name)
    def keys(self):             return self.__dict__.keys()
    def values(self):           return self.__dict__.values()
    def items(self):            return self.__dict__.items()
    def __repr__(self):  # Useful for printing 
        str = ''
        for name,value in self.__dict__.iteritems():
            if value == None:
                pass
            elif (isinstance(value,Setting)):
                str += '{0}=...\n'.format(name)
            elif (isinstance(value,basestring)):
                str += '{0}=\'{1}\'\n'.format(name,value)
            else:
                str += '{0}={1}\n'.format(name,value)
        return str

def enumerate_settings(*settings_spec):
    '''
    Input: Config() instances spec0,spec1,...specN where the properties of each 
           are lists of parameter settings.
    Output: A shuffled list containing all combinations of parameters in spec0,
            plus all combinations of parameters in spec1, etc.
    '''
    variants = []
    for spec in settings_spec:
        names,combos,nones = [],[],[]
        for name,settings in spec.items():
            names.extend([(name,key) for key in settings.keys()])
            combos.extend([value for dummy,value in settings.items()])
        
        # For every combination of setting values, 
        for combo in itertools.product(*combos):
            variant = {}
            for name,key in names: 
                variant[name] = Setting()
            for i in range(len(combo)):
                name,key = names[i]
                variant[name][key] = combo[i]
            variants.append(variant)

    random.shuffle(variants)
    return variants

def make_model(setting,Xshape,Yshape):
    '''Given a particular Settings object specifying a model, 
       constructs a corresponding NeuralNet instance'''

    def check_list_len(name,length):
        assert((not isinstance(setting[name],list)) or len(setting[name]) == length)

    # Make sure all the lists are the right length
    K = len(setting.activation) # hidden + output layers
    check_list_len('size',K-1)  # hidden layers only
    for name in ('L1','L2','maxnorm','dropout','sparsity','init_scale'):
        check_list_len(name,K) 

    # If one of these settings is a scalar and not a list, then it means
    # we should apply it to all layers and it is a 'default' value
    defaults = {}
    for name in ('L1','L2','maxnorm','dropout','sparsity','init_scale'):
        if not isinstance(setting[name],list):
            defaults[name] = setting[name]

    # Pull out all settings that specify the input layer
    cfg = NeuralNetCfg(**defaults)
    cfg.input(Xshape,dropout=(setting.dropout[0] if isinstance(setting.dropout,list) else setting.dropout))

    # Pull out the settings for each hidden layer
    for k in range(K-1):
        args = {}
        # For settings that have values defined for layers 1:K
        for name in ('size','activation','L1','L2','maxnorm','sparsity','init_scale'):
            if isinstance(setting[name],list):
                args[name] = setting[name][k]
        if isinstance(setting.dropout,list):
            args['dropout'] = setting.dropout[k+1]   # skip dropout setting for input layer
        cfg.hidden(**args)

    # Finally, pull out the settings for the output layer
    output_args = { 'size' : Yshape }
    for name in ('activation','L1','L2','maxnorm','init_scale'):
        if isinstance(setting[name],list):
            output_args[name] = setting[name][K-1]
    cfg.output(**output_args)

    return NeuralNet(cfg)


#######################################################

main()


