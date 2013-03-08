from numpy import *
import BigMat as bm
from Util import ensure_dir
import collections
import time
from time import time as now
import logging
import os,shutil

# Import and configure everything we need from matplotlib
publish_mode = False
import matplotlib
#matplotlib.use("agg")
matplotlib.rcParams.update({'font.size': 9, 'font.family': 'serif', 'text.usetex' : publish_mode})
from matplotlib.pyplot import *
from matplotlib.ticker import NullFormatter
from matplotlib import colors as colors
from matplotlib import cm
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg as FigureCanvas
import Tkinter as Tk

_curr_logfile = None
_num_logfile_entries = 0
_best_logfile_errors = {}
_logs_dir = "logs"

def open_logfile(prefix='train',msg=""):
    global _curr_logfile
    global _num_logfile_entries
    global _best_logfile_errors
    global _logs_dir

    dir = ensure_dir(_logs_dir)
    _curr_logfile = "%s/%s-%s.html" % (dir,prefix,time.strftime("%m%d%H%M",time.localtime()))
    with open(_curr_logfile,"w") as f:
        f.write("<html><head><title>%s</title></head>\n<body>\n" % _curr_logfile)
        f.write("<center style='font-size:120%%'>%s</center>\n" % _curr_logfile)
        f.write("<center style='font-size:120%%'>%s</center>\n" % msg)
        f.flush()
        f.close()
    _num_logfile_entries = 0
    _best_logfile_errors = {}
    return _curr_logfile

def close_logfile():
    global _best_logfile_errors
    str = '<hr><div><table cellspacing=6 cellpadding=0><tr><td></td><td><b>train</b></td><td><b>valid</b></td><td><b>test</b></td></tr>'
    best_errors = sorted(_best_logfile_errors.items(),key = lambda item: item[1][1])
    for test in best_errors:
        str += '<tr>'
        str += '<td><a href="#case%d">RUN %d:</a></td>' % (test[0],test[0])
        str += ('<td>%.2f%%</td>' % test[1][0]) if test[1][0] != None else '<td></td>'
        str += ('<td>%.2f%%</td>' % test[1][1]) if test[1][1] != None else '<td></td>'
        str += ('<td>%.2f%%</td>' % test[1][2]) if test[1][2] != None else '<td></td>'
        str += '</tr>'
    str += '</table></div>\n'
    str += '</body></html>\n'

    with open(_curr_logfile,"a") as f:
        f.write(str)
        f.flush()
        f.close()


#######################################################

class TrainingReport(object):

    def __init__(self,trainer,verbose=True,interval=10,visualize=False,log_mode='none',callback=None):
        self.trainer = trainer
        self.verbose = verbose
        self.visualize = visualize
        self.interval  = interval
        self.log_mode = log_mode  # "none", "html", "html_anim" for animated gifs
        self.callback = callback
        self._callback_msg = None
        self._start_time = now()
        self._screenshots = []
        self._best_validerror = inf

        if visualize:
            self._window = TrainingReportWindow(trainer)

    def __call__(self,event):
        '''
        Collects statistics about the model currently being trained by
        the 'trainer'. 
        '''
        # Only update stats at certain epoch intervals, for the sake of training speed
        if event == "epoch" and mod(self.trainer.epoch,self.interval) != 0:
            return

        stats = {}
        stats["time"]  = now() - self._start_time
        stats["epoch"] = self.trainer.epoch
        stats["learn_rate"] = self.trainer.learn_rate
        stats["momentum"]   = self.trainer.momentum
        for fold in ('train','valid','test'):
            stats[fold] = self._collect_stats_on_fold(fold)

        if self.callback:
            self._callback_msg = self.callback(event,stats)

        self.log(event,stats)

    def log(self,event,stats):
        # common prologue
        msg = '%5.1fs: ' % stats['time']
        if event == 'epoch': msg += '[%3d]' % stats['epoch']
        else:                msg += event
        trstats = stats['train']

        if self.trainer.task() == "classification":
            # classification-specific output format
            msg += ': err=%.2f%%' % trstats['error rate']
            if stats['valid']:  msg += '/%.2f%%' % stats['valid']['error rate']
            if stats['test']:   msg += '/%.2f%%' % stats['test']['error rate']
        else:
            # regression-specific output format
            msg += ': loss=%.3f' % trstats['loss']
            if stats['valid']:  msg += '/%.3f' % stats['valid']['loss']
            if stats['test']:   msg += '/%.3f' % stats['test']['loss']

        # common epilogue: training loss + regularizer + penalty
        msg += '; (%.3f+%.3f+%.3f)' % (trstats['loss'],trstats['regularizer'],trstats['penalty'])

        # common epilogue: learning rate and momentum
        msg += ' r=%.3f' % stats['learn_rate']
        if stats['momentum'] > 0.0:
            msg += ' m=%.2f' % stats['momentum']

        logger = logging.getLogger()
        logger.info(msg)

        if self.visualize:
            self._window.log(event,stats,msg)

        self._update_log(event,stats,msg)

    ########################################################

    def _collect_stats_on_fold(self,fold):
        '''
        Give a particular fold (training/testing) this evaluates the model using
        the current fold, and collects statistics about how the model is performing.
        '''
        data = self.trainer.data[fold]
        if data.size == 0:
            return None
        X,Y = data
        model = self.trainer.model
        H = model.eval(X,want_hidden=True)
        stats = {}
        stats["H"]           = [bm.as_numpy(Hi).copy() for Hi in H]  # make a copy of hidden activations
        stats["loss"]        = model.loss(H[-1],Y)  # scalar loss value
        stats["regularizer"] = model.regularizer(H) # scalar hidden unit regularization penalty
        stats["penalty"]     = model.penalty()      # scalar weight penalty
        if self.trainer.task() == "classification":
            stats["error rate"] = 100*count_nonzero(array(argmax(H[-1],axis=1)) != argmax(data.Y,axis=1)) / float(data.size)
        return stats

    def _update_log(self,event,stats,msg):
        global _num_logfile_entries
        global _best_logfile_errors

        if self.log_mode == 'none':
            return

        str = ""
        if event == 'start':
            # If this is a new training run, 
            _num_logfile_entries += 1
            id = _num_logfile_entries
            bgcolors = ['#eeeeee','#eeeeff','#eeffee','#ffeeee','#ffffee','#ffeeff','#eeffff','#ffffff']
            str += '\n\n\n<hr/><div style="background:%s"><center><a name="case%d"></a><b style="font-size:150%%">RUN %d</b></center>\n' % (bgcolors[mod(_num_logfile_entries,len(bgcolors))],_num_logfile_entries,_num_logfile_entries)
            if self._callback_msg:
                str += self._callback_msg

        # Take a screenshot of the current figure, possibly for future use
        if self.log_mode == "html_anim" or event == "stop" and self.visualize:
            screenshot = self._window.save_figure()
            self._screenshots.append(screenshot)

        # If this update has the lowest validation error for this run, remember 
        # the ideal early stopping point. Oh god this code is so horrible :(
        #
        if stats["valid"] != None:
            id = _num_logfile_entries
            errname  = "error rate" if stats["valid"].has_key('error rate') else "loss"
            errors = [ (stats[fold][errname] if stats[fold] != None else None) for fold in ('train','valid','test')]
            if (not _best_logfile_errors.has_key(id)) or stats["valid"][errname] < _best_logfile_errors[id][1]:
                _best_logfile_errors[id] = errors
        
        # Add the output 'msg' to the log
        str += "<pre style='margin:0;padding:0;font-weight:%s'>%s</pre>\n" % ("bold" if event == "stop" else "normal",msg.strip())

        if event == 'stop':
            # Insert an image or an animation, and close the outer-most DIV element
            if len(self._screenshots) > 0:
                thumbfile = self._screenshots[-1]
                thumburl = os.path.join(*(thumbfile.split('/')[1:]))
                animurl  = thumburl
                if len(self._screenshots) > 1:
                    animfile = os.path.splitext(thumbfile)[0] + '-anim.gif'
                    animurl = os.path.join(*(animfile.split('/')[1:]))
                    inputs = os.path.split(thumbfile)[0] + '/img-%05d-%%02d.png[0-%d]' % (_num_logfile_entries,len(self._screenshots)-1)
                    os.system('convert -delay 30 -coalesce -layers optimize ' + inputs + ' ' + animfile)

                    # Always remove screenshots that are no longer needed
                    for screenshot in self._screenshots[:-1]:
                        os.remove(screenshot)
                
                str += "<div><a href='%s'><img src='%s' border=0/></a></div>\n" % (animurl,thumburl)


            str += "</div>\n"
        
        self._append_to_log(str)

    def _append_to_log(self,str):
        global _curr_logfile
        with open(_curr_logfile,"a") as f:
            f.write(str)
            f.flush()
            f.close()


##############################################################################
#                               WINDOW
##############################################################################


class TrainingReportWindow(Tk.Frame):
    '''
    A window that visualizes the current state of training progress.
    '''
    def __init__(self,trainer):
        Tk.Frame.__init__(self)
        self._unique_id = 0

        # Set up a 2x2 grid, where each cell will have its own kind of figure
        self.master.rowconfigure(0,weight=1)
        self.master.rowconfigure(1,weight=1)
        self.master.columnconfigure(0,weight=1)
        self.master.columnconfigure(1,weight=1)
        dpi = 80.0
        self.plots = {}
        row0_ht = 200
        row1_ht = 120

        # Add error plot in top-left cell
        self.plots["errors"] = TrainingReportErrorPlot(self.master,(300,row0_ht),dpi,trainer.task())
        self.plots["errors"].canvas.get_tk_widget().grid(row=0,column=0,sticky=Tk.N+Tk.S+Tk.E+Tk.W)

        # Feature grid in top-right cell
        if trainer.data.Xshape[0] > 1 and trainer.data.Xshape[1]:
            self.plots["feat"] = TrainingReportFeatureGrid(self.master,(300,row0_ht),dpi,trainer.model,trainer.data.Xshape)
            self.plots["feat"].canvas.get_tk_widget().grid(row=0,column=1,sticky=Tk.N+Tk.S+Tk.E+Tk.W)

        # *Weight* statistics in bottom-left cell
        get_weightmats = lambda event,stats: [bm.as_numpy(abs(layer.W)) for layer in trainer.model.weights]
        weight_percentiles =  list(100*(1-linspace(0.1,.9,10)**1.5))
        self.plots["wstats"] = TrainingReportPercentiles(self.master,(300,row1_ht),dpi,get_weightmats,weight_percentiles)
        self.plots["wstats"].canvas.get_tk_widget().grid(row=1,column=0,sticky=Tk.N+Tk.S+Tk.E+Tk.W)


        # *Hidden activity* statistics in bottom-right cell
        get_hidden = lambda event,stats: stats["train"]["H"]
        hidden_percentiles =  list(100*(1-linspace(0.1,.9,10)**1.5))
        self.plots["hstats"] = TrainingReportPercentiles(self.master,(300,row1_ht),dpi,get_hidden,hidden_percentiles)
        self.plots["hstats"].canvas.get_tk_widget().grid(row=1,column=1,sticky=Tk.N+Tk.S+Tk.E+Tk.W)

        self.master.protocol("WM_DELETE_WINDOW",self._close_window)
        self.master.geometry('+%d+%d' % (0,180))
        self.master.title("Training Report")
        self.update()
        self._redraw_interval = 500
        self.after(self._redraw_interval,self.redraw)

    def log(self,event,stats,msg):
        for plot in self.plots.values():
            plot.log(event,stats)
        self.redraw()

    def _quit_program(self):
        self.master.quit()
        self.master.destroy()
        quit()

    def _close_window(self):
        self.want_quit = True  # quit later
        self.after(5,self._quit_program)

    def redraw(self):
        for plot in self.plots.values():
            plot.redraw()
        self.update()
        self.update_idletasks()
        self.after(self._redraw_interval,self.redraw)

    def save_figure(self):
        global _curr_logfile
        global _num_logfile_entries
        dir = ensure_dir(os.path.splitext(_curr_logfile)[0])
        filename = dir + ('/img-%05d-%02d.png'     % (_num_logfile_entries,self._unique_id))
        tempname = dir + ('/img-%05d-%02d-%%s.png' % (_num_logfile_entries,self._unique_id))
        self._unique_id += 1
        
        # First save the individual Figure canvases to files
        pnames = ('errors','feat','wstats','hstats')
        fnames = []
        for pname in pnames:
            fnames.append(tempname % pname)
            plot = self.plots[pname]
            plot.savefig(fnames[-1],dpi=80)

        # Then use ImageMagick to put them together again
        cmd = 'montage'
        for fname in fnames:
            cmd += ' ' + fname
        cmd += ' -tile 2x2 -geometry +0+0 ' + filename
        os.system(cmd)

        # And delete the temporary images of separate parts of the overall figure
        for fname in fnames:
            os.remove(fname)

        return filename




##############################################################################
#                               ERROR PLOT
##############################################################################



class TrainingReportErrorPlot(Figure):

    def __init__(self,master,size,dpi,task):
        Figure.__init__(self,figsize=(size[0]/dpi,size[1]/dpi),dpi=dpi,facecolor='w',edgecolor='b',frameon=True,linewidth=0)
        FigureCanvas(self,master=master)
        self.master = master
        self._errors = collections.OrderedDict()
        self._dirty = True
        self._task = task
        self.add_subplot(111,axisbg='w')
        
    def log(self,event,stats):
        yaxis = "error rate" if self._task == "classification" else "loss"
        for fold in ('train','valid','test'):
            if stats[fold] == None:
                continue
            # Add the point (epoch,errors) to the plot series for this fold
            x,y = stats["epoch"],stats[fold][yaxis]
            series = self._errors.setdefault(fold,[[],[]])  # list of X values, list of Y values
            series[0].append(x)
            series[1].append(y)
        self._dirty = True

    def redraw(self):
        if self._dirty:
            self._dirty = False
            ax = self.axes[0]
            ax.cla()
            ax.hold("on")
            ax.set_xlabel('epoch')
            ax.set_ylabel("error rate" if self._task == "classification" else "loss")
            colours = [[0.0,0.2,0.8],  # train
                       [0.4,0.1,0.5],  # valid
                       [1.0,0.0,0.0]]  # test
            styles = ['-','--',':']
            minerr,maxerr = 1e10,-1e10
            for series,colour,style in zip(self._errors.items(),colours,styles):
                X,Y = series[1]
                ax.semilogy(X,Y,color=colour,linestyle=style,label=series[0]);
                maxerr = max(maxerr,max(Y))
                minerr = min(minerr,min(Y))
            minerr = max(0.0001 if self._task == "regression" else 0.01,minerr)
            maxerr = max(0.001+minerr,maxerr)

            ax.set_ylim([10**floor(log10(minerr)),10**ceil(log10(maxerr))])
            ax.grid(True,which='major',axis='y',linestyle=':',color=[0.2,0.2,0.2])
            ax.grid(True,which='minor',axis='y',linestyle=':',color=[0.8,0.8,0.8])
            ax.legend()
            ax.set_position([0.125,0.14,.83,.82])
            self.canvas.draw()
    

##############################################################################
#                               FEATURE GRID
##############################################################################


class TrainingReportFeatureGrid(Figure):

    def __init__(self,master,size,dpi,model,inshape):
        Figure.__init__(self,figsize=(size[0]/dpi,size[1]/dpi),dpi=dpi,facecolor='w',edgecolor='b',frameon=True,linewidth=0)
        FigureCanvas(self,master=master)
        self.master = master
        self._dirty = True
        self._model = model
        self._feat  = None
        self._featrange = None
        self._inshape = inshape
        self._sorted = True
        self.add_subplot(111,axisbg='w')
        
    def log(self,event,stats):
        W,b = self._model.weights[0]
        self._feat  = bm.as_numpy(W).copy() # col[i] contains weights entering unit i in first hidden layer
        self._featrange = (min(self._feat.ravel()),max(self._feat.ravel()))
        #self._feat += as_floatx(b)        # add bias
        self._feat = self._feat.reshape(self._inshape + tuple([-1]))
        self._dirty = True

    def redraw(self):
        if self._dirty:
            self._dirty = False
            feat = self._feat
            self.clf()

            if self._sorted:
                # Sort by decreasing variance
                variances = [-var(feat[:,:,j].ravel()) for j in range(feat.shape[2])]
                feat = feat[:,:,argsort(variances)]

            # Convert list of features into a grid of images, fitting the current drawing canvas
            wd,ht = self.canvas.get_width_height()
            zoom = max(1,16//max(feat.shape[0:2]))
            img = self._feat2grid(feat,zoom,1.0,[1.0*wd-2,ht-30])

            # Draw the image centered
            x0,y0 = (wd-img.shape[1])/2, (ht-img.shape[0])/2
            self.figimage(img,x0,y0,None,None,cm.gray,zorder=2)

            # Print the range of the colormap we're seeing
            self.text(float(x0)/wd,float(y0-15)/ht,'[%.3f,%.3f]' % self._featrange,zorder=5)
            self.text(float(x0)/wd,float(y0+img.shape[0]+5)/ht,'features',zorder=5)

            self.canvas.draw()
    


    def _feat2grid(self,feat,zoom=1.0,gamma=1.0,bbox=[300,300],want_transpose=False,outframe=True,pad=1):
        '''
        Like feature2img, except returns a single image with
        a grid layout, instead of a list of individual images
        '''
        ht,wd,n = feat.shape
        wd *= zoom
        ht *= zoom
        opad = pad if outframe else 0
        bbwd,bbht = bbox
        maxcols = int(bbwd-2*opad+pad)//(wd+pad)
        maxrows = int(bbht-2*opad+pad)//(ht+pad)
        if want_transpose: numrows,numcols = self._gridsize(maxrows,maxcols,n)
        else:              numcols,numrows = self._gridsize(maxcols,maxrows,n)

        # Convert 3D array features into a 3D array of images
        img = self._feat2img(feat[:,:,:min(n,numrows*numcols)],zoom=zoom,gamma=gamma)

        # Pull each image out and place it within a grid cell, leaving space for padding
        framecolor = 0
        numchannels = img[0].shape[2]
        grid = zeros([numrows*(ht+pad)-pad+2*opad,numcols*(wd+pad)-pad+2*opad,numchannels],dtype=ubyte) + framecolor
        for i in range(len(img)):
            ri = (i % numrows) if     want_transpose else floor(i / numcols)
            ci = (i % numcols) if not want_transpose else floor(i / numrows)
            row0 = ri * (ht+pad) + opad 
            col0 = ci * (wd+pad) + opad
            grid[row0:row0+ht,col0:col0+wd,:] = img[i]

        if numchannels==1:
            grid.shape = grid.shape[0:2]  # matplotlib's figimage doesn't like extra dimension on greyscale images
        return grid
    


    def _gridsize(self,max1,max2,n):
        '''Calculates the dimensions of the grid, in 'cells', for n items'''
        max1 = float(max1); max2 = float(max2); n = float(n)
        if max1 <= 1:
            return (0,0)
        num1 = min(max1,n)
        num2 = 1.0
        while ceil(n/(max1-1)) <= max1-1 and ceil(n/(max1-1)) <= max2 and ceil(n/(max1-1)) > num2:
            num1 -= 1
            num2 = ceil(n/num1)
        if num1 == 0:
            return (0,0)
        num2 = min(max2,ceil(n/num1))
        return (int(num1),int(num2))


    def _feat2img(self,feat,zoom,gamma):
        '''
        Given an NxMxT stack of features (filters), splits it
        into a list of NxM images.
        '''
        valrange = max(abs(feat.ravel())) # will scale [0,valrange] to [0,255]
        if valrange <= 1e-10:
            valrange = 1
        ht,wd,n = feat.shape # n = number of features
        normalize = True
        img = []
        for i in range(n):
            I = reshape(feat[:,:,i],(ht,wd,1))
            if normalize: I = sign(I)*pow(abs(I)/valrange,gamma)/2*255+127
            else:         I = 255*pow((I+1)/2,gamma)
            I = uint8(minimum(255,maximum(0,I)))
            if zoom != 1:
                I = repeat(I,zoom,axis=0)
                I = repeat(I,zoom,axis=1)
            img.append(I)
        return img





##############################################################################
#                               WEIGHT STATISTICS
##############################################################################


class TrainingReportPercentiles(Figure):

    def __init__(self,master,size,dpi,get_matrices_fn,percentiles):
        Figure.__init__(self,figsize=(size[0]/dpi,size[1]/dpi),dpi=dpi,facecolor='w',edgecolor='b',frameon=True,linewidth=0)
        FigureCanvas(self,master=master)
        self.master = master
        self._dirty = True
        self._get_matrices_fn = get_matrices_fn
        self._P = []
        self._t = percentiles
        self.add_subplot(111,axisbg='w')
        
    def log(self,event,stats):
        self._P = []
        matrices = self._get_matrices_fn(event,stats)
        for A in matrices:
            P = make_matrix_percentiles(A.transpose(),self._t)
            self._P.append(P)
        self._dirty = True

    def redraw(self):
        if self._dirty:
            self._dirty = False
            self.clf()

            # Convert list of features into a grid of images, fitting the current drawing canvas
            wd,ht = self.canvas.get_width_height()
            nlayer = len(self._P)
            for k in range(nlayer):
                P = self._P[k]
                Prange = (P.ravel().min(),P.ravel().max())
                P -= Prange[0]
                if Prange[1] != Prange[0]:
                    P *= 255/(Prange[1]-Prange[0])
                img = asarray(P,dtype='uint8')
                zoom = 8
                img = repeat(img,zoom,axis=0)
                img = repeat(img,zoom,axis=1)

                x0,y0 = (k+0.5)*(wd-img.shape[1])/nlayer, (ht-img.shape[0])/2
                self.figimage(img,x0,y0,None,None,cm.gray,zorder=2)

                # Print the range of the colormap we're seeing
                self.text(float(x0)/wd,float(y0-15)/ht,'[%.3f,%.3f]' % Prange,zorder=5)
                self.text(float(x0)/wd,float(y0+img.shape[0]+5)/ht,'layer%d' % k,zorder=5)

            self.canvas.draw()
    











# Input: n-dimensional matrix A and and m percentile thresholds in list t. 
# Returns an n-dimensional matrix P, where each dimension has size m, 
# where element P[j1,j2,...jn] is 
#       the t[j1]-percentile along dimension 1
#    OF the t[j1]-percentile along dimension 2
#       ...
#    OF the t[jn]-percentile along dimension n
#
# So if A were 8x6 (2 dimensions) and t=[1,10,50] then P is a 3x3 matrix
# where P[i,j] = percentile(percentile(A,t[j],axis=1),t[i],axis=0)
#
def make_matrix_percentiles(A,t):
    n,m = A.ndim,len(t)
    if n == 0:
        return A # any percentile of a single number is just that number

    # First collect all the t[j] percentiles of A along the last dimension
    # Result: list of length m, each item being an (n-1)-dimensional matrix
    Ap = percentile(A,t,n-1)

    # For each Ap[j] matrix, compute its (n-1)-dimensional P matrix 
    # (sides of length m) using recursion
    Pp = []
    for Apj in Ap:
        Pp.append(make_matrix_percentiles(Apj,t))

    # Finally, stack each Pp
    newshape = tuple([m]) + Pp[0].shape
    P = ndarray(newshape,dtype=A.dtype)
    for j in range(m):
        P[j] = Pp[j]

    return P


#########################################################################
# Set up default logging config

_log = logging.getLogger()

def setup_logging(filename,clear=False):

    if clear:
        with open(filename, 'w'):
            pass  # clear the log file if it already exists

    log_file = logging.FileHandler(filename)
    log_file.setLevel(logging.DEBUG)

    log_console = logging.StreamHandler(sys.stdout)
    log_console.setLevel(logging.DEBUG)

    _log.setLevel(logging.DEBUG)
    _log.addHandler(log_file)
    _log.addHandler(log_console)

setup_logging('basic_learn.log',clear=True)
