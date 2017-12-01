import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.sparse import issparse, spdiags, coo_matrix, csc_matrix
from past.utils import old_div
from scipy.ndimage.measurements import center_of_mass
import tifffile as tiff
import os, sys
import matplotlib.gridspec as gridspec
import scipy.io as sio

try:
    import bokeh
    import bokeh.plotting as bpl
    from bokeh.models import CustomJS, ColumnDataSource, Range1d
except:
    print("Bokeh could not be loaded. Either it is not installed or you are not running within a notebook")


def run_foopsi(root):
    
    file_name= root.data.file_name
    data = np.load(file_name)
    traces = data['traces'].T
    centres = data['cm']
    print centres[0]
    
    print len(traces)
    
    #lengths = [3000,3500,3500,3000,3000,3000]
    #import imp
    #cm = imp.load_source('caiman', '/home/cat/code/CaImAn/')
    
    caiman_path = np.loadtxt('caiman_folder_location.txt', dtype=str)
    sys.path.append(str(caiman_path)+'/')

    import caiman as cm
    from caiman.source_extraction.cnmf.deconvolution import constrained_foopsi
    
    c_array = []
    raster_array = []
    for n, trace in enumerate(traces):
    #for n, trace in enumerate(traces[:10]):
        print " ... cell: ", n
        #trace+=20
        temp_c = []
        temp_raster = []
        
        if True:                               
            #c, bl, c1, g, sn, sp, lam = constrained_foopsi(trace,method = 'cvxpy', p=1)
            c, bl, c1, g, sn, sp, lam = constrained_foopsi(trace, p=1)
            temp_c.extend(c)
            temp_raster.extend(sp)    
        
        #This runs deconvolution chunkwise and setsbaseline based on mean of the input traces - to account for differences in light intensity across frames...
        else:                                   
            for l in range(len(lengths)):
                print np.sum(lengths[0:l]), np.sum(lengths[0:l+1])
                c, bl, c1, g, sn, sp, lam = constrained_foopsi(trace[np.sum(lengths[0:l]):np.sum(lengths[0:l+1])]+abs(np.mean(trace)),p=2)
                temp_c.extend(c)
                temp_raster.extend(sp)

        c_array.append(temp_c)
        raster_array.append(temp_raster)

    raster_array = np.array(raster_array)
    c_array=np.array(c_array)
    #Compute 
    print "...extracting binary rasters..."
    rasters_array = []
    thresholds = np.float32([root.data.foopsi_threshold])
    for threshold in thresholds:
        print "... generating rasters for threshold: ", threshold
        rasters = np.zeros(raster_array.shape,dtype=np.float32)
        for k in range(len(rasters)):
            indexes = np.where(raster_array[k]>threshold)[0]
            rasters[k][indexes]=1
        rasters_array.append(rasters)

    np.savez(file_name[:-4]+"_deconvolved_data_thr"+str(root.data.foopsi_threshold), original_traces=traces, deconvolved_traces=c_array, foopsi_probabilities=raster_array, rasters=rasters_array[0],centres=centres)

    sio.savemat(file_name[:-4]+'_deconvolved_data_thr'+str(root.data.foopsi_threshold)+'.mat', {'original_traces':traces, 'deconvolved_traces':c_array, 'foopsi_probabilities':raster_array,'rasters':rasters_array[0],'centres':centres})

def view_rasters(root):
    print "...View rasters ..."
    
    root.deconvolved_filename = root.data.file_name[:-4]+"_deconvolved_data_thr"+str(root.data.foopsi_threshold)+".npz"
    data = np.load(root.deconvolved_filename)
    rasters = data['rasters']
    print rasters.shape

    ax=plt.subplot(1,1,1)
    for k in range(len(rasters)):
        print len(rasters[k])
        indexes = np.where(rasters[k]>0)[0]
        plt.scatter(indexes, [k]*len(indexes))
    
    plt.xlim(-1,len(rasters[0])+1)
    plt.ylim(-1,len(rasters)+1)
    plt.xlabel("Frames", fontsize=25)
    plt.ylabel("Neuron ID", fontsize=25)
    ax.tick_params(axis='both', which='both', labelsize=25)
    plt.title("Foopsi threshold: "+str(root.data.foopsi_threshold), fontsize=25)
    plt.show()
            

def convert_tif_npy(file_name):
    images = tiff.imread(file_name)
    np.save(file_name[:-4], images)

    tiff.imsave(file_name[:-4]+"_500frames.tif", images[:500])
    np.save(file_name[:-4]+"_500frames.npy", images[:500])

def Ensemble_detection(root):
    print "...Ensemble detection ... (not implemented)"

def About():
    tkMessageBox.showinfo("About", "CaImAn Ver 1.0 ...")

def com(A, d1, d2):
    """Calculation of the center of mass for spatial components

     Inputs:
     ------
     A:   np.ndarray
          matrix of spatial components (d x K)

     d1:  int
          number of pixels in x-direction

     d2:  int
          number of pixels in y-direction

     Output:
     -------
     cm:  np.ndarray
          center of mass for spatial components (K x 2)
    """
    from past.utils import old_div

    nr = np.shape(A)[-1]
    Coor = dict()
    Coor['x'] = np.kron(np.ones((d2, 1)), np.expand_dims(list(range(d1)), axis=1))
    Coor['y'] = np.kron(np.expand_dims(list(range(d2)), axis=1), np.ones((d1, 1)))
    cm = np.zeros((nr, 2))        # vector for center of mass
    
    cm[:, 0] = old_div(np.dot(Coor['x'].T, A), A.sum(axis=0))
    cm[:, 1] = old_div(np.dot(Coor['y'].T, A), A.sum(axis=0))

    return cm
    
def plot_contours(A, Cn, thr=None, thr_method='max', maxthr=0.2, nrgthr=0.9, display_numbers=True, max_number=None,
                  cmap=None, swap_dim=False, colors='w', vmin=None, vmax=None, **kwargs):
    """Plots contour of spatial components against a background image and returns their coordinates

     Parameters:
     -----------
     A:   np.ndarray or sparse matrix
               Matrix of Spatial components (d x K)

     Cn:  np.ndarray (2D)
               Background image (e.g. mean, correlation)

     thr_method: [optional] string
              Method of thresholding: 
                  'max' sets to zero pixels that have value less than a fraction of the max value
                  'nrg' keeps the pixels that contribute up to a specified fraction of the energy

     maxthr: [optional] scalar
                Threshold of max value

     nrgthr: [optional] scalar
                Threshold of energy

     thr: scalar between 0 and 1
               Energy threshold for computing contours (default 0.9)
               Kept for backwards compatibility. If not None then thr_method = 'nrg', and nrgthr = thr

     display_number:     Boolean
               Display number of ROIs if checked (default True)

     max_number:    int
               Display the number for only the first max_number components (default None, display all numbers)

     cmap:     string
               User specifies the colormap (default None, default colormap)

     Returns:
     --------
     Coor: list of coordinates with center of mass, contour plot coordinates and bounding box for each component
    """
    if issparse(A):
        A = np.array(A.todense())
    else:
        A = np.array(A)

    if swap_dim:
        Cn = Cn.T
        print('Swapping dim')

    d1, d2 = np.shape(Cn)
    d, nr = np.shape(A)
    print "# neurons: ", nr
    if max_number is None:
        max_number = nr

    #if thr is not None:
    #    thr_method = 'nrg'
    #    nrgthr = thr
    #    warn("The way to call utilities.plot_contours has changed. Look at the definition for more details.")

    x, y = np.mgrid[0:d1:1, 0:d2:1]

    
        
    ax = plt.gca()
    if vmax is None and vmin is None:
        plt.imshow(Cn, interpolation=None, cmap=cmap,
                  vmin=np.percentile(Cn[~np.isnan(Cn)], 1), vmax=np.percentile(Cn[~np.isnan(Cn)], 99))
    else:
        plt.imshow(Cn, interpolation=None, cmap=cmap,
                  vmin=vmin, vmax=vmax)
    
    coordinates = []
    cm = com(A, d1, d2)
    for i in range(np.minimum(nr, max_number)):
        print i,
    
        pars = dict(kwargs)
        if thr_method == 'nrg':
            indx = np.argsort(A[:, i], axis=None)[::-1]
            cumEn = np.cumsum(A[:, i].flatten()[indx]**2)
            cumEn /= cumEn[-1]
            Bvec = np.zeros(d)
            Bvec[indx] = cumEn
            thr = nrgthr

        else:  # thr_method = 'max'
            if thr_method != 'max':
                warn("Unknown threshold method. Choosing max")
            Bvec = A[:, i].flatten()
            Bvec /= np.max(Bvec)
            thr = maxthr

        if swap_dim:
            Bmat = np.reshape(Bvec, np.shape(Cn), order='C')
        else:
            Bmat = np.reshape(Bvec, np.shape(Cn), order='F')
        cs = plt.contour(y, x, Bmat, [thr], colors=colors)
        
        # this fix is necessary for having disjoint figures and borders plotted correctly
        p = cs.collections[0].get_paths()
        v = np.atleast_2d([np.nan, np.nan])
        for pths in p:
            vtx = pths.vertices
            num_close_coords = np.sum(np.isclose(vtx[0, :], vtx[-1, :]))
            if num_close_coords < 2:
                if num_close_coords == 0:
                    # case angle
                    newpt = np.round(old_div(vtx[-1, :], [d2, d1])) * [d2, d1]
                    #import ipdb; ipdb.set_trace()
                    vtx = np.concatenate((vtx, newpt[np.newaxis, :]), axis=0)

                else:
                    # case one is border
                    vtx = np.concatenate((vtx, vtx[0, np.newaxis]), axis=0)
                    #import ipdb; ipdb.set_trace()

            v = np.concatenate((v, vtx, np.atleast_2d([np.nan, np.nan])), axis=0)

        pars['CoM'] = np.squeeze(cm[i, :])
        pars['coordinates'] = v
        pars['bbox'] = [np.floor(np.min(v[:, 1])), np.ceil(np.max(v[:, 1])),
                        np.floor(np.min(v[:, 0])), np.ceil(np.max(v[:, 0]))]
        pars['neuron_id'] = i + 1
        coordinates.append(pars)


    if display_numbers:
        for i in range(np.minimum(nr, max_number)):
            if swap_dim:
                ax.text(cm[i, 0], cm[i, 1], str(i + 1), color=colors)
            else:
                ax.text(cm[i, 1], cm[i, 0], str(i + 1), color=colors)

    plt.show()

    return coordinates
    


def PointsInCircum(r,n=100):
    import math
    from math import pi
    return [(math.cos(2*pi/n*x)*r,math.sin(2*pi/n*x)*r) for x in xrange(0,n+1)]   

#************************* GUI TO CORRECT ROIS; USING EXISTING CAIMAN CODE *******************
def correct_ROIs(file_name, A, Cn, thr=None, thr_method='max', maxthr=0.2, nrgthr=0.9, display_numbers=True, max_number=None,
                  cmap=None, swap_dim=False, colors='grey', vmin=None, vmax=None, **kwargs):
    """Plots contour of spatial components against a background image and returns their coordinates

     Parameters:
     -----------
     A:   np.ndarray or sparse matrix
               Matrix of Spatial components (d x K)

     Cn:  np.ndarray (2D)
               Background image (e.g. mean, correlation)

     thr_method: [optional] string
              Method of thresholding: 
                  'max' sets to zero pixels that have value less than a fraction of the max value
                  'nrg' keeps the pixels that contribute up to a specified fraction of the energy

     maxthr: [optional] scalar
                Threshold of max value

     nrgthr: [optional] scalar
                Threshold of energy

     thr: scalar between 0 and 1
               Energy threshold for computing contours (default 0.9)
               Kept for backwards compatibility. If not None then thr_method = 'nrg', and nrgthr = thr

     display_number:     Boolean
               Display number of ROIs if checked (default True)

     max_number:    int
               Display the number for only the first max_number components (default None, display all numbers)

     cmap:     string
               User specifies the colormap (default None, default colormap)

     Returns:
     --------
     Coor: list of coordinates with center of mass, contour plot coordinates and bounding box for each component
    """
    
    global nearest_cell, previous_cell, l_width, ylim_max, ylim_min, y_array, x_array, Bmat_array, thr_array, traces, cm, img1, images_kalman, color_selected, img_data, ax, ax2, ax3, add_cell_flag


    if issparse(A):
        A = np.array(A.todense())
    else:
        A = np.array(A)

    if swap_dim:
        Cn = Cn.T
        print('Swapping dim')

    d1, d2 = np.shape(Cn)
    d, nr = np.shape(A)
    print "# neurons: ", nr
    if max_number is None:
        max_number = nr

    #if thr is not None:
    #    thr_method = 'nrg'
    #    nrgthr = thr
    #    warn("The way to call utilities.plot_contours has changed. Look at the definition for more details.")

    x, y = np.mgrid[0:d1:1, 0:d2:1]
    
    def reload_data():
        global cm, traces, images_kalman
        cm = com(A, d1, d2)

        print Cn.shape
        print file_name
        print file_name.replace("_processed.npz",'.npy')
        images_kalman = np.load(file_name.replace("_processed.npz",'.npy'))
        print images_kalman.shape
        
        traces = np.load(file_name[:-4]+"_traces.npy")
        print traces.shape
    
    reload_data()
    
    #************* PLOT AVERAGE DATA MOVIES ************
    from matplotlib.widgets import Slider, Button, RadioButtons

    #fig, ax = plt.subplots()
    fig = plt.figure()
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    gs = gridspec.GridSpec(2,4)
    #ax1 = plt.subplot(self.gs[0:2,0:2])
    
    #Setup neuron rasters plot

    f0 = 0
    l_width=1
    ylim_max=500
    ylim_min=-200
    previous_cell=0
    add_cell_flag=False
    nearest_cell=(previous_cell+1)

    img_data = images_kalman
    print img_data.shape

    #ax=plt.subplot(1,2,1)
    ax = plt.subplot(gs[0:2,0:2])
    img1 = ax.imshow(img_data[0], cmap='viridis')

    #BOTTOM TRACES
    ax2 = plt.subplot(gs[1:2,2:4])
    img2, = ax2.plot(traces[:,0])
    plt.ylim(-50,400)
    plt.xlim(0,len(images_kalman))

    #TOP TRACES
    ax3 = plt.subplot(gs[0:1,2:4])
    img3, = ax3.plot(traces[:,0])
    plt.ylim(-50,400)
    plt.xlim(0,len(images_kalman))
    
    #SLIDER WINDOW
    axcolor = 'lightgoldenrodyellow'
    axframe = plt.axes([0.12, 0.02, 0.35, 0.03])#, facecolor=axcolor)
    frame = Slider(axframe, 'frame', 0, len(img_data), valinit=f0)


    #********* PRELOAD CONTOUR VALS **************
    y_array=[]
    x_array=[]
    Bmat_array=[]
    thr_array=[]
    thr_method == 'nrg'
    colors_white='w'
    #cm = cm[:nr]
    
    #for i in range(np.minimum(nr, max_number)):
    for k in range(nr):
    #for k in range(10):                                         #this index matches Bokeh plot numbering
        i=k
        print "cell: ", k, " coords: ", cm[i,1],cm[i,0]
        
        pars = dict(kwargs)
        if thr_method == 'nrg':
            indx = np.argsort(A[:, i], axis=None)[::-1]
            cumEn = np.cumsum(A[:, i].flatten()[indx]**2)
            cumEn /= cumEn[-1]
            Bvec = np.zeros(d)
            Bvec[indx] = cumEn
            thr = nrgthr

        else:  # thr_method = 'max'
            if thr_method != 'max':
                warn("Unknown threshold method. Choosing max")
            Bvec = A[:, i].flatten()
            Bvec /= np.max(Bvec)
            thr = maxthr

        #if swap_dim:
        #    Bmat = np.reshape(Bvec, np.shape(Cn), order='C')
        #else:
        Bmat = np.reshape(Bvec, np.shape(Cn), order='F')
        
        y_array.append(y)
        x_array.append(x)
        thr_array.append(thr)
        Bmat_array.append(Bmat)
        cs = ax.contour(y, x, Bmat, [thr], linewidth=l_width, colors=colors)
        #ax.text(cm[i, 1], cm[i, 0], str(i + 1), color=colors)
        ax.text(cm[i, 1], cm[i, 0], str(i), color=colors_white)
    
    Bmat_array = np.array(Bmat_array)
    thr_array = np.array(thr_array)
    y_array = np.array(y_array)
    x_array = np.array(x_array)



    #***********************************************************************************
    #**************************** REDRAW TRACES FUNCTION  ******************************
    #***********************************************************************************
    def redraw_traces():
        ''' Function to redraw traces on right panels; called by various tools
        '''
        global nearest_cell, previous_cell, l_width, ylim_max,  ylim_min, y_array, x_array, Bmat_array, thr_array, traces, cm, img1, img_data, ax, ax2, ax3
        

        #********** PREVIOUS NEURON ************
        ax3.cla()
        ax3.plot([int(frame.val),int(frame.val)],[ylim_min,ylim_max])
        ax3.set_ylim(ylim_min, ylim_max)
        ax3.set_title("Cell: "+str(previous_cell), fontsize=15)
        ax3.set_xlim(0,len(traces))

        #Plot original traces
        ax3.plot(traces[:,previous_cell]+50, color='red', alpha=0.8)
        
        #Plot spikes
        derivative = traces[:,previous_cell][1:]-traces[:,previous_cell][:-1]
        der_std = np.std(derivative)
        spikes = np.where(derivative>(der_std*3))[0]
        ax3.vlines(spikes,[0],[-100])
        

        #************** CURRENT NEURON *********
        ax2.cla()
        ax2.set_ylim(ylim_min, ylim_max)
        ax2.plot([int(frame.val),int(frame.val)],[ylim_min,ylim_max])
        ax2.set_title("Cell: "+str(nearest_cell), fontsize=15)
        ax2.set_xlim(0,len(traces))

        #Plot original traces
        ax2.plot(traces[:,nearest_cell]+50, color='blue', alpha=0.8)
        #Plot spikes
        temp_trace = traces[:,nearest_cell]
        derivative = temp_trace[1:]-temp_trace[:-1]
        derivative = np.gradient(traces[:,nearest_cell])
        der_std = np.std(derivative)
        ax2.plot(derivative)
        spikes = np.where(derivative>(der_std*3))[0]
        ax2.plot([0,len(traces)], [der_std*3,der_std*3], 'r--', color='red')
        ax2.vlines(spikes,[0],[-100])
        

        fig.canvas.draw()


    #***********************************************************************************
    #**************************** RESET FUNCTION  **************************************
    #***********************************************************************************
    def reset_function():
        ''' Reset function called by various buttons to redraw everything
        '''
        global nearest_cell, previous_cell, l_width, ylim_max,  ylim_min, y_array, x_array, Bmat_array, thr_array, traces, cm, img1, img_data, ax, ax2, ax3
        print "...reset function called ..."
        print "...nearest_cell: ", nearest_cell
        print "...previous_cell: ", previous_cell
        
        #******* Redraw movie panel
        ax.cla()
        img1 = ax.imshow(img_data[0], cmap=color_selected, interpolation='sinc')
        for c in range(len(x_array)):
            ax.contour(y_array[c], x_array[c], Bmat_array[c], [thr_array[c]], colors=colors)
            ax.text(cm[c, 1], cm[c, 0], str(c), color=colors_white)

        ax.set_title("# Cells: "+str(len(x_array)), fontsize=15)

        #*************** DRAW NEW CONTROUS **************
        ax.contour(y_array[previous_cell], x_array[previous_cell], Bmat_array[previous_cell], [thr_array[previous_cell]], linewidths=l_width, colors='red',alpha=0.9)
        ax.contour(y_array[nearest_cell], x_array[nearest_cell], Bmat_array[nearest_cell], [thr_array[nearest_cell]], linewidths=l_width, colors='blue',alpha=0.9)

        redraw_traces()


    #***********************************************************************************
    #**************************** SELECT NEURON BUTTON *********************************
    #***********************************************************************************
    def callback(event):
        ''' This function finds the nearest neuron to the mouse click location
        '''
        global nearest_cell, previous_cell, l_width, ylim_max, ylim_min, y_array, x_array, Bmat_array, thr_array, traces, cm, img1, img_data, ax, ax2, ax3, add_cell_flag
        
        if event.inaxes is not None:
            print "...button press: ", event.ydata, event.xdata
            
            if ax !=event.inaxes: 
                print " click outside image box "
                return

            print " click inside image box "
                
            if add_cell_flag:
                print "... adding cell...", add_cell_flag
                add_cell_flag=False

                
                #***********Clear previous 2 cells *******
                ax.contour(y_array[previous_cell], x_array[previous_cell], Bmat_array[previous_cell], [thr_array[previous_cell]], linewidth=l_width, colors=colors)
                ax.contour(y_array[nearest_cell], x_array[nearest_cell], Bmat_array[nearest_cell], [thr_array[nearest_cell]], linewidth=l_width, colors=colors)

                print " # cells: ", len(y_array)
            
                circle_size=5
                #Define N points on a circle centred at mouse click; shift circle to location
                points = np.vstack(PointsInCircum(circle_size,n=20))
                points = points + [int(event.ydata), int(event.xdata)]

                area_coords = []
                for i in range(len(points)):
                    area_coords.append((points[i][0], points[i][1]))
                    
                #Plot recent area
                temp_coords = np.array(area_coords)
                ax.plot(temp_coords[:,1],temp_coords[:,0],color='springgreen',linewidth=2)
                
                fig.canvas.draw()
                
                #*******Need to compute values and add them to arrays
                #Add centres
                cm_temp = [int(event.ydata), int(event.xdata)]
                cm = np.append(cm,[cm_temp], axis=0)                    #*** APpend centre of mass
                
                #Add grids
                x_array = np.append(x_array,[x_array[-1]],axis=0)       #*** Append ygrid
                y_array = np.append(y_array,[y_array[-1]],axis=0)       #*** Append xgrid
                print len(x_array[0]), len(y_array[0])

                #Add thrshold
                thr_array = np.append(thr_array,[thr_array[-1]], axis=0)

                #Add Bmatrix values:
                #this requires finding points inside drawn ROI
                from matplotlib.path import Path
                vertixes_path = Path(temp_coords)       #Define vertices from circle
                
                all_points = []
                for i in range(len(Bmat_array[0])):
                    for j in range(len(Bmat_array[0][0])):
                        all_points.append([i,j])
                print "len allpoints:" , len(all_points)
                        
                mask = vertixes_path.contains_points(all_points)
                mask = np.reshape(mask,(len(Bmat_array[0]),len(Bmat_array[0][0])))
                
                print Bmat_array.shape, mask.shape
                Bmat_array = np.append(Bmat_array, [mask], axis=0)       #*** Append Bmat (mask)
                
                #fig41 = plt.subplots()     #Check ROI drawn.
                #plt.imshow(mask)
                #plt.show()
                
                #Add traces:
                #filename = root.data.file_name #'/media/cat/4TB/in_vivo/rafa/alejandro/20171013/cropped_Registered_20171013to20171024_processed.npz'
                print file_name
                image_stack_1D = np.load(file_name)['Yr'].T
                
                mask_1d = np.ravel(mask.T)
                temp_trace = []
                for k in range(len(image_stack_1D)):
                    print "...frame: ", k
                    temp_trace.append(np.sum(image_stack_1D[k]*mask_1d))
                
                baseline = np.mean(temp_trace)
                temp_trace_dff = (temp_trace-baseline)/baseline*1E2
                traces = np.append(traces.T,[temp_trace_dff], axis=0).T     #*** Append traces

                if False:
                    temp_trace = np.array(temp_trace)
                    fig41 = plt.subplots()
                    plt.plot(temp_trace)
                    
                    plt.plot(temp_trace_dff)
                    plt.show()
                    
                    print traces.shape
                    print temp_trace.shape
                    print traces.shape
                                
                
                reset_function()

                #ax.contour(y_array, Bmat, [thr], linewidth=l_width, colors=colors)
                
                return
            
            else: 
                print "... selecting cells..."
                
                #***********Clear previous 2 cells *******
                ax.contour(y_array[previous_cell], x_array[previous_cell], Bmat_array[previous_cell], [thr_array[previous_cell]], linewidth=l_width, colors=colors)
                ax.contour(y_array[nearest_cell], x_array[nearest_cell], Bmat_array[nearest_cell], [thr_array[nearest_cell]], linewidth=l_width, colors=colors)

                previous_cell = nearest_cell
                nearest_cell = find_nearest_euclidean(cm, [event.ydata, event.xdata])
                print "nearest_cell: ", nearest_cell

                #Redraw new cells
                ax.contour(y_array[previous_cell], x_array[previous_cell], Bmat_array[previous_cell], [thr_array[previous_cell]], linewidths=l_width, colors='red',alpha=0.9)
                ax.contour(y_array[nearest_cell], x_array[nearest_cell], Bmat_array[nearest_cell], [thr_array[nearest_cell]], linewidths=l_width, colors='blue',alpha=0.9)

                redraw_traces()


    fig.canvas.callbacks.connect('button_press_event', callback)


    #***********************************************************************************
    #****************************** UPDATE CALCIUM MOVIE BUTTON ***********************
    #***********************************************************************************
    def update(val):
        ''' This updates calcium movie in left panel and traces in right panel
            - possible to speed it up? 
        '''
        global nearest_cell, previous_cell, l_width, ylim_max,  ylim_min, y_array, x_array, Bmat_array, thr_array, traces, cm, img1, img_data, ax, ax2, ax3
        img1.set_data(img_data[int(frame.val)])    
        img1.set_cmap(radio.value_selected)

        ax3.cla()
        ax3.plot(traces[:,previous_cell], color='red')
        ax3.plot([int(frame.val),int(frame.val)],[ylim_min,ylim_max])
        ax3.set_ylim(ylim_min, ylim_max)
        ax3.set_xlim(0,len(images_kalman))
        ax3.set_title("Cell: "+str(previous_cell), fontsize=15)

        ax2.cla()
        ax2.plot(traces[:,nearest_cell], color='blue')
        ax2.plot([int(frame.val),int(frame.val)],[-50,400])
        ax2.set_ylim(-50, 400)
        ax2.set_xlim(0,len(images_kalman))
        ax2.set_title("Cell: "+str(nearest_cell), fontsize=15)

        #fig.canvas.draw_idle()
        fig.canvas.draw()

    frame.on_changed(update)


    #***********************************************************************************
    #**************************** RESET NEURON BUTTON **********************************
    #***********************************************************************************
    resetax = plt.axes([0.025, 0.025, 0.03, 0.03])
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
    def reset_button(event):
        print "...reseting..."
        global nearest_cell, previous_cell, l_width, ylim_max, ylim_min, y_array, x_array, Bmat_array, thr_array, traces, cm, img1, color_selected, img_data, ax, ax2, ax3
        print len(y_array)

        previous_cell = 0
        nearest_cell=(previous_cell+1)%len(x_array)
        
        reset_function()

    button.on_clicked(reset_button)
    
    
    #***********************************************************************************
    #****************************** COLORS SELECTOR **************************************
    #***********************************************************************************
    rax = plt.axes([0.025, 0.75, 0.08, 0.08])#, facecolor=axcolor)
    radio = RadioButtons(rax, ('viridis', 'Greys_r', 'plasma'), active=0)
    color_selected = 'viridis'
    def colorfunc(label):
        global color_selected, img_data
        #img1.set_color(label)
        img1.set_data(img_data[int(frame.val)])    
        color_selected = radio.value_selected
        img1.set_cmap(color_selected)
        fig.canvas.draw_idle()
        #fig.canvas.draw()

    radio.on_clicked(colorfunc)


    #***********************************************************************************
    #****************************** NEXT NEURON BUTTON *********************************
    #***********************************************************************************

    next_neuron_ax = plt.axes([0.025, 0.405, 0.04, 0.03])
    button5 = Button(next_neuron_ax, 'Next\nNeuron', color=axcolor, hovercolor='0.975')
    
    def next_neuron(event):
        global nearest_cell, previous_cell, l_width, ylim_max, ylim_min, y_array, x_array, Bmat_array, thr_array, traces, cm, img1, color_selected, img_data, ax, ax2, ax3
        
        if nearest_cell==len(x_array)-1:
            print "... you are at the last cell..."
            return

        previous_cell = nearest_cell
        nearest_cell = nearest_cell+1
        print "next cell: ", nearest_cell

        reset_function()

    button5.on_clicked(next_neuron)


    #***********************************************************************************
    #****************************** PREVIOUS NEURON BUTTON *********************************
    #***********************************************************************************

    previous_neuron_ax = plt.axes([0.025, 0.365, 0.04, 0.03])
    button6 = Button(previous_neuron_ax, 'Previous\nNeuron', color=axcolor, hovercolor='0.975')
    
    def previous_neuron(event):
        global nearest_cell, previous_cell, l_width, ylim_max, ylim_min, y_array, x_array, Bmat_array, thr_array, traces, cm, img1, color_selected, img_data, ax, ax2, ax3

        if nearest_cell==0:
            print "... you are at the first cell..."
            return

        previous_cell=nearest_cell
        nearest_cell = nearest_cell-1

        reset_function()

    button6.on_clicked(previous_neuron)
    

    #***********************************************************************************
    #**************************** ADD NEURON BUTTON **********************************
    #***********************************************************************************
    add_cell_ax = plt.axes([0.025, 0.445, 0.03, 0.03])
    button8 = Button(add_cell_ax, 'Add\nNeuron', color=axcolor, hovercolor='0.975')
    def add_cell(event):
        print "...reseting..."
        global add_cell_flag
        
        add_cell_flag = True
        
    button8.on_clicked(add_cell)
    
    
    #***********************************************************************************
    #**************************** DELETE NEURON BUTTON *********************************
    #**********************************************************************************
    deleteax = plt.axes([0.025, 0.285, 0.03, 0.03])
    button2 = Button(deleteax, 'Delete', color=axcolor, hovercolor='0.975')
    
    def delete_cell(event):
        global nearest_cell, previous_cell, l_width, ylim_max, ylim_min, y_array, x_array, Bmat_array, thr_array, traces, cm, img1, color_selected, img_data, ax, ax2, ax3

        print "...deleting cell: ", nearest_cell

        print len(y_array)
        y_array=np.delete(y_array, nearest_cell, axis=0)
        x_array=np.delete(x_array, nearest_cell, axis=0)
        Bmat_array=np.delete(Bmat_array, nearest_cell, axis=0)
        thr_array=np.delete(thr_array, nearest_cell,axis=0)
        traces=np.delete(traces, nearest_cell,axis=1)
        cm = np.delete(cm, nearest_cell,axis=0)

        previous_cell=0
        nearest_cell=(previous_cell+1)%len(x_array)
        
        reset_function()

    button2.on_clicked(delete_cell)


    #***********************************************************************************
    #**************************** SAVE PROGRESS BUTTON *********************************
    #**********************************************************************************
    save_progress_ax = plt.axes([0.025, 0.065, 0.04, 0.03])
    button3 = Button(save_progress_ax, 'Save\nProgress', color=axcolor, hovercolor='0.975')
    
    def save_progress(event):
        global nearest_cell, previous_cell, l_width, ylim_max, ylim_min, y_array, x_array, Bmat_array, thr_array, traces, cm, img1, img_data, ax, ax2, ax3

        print "\n\n...saving progress...\n\n "
        np.savez(file_name[:-4]+"_saved_progress",  y_array=y_array, x_array=x_array, Bmat_array=Bmat_array, thr_array=thr_array, traces=traces, cm = cm)

    button3.on_clicked(save_progress)


    #***********************************************************************************
    #**************************** LOAD PROGRESS BUTTON *********************************
    #**********************************************************************************
    load_progress_ax = plt.axes([0.025, 0.105, 0.04, 0.03])
    button4 = Button(load_progress_ax, 'Load\nProgress', color=axcolor, hovercolor='0.975')
    
    def load_progress(event):
        global nearest_cell, previous_cell, l_width, ylim_max, ylim_min, y_array, x_array, Bmat_array, thr_array, traces, cm, img1, img_data, ax, ax2, ax3

        print "\n\n...loading in-progress file...\n\n "
        data = np.load(file_name[:-4]+"_saved_progress.npz")
        y_array=data['y_array']
        x_array=data['x_array']
        Bmat_array=data['Bmat_array']
        thr_array=data['thr_array']
        traces=data['traces']
        cm = data['cm']
        
        reset_function()

    button4.on_clicked(load_progress)


    #**********************************************************************************
    #**************************** EXPORT DATA *****************************************
    #**********************************************************************************
    export_ax = plt.axes([0.025, 0.145, 0.04, 0.03])
    button7 = Button(export_ax, 'Export\nData', color=axcolor, hovercolor='0.975')
    
    def export_data(event):
        global nearest_cell, previous_cell, l_width, ylim_max, ylim_min, y_array, x_array, Bmat_array, thr_array, traces, cm, img1, img_data, ax, ax2, ax3

        #print "...saving data in .txt "
        
        print "\n\n... exported ROIs file ... \n\n"
        np.savez(file_name[:-4]+"_ROIs",  y_array=y_array, x_array=x_array, Bmat_array=Bmat_array, thr_array=thr_array, traces=traces, cm = cm)
        sio.savemat(file_name[:-4]+'_ROIs.mat', {'y_array':y_array, 'x_array':x_array, 'Bmat_array':Bmat_array, 'thr_array':thr_array, 'traces':traces, 'centres':cm})
        
        #np.savetxt(file_name[:-4]+"_traces.txt", traces)
        #np.savetxt(file_name[:-4]+"_centres.txt", cm)

        #rasters = []
        #raster_array = np.zeros(traces.shape, dtype=np.float32)
        #print raster_array.shape
        ##fig2 = plt.figure()
        #for t in range(len(traces[0])):
            #derivative = traces[:,t][1:]-traces[:,t][:-1]
            #der_std = np.std(derivative)
            #spikes = np.where(derivative>(der_std*3))[0]
            #rasters.append(spikes)
            #print len(spikes)
            #print spikes
            #raster_array[:,t][spikes]=1
            ##plt.scatter(spikes, [t]*len(spikes))
        
        ##plt.show()
        
        #import scipy.io as sio
        
        #np.savetxt(file_name[:-4]+"_rasters.txt", raster_array)

    button7.on_clicked(export_data)

    plt.show()


def find_nearest_euclidean(array, value):
    #Returns index of closest neuron
    dists = np.sqrt(np.sum((array-value)**2, axis=1))
    return np.argmin(dists) #index of nearest cell
    
    
def save_traces(file_name, Yr, A, C, b, f, d1, d2, YrA = None, image_neurons=None, thr=0.99, denoised_color=None, cmap='viridis'):
    """
    Cat: Extraneous function to save calcium traces for each neuron; I COULDN"T FIGURE OUT HOW TO DO IT IN CAIMAN SO REUSING THIS CODE ***** CAN BE IMPROVED

    Parameters:
    -----------
    Yr: np.ndarray
        movie

    A,C,b,f: np.ndarrays
        outputs of matrix factorization algorithm

    d1,d2: floats
        dimensions of movie (x and y)

    YrA:   np.ndarray
        ROI filtered residual as it is given from update_temporal_components
        If not given, then it is computed (K x T)        

    image_neurons: np.ndarray
        image to be overlaid to neurons (for instance the average)

    thr: double
        threshold regulating the extent of the displayed patches

    denoised_color: string or None
        color name (e.g. 'red') or hex color code (e.g. '#F0027F')

    cmap: string
        name of colormap (e.g. 'viridis') used to plot image_neurons
    """
    colormap = mpl.cm.get_cmap(cmap)
    #colormap = cmap
    grayp = [mpl.colors.rgb2hex(m) for m in colormap(np.arange(colormap.N))]
    nr, T = C.shape
    #nA2 = np.ravel(A.power(2).sum(0))
    nA2 = np.ravel((A**2).sum(0))
    
    b = np.squeeze(b)
    f = np.squeeze(f)
    if YrA is None:
        Y_r = np.array(spdiags(old_div(1, nA2), 0, nr, nr) *
                   (A.T * np.matrix(Yr) -
                    (A.T * np.matrix(b[:, np.newaxis])) * np.matrix(f[np.newaxis]) -
                    A.T.dot(A) * np.matrix(C)) + C)
    else:
        Y_r = C + YrA
            

    x = np.arange(T)
    z = old_div(np.squeeze(np.array(Y_r[:, :].T)), 100)
    print "Traces shape: ", z.shape
    np.save(file_name[:-4]+"_traces", z)

def nb_view_patches(file_name, Yr, A, C, b, f, d1, d2, YrA = None, image_neurons=None, thr=0.99, denoised_color=None, cmap='viridis'):
    """
    Interactive plotting utility for ipython notebook

    Parameters:
    -----------
    Yr: np.ndarray
        movie

    A,C,b,f: np.ndarrays
        outputs of matrix factorization algorithm

    d1,d2: floats
        dimensions of movie (x and y)

    YrA:   np.ndarray
        ROI filtered residual as it is given from update_temporal_components
        If not given, then it is computed (K x T)        

    image_neurons: np.ndarray
        image to be overlaid to neurons (for instance the average)

    thr: double
        threshold regulating the extent of the displayed patches

    denoised_color: string or None
        color name (e.g. 'red') or hex color code (e.g. '#F0027F')

    cmap: string
        name of colormap (e.g. 'viridis') used to plot image_neurons
    """
    colormap = mpl.cm.get_cmap(cmap)
    #colormap = cmap
    grayp = [mpl.colors.rgb2hex(m) for m in colormap(np.arange(colormap.N))]
    nr, T = C.shape
    #nA2 = np.ravel(A.power(2).sum(0))
    nA2 = np.ravel((A**2).sum(0))
    
    b = np.squeeze(b)
    f = np.squeeze(f)
    if YrA is None:
        Y_r = np.array(spdiags(old_div(1, nA2), 0, nr, nr) *
                   (A.T * np.matrix(Yr) -
                    (A.T * np.matrix(b[:, np.newaxis])) * np.matrix(f[np.newaxis]) -
                    A.T.dot(A) * np.matrix(C)) + C)
    else:
        Y_r = C + YrA
            

    x = np.arange(T)
    z = old_div(np.squeeze(np.array(Y_r[:, :].T)), 100)
    print "Traces shape: ", z.shape
    np.save(file_name[:-4]+"_traces", z)
    
    
    if image_neurons is None:
        image_neurons = A.mean(1).reshape((d1, d2), order='F')

    #coors = get_contours(A, (d1, d2), thr, )
    #REUSING get_contours function from different place
    coors = get_contours(A, Cn=image_neurons, thr=None, thr_method='max', maxthr=0.2, nrgthr=0.9, display_numbers=True, max_number=None,
                  cmap=None, swap_dim=False, colors='w', vmin=None, vmax=None)
    cc1 = [cor['coordinates'][:, 0] for cor in coors]
    cc2 = [cor['coordinates'][:, 1] for cor in coors]
    c1 = cc1[0]
    c2 = cc2[0]

    # split sources up, such that Bokeh does not warn
    # "ColumnDataSource's columns must be of the same length"
    source = ColumnDataSource(data=dict(x=x, y=z[:, 0], y2=C[0] / 100))
    source_ = ColumnDataSource(data=dict(z=z.T, z2=C / 100))
    source2 = ColumnDataSource(data=dict(c1=c1, c2=c2))
    source2_ = ColumnDataSource(data=dict(cc1=cc1, cc2=cc2))

    callback = CustomJS(args=dict(source=source, source_=source_, source2=source2, source2_=source2_), code="""
            var data = source.get('data')
            var data_ = source_.get('data')
            var f = cb_obj.get('value')-1
            x = data['x']
            y = data['y']
            y2 = data['y2']

            for (i = 0; i < x.length; i++) {
                y[i] = data_['z'][i+f*x.length]
                y2[i] = data_['z2'][i+f*x.length]
            }

            var data2_ = source2_.get('data');
            var data2 = source2.get('data');
            c1 = data2['c1'];
            c2 = data2['c2'];
            cc1 = data2_['cc1'];
            cc2 = data2_['cc2'];

            for (i = 0; i < c1.length; i++) {
                   c1[i] = cc1[f][i]
                   c2[i] = cc2[f][i]
            }
            source2.trigger('change')
            source.trigger('change')
        """)
        
    print x.shape
    print z.shape
    y_array = z.T
    t = np.arange(len(x))
    for k in range(len(y_array)):
        #print x[k]
        #print z[k]
        plt.plot(t,y_array[k]+50*k)
    
        #x = np.arange(T)
        #z = old_div(np.squeeze(np.array(Y_r[:, :].T)), 100)
        
    plt.show()

    plot = bpl.figure(plot_width=600, plot_height=300)
    plot.line('x', 'y', source=source, line_width=1, line_alpha=0.6)
    if denoised_color is not None:
        plot.line('x', 'y2', source=source, line_width=1, line_alpha=0.6, color=denoised_color)

    slider = bokeh.models.Slider(start=1, end=Y_r.shape[0], value=1, step=1,
                                 title="Neuron Number", callback=callback)
    xr = Range1d(start=0, end=image_neurons.shape[1])
    yr = Range1d(start=image_neurons.shape[0], end=0)
    plot1 = bpl.figure(x_range=xr, y_range=yr, plot_width=800, plot_height=800)

    plot1.image(image=[image_neurons[::-1, :]], x=0,
                y=image_neurons.shape[0], dw=d2, dh=d1, palette=grayp)
    plot1.patch('c1', 'c2', alpha=0.6, color='purple', line_width=2, source=source2)


    bpl.show(bokeh.layouts.layout([[slider], [bokeh.layouts.row(plot1, plot)]]))

    return Y_r


def get_contours(A, Cn, thr=None, thr_method='max', maxthr=0.2, nrgthr=0.9, display_numbers=True, max_number=None,
                  cmap=None, swap_dim=False, colors='w', vmin=None, vmax=None, **kwargs):
    """Gets contour of spatial components and returns their coordinates

     Parameters:
     -----------
     A:   np.ndarray or sparse matrix
               Matrix of Spatial components (d x K)
     
	 dims: tuple of ints
               Spatial dimensions of movie (x, y[, z])
     
	 thr: scalar between 0 and 1
               Energy threshold for computing contours (default 0.9)

     Returns:
     --------
     Coor: list of coordinates with center of mass and
            contour plot coordinates (per layer) for each component
            
        
    #"""
    #Cat cutoff code above this...
    
    if issparse(A):
        A = np.array(A.todense())
    else:
        A = np.array(A)

    if swap_dim:
        Cn = Cn.T
        print('Swapping dim')

    d1, d2 = np.shape(Cn)
    d, nr = np.shape(A)
    print "# neurons: ", nr
    if max_number is None:
        max_number = nr

    #if thr is not None:
    #    thr_method = 'nrg'
    #    nrgthr = thr
    #    warn("The way to call utilities.plot_contours has changed. Look at the definition for more details.")

    x, y = np.mgrid[0:d1:1, 0:d2:1]

        
    ax = plt.gca()
    if vmax is None and vmin is None:
        plt.imshow(Cn, interpolation=None, cmap=cmap,
                  vmin=np.percentile(Cn[~np.isnan(Cn)], 1), vmax=np.percentile(Cn[~np.isnan(Cn)], 99))
    else:
        plt.imshow(Cn, interpolation=None, cmap=cmap,
                  vmin=vmin, vmax=vmax)
    
    coordinates = []
    cm = com(A, d1, d2)
    for i in range(np.minimum(nr, max_number)):
        print i
    
        pars = dict(kwargs)
        if thr_method == 'nrg':
            indx = np.argsort(A[:, i], axis=None)[::-1]
            cumEn = np.cumsum(A[:, i].flatten()[indx]**2)
            cumEn /= cumEn[-1]
            Bvec = np.zeros(d)
            Bvec[indx] = cumEn
            thr = nrgthr

        else:  # thr_method = 'max'
            if thr_method != 'max':
                warn("Unknown threshold method. Choosing max")
            Bvec = A[:, i].flatten()
            Bvec /= np.max(Bvec)
            thr = maxthr

        if swap_dim:
            Bmat = np.reshape(Bvec, np.shape(Cn), order='C')
        else:
            Bmat = np.reshape(Bvec, np.shape(Cn), order='F')
        cs = plt.contour(y, x, Bmat, [thr], colors=colors)
        
        # this fix is necessary for having disjoint figures and borders plotted correctly
        p = cs.collections[0].get_paths()
        v = np.atleast_2d([np.nan, np.nan])
        for pths in p:
            vtx = pths.vertices
            num_close_coords = np.sum(np.isclose(vtx[0, :], vtx[-1, :]))
            if num_close_coords < 2:
                if num_close_coords == 0:
                    # case angle
                    newpt = np.round(old_div(vtx[-1, :], [d2, d1])) * [d2, d1]
                    #import ipdb; ipdb.set_trace()
                    vtx = np.concatenate((vtx, newpt[np.newaxis, :]), axis=0)

                else:
                    # case one is border
                    vtx = np.concatenate((vtx, vtx[0, np.newaxis]), axis=0)
                    #import ipdb; ipdb.set_trace()

            v = np.concatenate((v, vtx, np.atleast_2d([np.nan, np.nan])), axis=0)
        pars['CoM'] = np.squeeze(cm[i, :])
        pars['coordinates'] = v
        pars['bbox'] = [np.floor(np.min(v[:, 1])), np.ceil(np.max(v[:, 1])),
                        np.floor(np.min(v[:, 0])), np.ceil(np.max(v[:, 0]))]
        pars['neuron_id'] = i + 1
        coordinates.append(pars)

    plt.close()

    #if display_numbers:
    #    for i in range(np.minimum(nr, max_number)):
    #        if swap_dim:
    #            ax.text(cm[i, 0], cm[i, 1], str(i + 1), color=colors)
    #        else:
    #            ax.text(cm[i, 1], cm[i, 0], str(i + 1), color=colors)

    return coordinates
    
    
