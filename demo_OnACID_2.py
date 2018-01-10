#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 17:52:23 2017
Basic demo for the OnACID algorithm using CNMF initialization. For a more 
complete demo check the script demo_OnACID_mesoscope.py
@author: jfriedrich & epnev
"""

import numpy as np
import pylab as pl
import caiman as cm
from caiman.source_extraction import cnmf as cnmf
from caiman.utils.visualization import view_patches_bar, plot_contours
from copy import deepcopy
from scipy.special import log_ndtr

import sys, os

#********************************************************************************************************************************
#********************************************************************************************************************************
#********************************************************************************************************************************

#%%
#fname = './example_movies/demoMovie.tif'
fname = sys.argv[1] #'/mnt/244f644c-15b8-46be-8f8a-50b5b2d8c6e1/in_vivo/rafa/alejandro/G2M5/20170511/000/G2M5_C1V1_GCaMP6s_20170511_000.tif'
print ("...running demo_OnACID on file: ", fname)

Y = cm.load(fname).astype(np.float32)                   # 
Cn = cm.local_correlations(Y.transpose(1, 2, 0))        # used as a background image


#Load parameters from .npz file
if True:
    print (sys.argv[1][:-4]+"_runtime_params.npz")
    params = np.load(sys.argv[1][:-4]+"_runtime_params.npz")
    
    print (type(params['merge_thr']))
    merge_thresh=np.float32(params['merge_thr']); print (merge_thresh)
    initbatch = np.int32(params['initibatch']); print (initbatch)
    patch_size = np.int32(params['patch_size']); print (patch_size)
    stride=np.int32(params['stride']); print (stride)
    K=np.int32(params['K']); print (K)
    
    frame_rate=np.float32(params['frame_rate']); print (frame_rate)
    decay_time=np.float32(params['decay_time']); print (decay_time)

    gSig=params['neuron_size']; print (type(gSig))
    gSig=values = [int(x) for x in str(gSig).split(',') if x]
    print (gSig)
    print (type(gSig))
    #gSig=[6,6]
    #print gSig
    #print type(gSig)
    
    p=np.int32(params['AR_dynamics']); print (p)
    min_SNR=np.float32(params['min_SNR']); print (min_SNR)
    rval_thr = np.float32(params['rval_threshold']); print (rval_thr)
    gnb=np.int32(params['no_bkgr_components']); print (gnb)
    
    max_comp_update_shape=params['no_updated_shapes']
    if max_comp_update_shape =='inf': max_comp_update_shape = np.inf
    else: max_comp_update_shape = np.float32(max_comp_update_shape)
    print (max_comp_update_shape)
    
    expected_comps=np.int32(params['no_expected_shapes']); print (expected_comps)
    N_samples=np.float32(params['no_timesteps']); print (N_samples)
    thresh_fitness_raw=np.float32(params['exceptionality_threshold']); print (thresh_fitness_raw)
    
    T1=params['total_len_file']
    if T1=='all': T1 = Y.shape[0]
    else: T1=np.int32(T1)

    cnn_mode_filename = str(params['caiman_location']) +'/use_cases/CaImAnpaper/cnn_model'

else: #Default parameters
    #%% set up some parameters

    fr = 10                                                             # frame rate (Hz)
    decay_time = 0.5                                                    # approximate length of transient event in seconds
    gSig = [6,6]                                                        # expected half size of neurons
    p = 1                                                               # order of AR indicator dynamics
    min_SNR = 3.5                                                       # minimum SNR for accepting new components
    rval_thr = 0.90                                                     # correlation threshold for new component inclusion
    gnb = 3                                                             # number of background components

    # set up some additional supporting parameters needed for the algorithm (these are default values but change according to dataset characteristics)

    max_comp_update_shape = np.inf                                      # number of shapes to be updated each time (put this to a finite small value to increase speed)
    expected_comps = 250                                                 # maximum number of expected components used for memory pre-allocation (exaggerate here)
    N_samples = np.ceil(fr*decay_time)                                  # number of timesteps to consider when testing new neuron candidates
    thresh_fitness_raw = log_ndtr(-min_SNR)*N_samples                   # exceptionality threshold
    T1 = Y.shape[0]                                                     # total length of file

    # set up CNMF initialization parameters 

    merge_thresh = 0.8                                                  # merging threshold, max correlation allowed
    initbatch = 20000                                                     # number of frames for initialization (presumably from the first file)
    patch_size = 32                                                     # size of patch
    stride = 3                                                          # amount of overlap between patches
    K = 4                                                               # max number of components in each patch


#********************************************************************************************************************************
#********************************************************************************************************************************
#********************************************************************************************************************************
#%% obtain initial batch file used for initialization

fname_new = Y[:initbatch].save('demo.mmap', order='C')              # memory map file (not needed)
Yr, dims, T = cm.load_memmap(fname_new)
images = np.reshape(Yr.T, [T] + list(dims), order='F')
Cn_init = cm.local_correlations(np.reshape(Yr, dims + (T,), order='F'))


#%% RUN (offline) CNMF algorithm on the initial batch
pl.close('all')
cnm_init = cnmf.CNMF(2, k=K, gSig=gSig, merge_thresh=merge_thresh,
                     p=p, rf=patch_size//2, stride=stride, skip_refinement=False,
                     normalize_init=False, options_local_NMF=None,
                     minibatch_shape=100, minibatch_suff_stat=5,
                     update_num_comps=True, rval_thr=rval_thr,
                     thresh_fitness_delta=-50, gnb = gnb,
                     thresh_fitness_raw=thresh_fitness_raw,
                     batch_update_suff_stat=True, max_comp_update_shape=max_comp_update_shape)

cnm_init = cnm_init.fit(images)

print(('Number of components:' + str(cnm_init.A.shape[-1])))

if False:
    pl.figure()
    crd = plot_contours(cnm_init.A.tocsc(), Cn_init, thr=0.9)


#%% run (online) OnACID algorithm 

cnm = deepcopy(cnm_init)
cnm._prepare_object(np.asarray(Yr), T1, expected_comps)
t = cnm.initbatch

Y_ = cm.load(fname)[initbatch:].astype(np.float32)
for frame_count, frame in enumerate(Y_):
    cnm.fit_next(t, frame.copy().reshape(-1, order='F'))
    t += 1

#%% extract the results
    
C, f = cnm.C_on[cnm.gnb:cnm.M], cnm.C_on[:cnm.gnb]
A, b = cnm.Ab[:, cnm.gnb:cnm.M], cnm.Ab[:,:cnm.gnb]
print(('Number of components:' + str(A.shape[-1])))

#%% pass through the CNN classifier with a low threshold (keeps clearer neuron shapes and excludes processes)
use_CNN = False
if use_CNN:
    print ("... using CNN classifier ...")
    thresh_cnn = 0.1                                                                # threshold for CNN classifier
    from caiman.components_evaluation import evaluate_components_CNN 
    predictions,final_crops = evaluate_components_CNN(A,dims,gSig,model_name = '/home/cat/Downloads/CaImAn/use_cases/CaImAnpaper/cnn_model')
    A_exclude, C_exclude = A[:,predictions[:,1]<thresh_cnn], C[predictions[:,1]<thresh_cnn]
    A, C = A[:,predictions[:,1]>=thresh_cnn], C[predictions[:,1]>=thresh_cnn]
    noisyC = cnm.noisyC[cnm.gnb:cnm.M]
    YrA = noisyC[predictions[:,1]>=thresh_cnn] - C
else:
    print ("...skipping CNN classifier...")
    YrA = cnm.noisyC[cnm.gnb:cnm.M] - C

#%% plot results
if False:
    pl.figure()
    crd = cm.utils.visualization.plot_contours(A, Cn, thr=0.9)

    view_patches_bar(Yr, A, C, b, f,
                     dims[0], dims[1], YrA, img=Cn)


#SAVE DATA
print ("\n\n...saving .npz file containing all processed data: ", fname[:-4]+"_processed.npz")
np.savez(fname[:-4]+"_processed.npz", C=C, A=A, Y_=Y_,Yr=Yr, b=cnm.b, f=cnm.f,YrA=cnm.noisyC[cnm.gnb:cnm.M] - C, Cn=Cn)

print ("\n... Clean exit!...\n\n\n")


