# -*- coding: utf-8 -*-

import time
import os
import numpy as np
import copy
import re
import sys
import pickle
from scipy import stats
from scipy import constants as cst
from scipy import stats as st

sys.path.insert(0,'input_files')
import config

import xobjects as xo
import xtrack as xt
import xfields as xf
import xline
from PyHEADTAIL.trackers.transverse_tracking import TransverseSegmentMap
from PyHEADTAIL.particles.slicing import UniformBinSlicer
from PyHEADTAIL.particles.slicing import UniformChargeSlicer
sys.path.insert(0,'src')
import src
import profiler

import argparse

from mpi4py import MPI



def slice_slice_interaction(particles_b1, particles_b2, slice_mask_b1, slice_mask_b2, s1_idx, s2_idx, xfields_beambeam, b1_means_dict, b2_means_dict, turn, measure_time, time_elapsed_dict):
"""
Code for singel slice-slice interaction
:param particles_bX: xtrack object of full8 beam particles
:param slice_mask_bX: numpy array of size (# macroparts in current size), containing integer indices of slice macroparts.
:param sX_idx: int, index of slice in beam
:param xfields_beambeam: dict containing xfields beambeam elements
:param bX_means_dict: dict containing floats for the mean coordinates (x,x') of the full beam
:param turn: int, turn number, for verbose
:param measure_time: profiler object, to measure elapsed time between different steps
:param time_elapsed_dict: dict, keys are strings denoting simulation steps and values are a single float that is the elapsed time 

:return particles_bX: modified xtrack particle object with the slice coordinates updated with the kick
"""
    measure_time.start()
    particles_b1.state[slice_mask_b1] = 1000 + s1
    particles_b2.state[slice_mask_b2] = 1000 + s2
    src.add_new_value(time_elapsed_dict, "Set slice states", measure_time.stop())

    # update xsuite elements with current slice indices
    measure_time.start()
    xfields_beambeam["iptocp_ip1_b1"].slice_id = s1
    xfields_beambeam["iptocp_ip1_b2"].slice_id = s2
    xfields_beambeam["cptoip_ip1_b1"].slice_id = s1
    xfields_beambeam["cptoip_ip1_b2"].slice_id = s2
    xfields_beambeam["strongstrong3d_ip1_b1"].slice_id = s1
    xfields_beambeam["strongstrong3d_ip1_b2"].slice_id = s2
    src.add_new_value(time_elapsed_dict, "Update elements with slice indices", measure_time.stop())

    # Measure boosted slice properties at IP
    measure_time.start()
    s1_means_dict, s1_sigmas_dict = src.get_stats(particles_b1, mask=slice_mask_b1)
    s2_means_dict, s2_sigmas_dict = src.get_stats(particles_b2, mask=slice_mask_b2)
    src.add_new_value(time_elapsed_dict, "Get slice moments", measure_time.stop())

    # for transforming the reference frame into w.r.t other slice's centroid at the IP
    measure_time.start()
    xfields_beambeam["iptocp_ip1_b1"].x_bb_centroid  =  s2_means_dict["x"] 
    xfields_beambeam["iptocp_ip1_b2"].x_bb_centroid  =  s1_means_dict["x"] 
    xfields_beambeam["iptocp_ip1_b1"].y_bb_centroid  =  s2_means_dict["y"] 
    xfields_beambeam["iptocp_ip1_b2"].y_bb_centroid  =  s1_means_dict["y"] 
    xfields_beambeam["iptocp_ip1_b1"].px_bb_centroid =  s2_means_dict["px"]
    xfields_beambeam["iptocp_ip1_b2"].px_bb_centroid =  s1_means_dict["px"]
    xfields_beambeam["iptocp_ip1_b1"].py_bb_centroid =  s2_means_dict["py"]
    xfields_beambeam["iptocp_ip1_b2"].py_bb_centroid =  s1_means_dict["py"]
    xfields_beambeam["iptocp_ip1_b1"].z_bb_centroid  =  s2_means_dict["z"] 
    xfields_beambeam["iptocp_ip1_b2"].z_bb_centroid  =  s1_means_dict["z"] 
    xfields_beambeam["iptocp_ip1_b1"].z_centroid     =  s1_means_dict["z"] 
    xfields_beambeam["iptocp_ip1_b2"].z_centroid     =  s2_means_dict["z"] 
    
    xfields_beambeam["cptoip_ip1_b1"].x_bb_centroid  =  s2_means_dict["x"] 
    xfields_beambeam["cptoip_ip1_b2"].x_bb_centroid  =  s1_means_dict["x"] 
    xfields_beambeam["cptoip_ip1_b1"].y_bb_centroid  =  s2_means_dict["y"] 
    xfields_beambeam["cptoip_ip1_b2"].y_bb_centroid  =  s1_means_dict["y"] 
    xfields_beambeam["cptoip_ip1_b1"].px_bb_centroid =  s2_means_dict["px"]
    xfields_beambeam["cptoip_ip1_b2"].px_bb_centroid =  s1_means_dict["px"]
    xfields_beambeam["cptoip_ip1_b1"].py_bb_centroid =  s2_means_dict["py"]
    xfields_beambeam["cptoip_ip1_b2"].py_bb_centroid =  s1_means_dict["py"]
    xfields_beambeam["cptoip_ip1_b1"].z_bb_centroid  =  s2_means_dict["z"] 
    xfields_beambeam["cptoip_ip1_b2"].z_bb_centroid  =  s1_means_dict["z"]   
    xfields_beambeam["cptoip_ip1_b1"].z_centroid     =  s1_means_dict["z"] 
    xfields_beambeam["cptoip_ip1_b2"].z_centroid     =  s2_means_dict["z"] 
    
    xfields_beambeam["iptocp_ip1_b1"].x_full_bb_centroid  =  b2_means_dict["x"]
    xfields_beambeam["iptocp_ip1_b2"].x_full_bb_centroid  =  b1_means_dict["x"]
    xfields_beambeam["iptocp_ip1_b1"].y_full_bb_centroid  =  b2_means_dict["y"]
    xfields_beambeam["iptocp_ip1_b2"].y_full_bb_centroid  =  b1_means_dict["y"]
    
    xfields_beambeam["cptoip_ip1_b1"].x_full_bb_centroid  =  b2_means_dict["x"]
    xfields_beambeam["cptoip_ip1_b2"].x_full_bb_centroid  =  b1_means_dict["x"]
    xfields_beambeam["cptoip_ip1_b1"].y_full_bb_centroid  =  b2_means_dict["y"]
    xfields_beambeam["cptoip_ip1_b2"].y_full_bb_centroid  =  b1_means_dict["y"] 
    src.add_new_value(time_elapsed_dict, "Update iptocp elements", measure_time.stop())

    # get sigmas at IP
    measure_time.start()
    sigma_matrix_ip_b1 = particles_b1.get_sigma_matrix(mask=slice_mask_b1) # boosted sigma matrix at IP
    sigma_matrix_ip_b2 = particles_b2.get_sigma_matrix(mask=slice_mask_b2) # boosted sigma matrix at IP
    src.add_new_value(time_elapsed_dict, "Get slice sigmas at IP", measure_time.stop())

    # store z coords at IP
    measure_time.start()
    particles_b1_z_storage = np.copy(particles_b1.zeta)
    particles_b2_z_storage = np.copy(particles_b2.zeta)
    src.add_new_value(time_elapsed_dict, "Save z coords", measure_time.stop())

    # transport CP using the z centroids, now all z coords are .5(z_c1-z_c2)
    measure_time.start()
    xfields_beambeam["iptocp_ip1_b1"].track(particles_b1)
    xfields_beambeam["iptocp_ip1_b2"].track(particles_b2)
    src.add_new_value(time_elapsed_dict, "Transport to CP", measure_time.stop())

    # get sigmas at CP
    measure_time.start()
    sigma_matrix_cp_b1 = particles_b1.get_sigma_matrix(mask=slice_mask_b1)
    sigma_matrix_cp_b2 = particles_b2.get_sigma_matrix(mask=slice_mask_b2)
    src.add_new_value(time_elapsed_dict, "Get slice sigmas at CP", measure_time.stop())

    # update beambeam with sigmas 
    measure_time.start()
    xfields_beambeam["strongstrong3d_ip1_b1"].update(n_macroparts_bb=beam_params["bunch_intensity"]/beam_params["n_macroparticles_b1"]*len(slice_b2_indices), 
                                              sigma_matrix_ip = sigma_matrix_ip_b2,
                                              sigma_matrix_cp = sigma_matrix_cp_b2, verbose_info=turn)
    xfields_beambeam["strongstrong3d_ip1_b2"].update(n_macroparts_bb=beam_params["bunch_intensity"]/beam_params["n_macroparticles_b2"]*len(slice_b1_indices),
                                  sigma_matrix_ip = sigma_matrix_ip_b1,
                                  sigma_matrix_cp = sigma_matrix_cp_b1, verbose_info=turn)
    src.add_new_value(time_elapsed_dict, "Update beambeam elements", measure_time.stop())

    # need to update individual z coords
    measure_time.start()
    xfields_beambeam["strongstrong3d_ip1_b1"].track(particles_b1)
    xfields_beambeam["strongstrong3d_ip1_b2"].track(particles_b2)
    src.add_new_value(time_elapsed_dict, "Beambeam kick", measure_time.stop())

    # transport back from CP to IP
    measure_time.start()
    xfields_beambeam["cptoip_ip1_b1"].track(particles_b1)
    xfields_beambeam["cptoip_ip1_b2"].track(particles_b2)
    src.add_new_value(time_elapsed_dict, "Transport back to IP", measure_time.stop())

    # update individual z coords
    measure_time.start()
    particles_b1.zeta = np.copy(particles_b1_z_storage)
    particles_b2.zeta = np.copy(particles_b2_z_storage)
    src.add_new_value(time_elapsed_dict, "Update z coords", measure_time.stop())

    # set back slice state
    measure_time.start()
    particles_b1.state[slice_mask_b1] = 1
    particles_b2.state[slice_mask_b2] = 1            
    src.add_new_value(time_elapsed_dict, "Set back slice states", measure_time.stop())

    return particles_b1, particles_b2
