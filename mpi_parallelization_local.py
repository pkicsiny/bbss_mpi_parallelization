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
from mpi4py.util import pkl5

"""
init mpi environment
"""
comm = MPI.COMM_WORLD
#comm = pkl5.Intracomm(MPI.COMM_WORLD)

rank = comm.Get_rank()
size = comm.Get_size()
print("rank: ", rank, "size:", size)


"""
29/10/21: save now the avg. coordinates, coordinates of 1000 macroparts + the test particles so that I can still plot the tune footprint. This way the file size is around 0.5GB per job.
"""

context = xo.ContextCpu(omp_num_threads=0)
xt.enable_pyheadtail_interface()

"""
beam settings
"""
beam_params = config.hl_lhc_params

beam_params["sigma_x"]     = np.sqrt(beam_params["physemit_x"]*beam_params["betastar_x"])
beam_params["sigma_px"]    = np.sqrt(beam_params["physemit_x"]/beam_params["betastar_x"])
beam_params["sigma_y"]     = np.sqrt(beam_params["physemit_y"]*beam_params["betastar_y"])
beam_params["sigma_py"]    = np.sqrt(beam_params["physemit_y"]/beam_params["betastar_y"])

"""
simulations settings
"""

sim_params = {"context": context,
              "n_turns": int(1),  # ~4k turns
              "n_slices": 2,
              "use_strongstrong": 1,
              "rank": -1,        
            }

assert size > sim_params["n_slices"], "Number of MPI processes ({}) should be larger than number of beam slices ({}).".format(size, sim_params["n_slices"])

beam_params["n_macroparticles_b1"] = int(10)
beam_params["n_macroparticles_b2"] = int(10)

print("Number of slices: {}".format(sim_params["n_slices"]))
print("Number of macroparts: {}".format(beam_params["n_macroparticles_b1"]))


print("Initializing beam elements and particles on all processes...")    
xtrack_arc = src.init_arcs(beam_params)
xfields_beambeam = src.init_beambeam(beam_params, sim_params)
xtrack_particles = src.init_particles(beam_params, sim_params, draw_random=True)
  
sim_params.pop('context', None)
 
particles_6dss_dict_b1 = copy.deepcopy(xtrack_particles["b1"].to_dict())
particles_6dss_dict_b2 = copy.deepcopy(xtrack_particles["b2"].to_dict())
particles_6dss_b1 = xt.Particles(_context=context, **particles_6dss_dict_b1)
particles_6dss_b2 = xt.Particles(_context=context, **particles_6dss_dict_b2)
    
s1_list = [list(range(max(0,k+1-sim_params["n_slices"]), min(k+1,sim_params["n_slices"]))) for k in range(2*sim_params["n_slices"]-1)]
s2_list = [list(reversed(l)) for l in s1_list]

n_timesteps = 2*sim_params["n_slices"]-1
n_processes = [1+min(i, n_timesteps-1-i) for i in range(n_timesteps)]
max_processes = sim_params["n_slices"] + 1
print("{} timesteps, {} processes, {} max processes".format(n_timesteps, n_processes, max_processes))
# do the tracking on the root process
if rank == 0:
    for turn in range(sim_params["n_turns"]):
        print("Turn {} on rank {}".format(turn, rank))
        """
        beambeam
        """
    
        # for beambeam, reverse x and px
        particles_6dss_b1.x *= -1
        particles_6dss_b1.px *= -1
        
        # boost full beam
        xfields_beambeam["boost_ip1_b1"].track(particles_6dss_b1)
        xfields_beambeam["boost_ip1_b2"].track(particles_6dss_b2) 
    
        # boosted full beam stats
        b1_means_dict, b1_sigmas_dict = src.get_stats(particles_6dss_b1)
        b2_means_dict, b2_sigmas_dict = src.get_stats(particles_6dss_b2)
    
        # reslice beams every turn bc content can change. for this particles need to be up to date so it cant go into worker
        slicer = UniformChargeSlicer(sim_params["n_slices"])
        slices_b1 = particles_6dss_b1.get_slices(slicer)
        slices_b2 = particles_6dss_b2.get_slices(slicer)
 
        # one timestep: 2*N_slices - 1 timesteps
        for t in range(n_timesteps):
            print("Turn {}, timestep {}: Spawning {} processes".format(turn, t, n_processes[t]))

            # length of this equals n_processes[t], each element of these to one process
            s1_list_t = s1_list[t]
            s2_list_t = s2_list[t]

            # send to worker processes, only use as many pros as many overlapping slice pairs
            for r in range(1, 1+n_processes[t]):

                # send mask (list of indices of one slice)
                slice_b1_indices = slices_b1.particle_indices_of_slice(s1_list_t[r-1])
                slice_b2_indices = slices_b2.particle_indices_of_slice(s2_list_t[r-1])
                slice_mask_b1 = slice_b1_indices[np.where(particles_6dss_b1.state[slice_b1_indices])]
                slice_mask_b2 = slice_b2_indices[np.where(particles_6dss_b2.state[slice_b2_indices])]

                msg_tag = 1000*t + r

                print("Turn {}, timestep {}: root send {} to rank {} with tag {}".format(turn, t, particles_6dss_b1.x, r, msg_tag)) 
                comm.send(particles_6dss_b1.x, dest=r, tag=msg_tag)
                comm.send(slice_mask_b1, dest=r, tag=10*msg_tag)
    
            for r in range(1, 1+n_processes[t]):
                msg_tag = 1000*t + r

                # receive updated coords from worker processes
                print("Turn {}, timestep {}: listening on root from rank {} for tag {}".format(turn, t, r, msg_tag))
                particles_6dss_b1.x = comm.recv(source=r, tag=msg_tag)
                slice_mask_b1 = comm.recv(source=r, tag=10*msg_tag)
                print("Turn {}, timestep {}: received {} on root from rank {} with tag {}".format(turn, t, particles_6dss_b1.x, r, msg_tag))
 
        # inverse boost full beam after interaction
        xfields_beambeam["boostinv_ip1_b1"].track(particles_6dss_b1)
        xfields_beambeam["boostinv_ip1_b2"].track(particles_6dss_b2)
     
        # for beambeam, reverse back x and px
        particles_6dss_b1.x *= -1
        particles_6dss_b1.px *= -1
    
        """
        arc
        """
        
        #track both bunches from IP1 to IP2
        xtrack_arc["section1_b1"].track(particles_6dss_b1)
        xtrack_arc["section1_b2"].track(particles_6dss_b2)
        xtrack_arc["section2_b1"].track(particles_6dss_b1)
        xtrack_arc["section2_b2"].track(particles_6dss_b2)
    
# worker processes receive slices and do the slice slice and send back the updated coords
#inputs: particles_6dss_b1, slices_b1, s1, xfields_beambeam, beam_params, sim_params
# beam_params: global, doesnt change
# sim_params: global, doesnt change
# xfields_beambeam: global, changes so I can send the new arguments each turn and update the element. Actually nothing is needed from outside except the slice mask.
# s1, s2: slice indices are parallel, elements from s1_list[t], s1_list global and doesnt change. local in root, have to be sent as s1_list_t[r]. s1, s2 needed to update iptocp
# slices_b1, slices_b2: local in root, have to send.
# Send mask of list of slice indices instead of s1 and slices_b1
# particles_6dss_b1, particles_6dss_b2: global but change is local in root. Need to send all coords to all processes for every update

elif rank < max_processes:
   
    for turn in range(sim_params["n_turns"]):
        for t in range(n_timesteps):
            for r in range(1, 1+n_processes[t]):
                if r == rank:
                    msg_tag = 1000*t + rank
         
                    # there are many sliceslice interactions assigned to one worker within one turn 
                    # start by receiving the slice coords from root process then do 1 slice slice interaction
                    print("Turn {}, timestep {}: listening on rank {} for tag {}".format(turn, t, rank, msg_tag))
                    particles_6dss_b1.x = comm.recv(source=0, tag=msg_tag)
                    slice_mask_b1 = comm.recv(source=0, tag=10*msg_tag)
                    print("Turn {}, timestep {}: received {} on rank {} with tag {}".format(turn, t, particles_6dss_b1.x, rank, msg_tag)) 
                    comm.send(particles_6dss_b1.x, dest=0, tag=msg_tag)
                    comm.send(slice_mask_b1, dest=0, tag=10*msg_tag)
                    print("Turn {}, timestep {}: sent {} to root from rank {} with tag {}".format(turn, t, particles_6dss_b1.x, rank, msg_tag))
    """
        # select alive slice particles
        slice_b1_indices = slices_b1.particle_indices_of_slice(s1)
        slice_b2_indices = slices_b2.particle_indices_of_slice(s2)
        
        slice_mask_b1 = slice_b1_indices[np.where(particles_6dss_b1.state[slice_b1_indices])]
        slice_mask_b2 = slice_b2_indices[np.where(particles_6dss_b2.state[slice_b2_indices])]
        
        # encode slice index into state attribute
        particles_6dss_b1.state[slice_mask_b1] = 1000 + s1
        particles_6dss_b2.state[slice_mask_b2] = 1000 + s2
        
        # update xsuite elements with current slice indices
        xfields_beambeam["iptocp_ip1_b1"].slice_id = s1
        xfields_beambeam["iptocp_ip1_b2"].slice_id = s2
        xfields_beambeam["cptoip_ip1_b1"].slice_id = s1
        xfields_beambeam["cptoip_ip1_b2"].slice_id = s2
        xfields_beambeam["strongstrong3d_ip1_b1"].slice_id = s1
        xfields_beambeam["strongstrong3d_ip1_b2"].slice_id = s2
        
        # Measure boosted slice properties at IP
        s1_means_dict, s1_sigmas_dict = src.get_stats(particles_6dss_b1, mask=slice_mask_b1)
        s2_means_dict, s2_sigmas_dict = src.get_stats(particles_6dss_b2, mask=slice_mask_b2)
        
        # for transforming the reference frame into w.r.t other slice's centroid at the IP
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
        
        # get sigmas at IP
        sigma_matrix_ip_b1 = particles_6dss_b1.get_sigma_matrix(mask=slice_mask_b1) # boosted sigma matrix at IP
        sigma_matrix_ip_b2 = particles_6dss_b2.get_sigma_matrix(mask=slice_mask_b2) # boosted sigma matrix at IP
        
        # store z coords at IP
        particles_b1_z_storage = np.copy(particles_6dss_b1.zeta)
        particles_b2_z_storage = np.copy(particles_6dss_b2.zeta)
        
        # transport CP using the z centroids, now all z coords are .5(z_c1-z_c2)
        xfields_beambeam["iptocp_ip1_b1"].track(particles_6dss_b1)
        xfields_beambeam["iptocp_ip1_b2"].track(particles_6dss_b2)
        
        # get sigmas at CP
        sigma_matrix_cp_b1 = particles_6dss_b1.get_sigma_matrix(mask=slice_mask_b1)
        sigma_matrix_cp_b2 = particles_6dss_b2.get_sigma_matrix(mask=slice_mask_b2)
        
        # update beambeam with sigmas 
        xfields_beambeam["strongstrong3d_ip1_b1"].update(n_macroparts_bb=beam_params["bunch_intensity"]/beam_params["n_macroparticles_b1"]*len(slice_b2_indices), 
                                                  sigma_matrix_ip = sigma_matrix_ip_b2,
                                                  sigma_matrix_cp = sigma_matrix_cp_b2, verbose_info=turn)
        xfields_beambeam["strongstrong3d_ip1_b2"].update(n_macroparts_bb=beam_params["bunch_intensity"]/beam_params["n_macroparticles_b2"]*len(slice_b1_indices),
                                      sigma_matrix_ip = sigma_matrix_ip_b1,
                                      sigma_matrix_cp = sigma_matrix_cp_b1, verbose_info=turn)
        
        # need to update individual z coords
        xfields_beambeam["strongstrong3d_ip1_b1"].track(particles_6dss_b1)
        xfields_beambeam["strongstrong3d_ip1_b2"].track(particles_6dss_b2)
        
        # transport back from CP to IP
        xfields_beambeam["cptoip_ip1_b1"].track(particles_6dss_b1)
        xfields_beambeam["cptoip_ip1_b2"].track(particles_6dss_b2)
        
        # update individual z coords
        particles_6dss_b1.zeta = np.copy(particles_b1_z_storage)
        particles_6dss_b2.zeta = np.copy(particles_b2_z_storage)
        
        # set back slice state
        particles_6dss_b1.state[slice_mask_b1] = 1
        particles_6dss_b2.state[slice_mask_b2] = 1            
    """
print("Rank {} finished".format(rank))
