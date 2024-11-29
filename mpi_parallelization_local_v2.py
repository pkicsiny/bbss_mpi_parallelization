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
#from mpi4py.util import pkl5

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
              "n_slices": 3,
              "use_strongstrong": 1,
              "rank": -1,        
            }

assert size > sim_params["n_slices"], "Number of MPI processes ({}) should be larger than number of beam slices ({}).".format(size, sim_params["n_slices"])

beam_params["n_macroparticles_b1"] = int(3)
beam_params["n_macroparticles_b2"] = int(3)

print("Number of slices: {}".format(sim_params["n_slices"]))
print("Number of macroparts: {}".format(beam_params["n_macroparticles_b1"]))


print("Initializing beam elements and particles on all processes...")    
xtrack_arc = src.init_arcs(beam_params)
xfields_beambeam = src.init_beambeam(beam_params, sim_params)
xtrack_particles = src.init_particles(beam_params, sim_params, draw_random=True)
  
particles_dict_b1 = copy.deepcopy(xtrack_particles["b1"].to_dict())
particles_dict_b2 = copy.deepcopy(xtrack_particles["b2"].to_dict())
particles_b1 = xt.Particles(_context=context, **particles_dict_b1)
particles_b2 = xt.Particles(_context=context, **particles_dict_b2)
    
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
        particles_b1.x *= -1
        particles_b1.px *= -1
        
        # boost full beam
        xfields_beambeam["boost_ip1_b1"].track(particles_b1)
        xfields_beambeam["boost_ip1_b2"].track(particles_b2) 
    
        # boosted full beam stats
        b1_means_dict, b1_sigmas_dict = src.get_stats(particles_b1)
        b2_means_dict, b2_sigmas_dict = src.get_stats(particles_b2)
    
        # reslice beams every turn bc content can change. for this particles need to be up to date so it cant go into worker
        slicer = UniformChargeSlicer(sim_params["n_slices"])
        slices_b1 = particles_b1.get_slices(slicer)
        slices_b2 = particles_b2.get_slices(slicer)
 
        # one timestep: 2*N_slices - 1 timesteps
        for t in range(n_timesteps):
            print("Turn {}, timestep {}: Using {} processes".format(turn, t, n_processes[t]))

            # length of this equals n_processes[t], each element of these to one process
            s1_list_t = s1_list[t]
            s2_list_t = s2_list[t]

            # send to worker processes, only use as many pros as many overlapping slice pairs
            for r in range(1, 1+n_processes[t]):
 
                s1_idx = s1_list_t[r-1]
                s2_idx = s2_list_t[r-1]

                # send mask (list of indices of one slice), send s1 and slice_b1_indices bc. slices_bx cannot be sent
                s1_part_indices = slices_b1.particle_indices_of_slice(s1_idx)
                s2_part_indices = slices_b2.particle_indices_of_slice(s2_idx)

                # get unique message tags
                tag_x_1             = int(11*1e7 + t*1e3 + r)
                tag_px_1            = int(12*1e7 + t*1e3 + r)
                tag_y_1             = int(13*1e7 + t*1e3 + r)
                tag_py_1            = int(14*1e7 + t*1e3 + r)
                tag_z_1             = int(15*1e7 + t*1e3 + r)
                tag_delta_1         = int(16*1e7 + t*1e3 + r)
                tag_x_2             = int(21*1e7 + t*1e3 + r)
                tag_px_2            = int(22*1e7 + t*1e3 + r)
                tag_y_2             = int(23*1e7 + t*1e3 + r)
                tag_py_2            = int(24*1e7 + t*1e3 + r)
                tag_z_2             = int(25*1e7 + t*1e3 + r)
                tag_delta_2         = int(26*1e7 + t*1e3 + r)
                tag_s1              = int(31*1e7 + t*1e3 + r)
                tag_s2              = int(32*1e7 + t*1e3 + r)
                tag_s1_part_indices = int(41*1e7 + t*1e3 + r)
                tag_s2_part_indices = int(42*1e7 + t*1e3 + r)

                print("Turn {}, timestep {}: root send {} to rank {} with tag {}".format(turn, t, particles_b1.x, r, tag_x_1)) 
                comm.send(particles_b1.x,     dest=r, tag=tag_x_1            )
                comm.send(particles_b1.px,    dest=r, tag=tag_px_1           )
                comm.send(particles_b1.y,     dest=r, tag=tag_y_1            )
                comm.send(particles_b1.py,    dest=r, tag=tag_py_1           )
                comm.send(particles_b1.z,     dest=r, tag=tag_z_1            )
                comm.send(particles_b1.delta, dest=r, tag=tag_delta_1        )
                comm.send(particles_b2.x,     dest=r, tag=tag_x_2            )
                comm.send(particles_b2.px,    dest=r, tag=tag_px_2           )
                comm.send(particles_b2.y,     dest=r, tag=tag_y_2            )
                comm.send(particles_b2.py,    dest=r, tag=tag_py_2           )
                comm.send(particles_b2.z,     dest=r, tag=tag_z_2            )
                comm.send(particles_b2.delta, dest=r, tag=tag_delta_2        )
                comm.send(s1_idx,             dest=r, tag=tag_s1             )
                comm.send(s2_idx,             dest=r, tag=tag_s2             )
                comm.send(s1_part_indices,    dest=r, tag=tag_s1_part_indices)
                comm.send(s2_part_indices,    dest=r, tag=tag_s2_part_indices)

            for r in range(1, 1+n_processes[t]):

                # get unique message tags
                tag_x_1             = int(11*1e7 + t*1e3 + r)
                tag_px_1            = int(12*1e7 + t*1e3 + r)
                tag_y_1             = int(13*1e7 + t*1e3 + r)
                tag_py_1            = int(14*1e7 + t*1e3 + r)
                tag_z_1             = int(15*1e7 + t*1e3 + r)
                tag_delta_1         = int(16*1e7 + t*1e3 + r)
                tag_x_2             = int(21*1e7 + t*1e3 + r)
                tag_px_2            = int(22*1e7 + t*1e3 + r)
                tag_y_2             = int(23*1e7 + t*1e3 + r)
                tag_py_2            = int(24*1e7 + t*1e3 + r)
                tag_z_2             = int(25*1e7 + t*1e3 + r)
                tag_delta_2         = int(26*1e7 + t*1e3 + r)
                tag_s1              = int(31*1e7 + t*1e3 + r)
                tag_s2              = int(32*1e7 + t*1e3 + r)
                tag_s1_part_indices = int(41*1e7 + t*1e3 + r)
                tag_s2_part_indices = int(42*1e7 + t*1e3 + r)

                # receive updated coords from worker processes
                print("Turn {}, timestep {}: listening on root from rank {} for tag {}".format(turn, t, r, tag_x_1))

                particles_b1.x     = comm.recv(source=r, tag=tag_x_1    )
                particles_b1.px    = comm.recv(source=r, tag=tag_px_1   )
                particles_b1.y     = comm.recv(source=r, tag=tag_y_1    )
                particles_b1.py    = comm.recv(source=r, tag=tag_py_1   )
                particles_b1.z     = comm.recv(source=r, tag=tag_z_1    )
                particles_b1.delta = comm.recv(source=r, tag=tag_delta_1)
                particles_b2.x     = comm.recv(source=r, tag=tag_x_2    )
                particles_b2.px    = comm.recv(source=r, tag=tag_px_2   )
                particles_b2.y     = comm.recv(source=r, tag=tag_y_2    )
                particles_b2.py    = comm.recv(source=r, tag=tag_py_2   )
                particles_b2.z     = comm.recv(source=r, tag=tag_z_2    )
                particles_b2.delta = comm.recv(source=r, tag=tag_delta_2)

                print("Turn {}, timestep {}: received {} on root from rank {} with tag {}".format(turn, t, particles_b1.x, r, tag_x_1))
 
        # inverse boost full beam after interaction
        xfields_beambeam["boostinv_ip1_b1"].track(particles_b1)
        xfields_beambeam["boostinv_ip1_b2"].track(particles_b2)
     
        # for beambeam, reverse back x and px
        particles_b1.x *= -1
        particles_b1.px *= -1
    
        """
        arc
        """
        
        #track both bunches from IP1 to IP2
        xtrack_arc["section1_b1"].track(particles_b1)
        xtrack_arc["section1_b2"].track(particles_b2)
        xtrack_arc["section2_b1"].track(particles_b1)
        xtrack_arc["section2_b2"].track(particles_b2)
    
# worker processes receive slices and do the slice slice and send back the updated coords
#inputs: particles_b1, slices_b1, s1, xfields_beambeam, beam_params, sim_params
# beam_params: global, doesnt change
# sim_params: global, doesnt change
# xfields_beambeam: global, changes so I can send the new arguments each turn and update the element. Actually nothing is needed from outside except the slice mask.
# s1, s2: slice indices are parallel, elements from s1_list[t], s1_list global and doesnt change. local in root, have to be sent as s1_list_t[r]. s1, s2 needed to update iptocp
# slices_b1, slices_b2: local in root, have to send.
# Send mask of list of slice indices instead of s1 and slices_b1
# particles_b1, particles_b2: global but change is local in root. Need to send all coords to all processes for every update

elif rank < max_processes:
   
    for turn in range(sim_params["n_turns"]):
        for t in range(n_timesteps):
            for r in range(1, 1+n_processes[t]):
                if r == rank:

                    # get unique message tags
                    tag_x_1             = int(11*1e7 + t*1e3 + r)
                    tag_px_1            = int(12*1e7 + t*1e3 + r)
                    tag_y_1             = int(13*1e7 + t*1e3 + r)
                    tag_py_1            = int(14*1e7 + t*1e3 + r)
                    tag_z_1             = int(15*1e7 + t*1e3 + r)
                    tag_delta_1         = int(16*1e7 + t*1e3 + r)
                    tag_x_2             = int(21*1e7 + t*1e3 + r)
                    tag_px_2            = int(22*1e7 + t*1e3 + r)
                    tag_y_2             = int(23*1e7 + t*1e3 + r)
                    tag_py_2            = int(24*1e7 + t*1e3 + r)
                    tag_z_2             = int(25*1e7 + t*1e3 + r)
                    tag_delta_2         = int(26*1e7 + t*1e3 + r)
                    tag_s1              = int(31*1e7 + t*1e3 + r)
                    tag_s2              = int(32*1e7 + t*1e3 + r)
                    tag_s1_part_indices = int(41*1e7 + t*1e3 + r)
                    tag_s2_part_indices = int(42*1e7 + t*1e3 + r)

        
                    # there are many sliceslice interactions assigned to one worker within one turn 
                    # start by receiving the slice coords from root process then do 1 slice slice interaction
                    print("Turn {}, timestep {}: listening on rank {} for tag {}".format(turn, t, rank, tag_x_1))

                    b1_x               = comm.recv(source=0, tag=tag_x_1            )
                    b1_px              = comm.recv(source=0, tag=tag_px_1           )
                    b1_y               = comm.recv(source=0, tag=tag_y_1            )
                    b1_py              = comm.recv(source=0, tag=tag_py_1           )
                    b1_z               = comm.recv(source=0, tag=tag_z_1            )
                    b1_delta           = comm.recv(source=0, tag=tag_delta_1        )
                    b2_x               = comm.recv(source=0, tag=tag_x_2            )
                    b2_px              = comm.recv(source=0, tag=tag_px_2           )
                    b2_y               = comm.recv(source=0, tag=tag_y_2            )
                    b2_py              = comm.recv(source=0, tag=tag_py_2           )
                    b2_z               = comm.recv(source=0, tag=tag_z_2            )
                    b2_delta           = comm.recv(source=0, tag=tag_delta_2        )
                    s1_idx             = comm.recv(source=0, tag=tag_s1             )
                    s2_idx             = comm.recv(source=0, tag=tag_s2             )
                    s1_part_indices    = comm.recv(source=0, tag=tag_s1_part_indices)
                    s2_part_indices    = comm.recv(source=0, tag=tag_s2_part_indices)

                    print("Turn {}, timestep {}: received {} on rank {} with tag {}".format(turn, t, b1_x, rank, tag_x_1)) 

                    # build xtrack particles object from coordinate arrays. beam_params and sim_params are the same on every process and never change
                    particles_b1 = xt.Particles(
                        _context = sim_params["context"], 
                        q0       = beam_params["q_b1"],
                        p0c      = beam_params["p0c"],
                        x        = b1_x,
                        px       = b1_px,
                        y        = b1_y,
                        py       = b1_py,
                        zeta     = b1_z,
                        delta    = b1_delta,
                        )

                    particles_b2 = xt.Particles(
                        _context = sim_params["context"], 
                        q0       = beam_params["q_b2"],
                        p0c      = beam_params["p0c"],
                        x        = b2_x,
                        px       = b2_px,
                        y        = b2_y,
                        py       = b2_py,
                        zeta     = b2_z,
                        delta    = b2_delta,
                        )


                    # do stuff
                    particles_b1.x *= -1

                    # send back modified coords
                    comm.send(particles_b1.x,     dest=0, tag=tag_x_1    )
                    comm.send(particles_b1.px,    dest=0, tag=tag_px_1   )
                    comm.send(particles_b1.y,     dest=0, tag=tag_y_1    )
                    comm.send(particles_b1.py,    dest=0, tag=tag_py_1   )
                    comm.send(particles_b1.z,     dest=0, tag=tag_z_1    )
                    comm.send(particles_b1.delta, dest=0, tag=tag_delta_1)
                    comm.send(particles_b2.x,     dest=0, tag=tag_x_2    )
                    comm.send(particles_b2.px,    dest=0, tag=tag_px_2   )
                    comm.send(particles_b2.y,     dest=0, tag=tag_y_2    )
                    comm.send(particles_b2.py,    dest=0, tag=tag_py_2   )
                    comm.send(particles_b2.z,     dest=0, tag=tag_z_2    )
                    comm.send(particles_b2.delta, dest=0, tag=tag_delta_2)
                    print("Turn {}, timestep {}: sent {} to root from rank {} with tag {}".format(turn, t, particles_b1.x, rank, tag_x_1))
    """
        slice_mask_b1 = slice_b1_indices[np.where(particles_b1.state[slice_b1_indices])]
        slice_mask_b2 = slice_b2_indices[np.where(particles_b2.state[slice_b2_indices])]
        
        # encode slice index into state attribute
        particles_b1.state[slice_mask_b1] = 1000 + s1
        particles_b2.state[slice_mask_b2] = 1000 + s2
        
        # update xsuite elements with current slice indices
        xfields_beambeam["iptocp_ip1_b1"].slice_id = s1
        xfields_beambeam["iptocp_ip1_b2"].slice_id = s2
        xfields_beambeam["cptoip_ip1_b1"].slice_id = s1
        xfields_beambeam["cptoip_ip1_b2"].slice_id = s2
        xfields_beambeam["strongstrong3d_ip1_b1"].slice_id = s1
        xfields_beambeam["strongstrong3d_ip1_b2"].slice_id = s2
        
        # Measure boosted slice properties at IP
        s1_means_dict, s1_sigmas_dict = src.get_stats(particles_b1, mask=slice_mask_b1)
        s2_means_dict, s2_sigmas_dict = src.get_stats(particles_b2, mask=slice_mask_b2)
        
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
        sigma_matrix_ip_b1 = particles_b1.get_sigma_matrix(mask=slice_mask_b1) # boosted sigma matrix at IP
        sigma_matrix_ip_b2 = particles_b2.get_sigma_matrix(mask=slice_mask_b2) # boosted sigma matrix at IP
        
        # store z coords at IP
        particles_b1_z_storage = np.copy(particles_b1.zeta)
        particles_b2_z_storage = np.copy(particles_b2.zeta)
        
        # transport CP using the z centroids, now all z coords are .5(z_c1-z_c2)
        xfields_beambeam["iptocp_ip1_b1"].track(particles_b1)
        xfields_beambeam["iptocp_ip1_b2"].track(particles_b2)
        
        # get sigmas at CP
        sigma_matrix_cp_b1 = particles_b1.get_sigma_matrix(mask=slice_mask_b1)
        sigma_matrix_cp_b2 = particles_b2.get_sigma_matrix(mask=slice_mask_b2)
        
        # update beambeam with sigmas 
        xfields_beambeam["strongstrong3d_ip1_b1"].update(n_macroparts_bb=beam_params["bunch_intensity"]/beam_params["n_macroparticles_b1"]*len(slice_b2_indices), 
                                                  sigma_matrix_ip = sigma_matrix_ip_b2,
                                                  sigma_matrix_cp = sigma_matrix_cp_b2, verbose_info=turn)
        xfields_beambeam["strongstrong3d_ip1_b2"].update(n_macroparts_bb=beam_params["bunch_intensity"]/beam_params["n_macroparticles_b2"]*len(slice_b1_indices),
                                      sigma_matrix_ip = sigma_matrix_ip_b1,
                                      sigma_matrix_cp = sigma_matrix_cp_b1, verbose_info=turn)
        
        # need to update individual z coords
        xfields_beambeam["strongstrong3d_ip1_b1"].track(particles_b1)
        xfields_beambeam["strongstrong3d_ip1_b2"].track(particles_b2)
        
        # transport back from CP to IP
        xfields_beambeam["cptoip_ip1_b1"].track(particles_b1)
        xfields_beambeam["cptoip_ip1_b2"].track(particles_b2)
        
        # update individual z coords
        particles_b1.zeta = np.copy(particles_b1_z_storage)
        particles_b2.zeta = np.copy(particles_b2_z_storage)
        
        # set back slice state
        particles_b1.state[slice_mask_b1] = 1
        particles_b2.state[slice_mask_b2] = 1            
    """
print("Rank {} finished".format(rank))
