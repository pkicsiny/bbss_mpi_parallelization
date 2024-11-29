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

beam_params["n_macroparticles_b1"] = int(1e3)
beam_params["n_macroparticles_b2"] = int(1e3)

beam_params["sigma_x"]     = np.sqrt(beam_params["physemit_x"]*beam_params["betastar_x"])
beam_params["sigma_px"]    = np.sqrt(beam_params["physemit_x"]/beam_params["betastar_x"])
beam_params["sigma_y"]     = np.sqrt(beam_params["physemit_y"]*beam_params["betastar_y"])
beam_params["sigma_py"]    = np.sqrt(beam_params["physemit_y"]/beam_params["betastar_y"])

"""
simulations settings
"""

sim_params = {"context": context,
              "n_turns": int(1000),  # ~4k turns
              "n_slices": int(8),
              "use_strongstrong": 1,
              "min_sigma_diff": 1,
           }

"""
dont touch from here
"""

assert size > sim_params["n_slices"], "Number of MPI processes ({}) should be larger than number of beam slices ({}).".format(size, sim_params["n_slices"])

s1_list = [list(range(max(0,k+1-sim_params["n_slices"]), min(k+1,sim_params["n_slices"]))) for k in range(2*sim_params["n_slices"]-1)]
s2_list = [list(reversed(l)) for l in s1_list]

n_timesteps = 2*sim_params["n_slices"]-1
n_processes = [1+min(i, n_timesteps-1-i) for i in range(n_timesteps)]
max_processes = sim_params["n_slices"] + 1

print("Number of slices: {}".format(sim_params["n_slices"]))
print("Number of macroparts: {}".format(beam_params["n_macroparticles_b1"]))
print("{} timesteps, {} processes, {} max processes".format(n_timesteps, n_processes, max_processes))

print("Initializing beam elements and particles on all processes...")    
xfields_beambeam = src.init_beambeam(beam_params, sim_params)

# buffer for worker processes
xtrack_particles = src.init_particles(beam_params, sim_params, draw_random=True)
  
particles_dict_b1 = copy.deepcopy(xtrack_particles["b1"].to_dict())
particles_dict_b2 = copy.deepcopy(xtrack_particles["b2"].to_dict())
particles_b1 = xt.Particles(_context=context, **particles_dict_b1)
particles_b2 = xt.Particles(_context=context, **particles_dict_b2)
 
# do the tracking on the root process
if rank == 0:
    xtrack_arc = src.init_arcs(beam_params)
   
    # profiler settings
    measure_time      = profiler.profiler()
    measure_time_tot  = profiler.profiler()
    measure_time_ss   = profiler.profiler()
    time_elapsed_dict = {}
    time_elapsed_ss   = {}
    time_elapsed_tot  = {}

    save_n_coords = int(1e3)
    coords = {
               "b1": {"x":          np.zeros((sim_params["n_turns"]+1, save_n_coords), dtype=float),
                      "px":         np.zeros((sim_params["n_turns"]+1, save_n_coords), dtype=float),
                      "y":          np.zeros((sim_params["n_turns"]+1, save_n_coords), dtype=float),
                      "py":         np.zeros((sim_params["n_turns"]+1, save_n_coords), dtype=float),
                      "z":          np.zeros((sim_params["n_turns"]+1, save_n_coords), dtype=float),
                      "delta":      np.zeros((sim_params["n_turns"]+1, save_n_coords), dtype=float),
                      "x_mean":     np.zeros((sim_params["n_turns"]+1), dtype=float),
                      "px_mean":    np.zeros((sim_params["n_turns"]+1), dtype=float),
                      "y_mean":     np.zeros((sim_params["n_turns"]+1), dtype=float),
                      "py_mean":    np.zeros((sim_params["n_turns"]+1), dtype=float),
                      "z_mean":     np.zeros((sim_params["n_turns"]+1), dtype=float),
                      "delta_mean": np.zeros((sim_params["n_turns"]+1), dtype=float)},              
               "b2": {"x":          np.zeros((sim_params["n_turns"]+1, save_n_coords), dtype=float),
                      "px":         np.zeros((sim_params["n_turns"]+1, save_n_coords), dtype=float),
                      "y":          np.zeros((sim_params["n_turns"]+1, save_n_coords), dtype=float),
                      "py":         np.zeros((sim_params["n_turns"]+1, save_n_coords), dtype=float),
                      "z":          np.zeros((sim_params["n_turns"]+1, save_n_coords), dtype=float),
                      "delta":      np.zeros((sim_params["n_turns"]+1, save_n_coords), dtype=float),
                      "x_mean":     np.zeros((sim_params["n_turns"]+1), dtype=float),
                      "px_mean":    np.zeros((sim_params["n_turns"]+1), dtype=float),
                      "y_mean":     np.zeros((sim_params["n_turns"]+1), dtype=float),
                      "py_mean":    np.zeros((sim_params["n_turns"]+1), dtype=float),
                      "z_mean":     np.zeros((sim_params["n_turns"]+1), dtype=float),
                      "delta_mean": np.zeros((sim_params["n_turns"]+1), dtype=float)},
              }

    measure_time_tot.start()
    for turn in range(sim_params["n_turns"]):
        print("Turn {} on rank {}".format(turn, rank))
        """
        beambeam
        """

        #Record positions for post-processing
        measure_time.start()
        src.record_coordinates(coords, particles_b1, particles_b2, turn_idx=turn)
        src.add_new_value(time_elapsed_dict, "Record coords", measure_time.stop())
 
        # for beambeam, reverse x and px
        measure_time.start()
        particles_b1.x *= -1
        particles_b1.px *= -1
        src.add_new_value(time_elapsed_dict, "Swap x and px sign", measure_time.stop())   

        # boost full beam
        measure_time.start()
        xfields_beambeam["boost_ip1_b1"].track(particles_b1)
        xfields_beambeam["boost_ip1_b2"].track(particles_b2) 
        src.add_new_value(time_elapsed_dict, "Boost", measure_time.stop())
    
        # boosted full beam stats
        measure_time.start()
        b1_means_dict, b1_sigmas_dict = src.get_stats(particles_b1)
        b2_means_dict, b2_sigmas_dict = src.get_stats(particles_b2)
        src.add_new_value(time_elapsed_dict, "Get beam moments", measure_time.stop())
    
        # reslice beams every turn bc content can change. for this particles need to be up to date so it cant go into worker
        measure_time.start()
        slicer = UniformChargeSlicer(sim_params["n_slices"])
        slices_b1 = particles_b1.get_slices(slicer)
        slices_b2 = particles_b2.get_slices(slicer)
        src.add_new_value(time_elapsed_dict, "Slice beams", measure_time.stop())

        if turn == 0:
            print("IP 1 Beam 1 macroparts per slice: {}".format(slices_b1.n_macroparticles_per_slice))
            print("IP 1 Beam 2 macroparts per slice: {}".format(slices_b2.n_macroparticles_per_slice))


        measure_time_ss.start()
        # one timestep: 2*N_slices - 1 timesteps
        for t in range(n_timesteps):
            #print("Turn {}, timestep {}: Using {} processes".format(turn, t, n_processes[t]))

            # length of this equals n_processes[t], each element of these to one process
            measure_time.start()
            s1_list_t = s1_list[t]
            s2_list_t = s2_list[t]
            src.add_new_value(time_elapsed_dict, "Get overlapping slice indices", measure_time.stop())

            # init slice mask list of lists on rank 0
            measure_time.start()
            slice_mask_b1 = []
            slice_mask_b2 = []
            src.add_new_value(time_elapsed_dict, "Init slice mask list", measure_time.stop())

            # send to worker processes, only use as many pros as many overlapping slice pairs
            for r in range(1, 1+n_processes[t]):
 
                measure_time.start()
                s1_idx = s1_list_t[r-1]
                s2_idx = s2_list_t[r-1]
                src.add_new_value(time_elapsed_dict, "Get slice pair indices for send", measure_time.stop())

                # send mask (list of indices of one slice), send s1 and s1_part_indices bc. slices_bx cannot be sent
                measure_time.start()
                s1_part_indices = slices_b1.particle_indices_of_slice(s1_idx)
                s2_part_indices = slices_b2.particle_indices_of_slice(s2_idx)
                src.add_new_value(time_elapsed_dict, "Get slice particle indices", measure_time.stop())

                measure_time.start()
                slice_mask_b1.append(s1_part_indices[np.where(particles_b1.state[s1_part_indices])])
                slice_mask_b2.append(s2_part_indices[np.where(particles_b2.state[s2_part_indices])])
                src.add_new_value(time_elapsed_dict, "Get slice particle mask", measure_time.stop())
 
                # get unique message tags
                measure_time.start()
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
                tag_slice_mask_b1   = int(41*1e7 + t*1e3 + r)
                tag_slice_mask_b2   = int(42*1e7 + t*1e3 + r)
                tag_b1_means_dict   = int(51*1e7 + t*1e3 + r)
                tag_b2_means_dict   = int(52*1e7 + t*1e3 + r)
                src.add_new_value(time_elapsed_dict, "Create msg tags for send", measure_time.stop())

                #print("Turn {}, timestep {}: root send {} to rank {} with tag {}".format(turn, t, particles_b1.x[0], r, tag_x_1)) 
                measure_time.start()
                comm.send(particles_b1.x,     dest=r, tag=tag_x_1            )
                src.add_new_value(time_elapsed_dict, "send particles_b1.x", measure_time.stop())
                measure_time.start()
                comm.send(particles_b1.px,    dest=r, tag=tag_px_1           )
                src.add_new_value(time_elapsed_dict, "send particles_b1.px", measure_time.stop())
                measure_time.start()
                comm.send(particles_b1.y,     dest=r, tag=tag_y_1            )
                src.add_new_value(time_elapsed_dict, "send particles_b1.y", measure_time.stop())
                measure_time.start()
                comm.send(particles_b1.py,    dest=r, tag=tag_py_1           )
                src.add_new_value(time_elapsed_dict, "send particles_b1.py", measure_time.stop())
                measure_time.start()
                comm.send(particles_b1.zeta,  dest=r, tag=tag_z_1            )
                src.add_new_value(time_elapsed_dict, "send particles_b1.zeta", measure_time.stop())
                measure_time.start()
                comm.send(particles_b1.delta, dest=r, tag=tag_delta_1        )
                src.add_new_value(time_elapsed_dict, "send particles_b1.delta", measure_time.stop())
                measure_time.start()
                comm.send(particles_b2.x,     dest=r, tag=tag_x_2            )
                src.add_new_value(time_elapsed_dict, "send particles_b2.x", measure_time.stop())
                measure_time.start()
                comm.send(particles_b2.px,    dest=r, tag=tag_px_2           )
                src.add_new_value(time_elapsed_dict, "send particles_b2.px", measure_time.stop())
                measure_time.start()
                comm.send(particles_b2.y,     dest=r, tag=tag_y_2            )
                src.add_new_value(time_elapsed_dict, "send particles_b2.y", measure_time.stop())
                measure_time.start()
                comm.send(particles_b2.py,    dest=r, tag=tag_py_2           )
                src.add_new_value(time_elapsed_dict, "send particles_b2.py", measure_time.stop())
                measure_time.start()
                comm.send(particles_b2.zeta,  dest=r, tag=tag_z_2            )
                src.add_new_value(time_elapsed_dict, "send particles_b2.zeta", measure_time.stop())
                measure_time.start()
                comm.send(particles_b2.delta, dest=r, tag=tag_delta_2        )
                src.add_new_value(time_elapsed_dict, "send particles_b2.delta", measure_time.stop())
                measure_time.start()
                comm.send(s1_idx,             dest=r, tag=tag_s1             )
                src.add_new_value(time_elapsed_dict, "send s1_idx", measure_time.stop())
                measure_time.start()
                comm.send(s2_idx,             dest=r, tag=tag_s2             )
                src.add_new_value(time_elapsed_dict, "send s2_idx", measure_time.stop())
                measure_time.start()
                comm.send(slice_mask_b1[-1],      dest=r, tag=tag_slice_mask_b1  )
                src.add_new_value(time_elapsed_dict, "send slice_mask_b1[-1]", measure_time.stop())
                measure_time.start()
                comm.send(slice_mask_b2[-1],      dest=r, tag=tag_slice_mask_b2  )
                src.add_new_value(time_elapsed_dict, "send slice_mask_b2[-1]", measure_time.stop())
                measure_time.start()
                comm.send(b1_means_dict,      dest=r, tag=tag_b1_means_dict  )
                src.add_new_value(time_elapsed_dict, "send b1_means_dict", measure_time.stop())
                measure_time.start()
                comm.send(b2_means_dict,      dest=r, tag=tag_b2_means_dict  )
                src.add_new_value(time_elapsed_dict, "send b2_means_dict", measure_time.stop())

            for r in range(1, 1+n_processes[t]):

                # get unique message tags
                measure_time.start() 
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
                tag_slice_mask_b1   = int(41*1e7 + t*1e3 + r)
                tag_slice_mask_b2   = int(42*1e7 + t*1e3 + r)
                tag_b1_means_dict   = int(51*1e7 + t*1e3 + r)
                tag_b2_means_dict   = int(52*1e7 + t*1e3 + r)
                src.add_new_value(time_elapsed_dict, "Create msg tags for recv", measure_time.stop())

                # receive updated coords from worker processes
                #print("Turn {}, timestep {}: listening on root from rank {} for tag {}".format(turn, t, r, tag_x_1))

                # slice mask b1 is always different for each rank
                measure_time.start()
                slice_mask_b1_r = slice_mask_b1[r-1]
                slice_mask_b2_r = slice_mask_b2[r-1]
                src.add_new_value(time_elapsed_dict, "Get slice pair indices for recv", measure_time.stop())

                measure_time.start()
                particles_b1.x[slice_mask_b1_r]     = comm.recv(source=r, tag=tag_x_1    )
                src.add_new_value(time_elapsed_dict, "recv particles_b1.x[slice_mask_b1_r]", measure_time.stop())
                measure_time.start()
                particles_b1.px[slice_mask_b1_r]    = comm.recv(source=r, tag=tag_px_1   )
                src.add_new_value(time_elapsed_dict, "recv particles_b1.px[slice_mask_b1_r]", measure_time.stop())
                measure_time.start()
                particles_b1.y[slice_mask_b1_r]     = comm.recv(source=r, tag=tag_y_1    )
                src.add_new_value(time_elapsed_dict, "recv particles_b1.y[slice_mask_b1_r]", measure_time.stop())
                measure_time.start()
                particles_b1.py[slice_mask_b1_r]    = comm.recv(source=r, tag=tag_py_1   )
                src.add_new_value(time_elapsed_dict, "recv particles_b1.py[slice_mask_b1_r]", measure_time.stop())
                measure_time.start()
                particles_b1.zeta[slice_mask_b1_r]  = comm.recv(source=r, tag=tag_z_1    )
                src.add_new_value(time_elapsed_dict, "recv particles_b1.zeta[slice_mask_b1_r]", measure_time.stop())
                measure_time.start()
                particles_b1.delta[slice_mask_b1_r] = comm.recv(source=r, tag=tag_delta_1)
                src.add_new_value(time_elapsed_dict, "recv particles_b1.delta[slice_mask_b1_r]", measure_time.stop())
                measure_time.start()
                particles_b2.x[slice_mask_b2_r]     = comm.recv(source=r, tag=tag_x_2    )
                src.add_new_value(time_elapsed_dict, "recv particles_b2.x[slice_mask_b2_r]", measure_time.stop())
                measure_time.start()
                particles_b2.px[slice_mask_b2_r]    = comm.recv(source=r, tag=tag_px_2   )
                src.add_new_value(time_elapsed_dict, "recv particles_b2.px[slice_mask_b2_r]", measure_time.stop())
                measure_time.start()
                particles_b2.y[slice_mask_b2_r]     = comm.recv(source=r, tag=tag_y_2    )
                src.add_new_value(time_elapsed_dict, "recv particles_b2.y[slice_mask_b2_r]", measure_time.stop())
                measure_time.start()
                particles_b2.py[slice_mask_b2_r]    = comm.recv(source=r, tag=tag_py_2   )
                src.add_new_value(time_elapsed_dict, "recv particles_b2.py[slice_mask_b2_r]", measure_time.stop())
                measure_time.start()
                particles_b2.zeta[slice_mask_b2_r]  = comm.recv(source=r, tag=tag_z_2    )
                src.add_new_value(time_elapsed_dict, "recv particles_b2.zeta[slice_mask_b2_r]", measure_time.stop())
                measure_time.start()
                particles_b2.delta[slice_mask_b2_r] = comm.recv(source=r, tag=tag_delta_2)
                src.add_new_value(time_elapsed_dict, "recv particles_b2.delta[slice_mask_b2_r]", measure_time.stop())
 
                #print("Turn {}, timestep {}: received {} on root from rank {} with tag {}".format(turn, t, particles_b1.x[0], r, tag_x_1))
        src.add_new_value(time_elapsed_ss, str(sim_params["n_slices"]), measure_time_ss.stop())

        # inverse boost full beam after interaction
        measure_time.start()
        xfields_beambeam["boostinv_ip1_b1"].track(particles_b1)
        xfields_beambeam["boostinv_ip1_b2"].track(particles_b2)
        src.add_new_value(time_elapsed_dict, "Inverse boost", measure_time.stop())
     
        # for beambeam, reverse back x and px
        measure_time.start()
        particles_b1.x *= -1
        particles_b1.px *= -1
        src.add_new_value(time_elapsed_dict, "Swap back x and px sign", measure_time.stop())

        """
        arc
        """
        
        #track both bunches from IP1 to IP2
        measure_time.start()
        xtrack_arc["section1_b1"].track(particles_b1)
        xtrack_arc["section1_b2"].track(particles_b2)
        xtrack_arc["section2_b1"].track(particles_b1)
        xtrack_arc["section2_b2"].track(particles_b2)
        src.add_new_value(time_elapsed_dict, "Track through arc", measure_time.stop())

    
    # After last turn
    measure_time.start()
    src.record_coordinates(coords, particles_b1, particles_b2)
    src.add_new_value(time_elapsed_dict, "Record coords", measure_time.stop())
 
    measure_time.start()
    coords_t = src.transpose_dict(coords)
    src.add_new_value(time_elapsed_dict, "Transpose coords", measure_time.stop()) 

    measure_time.start()
    out_dir = "/eos/home-p/pkicsiny/xsuite_simulations/mpi_parallelization"
    with open(os.path.join(out_dir, 'coords_dict_{}_slices_{}_macroparts_{}_turns.pickle'.format(sim_params["n_slices"], beam_params["n_macroparticles_b1"], sim_params["n_turns"])), 'wb') as handle:
        pickle.dump(coords_t, handle, protocol=pickle.HIGHEST_PROTOCOL)
    src.add_new_value(time_elapsed_dict, "Save output", measure_time.stop())
    print("Saved output dict.")

    pickle.dump(time_elapsed_dict, open(os.path.join(out_dir, "runtime_analysis_stepbystep_nslices_{}_nmacroparts_{}.pickle".format(n_slices, n_macroparts)), "wb"), protocol=pickle.HIGHEST_PROTOCOL) 
    pickle.dump(time_elapsed_ss, open(os.path.join(out_dir, "runtime_analysis_sliceslice_nslices_{}_nmacroparts_{}.pickle".format(n_slices, n_macroparts)), "wb"), protocol=pickle.HIGHEST_PROTOCOL) 
    pickle.dump(time_elapsed_tot, open(os.path.join(out_dir, "runtime_analysis_fullsim_nslices_{}_nmacroparts_{}.pickle".format(n_slices, n_macroparts)), "wb"), protocol=pickle.HIGHEST_PROTOCOL) 
    print("Saved profiler data.")

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
                    tag_slice_mask_b1   = int(41*1e7 + t*1e3 + r)
                    tag_slice_mask_b2   = int(42*1e7 + t*1e3 + r)
                    tag_b1_means_dict   = int(51*1e7 + t*1e3 + r)
                    tag_b2_means_dict   = int(52*1e7 + t*1e3 + r)

                    # there are many sliceslice interactions assigned to one worker within one turn 
                    # start by receiving the slice coords from root process then do 1 slice slice interaction
                    #print("Turn {}, timestep {}: listening on rank {} for tag {}".format(turn, t, rank, tag_x_1))

                    b1_x               = comm.recv(source=0, tag=tag_x_1            )
                    b1_px              = comm.recv(source=0, tag=tag_px_1           )
                    b1_y               = comm.recv(source=0, tag=tag_y_1            )
                    b1_py              = comm.recv(source=0, tag=tag_py_1           )
                    b1_zeta            = comm.recv(source=0, tag=tag_z_1            )
                    b1_delta           = comm.recv(source=0, tag=tag_delta_1        )
                    b2_x               = comm.recv(source=0, tag=tag_x_2            )
                    b2_px              = comm.recv(source=0, tag=tag_px_2           )
                    b2_y               = comm.recv(source=0, tag=tag_y_2            )
                    b2_py              = comm.recv(source=0, tag=tag_py_2           )
                    b2_zeta            = comm.recv(source=0, tag=tag_z_2            )
                    b2_delta           = comm.recv(source=0, tag=tag_delta_2        )
                    s1_idx             = comm.recv(source=0, tag=tag_s1             )
                    s2_idx             = comm.recv(source=0, tag=tag_s2             )
                    slice_mask_b1      = comm.recv(source=0, tag=tag_slice_mask_b1  )
                    slice_mask_b2      = comm.recv(source=0, tag=tag_slice_mask_b2  )
                    b1_means_dict      = comm.recv(source=0, tag=tag_b1_means_dict  )
                    b2_means_dict      = comm.recv(source=0, tag=tag_b2_means_dict  )

                    #print("Turn {}, timestep {}: received {} on rank {} with tag {}".format(turn, t, b1_x[0], rank, tag_x_1)) 

                    # build xtrack particles object from coordinate arrays. beam_params and sim_params are the same on every process and never change
                    particles_b1 = xt.Particles(
                        _context = sim_params["context"], 
                        q0       = beam_params["q_b1"],
                        p0c      = beam_params["p0c"],
                        x        = b1_x,
                        px       = b1_px,
                        y        = b1_y,
                        py       = b1_py,
                        zeta     = b1_zeta,
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
                        zeta     = b2_zeta,
                        delta    = b2_delta,
                        )

                    # do stuff
                   
                    # encode slice index into state attribute
                    particles_b1.state[slice_mask_b1] = 1000 + s1_idx
                    particles_b2.state[slice_mask_b2] = 1000 + s2_idx
                    
                    # update xsuite elements with current slice indices
                    xfields_beambeam["iptocp_ip1_b1"].slice_id = s1_idx
                    xfields_beambeam["iptocp_ip1_b2"].slice_id = s2_idx
                    xfields_beambeam["cptoip_ip1_b1"].slice_id = s1_idx
                    xfields_beambeam["cptoip_ip1_b2"].slice_id = s2_idx
                    xfields_beambeam["strongstrong3d_ip1_b1"].slice_id = s1_idx
                    xfields_beambeam["strongstrong3d_ip1_b2"].slice_id = s2_idx
                    
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
                    xfields_beambeam["strongstrong3d_ip1_b1"].update(n_macroparts_bb=beam_params["bunch_intensity"]/beam_params["n_macroparticles_b1"]*len(slice_mask_b2), 
                                                              sigma_matrix_ip = sigma_matrix_ip_b2,
                                                              sigma_matrix_cp = sigma_matrix_cp_b2, verbose_info=turn)
                    xfields_beambeam["strongstrong3d_ip1_b2"].update(n_macroparts_bb=beam_params["bunch_intensity"]/beam_params["n_macroparticles_b2"]*len(slice_mask_b1),
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

                    # send back modified coords
                    comm.send(particles_b1.x[slice_mask_b1],     dest=0, tag=tag_x_1    )
                    comm.send(particles_b1.px[slice_mask_b1],    dest=0, tag=tag_px_1   )
                    comm.send(particles_b1.y[slice_mask_b1],     dest=0, tag=tag_y_1    )
                    comm.send(particles_b1.py[slice_mask_b1],    dest=0, tag=tag_py_1   )
                    comm.send(particles_b1.zeta[slice_mask_b1],  dest=0, tag=tag_z_1    )
                    comm.send(particles_b1.delta[slice_mask_b1], dest=0, tag=tag_delta_1)
                    comm.send(particles_b2.x[slice_mask_b2],     dest=0, tag=tag_x_2    )
                    comm.send(particles_b2.px[slice_mask_b2],    dest=0, tag=tag_px_2   )
                    comm.send(particles_b2.y[slice_mask_b2],     dest=0, tag=tag_y_2    )
                    comm.send(particles_b2.py[slice_mask_b2],    dest=0, tag=tag_py_2   )
                    comm.send(particles_b2.zeta[slice_mask_b2],  dest=0, tag=tag_z_2    )
                    comm.send(particles_b2.delta[slice_mask_b2], dest=0, tag=tag_delta_2)
                    #print("Turn {}, timestep {}: sent {} to root from rank {} with tag {}".format(turn, t, particles_b1.x[0], rank, tag_x_1))
 

print("Rank {} finished".format(rank))
