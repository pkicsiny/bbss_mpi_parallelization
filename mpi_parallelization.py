# -*- coding: utf-8 -*-

import argparse

import numpy as np
import pandas as pd
from scipy import stats
from scipy import constants as cst
from scipy import stats as st
import os
import time
import pickle
import copy
import re
import sys

sys.path.append('/afs/cern.ch/work/p/pkicsiny/private/xsuite_simulations/global_source_code')
import input_files.config as config
import src.src as src
import src.profiler as profiler

# xsuite
import xobjects as xo
import xtrack as xt
import xfields as xf
import xline
from PyHEADTAIL.trackers.transverse_tracking import TransverseSegmentMap
from PyHEADTAIL.particles.slicing import UniformChargeSlicer
from PyHEADTAIL.particles.slicing import UniformBinSlicer

context = xo.ContextCpu(omp_num_threads=0)
xt.enable_pyheadtail_interface()


"""
29/10/21: save now the avg. coordinates, coordinates of 1000 macroparts + the test particles so that I can still plot the tune footprint. This way the file size is around 0.5GB per job.
"""

"""
get command line args
"""

parser = argparse.ArgumentParser()
parser.add_argument("--scan_params")
parser.add_argument("--nThreads")
parser.add_argument("--jobId")
parser.add_argument("--outputDirectory")
args = parser.parse_args()  # all are string
print("the arguments: {}".format(args))
scan_params_list = eval(args.scan_params)  # list
jobid = int(args.jobId)  # int
out_dir = args.outputDirectory  # string
print("parameter scans: {}, jobid: {}, output dir: {}".format(scan_params_list, jobid, out_dir))

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

# full crossing angles
n_slices         = 8 #n_slices_vec[int(jobid / len(n_macroparts_vec))]  # the curve
n_macroparts     = int(1e3) #n_macroparts_vec[jobid % len(n_macroparts_vec)]  # the data point on the curve

sim_params = {"context": context,
              "n_turns": int(1000),  # ~4k turns
              "n_slices": int(n_slices),
              "use_strongstrong": 1,       
              "min_sigma_diff": 1, 
            }

beam_params["n_macroparticles_b1"] = int(n_macroparts)
beam_params["n_macroparticles_b2"] = int(n_macroparts)

print("Number of slices: {}".format(sim_params["n_slices"]))
print("Number of macroparts: {}".format(beam_params["n_macroparticles_b1"]))


print("Initializing beam elements and particles...")    
xtrack_arc = src.init_arcs(beam_params)
xfields_beambeam = src.init_beambeam(beam_params, sim_params)
xtrack_particles = src.init_particles(beam_params, sim_params, draw_random=False)
    
particles_dict_b1 = copy.deepcopy(xtrack_particles["b1"].to_dict())
particles_dict_b2 = copy.deepcopy(xtrack_particles["b2"].to_dict())
particles_b1 = xt.Particles(_context=context, **particles_dict_b1)
particles_b2 = xt.Particles(_context=context, **particles_dict_b2)
    
s1_list = [list(range(max(0,k+1-sim_params["n_slices"]), min(k+1,sim_params["n_slices"]))) for k in range(2*sim_params["n_slices"]-1)]
s2_list = [list(reversed(l)) for l in s1_list]

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
       
    if turn%10 == 0:
        print("Turn {}".format(turn)) 

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

    # slice beams
    measure_time.start()
    slicer = UniformChargeSlicer(sim_params["n_slices"])
    slices_b1 = particles_b1.get_slices(slicer)
    slices_b2 = particles_b2.get_slices(slicer)
    src.add_new_value(time_elapsed_dict, "Slice beams", measure_time.stop())

    if turn == 0:
        print("IP 1 Beam 1 macroparts per slice: {}".format(slices_b1.n_macroparticles_per_slice))
        print("IP 1 Beam 2 macroparts per slice: {}".format(slices_b2.n_macroparticles_per_slice))

    # loop over slice configurations
    measure_time_ss.start()
    for s in range(2*sim_params["n_slices"]-1):
        
            # interact slices one by one
            for s1, s2 in zip(s1_list[s], s2_list[s]):
                
                # select alive slice particles
                measure_time.start()
                slice_b1_indices = slices_b1.particle_indices_of_slice(s1)
                slice_b2_indices = slices_b2.particle_indices_of_slice(s2)
                src.add_new_value(time_elapsed_dict, "Get slice indices", measure_time.stop())
                
                measure_time.start()    
                slice_mask_b1 = slice_b1_indices[np.where(particles_b1.state[slice_b1_indices])]
                slice_mask_b2 = slice_b2_indices[np.where(particles_b2.state[slice_b2_indices])]
                src.add_new_value(time_elapsed_dict, "Get slice mask", measure_time.stop())

                # encode slice index into state attribute
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

              # end slice pair
        # end timestamp
    src.add_new_value(time_elapsed_ss, str(n_slices), measure_time_ss.stop())
    
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

src.add_new_value(time_elapsed_tot, str(n_slices), measure_time_tot.stop())

measure_time.start()
with open(os.path.join(out_dir, 'coords_dict_{}_slices_{}_macroparts_{}_turns.pickle'.format(sim_params["n_slices"], beam_params["n_macroparticles_b1"], sim_params["n_turns"])), 'wb') as handle:
    pickle.dump(coords_t, handle, protocol=pickle.HIGHEST_PROTOCOL)
src.add_new_value(time_elapsed_dict, "Save output", measure_time.stop())
print("Saved output dict.")


pickle.dump(time_elapsed_dict, open(os.path.join(out_dir, "runtime_analysis_stepbystep_nslices_{}_nmacroparts_{}.pickle".format(n_slices, n_macroparts)), "wb"), protocol=pickle.HIGHEST_PROTOCOL) 
pickle.dump(time_elapsed_ss, open(os.path.join(out_dir, "runtime_analysis_sliceslice_nslices_{}_nmacroparts_{}.pickle".format(n_slices, n_macroparts)), "wb"), protocol=pickle.HIGHEST_PROTOCOL) 
pickle.dump(time_elapsed_tot, open(os.path.join(out_dir, "runtime_analysis_fullsim_nslices_{}_nmacroparts_{}.pickle".format(n_slices, n_macroparts)), "wb"), protocol=pickle.HIGHEST_PROTOCOL) 
print("Saved outputs to: {}".format(out_dir))
