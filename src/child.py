from mpi4py import MPI
import numpy as np

comm = MPI.Comm.Get_parent()
size = comm.Get_size()
rank = comm.Get_rank()

s1_t = 0
#s1_t = comm.scatter(s1_t, root=0)

slice_indices = np.zeros(2, dtype=np.int64) -1 
comm.Scatterv(None, slice_indices, root=0)

print("Interacting slices are {}-{} on rank {}".format(slice_indices[0], slice_indices[1], rank))
comm.Disconnect()

"""   
# select alive slice particles
measure_time.start()
slice_b1_indices = slices_b1.particle_indices_of_slice(s1)
slice_b2_indices = slices_b2.particle_indices_of_slice(s2)
src.add_new_value(time_elapsed_dict, "Get slice indices", measure_time.stop())

measure_time.start()    
slice_mask_b1 = slice_b1_indices[np.where(particles_6dss_b1.state[slice_b1_indices])]
slice_mask_b2 = slice_b2_indices[np.where(particles_6dss_b2.state[slice_b2_indices])]
src.add_new_value(time_elapsed_dict, "Get slice mask", measure_time.stop())

# encode slice index into state attribute
measure_time.start()
particles_6dss_b1.state[slice_mask_b1] = 1000 + s1
particles_6dss_b2.state[slice_mask_b2] = 1000 + s2
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
s1_means_dict, s1_sigmas_dict = src.get_stats(particles_6dss_b1, mask=slice_mask_b1)
s2_means_dict, s2_sigmas_dict = src.get_stats(particles_6dss_b2, mask=slice_mask_b2)
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
sigma_matrix_ip_b1 = particles_6dss_b1.get_sigma_matrix(mask=slice_mask_b1) # boosted sigma matrix at IP
sigma_matrix_ip_b2 = particles_6dss_b2.get_sigma_matrix(mask=slice_mask_b2) # boosted sigma matrix at IP
src.add_new_value(time_elapsed_dict, "Get slice sigmas at IP", measure_time.stop())

# store z coords at IP
measure_time.start()
particles_b1_z_storage = np.copy(particles_6dss_b1.zeta)
particles_b2_z_storage = np.copy(particles_6dss_b2.zeta)
src.add_new_value(time_elapsed_dict, "Save z coords", measure_time.stop())

# transport CP using the z centroids, now all z coords are .5(z_c1-z_c2)
measure_time.start()
xfields_beambeam["iptocp_ip1_b1"].track(particles_6dss_b1)
xfields_beambeam["iptocp_ip1_b2"].track(particles_6dss_b2)
src.add_new_value(time_elapsed_dict, "Transport to CP", measure_time.stop())

# get sigmas at CP
measure_time.start()
sigma_matrix_cp_b1 = particles_6dss_b1.get_sigma_matrix(mask=slice_mask_b1)
sigma_matrix_cp_b2 = particles_6dss_b2.get_sigma_matrix(mask=slice_mask_b2)
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
xfields_beambeam["strongstrong3d_ip1_b1"].track(particles_6dss_b1)
xfields_beambeam["strongstrong3d_ip1_b2"].track(particles_6dss_b2)
src.add_new_value(time_elapsed_dict, "Beambeam kick", measure_time.stop())

# transport back from CP to IP
measure_time.start()
xfields_beambeam["cptoip_ip1_b1"].track(particles_6dss_b1)
xfields_beambeam["cptoip_ip1_b2"].track(particles_6dss_b2)
src.add_new_value(time_elapsed_dict, "Transport back to IP", measure_time.stop())

# update individual z coords
measure_time.start()
particles_6dss_b1.zeta = np.copy(particles_b1_z_storage)
particles_6dss_b2.zeta = np.copy(particles_b2_z_storage)
src.add_new_value(time_elapsed_dict, "Update z coords", measure_time.stop())

# set back slice state
measure_time.start()
particles_6dss_b1.state[slice_mask_b1] = 1
particles_6dss_b2.state[slice_mask_b2] = 1            
src.add_new_value(time_elapsed_dict, "Set back slice states", measure_time.stop())

"""
