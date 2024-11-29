# -*- coding: utf-8 -*-

import numpy as np
import os
import xtrack as xt
import xfields as xf

def add_new_value(elements_dict, key, value):
    try:
        elements_dict[key].append(value)
    except:
        elements_dict[key] = [value]


def init_arcs(beam_params, n_beams=2, n_arcs=2):
    """
    initializes the arc elements
    """
    xtrack_arc = {}
    for b in range(n_beams):
        for s in range(n_arcs):
            xtrack_arc["section{}_b{}".format(s+1, b+1)] = xt.LinearTransferMatrix(
                alpha_x_0            = 0,
                alpha_x_1            = 0,
                alpha_y_0            = 0,
                alpha_y_1            = 0,
                beta_x_0             = beam_params["betastar_x"],
                beta_x_1             = beam_params["betastar_x"],
                beta_y_0             = beam_params["betastar_y"],
                beta_y_1             = beam_params["betastar_y"],
                disp_x_0             = 0,
                disp_x_1             = 0,
                disp_y_0             = 0,
                disp_y_1             = 0,
                Q_x                  = beam_params["Qx"]/n_arcs,
                Q_y                  = beam_params["Qy"]/n_arcs,
                Q_s                  = -beam_params["Qz"]/n_arcs,
                beta_s               = beam_params["betastar_z"],
                energy_ref_increment = 0,
                energy_increment     = 0,
                )
            
    return xtrack_arc


def init_particles(beam_params, sim_params, n_beams=2, draw_random=False, save_random=False, input_path="input_files"):
    """
    initializes the particle ensembles
    """   
    random_numbers = {}
    for b in range(n_beams):
        if not draw_random:
            random_numbers["x_b{}".format(b+1)]     = np.loadtxt(os.path.join(input_path, "random_x_b{}.txt".format(b+1)    ))
            random_numbers["px_b{}".format(b+1)]    = np.loadtxt(os.path.join(input_path, "random_px_b{}.txt".format(b+1)   ))
            random_numbers["y_b{}".format(b+1)]     = np.loadtxt(os.path.join(input_path, "random_y_b{}.txt".format(b+1)    ))
            random_numbers["py_b{}".format(b+1)]    = np.loadtxt(os.path.join(input_path, "random_py_b{}.txt".format(b+1)   ))
            random_numbers["z_b{}".format(b+1)]     = np.loadtxt(os.path.join(input_path, "random_z_b{}.txt".format(b+1)    ))
            random_numbers["delta_b{}".format(b+1)] = np.loadtxt(os.path.join(input_path, "random_delta_b{}.txt".format(b+1)))
             
            if save_random:
                np.savetxt(os.path.join(input_path, "random_x_b{}.txt".format(b+1)    ), random_numbers["x_b{}".format(b+1)]    )
                np.savetxt(os.path.join(input_path, "random_px_b{}.txt".format(b+1)   ), random_numbers["px_b{}".format(b+1)]   )
                np.savetxt(os.path.join(input_path, "random_y_b{}.txt".format(b+1)    ), random_numbers["y_b{}".format(b+1)]    )
                np.savetxt(os.path.join(input_path, "random_py_b{}.txt".format(b+1)   ), random_numbers["py_b{}".format(b+1)]   ) 
                np.savetxt(os.path.join(input_path, "random_z_b{}.txt".format(b+1)    ), random_numbers["z_b{}".format(b+1)]    )
                np.savetxt(os.path.join(input_path, "random_delta_b{}.txt".format(b+1)), random_numbers["delta_b{}".format(b+1)])
        else:                                                         
            random_numbers["x_b{}".format(b+1)]     = np.random.randn(beam_params["n_macroparticles_b{}".format(b+1)])
            random_numbers["px_b{}".format(b+1)]    = np.random.randn(beam_params["n_macroparticles_b{}".format(b+1)])
            random_numbers["y_b{}".format(b+1)]     = np.random.randn(beam_params["n_macroparticles_b{}".format(b+1)])
            random_numbers["py_b{}".format(b+1)]    = np.random.randn(beam_params["n_macroparticles_b{}".format(b+1)])
            random_numbers["z_b{}".format(b+1)]     = np.random.randn(beam_params["n_macroparticles_b{}".format(b+1)])
            random_numbers["delta_b{}".format(b+1)] = np.random.randn(beam_params["n_macroparticles_b{}".format(b+1)])
    
    
    xtrack_particles = {}
    for b in range(n_beams):
        xtrack_particles["b{}".format(b+1)] = xt.Particles(
            _context = sim_params["context"], 
            q0       = beam_params["q_b{}".format(b+1)],
            p0c      = beam_params["p0c"],
            x        = np.sqrt(beam_params["physemit_x"]*beam_params["betastar_x"])*random_numbers["x_b{}".format(b+1)] ,
            px       = np.sqrt(beam_params["physemit_x"]/beam_params["betastar_x"])*random_numbers["px_b{}".format(b+1)],
            y        = np.sqrt(beam_params["physemit_y"]*beam_params["betastar_y"])*random_numbers["y_b{}".format(b+1)] ,
            py       = np.sqrt(beam_params["physemit_y"]/beam_params["betastar_y"])*random_numbers["py_b{}".format(b+1)],
            zeta     = beam_params["sigma_z"]*random_numbers["z_b{}".format(b+1)],
            delta    = beam_params["sigma_delta"]*random_numbers["delta_b{}".format(b+1)],
            )
    
    return xtrack_particles


def init_beambeam(beam_params, sim_params, n_beams=2, n_ips=1):
    """
    initializes the beambeam elements
    """
    xfields_beambeam = {}
    for b in range(n_beams):
        for i in range(n_ips):
            xfields_beambeam["boost_ip{}_b{}".format(i+1, b+1)] = xf.Boost3D(
                _context         = sim_params["context"],
                alpha            = beam_params["alpha"],
                phi              = beam_params["phi"],
                use_strongstrong = sim_params["use_strongstrong"],
                )

            xfields_beambeam["boostinv_ip{}_b{}".format(i+1, b+1)] = xf.BoostInv3D(
                _context         = sim_params["context"],
                alpha            = beam_params["alpha"],
                phi              = beam_params["phi"],
                use_strongstrong = sim_params["use_strongstrong"],
                )
        
            # these are sliced                        
            #xfields_beambeam["changeref_ip{}_b{}".format(i+1, b+1)] = xf.ChangeReference(
            #    is_sliced = int(bool(sim_params["n_slices"]-1)),
            #    )

            xfields_beambeam["iptocp_ip{}_b{}".format(i+1, b+1)] = xf.IPToCP3D(
                is_sliced = int(bool(sim_params["n_slices"]-1)),
                use_strongstrong = sim_params["use_strongstrong"],
                )

            xfields_beambeam["cptoip_ip{}_b{}".format(i+1, b+1)] = xf.CPToIP3D(
                is_sliced = int(bool(sim_params["n_slices"]-1)),
                use_strongstrong = sim_params["use_strongstrong"],
                )

            xfields_beambeam["strongstrong3d_ip{}_b{}".format(i+1, b+1)] = xf.StrongStrong3D(
                n_macroparts_bb    = beam_params["bunch_intensity"]/beam_params["n_macroparticles_b{}".format(b+1)], # intensity of beambeam kick
                q0_bb              = beam_params["q_b{}".format(n_beams-b)],
                min_sigma_diff     = sim_params["min_sigma_diff"],
                threshold_singular = 1e-28,  # must be larger than 0 but small
                is_sliced          = int(bool(sim_params["n_slices"]-1)),
                )
        
    return xfields_beambeam


def add_test_particle(beam_params, sim_params, xtrack_particles, x=0, px=0, y=0, py=0, z=0, delta=0, beam=1):

    x        = np.sqrt(beam_params["physemit_x"]*beam_params["betastar_x"])*x ,
    px       = np.sqrt(beam_params["physemit_x"]/beam_params["betastar_x"])*px,
    y        = np.sqrt(beam_params["physemit_y"]*beam_params["betastar_y"])*y ,
    py       = np.sqrt(beam_params["physemit_y"]/beam_params["betastar_y"])*py,
    zeta     = beam_params["sigma_z"]*z,
    delta    = beam_params["sigma_delta"]*delta,

    # redefine the beam with the new particle added to it
    xtrack_particles["b{}".format(beam)] = xt.Particles(
                _context = sim_params["context"], 
                q0       = beam_params["q_b{}".format(beam)],
                p0c      = beam_params["p0c"],
                x        = np.hstack((xtrack_particles["b{}".format(beam)].x,         x)),
                px       = np.hstack((xtrack_particles["b{}".format(beam)].px,       px)),
                y        = np.hstack((xtrack_particles["b{}".format(beam)].y,         y)),
                py       = np.hstack((xtrack_particles["b{}".format(beam)].py,       py)),
                zeta     = np.hstack((xtrack_particles["b{}".format(beam)].z,         z)),
                delta    = np.hstack((xtrack_particles["b{}".format(beam)].delta, delta)),
                test     = np.hstack((xtrack_particles["b{}".format(beam)].test, 1)),
                )


def record_coordinates(coords_dict, beam_1, beam_2, turn_idx=-1):
    """
    dimensions of coords_dict expected: (# var settings, # turns, # beam particles)
    29/10/21: add mean coordinate fields and only store particles equal to the length of the initialized dict.
    """    
    num_macroparts_b1 = np.shape(coords_dict["b1"]["x"])[-1]  # last dimenstion is num. particles
    num_macroparts_b2 = np.shape(coords_dict["b2"]["x"])[-1]  # last dimenstion is num. particles

    coords_dict["b1"]["x"]    [turn_idx] = beam_1.x[-num_macroparts_b1:]  # save last n particles so that it includes test particles
    coords_dict["b1"]["y"]    [turn_idx] = beam_1.y[-num_macroparts_b1:]
    coords_dict["b1"]["px"]   [turn_idx] = beam_1.px[-num_macroparts_b1:]
    coords_dict["b1"]["py"]   [turn_idx] = beam_1.py[-num_macroparts_b1:]
    coords_dict["b1"]["z"]    [turn_idx] = beam_1.z[-num_macroparts_b1:]
    coords_dict["b1"]["delta"][turn_idx] = beam_1.delta[-num_macroparts_b1:]
    coords_dict["b2"]["x"]    [turn_idx] = beam_2.x[-num_macroparts_b2:]
    coords_dict["b2"]["y"]    [turn_idx] = beam_2.y[-num_macroparts_b2:]
    coords_dict["b2"]["px"]   [turn_idx] = beam_2.px[-num_macroparts_b2:]
    coords_dict["b2"]["py"]   [turn_idx] = beam_2.py[-num_macroparts_b2:]
    coords_dict["b2"]["z"]    [turn_idx] = beam_2.z[-num_macroparts_b2:]
    coords_dict["b2"]["delta"][turn_idx] = beam_2.delta[-num_macroparts_b2:]

    coords_dict["b1"]["x_mean"]    [turn_idx] = beam_1.x.mean()
    coords_dict["b1"]["y_mean"]    [turn_idx] = beam_1.y.mean()
    coords_dict["b1"]["px_mean"]   [turn_idx] = beam_1.px.mean()
    coords_dict["b1"]["py_mean"]   [turn_idx] = beam_1.py.mean()
    coords_dict["b1"]["z_mean"]    [turn_idx] = beam_1.z.mean()
    coords_dict["b1"]["delta_mean"][turn_idx] = beam_1.delta.mean()
    coords_dict["b2"]["x_mean"]    [turn_idx] = beam_2.x.mean()
    coords_dict["b2"]["y_mean"]    [turn_idx] = beam_2.y.mean()
    coords_dict["b2"]["px_mean"]   [turn_idx] = beam_2.px.mean()
    coords_dict["b2"]["py_mean"]   [turn_idx] = beam_2.py.mean()
    coords_dict["b2"]["z_mean"]    [turn_idx] = beam_2.z.mean()
    coords_dict["b2"]["delta_mean"][turn_idx] = beam_2.delta.mean()



def transpose_dict(coords_dict):
    """
    29/10/21: add mean coordinate fileds, just copied from old dict
    """
   
    coords_dict_transposed = {key_1: {key_2: [] for key_2 in coords_dict[key_1]} for key_1 in coords_dict.keys()}
    
    coords_dict_transposed["b1"]["x"]     = np.transpose(    coords_dict["b1"]["x"], (1,0))
    coords_dict_transposed["b1"]["y"]     = np.transpose(    coords_dict["b1"]["y"], (1,0))
    coords_dict_transposed["b1"]["px"]    = np.transpose(   coords_dict["b1"]["px"], (1,0))
    coords_dict_transposed["b1"]["py"]    = np.transpose(   coords_dict["b1"]["py"], (1,0))
    coords_dict_transposed["b1"]["z"]     = np.transpose(    coords_dict["b1"]["z"], (1,0))
    coords_dict_transposed["b1"]["delta"] = np.transpose(coords_dict["b1"]["delta"], (1,0))
    
    coords_dict_transposed["b2"]["x"]     = np.transpose(    coords_dict["b2"]["x"], (1,0))
    coords_dict_transposed["b2"]["y"]     = np.transpose(    coords_dict["b2"]["y"], (1,0))
    coords_dict_transposed["b2"]["px"]    = np.transpose(   coords_dict["b2"]["px"], (1,0))
    coords_dict_transposed["b2"]["py"]    = np.transpose(   coords_dict["b2"]["py"], (1,0))
    coords_dict_transposed["b2"]["z"]     = np.transpose(    coords_dict["b2"]["z"], (1,0))
    coords_dict_transposed["b2"]["delta"] = np.transpose(coords_dict["b2"]["delta"], (1,0))

    # mean coordinates are left untouched
    coords_dict_transposed["b1"]["x_mean"]     =     coords_dict["b1"]["x_mean"]
    coords_dict_transposed["b1"]["y_mean"]     =     coords_dict["b1"]["y_mean"]
    coords_dict_transposed["b1"]["px_mean"]    =    coords_dict["b1"]["px_mean"]
    coords_dict_transposed["b1"]["py_mean"]    =    coords_dict["b1"]["py_mean"]
    coords_dict_transposed["b1"]["z_mean"]     =     coords_dict["b1"]["z_mean"]
    coords_dict_transposed["b1"]["delta_mean"] = coords_dict["b1"]["delta_mean"]
    
    coords_dict_transposed["b2"]["x_mean"]     =     coords_dict["b2"]["x_mean"]
    coords_dict_transposed["b2"]["y_mean"]     =     coords_dict["b2"]["y_mean"]
    coords_dict_transposed["b2"]["px_mean"]    =    coords_dict["b2"]["px_mean"]
    coords_dict_transposed["b2"]["py_mean"]    =    coords_dict["b2"]["py_mean"]
    coords_dict_transposed["b2"]["z_mean"]     =     coords_dict["b2"]["z_mean"]
    coords_dict_transposed["b2"]["delta_mean"] = coords_dict["b2"]["delta_mean"]
    
    return coords_dict_transposed


def print_info(turn, text, particles, n_macroparticles, print_to_file=False, overwrite_file=False):
    
    if not print_to_file:
        print(text)
        for ii in range(n_macroparticles):
            print("Macropart {}: x:     {:.10f}".format(    ii, particles.x[ii]))
            print("Macropart {}: px:    {:.10f}".format(   ii, particles.px[ii]))
            print("Macropart {}: y:     {:.10f}".format(    ii, particles.y[ii]))
            print("Macropart {}: py:    {:.10f}".format(   ii, particles.py[ii]))
            print("Macropart {}: z:     {:.10f}".format( ii, particles.zeta[ii]))
            print("Macropart {}: delta: {:.10f}".format(ii, particles.delta[ii]))
    else:  # write info to file
        fname = "outputs/coords_{}.txt".format(text.replace(" ", "_"))
        if not overwrite_file:
            try:
                os.remove(fname)
            except OSError:
                pass
            f = open(fname, "w")
        else:
            f = open(fname, "a")
            
        f.write("Turn {}\n{}\n".format(turn, text))
        for ii in range(n_macroparticles):
            f.write("Macropart {}: x:     {:.10f}\n".format(    ii, particles.x[ii]))
            f.write("Macropart {}: px:    {:.10f}\n".format(   ii, particles.px[ii]))
            f.write("Macropart {}: y:     {:.10f}\n".format(    ii, particles.y[ii]))
            f.write("Macropart {}: py:    {:.10f}\n".format(   ii, particles.py[ii]))
            f.write("Macropart {}: z:     {:.10f}\n".format( ii, particles.zeta[ii]))
            f.write("Macropart {}: delta: {:.10f}\n".format(ii, particles.delta[ii]))
        f.write("\n")
        f.close()


def test_inputs(params):
    betapereps_threshold = 6
    assert all(np.sqrt(params["betastar_x"]/params["physemit_x"]) > betapereps_threshold), "β/ε too small! Decrease ε or increase β!"


def get_stats(particles, mask=None):

    keys=["x", "y", "px", "py", "z"]
    beam_means_dict  = {key: None for key in keys}
    beam_sigmas_dict = {key: None for key in keys}

    if mask is None:
        beam_means_dict["x"],  beam_sigmas_dict["x"]   = xf.mean_and_std(particles.x[particles.test==0])
        beam_means_dict["y"],  beam_sigmas_dict["y"]   = xf.mean_and_std(particles.y[particles.test==0])
        beam_means_dict["px"], beam_sigmas_dict["px"]  = xf.mean_and_std(particles.px[particles.test==0])
        beam_means_dict["py"], beam_sigmas_dict["py"]  = xf.mean_and_std(particles.py[particles.test==0])
        beam_means_dict["z"],  beam_sigmas_dict["z"]   = xf.mean_and_std(particles.z[particles.test==0])
    else:
        beam_means_dict["x"],  beam_sigmas_dict["x"]   = xf.mean_and_std(particles.x [mask][particles.test[mask]==0])
        beam_means_dict["y"],  beam_sigmas_dict["y"]   = xf.mean_and_std(particles.y [mask][particles.test[mask]==0])
        beam_means_dict["px"], beam_sigmas_dict["px"]  = xf.mean_and_std(particles.px[mask][particles.test[mask]==0])
        beam_means_dict["py"], beam_sigmas_dict["py"]  = xf.mean_and_std(particles.py[mask][particles.test[mask]==0])
        beam_means_dict["z"],  beam_sigmas_dict["z"]   = xf.mean_and_std(particles.z [mask][particles.test[mask]==0])
        
    return beam_means_dict, beam_sigmas_dict


"""
direct Lorentz boost for the weak beam
here no crossing plane angle assumed (α=0)
"""
def boost_hirata(x, y, z, px, py, e, sphi, cphi, tphi, n):


    for i in range(n):
    
        # h = total energy of particle i
        a = ( px[i]**2 + py[i]**2 ) / ( 1 + e[i] )**2
        sqr1a = np.sqrt(1-a)
        h = ( 1 + e[i] ) * a / ( 1 + sqr1a )
        
        # transform momenta
        px[i] = (px[i] - tphi*h) / cphi
        py[i] /= cphi
        e[i] -= sphi*px[i]
        
        # h1d = pz* in pdf
        a1 = ( px[i]**2 + py[i]**2 ) / ( 1 + e[i] )**2
        sqr1a = np.sqrt(1-a1)
        h1d = ( 1 + e[i] ) * sqr1a
        
        # derivatives of transformed Hamiltonian (h1z=-hσ* ??)
        h1x = px[i] / h1d
        h1y = py[i] / h1d
        h1z = a1 / (( 1 + sqr1a ) * sqr1a )
        
        # update coordinates
        x1 = tphi * z[i] + ( 1 + sphi * h1x ) * x[i]
        y[i] += sphi * h1y * x[i]
        z[i] /= cphi - sphi * h1z * x[i]
        x[i] = x1
        

def boost(x, y, z, px, py, e, sphi, cphi, tphi, verbose=False):

    for i in range(len(x)):
            
        # h = total energy of particle i
        h = e[i] + 1 - np.sqrt( (1 + e[i])**2 - px[i]**2 - py[i]**2 )

        # transform momenta
        px_ = (px[i] - h*tphi) / cphi
        py_ = py[i] / cphi
        e_  = e[i] - px[i]*tphi + h*tphi**2
        px[i] = px_
        py[i] = py_
        e[i]  = e_
        
        if verbose:
            print("h:", h, "px_:", px_,"py_:", py_,"e_:",e_)
        
        # pz*
        pz = np.sqrt( (1 + e[i])**2 - px[i]**2 - py[i]**2 )
        
        # derivatives of transformed Hamiltonian (hz=-hσ* ??)
        hx = px[i] / pz
        hy = py[i] / pz
        hs = 1 - ( e[i] + 1 ) / pz
        
        # update coordinates
        x_ = ( 1 + hx * sphi ) * x[i] + tphi * z[i]
        y_ = hy * sphi * x[i] + y[i]
        z_ = hs * sphi * x[i] + z[i] / cphi
        x[i] = x_
        y[i] = y_
        z[i] = z_
        
"""
inverse Lorentz boost
"""
def boost_inv_hirata(x, y, z, px, py, e, sphi, cphi, tphi, n):
    
    for i in range(n):
        
        # h = total energy of particle i
        a1 = ( px[i]**2 + py[i]**2 ) / ( 1 + e[i] )**2
        sqr1a = np.sqrt(1-a1)
        h1d = ( 1 + e[i] ) * sqr1a
        h1 = ( 1 + e[i] ) * a1 / ( 1 + sqr1a )
        h1x = px[i] / h1d
        h1y = py[i] / h1d
        h1z = a1 / (( 1 + sqr1a ) * sqr1a )
        
        # update coordinates
        det = 1 + sphi * ( h1x - sphi * h1z )
        x[i] = ( x[i] - sphi * z[i] ) / det
        z[i] = cphi * ( z[i] + sphi * h1z * x[i] )
        y[i] -= sphi * h1y * x[i]
        
        # transform momenta
        e[i] += sphi * px[i] 
        px[i] = ( px[i] + sphi * h1 ) * cphi
        py[i] *= cphi

        
def boost_inv(x, y, z, px, py, e, sphi, cphi, tphi, verbose=False):
    
    for i in range(len(x)):
        
        # h = total energy of particle i
        pz = np.sqrt( ( 1 + e[i] )**2 - px[i]**2 - py[i]**2 )
        hx = px[i] / pz
        hy = py[i] / pz
        hs = 1 - ( e[i] + 1 ) / pz

        # update coordinates
        det = 1/cphi + ( hx - hs*sphi ) * tphi
        x_ = x[i] / cphi - tphi * z[i]
        y_ = -tphi*hy * x[i] + (1/cphi + tphi*(hx - hs*sphi)) * y[i] + tphi*hy*sphi * z[i]
        z_ = -hs*sphi*x[i] + (1+hx*sphi)*z[i]
        x[i] = x_/det
        y[i] = y_/det
        z[i] = z_/det
        
        
        # transform momenta
        h = ( e[i] + 1 - pz ) * cphi**2
        px_ = px[i]*cphi + h*tphi
        py_ = py[i]*cphi
        e_  = e[i] + px[i]*cphi*tphi
        px[i] = px_
        py[i] = py_ 
        e[i]  = e_
        
        if verbose:
            print("h_:", h, "px__:", px_, "py_:", py, "e__:",e_)


def slice_slice_interaction(particles_b1, particles_b2, slice_mask_b1, slice_mask_b2, s1_idx, s2_idx, beam_params, xfields_beambeam, b1_means_dict, b2_means_dict, turn, measure_time, time_elapsed_dict):
    """
    Code for singel slice-slice interaction
    :param particles_bX: xtrack object of full8 beam particles
    :param slice_mask_bX: numpy array of size (# macroparts in current size), containing integer indices of slice macroparts.
    :param sX_idx: int, index of slice in beam
    :param beam_params: dict, containing beam parameters
    :param xfields_beambeam: dict containing xfields beambeam elements
    :param bX_means_dict: dict containing floats for the mean coordinates (x,x') of the full beam
    :param turn: int, turn number, for verbose
    :param measure_time: profiler object, to measure elapsed time between different steps
    :param time_elapsed_dict: dict, keys are strings denoting simulation steps and values are a single float that is the elapsed time 
    
    :return particles_bX: modified xtrack particle object with the slice coordinates updated with the kick
    """

    measure_time.start()
    particles_b1.state[slice_mask_b1] = 1000 + s1_idx
    particles_b2.state[slice_mask_b2] = 1000 + s2_idx
    add_new_value(time_elapsed_dict, "Set slice states", measure_time.stop())

    # update xsuite elements with current slice indices
    measure_time.start()
    xfields_beambeam["iptocp_ip1_b1"].slice_id = s1_idx
    xfields_beambeam["iptocp_ip1_b2"].slice_id = s2_idx
    xfields_beambeam["cptoip_ip1_b1"].slice_id = s1_idx
    xfields_beambeam["cptoip_ip1_b2"].slice_id = s2_idx
    xfields_beambeam["strongstrong3d_ip1_b1"].slice_id = s1_idx
    xfields_beambeam["strongstrong3d_ip1_b2"].slice_id = s2_idx
    add_new_value(time_elapsed_dict, "Update elements with slice indices", measure_time.stop())

    # Measure boosted slice properties at IP
    measure_time.start()
    s1_means_dict, s1_sigmas_dict = get_stats(particles_b1, mask=slice_mask_b1)
    s2_means_dict, s2_sigmas_dict = get_stats(particles_b2, mask=slice_mask_b2)
    add_new_value(time_elapsed_dict, "Get slice moments", measure_time.stop())

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
    add_new_value(time_elapsed_dict, "Update iptocp elements", measure_time.stop())

    # get sigmas at IP
    measure_time.start()
    sigma_matrix_ip_b1 = particles_b1.get_sigma_matrix(mask=slice_mask_b1) # boosted sigma matrix at IP
    sigma_matrix_ip_b2 = particles_b2.get_sigma_matrix(mask=slice_mask_b2) # boosted sigma matrix at IP
    add_new_value(time_elapsed_dict, "Get slice sigmas at IP", measure_time.stop())

    # store z coords at IP
    measure_time.start()
    particles_b1_z_storage = np.copy(particles_b1.zeta)
    particles_b2_z_storage = np.copy(particles_b2.zeta)
    add_new_value(time_elapsed_dict, "Save z coords", measure_time.stop())

    # transport CP using the z centroids, now all z coords are .5(z_c1-z_c2)
    measure_time.start()
    xfields_beambeam["iptocp_ip1_b1"].track(particles_b1)
    xfields_beambeam["iptocp_ip1_b2"].track(particles_b2)
    add_new_value(time_elapsed_dict, "Transport to CP", measure_time.stop())

    # get sigmas at CP
    measure_time.start()
    sigma_matrix_cp_b1 = particles_b1.get_sigma_matrix(mask=slice_mask_b1)
    sigma_matrix_cp_b2 = particles_b2.get_sigma_matrix(mask=slice_mask_b2)
    add_new_value(time_elapsed_dict, "Get slice sigmas at CP", measure_time.stop())

    # update beambeam with sigmas 
    measure_time.start()
    xfields_beambeam["strongstrong3d_ip1_b1"].update(n_macroparts_bb=beam_params["bunch_intensity"]/beam_params["n_macroparticles_b2"]*len(slice_mask_b2), 
                                              sigma_matrix_ip = sigma_matrix_ip_b2,
                                              sigma_matrix_cp = sigma_matrix_cp_b2, verbose_info=turn)
    xfields_beambeam["strongstrong3d_ip1_b2"].update(n_macroparts_bb=beam_params["bunch_intensity"]/beam_params["n_macroparticles_b1"]*len(slice_mask_b1),
                                  sigma_matrix_ip = sigma_matrix_ip_b1,
                                  sigma_matrix_cp = sigma_matrix_cp_b1, verbose_info=turn)
    add_new_value(time_elapsed_dict, "Update beambeam elements", measure_time.stop())

    # need to update individual z coords
    measure_time.start()
    xfields_beambeam["strongstrong3d_ip1_b1"].track(particles_b1)
    xfields_beambeam["strongstrong3d_ip1_b2"].track(particles_b2)
    add_new_value(time_elapsed_dict, "Beambeam kick", measure_time.stop())

    # transport back from CP to IP
    measure_time.start()
    xfields_beambeam["cptoip_ip1_b1"].track(particles_b1)
    xfields_beambeam["cptoip_ip1_b2"].track(particles_b2)
    add_new_value(time_elapsed_dict, "Transport back to IP", measure_time.stop())

    # update individual z coords
    measure_time.start()
    particles_b1.zeta = np.copy(particles_b1_z_storage)
    particles_b2.zeta = np.copy(particles_b2_z_storage)
    add_new_value(time_elapsed_dict, "Update z coords", measure_time.stop())

    # set back slice state
    measure_time.start()
    particles_b1.state[slice_mask_b1] = 1
    particles_b2.state[slice_mask_b2] = 1            
    add_new_value(time_elapsed_dict, "Set back slice states", measure_time.stop())

    return particles_b1, particles_b2
