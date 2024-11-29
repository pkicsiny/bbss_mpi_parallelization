# -*- coding: utf-8 -*-

import numpy as np

# fcc-ee
fcc_params = {
"n_macroparticles_b1": int(1e3),
"n_macroparticles_b2": int(1e3), # 1e4-5, large intensity makes fft more ragged
"n_testparticles_b1": 0,
"n_testparticles_b2": 0,
"bunch_intensity": 2.3e11,  # 2.3e11 [1]
"energy": 182.5 ,  # 182.5 [GeV]
"physemit_x": 1.46e-9,  # 1.46e-9 [m]
"physemit_y": 2.9e-12,  # 2.9e-12 [m]
"sigma_z": 2.54e-3,  # 2.54e-3 [m]
"sigma_delta": 19.2e-4,  # 19.2e-4 [1]
"betastar_x": 1,  # 1 [1]
"betastar_y": 1.6e-3,  # 1.6e-3 [1]
"Qx": 0.108 ,  # 0.108 [1]
"Qy": 0.175 ,  # 0.175 [1]
"Qz": 0.0872,  # 0.0872 [1]
"p0c": 182.5e9,  # reference energy 182.5e9 [eV]
"alpha": 0,  # 0 [rad]
"phi": 0,  # 0 [rad]
"q_b1":1,  # 1 [e] 
"q_b2":-1  # -1 [e]
}
fcc_params["gamma"]      = fcc_params["energy"]/0.000511  # [1]
fcc_params["betastar_z"] = fcc_params["sigma_z"]/fcc_params["sigma_delta"]  # [1]

# hl-lhc
hl_lhc_params = {
"n_macroparticles_b1": int(1e3),
"n_macroparticles_b2": int(1e3), # 1e4-5, large intensity makes fft more ragged
"n_testparticles_b1": 0,
"n_testparticles_b2": 0,
"bunch_intensity": 2.2e11,  # 2.2e11 [1]
"energy": 7e3 ,  # 7e3 [GeV]
"sigma_z": 0.08,  # 0.08 [m]
"sigma_delta": 1E-4,  # [1]
"betastar_x": 1,  # 1 [1]
"betastar_y": 1,  # 1 [1]
"Qx": 0.31 ,  # 0.31 [1]
"Qy": 0.32 ,  # 0.32 [1]
"Qz": 1e-3,  # 0.31 [1]
"p0c": 7000e9, # reference energy 7000e9 [eV]
"alpha": 0,  # 0 [rad]
"phi": 0,  # 0 [rad]
"q_b1":1,  # 1 [e]
"q_b2":1  # 1 [e]
}
hl_lhc_params["gamma"]      = hl_lhc_params["energy"]/0.938  # [1]
hl_lhc_params["physemit_x"] = 2E-6/hl_lhc_params["gamma"]  # [m]
hl_lhc_params["physemit_y"] = 2E-6/hl_lhc_params["gamma"]  # [m]
hl_lhc_params["betastar_z"] = hl_lhc_params["sigma_z"]/hl_lhc_params["sigma_delta"]  # [1]

# Dimitry study
dimitry_params = {
"n_macroparticles_b1": int(1e3),
"n_macroparticles_b2": int(1e3), # 1e4-5, large intensity makes fft more ragged
"n_testparticles_b1": 0,
"n_testparticles_b2": 0,
"bunch_intensity": 8.873E+9,  # 8.873E+9 [1]
"energy": 0.51,  # 0.51 [GeV]
"sigma_z": 0.03,  # 0.03 [m]
"sigma_delta": 1E-4,  # 1E-4 [1]
"betastar_x": 1.5,  # 1.5 [m]
"betastar_y": 0.2,  # 0.2 [m]
"physemit_x" : 5e-7,  # 5e-7 [m]
"physemit_y" : 1e-9,  # 1e-9 [m]   
"Qx": 0.057 ,  # 0.057 [1]
"Qy": 0.097 ,  # 0.097 [1]
"Qz": 0.011,  # 0.011 [1]
"p0c": 0.51e9, # reference energy 7000e9 [eV]
"alpha": 0,  # 0 [rad]
"phi": 0,  # 0 [rad]
"q_b1":1,  # 1 [e]
"q_b2":-1  # 1 [e]
}
dimitry_params["gamma"]      = dimitry_params["energy"]/0.000511  # [1]
dimitry_params["betastar_z"] = dimitry_params["sigma_z"]/dimitry_params["sigma_delta"]  # [1]

dimitry_params_2 = {
"n_macroparticles_b1": int(1e3),
"n_macroparticles_b2": int(1e3), # 1e4-5, large intensity makes fft more ragged
"n_testparticles_b1": 0,
"n_testparticles_b2": 0,
"bunch_intensity": 2.0E+11, # 2.0E+11 [1]
"energy": 0.51,  # 51 [GeV]
"sigma_z": 0.003,  # 0.003 [m]
"sigma_delta": 1E-4,  # 1E-4 [1]
"betastar_x": 1,  # 1 [m]
"betastar_y": 1,  # 1 [m]
"physemit_x" : 2.25e-6,  # 2.25e-6 [m]
"physemit_y" : 1e-6,  # 1e-6 [m]   
"Qx": 0.057 ,  # 0.057 [1]
"Qy": 0.097 ,  # 0.097 [1]
"Qz": 0.011,  # 0.011 [1]
"p0c": 0.51e9, # reference energy 7000e9 [eV]
"alpha": 0,  # 0 [rad]
"phi": 0,  # 0 [rad]
"q_b1":1,  # 1 [e]
"q_b2":-1  # 1 [e]
}
dimitry_params_2["gamma"]      = dimitry_params_2["energy"]/0.000511  # [1]
dimitry_params_2["betastar_z"] = dimitry_params_2["sigma_z"]/dimitry_params_2["sigma_delta"]  # [1]


param_vec = {
"phi": np.linspace(0,0.5,6)*1e-3,  # [rad]
#"phi": np.array([0, .1, .2, .5, 1, 2, 5, 10, 20, 50, 1e2, 2e2, 5e2, 1e3])*1e-3,  # 14 [rad]
"alpha": np.array([0, np.pi/2]),  # [rad]
"betastar_x": np.array([1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]),  # [1]
"phi_y" : np.array([0, 1, 2, 3, 4, 5])*1e-3,  # 100, 200, 500, 1000])*1e-3,  # [rad]
"phi_x" : np.array([0, 20, 40, 60, 80, 100])*1e-3,  #, 200, 300, 400, 500, 600, 700, 800, 900, 1000])*1e-3,  #15 [rad]
"n_slices": np.array([1, 2, 5, 10, 20, 50, 100, 200]),  # 8 [1]
"n_macroparts": np.array([1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6]),  # 10 [1]
}
