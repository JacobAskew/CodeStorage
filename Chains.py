#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 14:54:13 2023

@author: jacobaskew
"""

# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib
# import corner
# import pickle

# keep = 500

# burn = 1000
# nwalkers = 100
# thin = 1
    
# filename = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/Modelling/Anisotropic/Anisotropic_Global_OM_Flipped/Anisotropic_Global_OM_Flipped_checkpoint_resume.pickle"
# # filename = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/Modelling/Anisotropic/Anisotropic_Global_Flipped/Anisotropic_Global_Flipped_checkpoint_resume.pickle"
# ## PICKLE ##
# with open(filename, 'rb') as f:
#     d=pickle.load(f)
    

# chain_array = d['chain_array']
# nwalker, nsteps, ndim = chain_array.shape
# ins = np.argwhere(chain_array[0, :, 0] > 0).squeeze()
# chain_array = chain_array[:, ins, :]
# # samples = np.array(chain_array)[:, ::thin, :].reshape((-1, ndim))
# samples = np.array(chain_array)[:, -int(keep/nwalkers*thin) :: thin, :].reshape((-1, ndim))

# # minimum_iteration = get_minimum_stable_itertion(self.mean_log_posterior, frac=self.convergence_inputs.mean_logl_frac)
# # discard_max = np.max([self.convergence_inputs.burn_in_fixed_discard, minimum_iteration])

# header = ['d', 's', 'A', 'vism_ra1', 'vism_dec1', 'vism_ra2', 'vism_dec2',
#        'vism_ra3', 'vism_dec3', 'vism_ra4', 'vism_dec4', 'vism_ra5',
#        'vism_dec5', 'vism_ra6', 'vism_dec6', 'vism_ra7', 'vism_dec7',
#        'KOM', 'OM', 'OMDOT', 'psi1', 'R1', 'psi2', 'R2', 'psi3', 'R3', 'psi4', 'R4',
#        'psi5', 'R5', 'psi6', 'R6', 'psi7', 'R7', 'TAUEFAC', 'DNUEFAC']

# # discard = 1750

# # nburn = 1000
# # thin2 = 1
# # iteration = d['iteration']
# # samples = np.array(chain_array)[:, discard + nburn : iteration : thin2, :].reshape((-1, ndim))


# # # data2 = []
# # for i in range(0, samples.shape[1]):
# #     ins = np.argwhere(samples[:, i] > 0)
# #     # data2.append(samples[ins, i])
# # # data2 = np.asarray(data2)

# # ## TEXT ##
# # sample_data = np.genfromtxt("/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/Modelling/Anisotropic/Anisotropic_Global_Flipped/Anisotropic_Global_Flipped_samples.txt", skip_header=0, encoding=None, dtype=str)
# # header = sample_data[0, :]
# # data = np.array(sample_data[1+burn::thin, :], dtype=float)

# # # chain = {}
# # # for i, p in enumerate(header):
# # #     chain[p] = data[:, i]
    

# # corner.corner(data, labels=header, show_titles=True)
# # plt.savefig("/Users/jacobaskew/Desktop/CORNER_txt.png")
# # plt.show()
# # plt.close()


# corner.corner(samples, labels=header, show_titles=True)
# plt.savefig("/Users/jacobaskew/Desktop/CORNER_Pickle.png")
# plt.show()
# plt.close()

# CODE THAT WILL WORK ON OZSTAR #

import numpy as np
import pickle

keep = 500
nwalkers = 100
thin = 1

for i in range(1, 1000):
    filename = "/fred/oz002/jaskew/0737_Project/Modelling/Anisotropic/Anisotropic_Global_Flipped_"+str(i)+"/Anisotropic_Global_Flipped_"+str(i)+"_checkpoint_resume.pickle"
    with open(filename, 'rb') as f:
        d=pickle.load(f)
    chain_array = d['chain_array']
    nwalker, nsteps, ndim = chain_array.shape
    ins = np.argwhere(chain_array[0, :, 0] > 0).squeeze()
    chain_array = chain_array[:, ins, :]
    samples = np.array(chain_array)[:, -int(keep/nwalkers*thin) :: thin, :].reshape((-1, ndim))


header = ['d', 's', 'A', 'vism_ra1', 'vism_dec1', 'vism_ra2', 'vism_dec2',
       'vism_ra3', 'vism_dec3', 'vism_ra4', 'vism_dec4', 'vism_ra5',
       'vism_dec5', 'vism_ra6', 'vism_dec6', 'vism_ra7', 'vism_dec7',
       'KOM', 'OM', 'OMDOT', 'psi1', 'R1', 'psi2', 'R2', 'psi3', 'R3', 'psi4', 'R4',
       'psi5', 'R5', 'psi6', 'R6', 'psi7', 'R7', 'TAUEFAC', 'DNUEFAC']


