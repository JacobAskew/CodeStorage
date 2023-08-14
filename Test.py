#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 15:02:27 2021

@author: jacobaskew
"""
import numpy as np

time_bins = 17
time_bin_length = 10
freq_bins = 9
freq_bin_length = 40
Tmin = []
Tmax = []
Fmin = []
Fmax = []

for i in range(0, time_bins):
    time = i * time_bin_length
    for ii in range(0, freq_bins):
        freq0 = ii * freq_bin_length
        freq1 = (ii - 1) * freq_bin_length
        Tmin.append(time)
        Tmax.append(time_bin_length+time)
        Fmin.append(1630-freq0)
        if ii == 0:
            Fmax.append(1682)
        else:
            Fmax.append(1630-freq1)

print("Tmin:", np.unique(Tmin))
print("Tmax:", np.unique(Tmax))
print("Fmin:", np.unique(Fmin))
print("Fmax:", np.unique(Fmax))
