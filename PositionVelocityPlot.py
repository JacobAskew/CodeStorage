#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 15:13:01 2023

@author: jacobaskew
"""
import datetime
import matplotlib.pyplot as plt
import math
import numpy as np

num = 86400*365.2425
num2 = 1/num
RAdata = np.cumsum(veff_ra*num2)
DECdata = np.cumsum(veff_dec*num2)


def rad_to_dms(rad):
    d = math.degrees(rad)
    m, s = divmod(d*3600, 60)
    d, m = divmod(m, 60)
    return "{}°{:02d}'{:06.3f}\"".format(int(d), int(m), s)


def rad_to_hms(rad):
    h = math.degrees(rad) / 15.0
    m = (h - math.floor(h)) * 60.0
    s = (m - math.floor(m)) * 60.0
    return "{:02d}:{:02d}:{:02d}".format(int(h), int(m), round(s, 2))


# List of HMS strings
hms_listRA = rad_to_hms(RAdata)
dms_listDEC = rad_to_dms(DECdata)
# Convert HMS strings to datetime.time objects
time_list = []
for hms in hms_listRA:
    h, m, s = map(int, hms.split(':'))
    time_list.append(datetime.time(hour=h, minute=m, second=s))
# Create x-axis values from datetime.time objects
x_values = []
for time_obj in time_list:
    x_values.append(datetime.datetime.combine(datetime.date.today(), time_obj))

decimal_degrees_list = []
for dms in dms_listDEC:
    d, m, s = map(float, dms[:-1].split('°') + dms[:-1].split('\''))
    decimal_degrees_list.append(d + m/60 + s/3600)

# Plot x-values vs. y-values
plt.plot(x_values, decimal_degrees_list)
plt.xlabel("RA")
plt.ylabel("DEC")
plt.show()

