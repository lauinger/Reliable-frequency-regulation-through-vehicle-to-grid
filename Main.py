# -*- coding: utf-8 -*-
"""
Created on Wed May  8 17:20:45 2019

@author: lauinger
"""

from my_functions import Sweep_y0_p
import numpy as np
# Simulation year in {2015, 2016, 2017, 2018}
Years = [2018]
# Battery Size in kWh
Battery = [50]
# Charger Power in kW
Charger = [7]
# Number of trading intervals between activation periods
Dk = [12, 16]
# Desired normalized state-of-charge at the end of the planning horizon in %
Y_star = np.array([50, 52, 54, 56, 58, 60])
# Price for deviations from Y_star in Euro/kWh
P_star = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]

for year in Years:
    for charger in Charger:
        for battery in Battery:
            for dk in Dk:
                print('Year: '+str(year)+' Battery: '+str(battery)+' Charger: '+str(charger)+' dk: '+str(dk))
                Sweep_y0_p(charger, battery, dk, Y_star, P_star, year, uni = False,\
                           losses = True, regulation = True, robust = True)