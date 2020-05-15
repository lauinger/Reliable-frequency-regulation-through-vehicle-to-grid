# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 16:32:44 2020

@author: lauinger
"""

from my_functions import Simulation
from my_functions import Loop_Simulation

import time

# --------------- PARAMETER LIST ------------------
# Simulation year in {2015, 2016, 2017, 2018, 2019}
year = 2019
# Battery size in kWh
battery = 50
# Charger power in kW
charger = 7
# Target state-of-charge in % of the battery size
y_star = 54
y_list = [50, 52, 54, 56, 58, 60]
# Deviation penalty in Euro/kWh
p_star = 0.15
p_list = [0.15, 0.20, 0.25]
# Yearly mileage (1 corresponds to 10,000 km)
drive = 1
# EU activation ratio in h: either 0.25 or 0.5
gmm = 0.5
# EU regulation cycle in h
Gmm = 2.5
# Reduced activation ratio in h
gmmh = 0.5
# Prolonged regulation cycle in h
Gmmh = 24
# Penalty for non-delivery of regulation power: either 'Exclusion' or 'Fine'
penalty = 'Exclusion'
# Penalty parameters for the calculation of the fine
kpen = 5 # Penalty factor (-)
py = 7.5 # Reserve price (Euro/kWh)

# ------------ Boolean Variables
uni = False
losses = True
regulation = True
robust = True
plan_losses = True
# ------------ Save results and verbose
save_result = True
verbose = True
# ------------ Parameter sweep: either 'nested' or 'sequential'
sweep = 'sequential'

#  ------------ Runs: either 'single' or 'multi'
runs = 'single'

# --------------- RUN THE SIMULATION ------------------
print('------------------------------------')
print('Year: '+str(year)+', Battery: '+str(battery)
      +', Charger: '+str(charger)+', Drive: '+str(drive))
print('gmm: '+str(gmm)+', Gmm: '+str(Gmm)+', gmmh: '+str(gmmh)+', Gmmh: '+str(Gmmh))
print('Penalty: '+penalty+', kpen: '+str(kpen)+', py: '+str(py))
print('Uni: '+str(uni)+', Losses: '+str(losses)+', Regulation: '+str(regulation)
      +', Robust: '+str(robust)+', Plan losses: '+ str(plan_losses))
print('------------------------------------')

# begin time measurement
start = time.time()

if runs == 'single':
    print('p_star: '+str(p_star)+', y_star: '+str(y_star))
    print('------------------------------------')
    profit_y0 = Simulation(charger, battery, gmm, Gmm, gmmh, Gmmh,
                           y_star, p_star, year, drive, uni, losses,
                           regulation, robust, penalty, kpen, py,
                           plan_losses, save_result, verbose)
if runs == 'multi':
    print('p_list: '+str(p_list))
    print('y_list: '+str(y_list))
    print('Sweep: '+sweep)
    print('------------------------------------')
    HM = Loop_Simulation(charger, battery, gmm, Gmm, gmmh, Gmmh, y_list, p_list,
                         year, drive, uni, losses, regulation, robust, penalty, 
                         kpen, py, plan_losses, save_result, sweep)

# end time measurement
end = time.time()
print('------------------------------------')
print('Execution time : '+str(round((end - start)/60))+'min')
print('------------------------------------')
