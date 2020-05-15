# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 16:30:57 2020

@author: lauinger
"""

from gurobipy import *
import pandas as pd
import numpy as np
import datetime as datetime

############################################################################

def Loop_Simulation(charger, battery, gmm, Gmm, gmmh, Gmmh, y_list, p_list,
                    year, drive = 1, uni = False, losses = True, regulation = True,
                    robust = True, penalty = 'Exclusion', kpen = 5, py = 7.5,
                    plan_losses = True, save_result = False, sweep = 'nested'):
    """
    Loop_Simulation(charger, battery, gmm, Gmm, gmmh, Gmmh, y_list, 
                    p_list, year, drive = 1, uni = False, losses = True, 
                    regulation = True, robust = True, penalty = 'Exclusion', 
                    kpen = 5, py = 7.5, plan_losses = True, save_result = False)
    
    Computes the cumulative profit of vehicle-to-grid for each combination of 
    the elements in the lists p_star_list and y_star_list for the 'year'.
    
    Return an error message and np.nan if a combination causes an error.
    
    INPUTS:
        charger:    charger power (kW)
        battery:    battery capacity (kWh)
        gmm:        activation period (h)
        Gmm:        regulation cycle (h)
        gmmh:       reduced activation period (h)
        Gmmh:       reduced regulation cycle (h)
        y_list:     list of state-of-charge target (%)
        p_list:     list of deviation penalties (EUR/kWh)
        year:       element of {2015, 2016, 2017, 2018, 2019}
        drive:      multiplier for yearly energy consumption for driving - 
                        default is 2,000 kWh/yr or 10,000 km/yr
        uni:        uni- or bidirectional charger
        losses:     with or without conversion losses
        regulation: active or not active on the regulation market
        robust:     constraint protection for gmm and Gmm if true, 
                        and for gmmh and Gmmh otherwise
        penalty:    penalty for non-delivery of promised regulation power,
                        either "Exclusion" or "Fine"
        kpen:       penalty factor set by the TSO--multiplies the availabitity price
        py:         price for instantaneous charging (EUR/kWh)
        plan_losses:does the planning problem consider losses?
        save_result:whether or not to save the results in a .h5 file
        sweep:      'nested' or 'sequential' for-loops
    
    OUTPUTS:
        HM: dataframe with the cumulative profit of vehicle-to-grid for each
        combination of the elements in the lists p_star_list and y_star_list
        for the 'year'.
        
        This dataframe is saved under the following name:
        HM_'+str(year)+'_'+str(charger)+'kW_'+str(battery)+'kWh_'+str(gmm)+'gmm_'
            +str(Gmm)+'Gmm'_+str(drive)+'dr_'+str(penalty)+'_'+str(kpen)+'kpen_'
            +str(py)+'py_'+str(uni)+'_uni_'+str(losses)+'_losses_'+str(regulation)
            +'_regulation_'+str(robust)+'_robust_'+str(plan_losses)+'_plan_losses_'
            +str(sweep)+'.h5'
    """
    # Declare the results dataframe
    HM = pd.DataFrame(np.nan, index=y_list, columns=p_list)
    # Set-Up a progress counter
    progress = 1
    # nested search
    if sweep == 'nested':
        # Sweep over p_star
        for p_star in p_list:
            # Sweep over y_star
            for y_star in y_list:
                # Run the simulation
                profit_y0 = Simulation(charger, battery, gmm, Gmm, gmmh, Gmmh,
                                       y_star, p_star, year, drive, uni, losses,
                                       regulation, robust, penalty, kpen, py,
                                       plan_losses, save_result, verbose=False)
                # Check for error message
                if type(profit_y0) == str:
                    print(profit_y0)
                else:
                    HM.loc[y_star, p_star] = profit_y0['Profit'].values[-1]
                # Show profit and progress message
                print('y_star: '+str(y_star)+', p_star: '+str(p_star)
                       +', profit: '+str(round(HM.loc[y_star, p_star],2)))
                print('Progress ', int(100*progress/(len(y_list)*len(p_list))))
                print('------------------------------------')
                progress = progress + 1
    # sequential search
    elif sweep == 'sequential': 
        # start with the highest SOC, last element in the list
        y_star = y_list[-1]
        # Sweep over p_star
        for p_star in p_list:
            # Run the simulation
            profit_y0 = Simulation(charger, battery, gmm, Gmm, gmmh, Gmmh,
                                   y_star, p_star, year, drive, uni, losses,
                                   regulation, robust, penalty, kpen, py,
                                   plan_losses, save_result, verbose=False)
            # Check for error message
            if type(profit_y0) == str:
                print(profit_y0)
            else:
                HM.loc[y_star, p_star] = profit_y0['Profit'].values[-1]
            # Show profit and progress message
            print('y_star: '+str(y_star)+', p_star: '+str(p_star)
                   +', profit: '+str(round(HM.loc[y_star, p_star],2)))
            print('Progress ', int(100*progress/(len(y_list) + len(p_list) - 1)))
            print('------------------------------------')
            progress = progress + 1
        # identify the best p_star
        p_star = HM.loc[y_star].idxmax()
        # Sweep over y_star
        for y_star in y_list[:-1]:
            # Run the simulation
            profit_y0 = Simulation(charger, battery, gmm, Gmm, gmmh, Gmmh,
                       y_star, p_star, year, drive, uni, losses,
                       regulation, robust, penalty, kpen, py,
                       plan_losses, save_result, verbose=False)
            # Check for error message
            if type(profit_y0) == str:
                print(profit_y0)
            else:
                HM.loc[y_star, p_star] = profit_y0['Profit'].values[-1]
            # Show profit and progress message
            print('y_star: '+str(y_star)+', p_star: '+str(p_star)
                   +', profit: '+str(round(HM.loc[y_star, p_star],2)))
            print('Progress ', int(100*progress/(len(y_list) + len(p_list) - 1)))
            print('------------------------------------')
            progress = progress + 1
    # misspecified sweep
    else:
        print('ERROR: sweep must be either \'nested\' or \'sequential\'.')
    # Save the heatmap
    if save_result:
        HM.to_hdf('Results/HM_'+str(year)+'_'+str(charger)+'kW_'+str(battery)+'kWh_'
                  +str(gmm)+'gmm_'+str(Gmm)+'Gmm_'+str(drive)+'dr_'
                  +str(penalty)+'_'+str(kpen)+'kpen_'+str(py)+'py_'
                  +str(uni)+'_uni_'+str(losses)+'_losses_'
                  +str(regulation)+'_regulation_'+str(robust)+'_robust_'
                  +str(plan_losses)+'_plan_losses_'+str(sweep)+'.h5', key='HM')
    # Return the results
    return HM

############################################################################

def Simulation(charger, battery, gmm, Gmm, gmmh, Gmmh, y_star, p_star, year,\
               drive = 1, uni = False, losses = True, regulation = True,\
               robust = True, penalty = 'Exclusion', kpen = 5, py = 7.5,\
               plan_losses = True, save_result = False, verbose = True):
    """
    Simulation(charger, battery, gmm, Gmm, gmmh, Gmmh, y_star, p_star, year,\
               drive = 1, uni = False, losses = True, regulation = True,\
               robust = True, penalty = 'Exclusion', kpen = 5, py = 7.5,\
               plan_losses = True, save_result = False, verbose = True)
        
    Computes the state-of-charge and cumulative profit at the end of each
    day of the 'year'.
    
    INPUTS:
        charger:    charger power (kW)
        battery:    battery capacity (kWh)
        gmm:        activation period (h)
        Gmm:        regulation cycle (h)
        gmmh:       reduced activation period (h)
        Gmmh:       reduced regulation cycle (h)
        y_star:     state-of-charge target (%)
        p_star:     deviation penalty (EUR/kWh)
        year:       element of {2015, 2016, 2017, 2018, 2019}
        drive:      multiplier for yearly energy consumption for driving - 
                        default is 2,000 kWh/yr or 10,000 km/yr
        uni:        uni- or bidirectional charger
        losses:     with or without conversion losses
        regulation: active or not active on the regulation market
        robust:     constraint protection for gmm and Gmm if true, 
                        and for gmmh and Gmmh otherwise
        penalty:    penalty for non-delivery of promised regulation power,
                        either "Exclusion" or "Fine"
        kpen:       penalty factor set by the TSO--multiplies the availabitity price
        py:         price for instantaneous charging (EUR/kWh)
        plan_losses:does the planning problem consider losses?
        save_result:whether or not to save the results in a .h5 file
        verbose:    whether or not to show comments
        
    OUTPUTS:
        profit_y0:  dataframe with the state-of-charge and the cumulative profit
                    at the end of each day as columns. Return an error message
                    if the problem is infeasible.
        
        This dataframe is saved under the following name:
        str(year)+'_'+str(y_star*100/battery)+'y_'+str(p_star)+'p_'
            +str(charger)+'kW_'+str(battery)+'kWh_'+str(gmm)+'gmm_'+str(Gmm)+'Gmm_'+
            +str(drive)+'dr_'+str(penalty)+'_'+str(kpen)+'kpen_'+str(py)+'py_'
            +str(uni)+'_uni_'+str(losses)+'_losses_'+str(regulation)+'_regulation_'
            +str(robust)+'_robust_'+str(plan_losses)+'_plan_losses'.h5'
    """
    # -------------------- Prepare simulation parameters -------------------
    # Load data
    pr = pd.read_hdf('pr.h5')
    pa = pd.read_hdf('pa.h5')
    pb = pd.read_hdf('pb.h5')
    ds = pd.read_hdf('ds.h5')
    
    # Load the frequency deviation signal depending on the EU activation period
    if Gmm == 2.5:
        delta = pd.read_hdf('delta_Gmm_2pt5hours.h5')
    elif Gmm == 5:
        delta = pd.read_hdf('delta_Gmm_5hours.h5')
    else:
        if verbose:
            print('ERROR: Gmm must be either 2.5 or 5')
        return 'ERROR: Gmm must be either 2.5 or 5'
    
    # Account for difference in driving
    ds['d'] = drive*ds['d']
    
    # Time Discretization, Planning Horizon Length
    dt = 0.5            # hours - planning time resolution
    ddt = 10            # seconds - simulation time resolution
    K = int(24/dt)      # length of the planning horizon
    
    # Minimum frequency deviation resolution
    delta_res = 10 / 200 # 10 out of 200 mHz
    
    # Battery Characteristics (kWh)
    y_max = 0.8*battery
    y_min = 0.2*battery
    
    # Charger efficiencies
    if losses:
        nc = 0.85
        nd = 0.85
    else:
        nc = 1
        nd = 1
        
    dn = 1/nd-nc
    
    # save a copy of the original gamma for the name of the results datafile
    gmm_org = gmm
    Gmm_org = Gmm
    
    # Planned efficiencies
    if plan_losses:
        nc_plan = nc
        nd_plan = nd
    else:
        nc_plan = 1
        nd_plan = 1
    dn_plan = 1/nd_plan - nc_plan 
    
    # Robustness
    if robust:
        gmm = gmm
        Gmm = Gmm
    else:
        gmm = gmmh
        Gmm = Gmmh
    
    # Utility Price
    uprice = "Tempo"
    
    # Convert target state-of-charge from % to kWh
    y_star = y_star*battery/100
    
    # Select test days
    if year == 2015:
        T = pd.date_range('01-01-2015 00:00:00', '12-31-2015', freq='D')
    elif year == 2016:
        T = pd.date_range('01-01-2016 00:00:00', '12-31-2016', freq='D')
    elif year == 2017:
        T = pd.date_range('01-01-2017 00:00:00', '12-31-2017', freq='D')
    elif year == 2018:
        T = pd.date_range('01-01-2018 00:00:00', '12-31-2018', freq='D')
    elif year == 2019:
        T = pd.date_range('01-01-2019 00:00:00', '12-31-2019', freq='D') 
    
    # Set-up profit counter and bidding market
    profit = 0
    # Declare result dataframe
    profit_y0 = pd.DataFrame(np.nan, index=T, columns = ['Profit', 'y0'])
    
    # initialize state-of-charge estimates
    y0 = y_star
    y0_min = y_star
    y0_max = y_star
    y0_hat_min = y_star
    y0_hat_max = y_star
    # initialize market participation
    if regulation:
        market = 'regulation'
    else:
        market = 'utility'
    
    # -------------------- Run the simulation ------------------------------
    for t_day in T:
        # For debugging
#        if t_day == T[3]:
#            return profit_y0
        t = pd.date_range(t_day, periods = K, freq='30min')
        # Charger Characteristics (kW)
        yc_max = charger*ds.loc[t,'s']          # charging
        if uni:
            yd_max = 0*ds.loc[t,'s']            # discharging
        else:
            yd_max = charger*ds.loc[t,'s']      # discharging
        
        # Assign initial state-of-charge to results dataframe
        profit_y0['y0'][t_day] = y0
        
        if market == 'regulation':
            res_bid = Reg(pr=pr.loc[t,:],pb=pb.loc[t,uprice], d=ds.d[t], K=K,\
                          y_max = y_max,y_min=y_min,yc_max=yc_max,yd_max=yd_max,\
                          y0_min = y0_min,y0_max = y0_max,\
                          y0_hat_min = y0_hat_min, y0_hat_max = y0_hat_max,\
                          y_star=y_star, p_star=p_star,\
                          nc=nc_plan,nd=nd_plan, dn=dn_plan, dt=dt,\
                          gmm = gmm, Gmm = Gmm, gmmh = gmmh, Gmmh = Gmmh)
            # Check whether the problem is infeasible: 
            infeasible = (res_bid['profit'] == 'Infeasible')
            
            if infeasible:
                if verbose:
                    print('ERROR: Infeasible bidding problem. Check whether \'ds.d\' is too high.')
                    print('Abort simulation.')
                return 'ERROR: Infeasible bidding problem. Check whether \'ds.d\' is too high.'
            
            # Calculate new state-of-charge if the bidding problem is feasible
            else:
                tt = pd.date_range(t_day, periods = K*dt/ddt*3600, freq='10s')
                res_soc = Calc_y0(t=t, tt=tt, dt=dt, ddt=ddt, d = ds.d[t], \
                                   xr=res_bid['xr'], xb=res_bid['xb'],\
                                   y0=y0, delta=delta.loc[tt,:], delta_res = delta_res,\
                                   K=K, nc = nc, nd = nd, dn = dn,\
                                   nc_plan = nc_plan, nd_plan = nd_plan, dn_plan = dn_plan,\
                                   gmm = gmm, Gmm = Gmm, gmmh = gmmh, Gmmh = Gmmh,\
                                   y_max = y_max, y_min = y_min)                        
                # For degugging
#                print(res_soc['y0'])
                if verbose:
                    print(str(t_day.date()) +' ymin : ' +str(round(res_soc['ymin'],2))+'kWh')
                    print(str(t_day.date()) +' ymax : ' +str(round(res_soc['ymax'],2))+'kWh')
                
                # Update profit
                profit = profit + res_bid['profit']
                
                # Calculate the fine for missing regulation power
                fine = dt * sum(res_soc['xr_m'] * kpen * pa['pa'].loc[t] \
                                   + res_soc['d_m'] * py)
                
                if fine > 0:
                    # Account for the penalty mechanisms
                    if penalty == 'Exclusion':
                        # Switch to the utility market
                        if verbose:
                            print('++++++++++++++++++++++++++++++++++++++++++++++++++++')
                            print(str(t_day.date()) +' Exclusion from regulation market')
                            print('++++++++++++++++++++++++++++++++++++++++++++++++++++')
                        market = 'utility'                        
                        
                    elif penalty == 'Fine':
                        if verbose:
                            print(str(t_day.date()) + ' Fine : ' + str(round(fine,2)) + 'EUR')
                        # Update profit
                        profit = profit - fine
                        
                    else:
                        if verbose:
                            print('ERROR: \'penalty\' must be either \'Exclusion\' or \'Fine\'')
                            print('Abort simulation.')
                        return 'ERROR: \'penalty\' must be either \'Exclusion\' or \'Fine\''
                    
                # Assign new state-of-charge
                y0_min = res_soc['y0_min']
                y0_max = res_soc['y0_max']
                y0_hat_min = res_soc['y0_hat_min']
                y0_hat_max = res_soc['y0_hat_max']
                y0 = res_soc['y0']
                # For debugging
#                print('y0 : ' + str(y0))
#                print('y0_min : ' + str(y0_min))
#                print('y0_max : ' + str(y0_max))
#                print('y0_hat_min : ' + str(y0_hat_min))
#                print('y0_hat_max : ' + str(y0_hat_max))

        if market == 'utility':
            res_utl = Utl(pb=pb.loc[t,uprice], d = ds.d[t], K = K,\
                                  y_max = y_max, y_min = y_min,\
                                  yc_max = yc_max, y0 = y0, nc = nc,\
                                  dt = dt, y_star = y_star, p_star = p_star)
            if res_utl['profit'] == 'Infeasible':
                if verbose:
                    print('ERROR: Infeasible utility purchases. Check whether \'ds.d\' is too high.')
                    print('Abort simulation.')
                return 'ERROR: Infeasible utility purchases. Check whether \'ds.d\' is too high.'
            else:
                profit = profit + res_utl['profit']
                y0 = res_utl['y0']           

        # assign resulting profit and state-of-charge to results
        profit_y0['Profit'][t_day] = profit
        profit_y0['y0'][t_day] = y0
        
        # Progress statement
        if verbose:
            print(str(t_day.date()) + ' Profit: ' + str(round(profit,2)) + 'EUR')
            print(str(t_day.date()) + ' SOC: ' + str(round(y0,2)) + 'kWh')
            print('----------------------------------------------------')
    
    # -------------------- Save the results --------------------------------
    profit_y0.to_hdf('Results/'+str(year)+'_'\
                     +str(y_star*100/battery)+'y_'+str(p_star)+'p_'\
                     +str(charger)+'kW_'+str(battery)+'kWh_'\
                     +str(gmm_org)+'gmm_'+str(Gmm_org)+'Gmm_'+str(drive)+'dr_'\
                     +str(penalty)+'_'+str(kpen)+'kpen_'+str(py)+'py_'\
                     +str(uni)+'_uni_'+str(losses)+'_losses_'\
                     +str(regulation)+'_regulation_'\
                     +str(robust)+'_robust_'\
                     +str(plan_losses)+'_plan_losses.h5',\
                     key='profit')
    # Return the results
    return profit_y0

############################################################################

def Calc_y0(t, tt, dt, ddt, d, xr, xb, y0, delta, delta_res, K, nc, nd, dn,\
            nc_plan, nd_plan, dn_plan, gmm, Gmm, gmmh, Gmmh, y_max, y_min):
    """
    Returns several measures of the state-of-charge at the end of the current day:
        'y0':         the true state-of-charge (kWh)
        'y0_min':     a conservative lower bound (kWh)
        'y0_max':     a conservative upper bound (kWh)
        'y0_hat_min': a less conservative lower bound (kWh)
        'y0_hat_max': a less conservative upper bound (kWh)
    In addition, it returns:
        'ymin':       the true minimum of the state-of-charge during the current day (kWh)
        'ymax':       the true maximum of the state-of-charge during the current day (kWh)
        'xr_m':       the amount of missing regulation power during each time interval (kW)
        'd_m':        the amount of missing power for driving during each time interval (kW)
    """

    # convert ddt into hours
    ddt = ddt/3600
    
    # --------------------- Calculate the true state-of-charge -------------    
    # build a dataframe of the market decisions for the current day
    df = pd.DataFrame({'xb': xb, 'xr':xr}, index=t)
    df['d'] = d[tt]
    # resample the dataframe from a 30min to a 10s resolution
    df = df.reindex(index = tt, method='ffill')
    df['delta'] = delta
    # add columns for the power missing for frequency regulation and for driving
    df['xr_m'] = 0   # power missing for frequency regulation
    df['d_m'] = 0    # power missing for driving
    # account for the lower and the upper bound on the battery state-of-charge
    df['y'] = np.nan # state-of-charge

    # --------------------- Non-robust case: missing power is possible
    if (gmm/Gmm == gmmh/Gmmh) | (nc != nc_plan) | (nd != nd_plan):
        # calculate soc and penalties iteratively
        for idx in df.index:
            # calculate state-of-charge at idx-1
            if idx == df.index[0]:
                y = y0
            else:
                y = df.y[idx - idx.freq]
            # case distinction: parking or driving
            if df.d[idx] == 0:  # parking
                # nominal power consumption
                x = df.xb[idx] + df.delta[idx] * df.xr[idx]
                # real power consumption when charging (note that the vehicle
                # ... is either charging or discharging but never both simultaneously)
                # ... the factor 1.001 is for numerical stability
                xhat = min( x, (1.001*y_max - y)/(nc * ddt) )
                # real power consumption when discharging
                # ... the factor 0.999 is for numerical stability
                xhat = max( xhat, nd*(0.999*y_min - y)/ddt )
                # missing reserve power
                df.loc[idx, 'xr_m'] = abs(x-xhat) / max(abs(df.delta[idx]), delta_res)
                # update y
                df.loc[idx, 'y'] = y + min(nc*xhat, xhat/nd) * ddt
            else: # driving
                # real power available for driving
                # the factor 0.999 is for numerical stability
                dhat = min( df.d[idx], (y - 0.999*y_min)/ddt )
                # missing power for driving
                df.loc[idx, 'd_m'] = df.d[idx] - dhat
                # update y
                df.loc[idx, 'y'] = y - dhat * ddt
    # --------------------- Robust case: missing power is impossible
    else:
        # calculate nominal charging
        yp = df['xb'] + df['delta'] * df['xr']
        # ... and discharging rates
        ym = - (df['xb'] + df['delta'] * df['xr'])
        # impose their non-negativity
        yp[yp < 0] = 0
        ym[ym < 0] = 0
        # update the state-of-charge accordingly        
        df['y'] = np.cumsum(nc*yp - ym/nd - df.d) * ddt + y0
                
    # --------------------- Calculate the bounds on the state-of-charge ----
    # observe state-of-charge at noon for the decision on new market bids
    t_obs = tt[0] + datetime.timedelta(hours=12,minutes=00)
    y_obs = df['y'][t_obs-datetime.timedelta(seconds=10)]
    # resample the true state-of-charge dataframe to a 30min resolution
    df_dt = df.resample('30min').mean()
    # define the time range at the end of which the bounds on the 
    # ... state-of-charge should hold
    t_est = pd.date_range(t_obs, t[-1], freq='30min')
    # extract the values of the market bids and the power consumption for 
    # ... driving for that time range
    d = df_dt.d[t_est]
    xb = np.array(df_dt.xb[t_est].values)
    xr = np.array(df_dt.xr[t_est].values)
    # Estimate the conservative bounds
    y0_max = Calc_y0_max(dt, d, xr, xb, y_obs, nc_plan, gmm, Gmm)
    y0_min = Calc_y0_min(dt, d, xr, xb, y_obs, nc_plan, dn_plan, gmm, Gmm)
    # Estimate the less conservative bounds
    y0_hat_max = Calc_y0_max(dt, d, xr, xb, y_obs, nc_plan, gmmh, Gmmh)
    y0_hat_min = Calc_y0_min(dt, d, xr, xb, y_obs, nc_plan, dn_plan, gmmh, Gmmh)
    # bound the forecasts by the lower and upper bounds on the batter capacity
    y0_min = max(y0_min, y_min)
    y0_hat_min = max(y0_hat_min, y_min)
    y0_max = min(y0_max, y_max)
    y0_hat_max = min(y0_hat_max, y_max)
    
    # --------------------- Return the results -----------------------------
    return {'y0': df['y'][-1], 'y0_min': y0_min, 'y0_max': y0_max,\
            'y0_hat_min': y0_hat_min, 'y0_hat_max': y0_hat_max,\
            'ymin': df['y'].min(), 'ymax': df['y'].max(),\
            'xr_m': df_dt['xr_m'], 'd_m': df_dt['d_m']}

############################################################################

def Calc_y0_max(dt, d, xr, xb, y0_obs, nc, gmm, Gmm):
    
#    print('Forecast upper bound on state-of-charge')

    k = len(xr)
#    print('k : '+str(k))
    
    # Gurobi Model
    m = Model("y0_max")
    # Maximization
    m.ModelSense = -1
    # Decision Variables
    delta = m.addVars(range(k), lb=0, ub=1, obj=nc*xr, name="delta")
    # Constraints
#    for l in range(k):
#        print(str(max(0, l+1 - round(Gmm/dt))) + '--' + str(l))
    m.addConstrs( (quicksum( delta[i] \
                    for i in range( max(0, l+1 - round(Gmm/dt)), l+1)) \
                    <= round(gmm/dt) for l in range(k)), "ymax" )
    # Disable logging to screen
    m.Params.OutputFlag = 0
    # Optimize
    m.optimize()
    
#    print('obj_val_ymax : ' + str(m.objVal))
#    print('y0_obs : ' + str(y0_obs))
    return y0_obs + (sum(nc*xb - d) + m.objVal)*dt

############################################################################

def Calc_y0_min(dt, d, xr, xb, y0_obs, nc, dn, gmm, Gmm):
    
#    print('Forecast lower bound on state-of-charge')
    
    k = len(xr)
#    print('k : '+str(k))
    # Gurobi Model
    m = Model("y0_min")
    # Maximization
    m.ModelSense = -1
    # Decision Variables
    delta = m.addVars(range(k), lb=0, ub=1,\
                      obj= (nc*xr + dn*np.maximum(xr-xb,0)) , name="delta")
    # Constraints
#    for l in range(k):
#        print(str(max(0, l+1 - round(Gmm/dt))) + '--' + str(l))
    m.addConstrs( (quicksum( delta[i] \
                            for i in range( max(0, l+1 - round(Gmm/dt)), l+1)) \
                    <= round(gmm/dt) for l in range(k)), "ymin" )
    # Disable logging to screen
    m.Params.OutputFlag = 0
    # Optimize
    m.optimize()
    
#    print('obj_val_ymin : ' + str(m.objVal))
    return y0_obs + (sum(nc*xb - d) - m.objVal)*dt

############################################################################

def Reg(pr, pb, d, K, y_max, y_min, yc_max, yd_max,\
        y0_min, y0_max, y0_hat_min, y0_hat_max, y_star, p_star,\
        nc, nd, dn, dt, gmm, Gmm, gmmh, Gmmh):
    
#    print('Build optimization problem')
    
    # Build tuplelist for dual variables
    idx = [(k,l) for k in range(K) for l in range(k+1)]
    
    # Setup Optimization Model
    m = Model("Reg")
    
    # Decision Variables and Objective (debug on real regulation prices)
#    xr = m.addVars(range(K), lb=0, obj=-dt*pr['pr'], name="xr")
    xr = m.addVars(range(K), lb=0, obj=-dt*pr['pr_est'], name="xr")
    xb = m.addVars(range(K), lb=0, obj=dt*pb, name="xb")
    z = m.addVar(obj=p_star, lb=-GRB.INFINITY, name="z")
    # Decision Rule Coefficients
    b = m.addVars(range(K), lb=-GRB.INFINITY, name="b")
    # Dual Variables - Vectors
    lbdp = m.addVars(range(K), lb=0, name="lbdp")
    lbdm = m.addVars(range(K), lb=0, name="lbdlm")
    thtp = m.addVars(range(K), lb=0, name="thtp")
    thtm = m.addVars(range(K), lb=0, name="thtm")
    # Dual Variables - Matrices
    Lbdp = m.addVars(idx, lb=0, name="Lbdp")
    Lbdm = m.addVars(idx, lb=0, name="Lbdm")
    Thtp = m.addVars(idx, lb=0, name="Thtp")
    Thtm = m.addVars(idx, lb=0, name="Thtm")

    # Power Constraints
    m.addConstrs( (xr[k] + xb[k] <= yc_max[k] for k in range(K)), "cmax" )
    m.addConstrs( (xr[k] - xb[k] <= yd_max[k] for k in range(K)), "dmax" )
    
    # Linearization Constraints
    m.addConstrs( (b[k] >= nc*xr[k] for k in range(K) ), "LDR_1" )
    m.addConstrs( (b[k] >= xr[k]/nd - dn*xb[k] for k in range(K) ), "LDR_2" )

    # Energy Constraints
    m.addConstrs( ( (quicksum( dt*(nc*xb[l] + Lbdp[k,l] - d[l] ) \
                              + gmm*Thtp[k,l] for l in range(k+1) ))\
                     <= (y_max - y0_max) for k in range(K)), "ymax")
    
    m.addConstrs( ( (quicksum( dt*(nc*xb[l] - Lbdm[k,l] - d[l] ) \
                              - gmm*Thtm[k,l] for l in range(k+1) ))\
                     >= (y_min - y0_min) for k in range(K)), "ymin")
    
    # Bounds on the deviation from the desired terminal state-of-charge
    m.addConstr( quicksum( dt*(nc*xb[k] + lbdp[k] - d[k] ) + gmmh*thtp[k] \
                          for k in range(K))\
                <= y_star - y0_hat_max + z, 'zh')
    m.addConstr( quicksum( dt*(nc*xb[k] - lbdm[k] - d[k]) - gmmh*thtm[k]\
                          for k in range(K))\
                >= y_star - y0_hat_min - z, 'zl')
    
    # Constraints from Dualization
    m.addConstrs( (Lbdp[k,l] + quicksum( Thtp[k,i] \
                   for i in range(l, 1+I(k, l, Gmm, dt))) \
                   >= nc*xr[l] for (k,l) in idx), "dual_yp")
    
    m.addConstrs( (Lbdm[k,l] + quicksum( Thtm[k,i] \
                   for i in range(l, 1+I(k, l, Gmm, dt))) \
                   >= b[l] for (k,l) in idx), "dual_ym")
       
    m.addConstrs( (lbdp[k] + quicksum( thtp[i] \
                   for i in range(k, 1+I(K-1, k, Gmmh, dt))) \
                   >= nc*xr[k] for k in range(K)), "dual_yph")
    
    m.addConstrs( (lbdm[k] + quicksum( thtm[i] \
                   for i in range(k, 1+I(K-1, k, Gmmh, dt))) \
                   >= b[k] for k in range(K)), "dual_ymh")

    # Disable logging to screen
    m.Params.OutputFlag = 0
    
#    print('Solve optimization problem')

    # Optimize
    m.optimize()
    
#    print('Retrieve solution')
#    print('z: '+str(z.X))
    # Update optimal solution
    if m.status == 2:
        # Read optimal variables
        xr_sol = []
        xb_sol = []
        for k in range(0,K):
            xr_sol.append(xr[k].X)
            xb_sol.append(xb[k].X)
        # Evaluate profit on real prices
        profit = dt*sum(pr['pr']*xr_sol - pb * xb_sol)
        return {'profit': profit, 'xr': xr_sol, 'xb': xb_sol}
    
    # Check for infeasibility
    else:
        profit = 'Infeasible'
        return {'profit': profit}
    
############################################################################

def I(k,l,Gmm,dt):
    return round(min(k, l + Gmm/dt - 1))

############################################################################
    
def Utl(pb, d, K, y_max, y_min, yc_max, y0, nc, dt, y_star, p_star):
    # Setup Optimization Model
    m = Model("Utl")

    # Decision Variables and Objective
    x = m.addVars(range(K), lb=0, ub=yc_max, obj=dt*pb, name='x')
    z = m.addVar(obj=p_star, name='z')
    
    # Constraints
    m.addConstrs( (quicksum( nc*x[l] - d[l] for l in range(k+1) ) \
                   <= (y_max - y0)/dt for k in range(K)), 'ymax')
    m.addConstrs( (quicksum( nc*x[l] - d[l] for l in range(k+1) ) \
                   >= (y_min - y0)/dt for k in range(K)), 'ymin')
    
    # Terminal State-of-Charge
    m.addConstr( quicksum( nc*x[k] - d[k] for k in range(K) ) \
                <= (y_star - y0 + z)/dt, 'zh' )
    m.addConstr( quicksum( nc*x[k] - d[k] for k in range(K) ) \
                >= (y_star - y0 - z)/dt, 'zl' ) 
    
    # Disable logging to screen
    m.Params.OutputFlag = 0
    # Optimize
    m.optimize()
    
    # Check for infeasibility
    if m.status != 2:
        profit = 'Infeasible'
        return {'profit': profit}
    else:
        # Update optimal solution--extract decision variables
        x_sol = []
        for k in range(0,K):
            x_sol.append(x[k].X)
        # Evaluate profit and terminal state-of-charge
        profit = -dt*sum(pb * x_sol)
        y0 = y0 + dt*sum(nc*np.array(x_sol) - d)
        return {'profit': profit, 'y0': y0}

############################################################################
