# -*- coding: utf-8 -*-
"""
Created on 8 May 2019

@author: lauinger
"""
from gurobipy import *
import pandas as pd
import numpy as np
import datetime as datetime

############################################################################

def Sweep_y0_p(charger, battery, dk, Y_star, P_star, year, uni = False,\
               losses = True, regulation = True, robust = True,\
               save_HM = True, plan_losses = True):
    """
    Computes the state-of-charge and cumulative profit at the end of each
    day of the 'year' for different values of the desired terminal state of
    charge and the price of deviations from it.
    
    INPUTS:
        charger:    charger power (kW)
        battery:    battery capacity (kWh)
        dk:         min. spacing between regulation calls (-)
        Y_star:     numpy array of desired terminal battery state-of-charge (%)
        P_star:     list of deviation prices (EUR/kWh)
        year:       element of {2015, 2016, 2017, 2018}
        uni:        uni- or bidirectional charger
        losses:     with or without conversion losses
        regulation: active or not active on the regulation market
        robust:     robust constraints, otherwise based on expected freq. deviation
        save_HM:    whether or not to save a dataframe of average daily profit
                    vs desired state of charge and deviation price
        plan_losses:does the planning problem consider losses?
        
    OUTPUTS:
		'Results/'+str(year)+'_'+str(y_star*100/battery)+'y_'+str(p_star)
                  +'p_'+str(charger)+'kW_'+str(battery)+'kWh_'+str(dk)+'dk_'
                        +str(uni)+'_uni_'+str(losses)+'_losses_'
                        +str(regulation)+'_regulation_'
                        +str(robust)+'_robust_'
                        +str(plan_losses)+'_plan_losses'.h5'
							 
		'Results/HM_'+str(year)+'_'+str(charger)
                        +'kW_'+str(battery)+'kWh_'+str(dk)+'dk_'
                        +str(uni)+'_uni_'+str(losses)+'_losses_'
                        +str(regulation)+'_regulation_'
                        +str(robust)+'_robust_'
                        +str(plan_losses)+'_plan_losses'.h5'
    """
    # Load data
    pr = pd.read_hdf('pr.h5')
    pb = pd.read_hdf('pb.h5')
    ds = pd.read_hdf('ds.h5')
    delta = pd.read_hdf('delta_10s.h5')
    
    # Time Discretization, Planning Horizon Length
    dt = 0.5            # hours - planning time resolution
    ddt = 10            # seconds - simulation time resolution
    K = int(24/dt)      # length of the planning horizon
    N = int(12.5/dt)    # length of the estimation horizon
    
    # Declare result vector
    Gamma_day = []
    Gamma_est = []
    
    # Select training days
    if year == 2015:
        T = pd.date_range('01-01-2016', '12-31-2016', freq='D')
    elif year == 2016:
        T = pd.date_range('01-01-2015', '12-31-2015', freq='D')
    elif year == 2017:
        T = pd.date_range('01-01-2015', '12-31-2016', freq='D')
    elif year == 2018:
        T = pd.date_range('01-01-2015', '12-31-2017', freq='D')
        
    for t in T:
        t_day = pd.date_range(t, periods = 8640, freq='10s')
        Gamma_day.append(np.mean(delta['delta'][t_day]**2))
        t_est = pd.date_range(t +datetime.timedelta(hours=11, minutes=30),\
                              t + datetime.timedelta(hours=24), freq='10s',\
                              closed='left')
        Gamma_est.append(np.mean(delta['delta'][t_est]**2))
        
    # Gamma for the variance of delta
    Gamma_day_mean = K*np.mean(np.array(Gamma_day))
    # Gamma_day_max = K*np.max(np.array(Gamma_day))
    Gamma_est_mean = N*np.mean(np.array(Gamma_est))
    # Gamma_est_max = N*np.max(np.array(Gamma_est))
    
    
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
    
    # Planned efficiencies
    if plan_losses:
        nc_plan = nc
        nd_plan = nd
    else:
        nc_plan = 1
        nd_plan = 1
    dn_plan = 1/nd_plan - nc_plan 
    
    # Utility Price
    uprice = "Tempo"
    
    # Define test desired state-of-charges
    Y_star = Y_star*battery/100
    
    # Select test days
    if year == 2015:
        T = pd.date_range('01-01-2015 00:00:00', '12-31-2015', freq='D')
    elif year == 2016:
        T = pd.date_range('01-01-2016 00:00:00', '12-31-2016', freq='D')
    elif year == 2017:
        T = pd.date_range('01-01-2017 00:00:00', '12-31-2017', freq='D')
    elif year == 2018:
        T = pd.date_range('01-01-2018 00:00:00', '12-31-2018', freq='D')
    
    # Declare result dataframe and set-up progress statements
    profit_y0_p = pd.DataFrame(np.nan, index=Y_star*100/battery, columns = P_star)
    progress = 1
    
    # Sweep over initial state-of-charge
    for y_star in Y_star:
        # Sweep over prices for constraint violation
        for p_star in P_star:
            # Set-up profit counter and bidding market
            profit = 0
            if regulation:
                market = 'regulation'
            else:
                market = 'utility'
            # Declare result dataframe
            profit_y0 = pd.DataFrame(np.nan, index=T, columns = ['Profit', 'y0'])
            # initialize state-of-charge estimates
            y0 = y_star
            y0_min = y_star
            y0_max = y_star
            y0_hat_min = y_star
            y0_hat_max = y_star
            # Sweep test days
            for t_day in T:
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
                    if robust:
                        # Base market bids on robust constraints
                        results = LDR(pr=pr.loc[t,:],pb=pb.loc[t,uprice],\
                              ds=ds.loc[t,:],K=K,dk=dk,\
                              y_max = y_max,y_min=y_min,yc_max=yc_max,yd_max=yd_max,\
                              y0_min = y0_min,y0_max = y0_max,\
                              y0_hat_min = y0_hat_min, y0_hat_max = y0_hat_max,\
                              y_star=y_star, p_star=p_star, Gamma=Gamma_day_mean,\
                              nc=nc_plan,nd=nd_plan,dn=dn_plan,dt=dt)
                    else:
                        # Base market bids on expected frequency deviations
                        results = LDR_Det(pr=pr.loc[t,:],pb=pb.loc[t,uprice],\
                              ds=ds.loc[t,:],K=K,\
                              y_max = y_max,y_min=y_min,yc_max=yc_max,yd_max=yd_max,\
                              y0_min = y0_min,y0_max = y0_max,\
                              y0_hat_min = y0_hat_min, y0_hat_max = y0_hat_max,\
                              y_star=y_star, p_star=p_star, Gamma=Gamma_day_mean,\
                              nc=nc_plan,nd=nd_plan,dn=dn_plan,dt=dt)
                            
                    # Check whether the problem is infeasible
                    if results['profit'] == 'Infeasible':
                        # abandon regulation market, purchase utility power only
                        results = PP(pb=pb.loc[t,uprice], ds=ds.loc[t,:], K=K,\
                          y_max=y_max, y_min=y_min, yc_max=yc_max,\
                          y0=y0, nc=nc_plan, dt=dt, change=True,\
                          y_star = y_star, p_star=p_star)
                        if results['profit'] == 'Infeasible':
                            break
                        else:
                            profit = profit + results['profit']
                            y0 = results['y0']
                            market = 'utility'
        
                    else:
                        # Update profit
                        profit = profit + results['profit']
                        # Calculate new state-of-charge
                        tt = pd.date_range(t_day, periods = K*dt/ddt*3600, freq='10s')
                        if robust:
                            results =  Calc_y0(t=t, tt=tt, dt=dt, ddt=ddt,\
                                               xr=results['xr'], xb=results['xb'],\
                                               y0=y0, delta=delta.loc[tt,:], d=ds.d[t],\
                                               K=K, dk=dk, nc = nc, nd = nd,\
                                               dn = dn, nc_plan=nc_plan, nd_plan=nd_plan,\
                                               dn_plan=dn_plan, Gamma=Gamma_est_mean)
                        else:
                            results =  Calc_y0_det(t=t, tt=tt, dt=dt, ddt=ddt,\
                                               xr=results['xr'], xb=results['xb'],\
                                               y0=y0, delta=delta.loc[tt,:], d=ds.d[t],\
                                               K=K, nc = nc, nd = nd, dn = dn,
                                               nc_plan=nc_plan, nd_plan=nd_plan, dn_plan=dn_plan,\
                                               Gamma=Gamma_est_mean)
                        # Assign new state-of-charge
                        y0_min = results['y0_min']
                        y0_max = results['y0_max']
                        y0_hat_min = results['y0_hat_min']
                        y0_hat_max = results['y0_hat_max']
                        y0 = results['y0']
                        
                elif market == 'utility':
                    results = PP(pb=pb.loc[t,uprice], ds=ds.loc[t,:], K=K,\
                          y_max=y_max, y_min=y_min, yc_max=yc_max,\
                          y0=y0, nc=nc_plan, dt=dt, change=False,\
                          y_star=y_star, p_star=p_star)
                    if results['profit'] == 'Infeasible':
                        break
                    else:
                        profit = profit + results['profit']
                        y0 = results['y0']
                    
                # assign resulting profit to results
                profit_y0['Profit'][t_day] = profit
            
            # Save data
            profit_y0.to_hdf('Results/'+str(year)+'_'\
                             +str(y_star*100/battery)+'y_'+str(p_star)+'p_'\
                             +str(charger)+'kW_'+str(battery)+'kWh_'+str(dk)+'dk_'\
                             +str(uni)+'_uni_'+str(losses)+'_losses_'\
                             +str(regulation)+'_regulation_'\
                             +str(robust)+'_robust_'\
                             +str(plan_losses)+'_plan_losses.h5',\
                             key='profit')
            
            # Assign profit to results dataframe
            profit_y0_p[p_star][y_star*100/battery] = profit/len(T)
            
            # Progress Statement
            print('Progress ', int(100*progress/(len(Y_star)*len(P_star))))
            progress = progress + 1
    
    # Save data
    if save_HM:
        profit_y0_p.to_hdf('Results/HM_'+str(year)+'_'+str(charger)\
                                 +'kW_'+str(battery)+'kWh_'+str(dk)+'dk_'\
                                 +str(uni)+'_uni_'+str(losses)+'_losses_'\
                                 +str(regulation)+'_regulation_'\
                                 +str(robust)+'_robust_'\
                                 +str(plan_losses)+'_plan_losses.h5',\
                                 key='profit')

############################################################################

def Calc_y0(t, tt, dt, ddt, xr, xb, y0, delta, K, dk, nc, nd, dn,\
            nc_plan, nd_plan, dn_plan, d, Gamma):
    # This function only works if Gamma < 1/length of the estimation horizon
    
    df = pd.DataFrame({'xb': xb, 'xr':xr}, index=t)
    df['d'] = d[tt]
    df = df.reindex(index = tt, method='ffill')
    df['delta'] = delta
    
    yp = df['xb'] + df['delta'] * df['xr']
    ym = - (df['xb'] + df['delta'] * df['xr'])
    yp[yp < 0] = 0
    ym[ym < 0] = 0
    
    df['yp'] = yp
    df['ym'] = ym
    
    df['y0'] = np.cumsum(nc*yp - ym/nd - df.d) * ddt/3600 + y0 

    t_obs = tt[0] + datetime.timedelta(hours=11,minutes=30)
    
    y0_obs = df['y0'][t_obs-datetime.timedelta(seconds=10)]

    df_dt = df.resample('30min').mean()
    
    t_est = pd.date_range(t_obs, t[-1], freq='30min')
    
    xb = np.array(df_dt.xb[t_est].values)
    xr = np.array(df_dt.xr[t_est].values)
    d = df_dt.d[t_est]
    
    y0_max = Calc_y0_max(dt, xr, xb, d, y0_obs, dk, nc_plan)
    y0_min = Calc_y0_min(dt, xr, xb, d, y0_obs, dk, nc_plan, dn_plan)

    idx = (Gamma*xr+xb).argsort()[-1]
    y0_hat_max = y0_obs + dt*(-sum(d) + nc_plan*( (Gamma*xr[idx] + xb[idx]) - xb[idx] + sum(xb)))
    
    idx = (Gamma*xr - xb).argsort()[-1]
    if Gamma*xr[idx] - xb[idx] > 0:
        ym = Gamma*xr[idx] - xb[idx]
    else:
        ym = 0
    y0_hat_min = y0_obs + dt*(-sum(d) - nc_plan*(Gamma*xr[idx] - xb[idx]) - dn_plan*ym + nc_plan*(sum(xb) - xb[idx]))
    
    return {'y0': df['y0'][-1], 'y0_min': y0_min, 'y0_max': y0_max, 'y0_hat_min': y0_hat_min, 'y0_hat_max': y0_hat_max}

############################################################################

def Calc_y0_max(dt, xr, xb, d, y0_obs, dk, nc):
    k = len(xr)
    # Gurobi Model
    m = Model("y0_max")
    # Maximization
    m.ModelSense = -1
    # Decision Variables
    delta = m.addVars(range(k), lb=0, ub=1, obj=xr, name="delta")
    # Constraints
    m.addConstrs( (quicksum( delta[i] for i in range( max(1,l-dk)-1, l+1)) <= 1 for l in range(k)), "ymax" )
    # Disable logging to screen
    m.Params.OutputFlag = 0
    # Optimize
    m.optimize()
    delta_sol = []
    for i in range(0,k):
        delta_sol.append(delta[i].X)
    
    return sum( nc*(np.array(delta_sol)*xr + xb) - d)*dt + y0_obs

############################################################################

def Calc_y0_min(dt, xr, xb, d, y0_obs, dk, nc, dn):
    k = len(xr)
    # Gurobi Model
    m = Model("y0_min")
    # Maximization
    m.ModelSense = -1
    # Decision Variables
    delta = m.addVars(range(k), lb=0, ub=1, obj= (nc*xr + dn*np.maximum(xr-xb,0)) , name="delta")
    # Constraints
    m.addConstrs( (quicksum( delta[i] for i in range( max(1,l-dk)-1, l+1)) <= 1 for l in range(k)), "ymin" )
    # Disable logging to screen
    m.Params.OutputFlag = 0
    # Optimize
    m.optimize()
    # Retrieve Optimal Values
    delta_sol = []
    for i in range(0,k):
        delta_sol.append(delta[i].X)
   
    return y0_obs - (m.objVal + sum(d - nc*xb))*dt
    
############################################################################

def Calc_y0_det(t, tt, dt, ddt, xr, xb, y0, delta, K, nc, nd, dn,\
                nc_plan, nd_plan, dn_plan, d, Gamma):
    # This function only works if Gamma < 1/length of the estimation horizon
    
    df = pd.DataFrame({'xb': xb, 'xr':xr}, index=t)
    df['d'] = d[tt]
    df = df.reindex(index = tt, method='ffill')
    df['delta'] = delta
    
    yp = df['xb'] + df['delta'] * df['xr']
    ym = - (df['xb'] + df['delta'] * df['xr'])
    yp[yp < 0] = 0
    ym[ym < 0] = 0
    
    df['yp'] = yp
    df['ym'] = ym
    
    df['y0'] = np.cumsum(nc*yp - ym/nd - df.d) * ddt/3600 + y0 

    t_obs = tt[0] + datetime.timedelta(hours=11,minutes=30)
    
    y0_obs = df['y0'][t_obs-datetime.timedelta(seconds=10)]

    df_dt = df.resample('30min').mean()
    
    t_est = pd.date_range(t_obs, t[-1], freq='30min')
    
    xb = np.array(df_dt.xb[t_est].values)
    xr = np.array(df_dt.xr[t_est].values)
    d = df_dt.d[t_est]

    idx = (Gamma*xr+xb).argsort()[-1]
    y0_hat_max = y0_obs + dt*(-sum(d) + nc_plan*( (Gamma*xr[idx] + xb[idx]) - xb[idx] + sum(xb)))
    
    idx = (Gamma*xr - xb).argsort()[-1]
    if Gamma*xr[idx] - xb[idx] > 0:
        ym = Gamma*xr[idx] - xb[idx]
    else:
        ym = 0
    y0_hat_min = y0_obs + dt*(-sum(d) - nc_plan*(Gamma*xr[idx] - xb[idx]) - dn_plan*ym + nc_plan*(sum(xb) - xb[idx]))
    
    y0_max = y0_hat_max
    y0_min = y0_hat_min
    
    return {'y0': df['y0'][-1], 'y0_min': y0_min, 'y0_max': y0_max, 'y0_hat_min': y0_hat_min, 'y0_hat_max': y0_hat_max}

############################################################################

def LDR(pr, pb, ds, K, dk, y_max, y_min, yc_max, yd_max,\
        y0_min, y0_max, y0_hat_min, y0_hat_max, y_star, p_star, Gamma,\
        nc,nd,dn,dt):
    
    # Build tuplelist for dual variables
    idx_d = [(k,l) for k in range(K) for l in range(k+1)]
    
    # Setup Optimization Model
    m = Model("LDR")
    
    # Decision Variables and Objective
    xr = m.addVars(range(K), lb=0, obj=-dt*pr['pr_est'], name="xr")
    xb = m.addVars(range(K), lb=0, obj=dt*pb, name="xb")
    zh = m.addVar(lb=0, obj=p_star, name="zh")
    zl = m.addVar(lb=0, obj=p_star, name="zl")
    # Decision Rule Coefficients
    a = m.addVars(range(K), lb=0, name="a")
    ah = m.addVars(range(K), lb=0, name="ah")
    # Dual Variables
    lbdh = m.addVar(lb=0, name="lbdh")
    lbdl = m.addVar(lb=0, name="lbdl")
    lbdp = m.addVars(idx_d, lb=0, name="lbdp")
    lbdm = m.addVars(idx_d, lb=0, name="lbdm")
    mup  = m.addVars(idx_d,  lb=0, name="mup")
    mum  = m.addVars(idx_d,  lb=0, name="mum")

    # Bounds on the deviation from the desired terminal state-of-charge
    # upper
    m.addConstr( Gamma*lbdh + quicksum( (nc*xb[k] - ds.d[k] ) for k in range(K))\
                <= (zh + y_star - y0_hat_max)/dt, 'zh')
    # lower
    m.addConstr( Gamma*lbdl + quicksum( (ds.d[k] - nc*xb[k] ) for k in range(K))\
                <= (zl - y_star + y0_hat_max)/dt, 'zl')
    # corresponding constraints on the dual variables lbdh and lbdl
    m.addConstrs( (lbdh >= nc*xr[k] for k in range(K)), 'lbdh' )
    m.addConstrs( (lbdl >= nc*xr[k] + dn*ah[k] for k in range(K)), 'lbdl' )
    
    # Robust Constraints
    # Power Constraints
    m.addConstrs( (xr[k] + xb[k] <= yc_max[k] for k in range(K)), "cmax" )
    m.addConstrs( (xr[k] - xb[k] <= yd_max[k] for k in range(K)), "dmax" )
    
    # Energy Constraints
    m.addConstrs( ( (quicksum( mup[k,l] + lbdp[k,l] + nc*xb[l] - ds['d'][l] for l in range(k+1) )) \
                     <= (y_max - y0_max)/dt for k in range(K)), "ymax")
    
    m.addConstrs( ( (quicksum( mum[k,l] + lbdm[k,l] - nc*xb[l] + ds['d'][l] for l in range(k+1) )) \
                     <= (y0_min - y_min)/dt for k in range(K)), "ymin")
    
    # Constraints from Dualization
    m.addConstrs( (lbdp[k,l] + quicksum( mup[k,i] for i in range(l, 1+min(l+dk,k))) \
                   >= nc*xr[l] for (k,l) in idx_d), "dualp")
    m.addConstrs( ( lbdm[k,l] + quicksum( mum[k,i] for i in range(l, 1+min(l+dk,k))) \
                   >= nc*xr[l] + dn*a[l] for (k,l) in idx_d), "dualm")
    
    # Decision Rule Constraints
    m.addConstrs( (a[k] >= xr[k] - xb[k] for k in range(K) ), "LDR a" )
    m.addConstrs( (ah[k] >= xr[k] - xb[k]/Gamma for k in range(K) ), "LDR ah" )

    # Disable logging to screen
    m.Params.OutputFlag = 0
    
    # Optimize
    m.optimize()
    
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

def LDR_Det(pr, pb, ds, K, y_max, y_min, yc_max, yd_max,\
        y0_min, y0_max, y0_hat_min, y0_hat_max, y_star, p_star, Gamma,\
        nc,nd,dn,dt):
    # This function only works if Gamma_day < 1/K
    
    # Setup Optimization Model
    m = Model("LDR")
    
    # Decision Variables and Objective
    xr = m.addVars(range(K), lb=0, obj=-dt*pr['pr_est'], name="xr")
    xb = m.addVars(range(K), lb=0, obj=dt*pb, name="xb")
    zh = m.addVar(lb=0, obj=p_star, name="zh")
    zl = m.addVar(lb=0, obj=p_star, name="zl")
    # Decision Rule Coefficients
    a = m.addVars(range(K), lb=0, name="a")
    ah = m.addVars(range(K), lb=0, name="ah")
    # Dual Variables
    lbdh = m.addVar(lb=0, name="lbdh")
    lbdl = m.addVar(lb=0, name="lbdl")
    lbdp = m.addVars(range(K), lb=0, name="lbdp")
    lbdm = m.addVars(range(K), lb=0, name="lbdm")

    # Bounds on the deviation from the desired terminal state-of-charge
    # upper
    m.addConstr( Gamma*lbdh + quicksum( (nc*xb[k] - ds.d[k] ) for k in range(K))\
                <= (zh + y_star - y0_hat_max)/dt, 'zh')
    # lower
    m.addConstr( Gamma*lbdl + quicksum( (ds.d[k] - nc*xb[k] ) for k in range(K))\
                <= (zl - y_star + y0_hat_max)/dt, 'zl')
    # corresponding constraints on the dual variables lbdh and lbdl
    m.addConstrs( (lbdh >= nc*xr[k] for k in range(K)), 'lbdh' )
    m.addConstrs( (lbdl >= nc*xr[k] + dn*ah[k] for k in range(K)), 'lbdl' )
    
    # Robust Constraints
    # Power Constraints
    m.addConstrs( (xr[k] + xb[k] <= yc_max[k] for k in range(K)), "cmax" )
    m.addConstrs( (xr[k] - xb[k] <= yd_max[k] for k in range(K)), "dmax" )
    
    # Energy Constraints
    m.addConstrs( ( Gamma*lbdp[k] + quicksum( nc*xb[l] - ds['d'][l] for l in range(k+1) ) \
                     <= (y_max - y0_max)/dt for k in range(K)), "ymax")
    
    m.addConstrs( ( Gamma*lbdm[k] + quicksum( - nc*xb[l] + ds['d'][l] for l in range(k+1) ) \
                     <= (y0_min - y_min)/dt for k in range(K)), "ymin")
    
    # Constraints from Dualization
    m.addConstrs( (lbdp[k] >= nc*xr[k] for k in range(K)), "dualp")
    m.addConstrs( (lbdm[k] >= nc*xr[k] + dn*a[k] for k in range(K)), "dualm")
    
    # Decision Rule Constraints
    m.addConstrs( (a[k] >= xr[k] - xb[k]/Gamma for k in range(K) ), "LDR a" )
    m.addConstrs( (ah[k] >= xr[k] - xb[k]/Gamma for k in range(K) ), "LDR ah" )

    # Disable logging to screen
    m.Params.OutputFlag = 0
    
    # Optimize
    m.optimize()
    
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
        
def PP(pb, ds, K, y_max, y_min, yc_max, y0, nc, dt, y_star, p_star, change):
    # Setup Optimization Model
    m = Model("PP_start")

    # Decision Variables and Objective
    x = m.addVars(range(K), lb=0, ub=yc_max, obj=dt*pb, name='x')
    zh = m.addVar(lb=0, obj=p_star, name='zh')
    zl = m.addVar(lb=0, obj=p_star, name='zl')
    
    # Constraints
    if change:
        m.addConstr( quicksum( nc*x[k] - ds['d'][k] for k in range(K) ) <= (y_max - y0)/dt, 'ymax')
        m.addConstr( quicksum(-nc*x[k] + ds['d'][k] for k in range(K) ) <= (y0 - y_min)/dt, 'ymin')
    else:
        m.addConstrs( (quicksum( nc*x[l] - ds['d'][l] for l in range(k+1) ) <= (y_max - y0)/dt for k in range(K)), 'ymax')
        m.addConstrs( (quicksum(-nc*x[l] + ds['d'][l] for l in range(k+1) ) <= (y0 - y_min)/dt for k in range(K)), 'ymin')
    
    # Terminal State-of-Charge
    m.addConstr( quicksum( nc*x[k] - ds['d'][k] for k in range(K) ) <= (zh + y_star - y0)/dt, 'zh' )
    m.addConstr( quicksum(-nc*x[k] - ds['d'][k] for k in range(K) ) <= (zl + y0 - y_star)/dt, 'zl' ) 
    
    # Disable logging to screen
    m.Params.OutputFlag = 0
    # Optimize
    m.optimize()
    
    # Update optimal solution
    if m.status == 2:
        # Read optimal variables
        x_sol = []
        for k in range(0,K):
            x_sol.append(x[k].X)
        
        # Evaluate profit and terminal state-of-charge
        profit = -dt*sum(pb * x_sol)
        y0 = y0 + dt*sum(nc*np.array(x_sol) - ds['d'])
        return {'profit': profit, 'y0': y0}
    
    # Check for infeasibility
    else:
        profit = 'Infeasible'
        return {'profit': profit}

############################################################################