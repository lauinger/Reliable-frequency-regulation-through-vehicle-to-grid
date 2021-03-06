# Reliable Frequency Regulation through Vehicle-to-Grid: From EU Legislation to Robust Optimization

This repository supplements the article "Reliable Frequency Regulation through Vehicle-to-Grid: From EU Legislation to Robust Optimization" by Dirk Lauinger, François Vuille, and Daniel Kuhn, available at https://arxiv.org/pdf/2005.06042.pdf. This project was funded by the [Institut VEDECOM](http://www.vedecom.fr/).

The code files are
1. `Main.py`,
2. `my_functions.py`.

The processed data files in the folder `Processed_data_excluding_frequency_deviations.zip` all have a time resolution of 30 minutes and span the years from 2015 to 2019. These files are:
1. `ds.h5`: the vehicle's state and power consumption for driving in kW,
2. `pa.h5`: the availability price in €/kW/h,
3. `pb.h5`: the utility price in €/kWh,
4. `pr.h5`: the estimated and true regulation price in €/kW/h.

The files `delta_10s.h5`, `delta_Gmm_2pt5hours.h5` and `delta_Gmm_5hours.h5` are each 250MB large and excluded from this repository for space reasons. The file `delta_10s.h5` can be constructed from the raw frequency recordings by running the jupyter notebook `Frequency.ipynb`. The uncertainty set dependent frequency deviation signals `delta_Gmm_2pt5hours.h5` and `delta_Gmm_5hours.h5` are constructed based on the original deviation recordings in `delta_10s.h5` by running the notebook `Adjust_delta.ipynb`. Both notebooks are in the folder `Data_Processing_and_Analysis.zip`, which also includes: `Availability_Price.ipynb`, `Delivery_Price.ipynb`, `Delivery_Payment.ipynb`, `Driving.ipynb`, `Frequency.ipynb`, and `Utility_Price.ipynb`. Each of these notebooks serves to convert raw data into processed data. The exact use of each notebook is described in its header. The notebooks provide links to the sources of the raw data. For convenience and reproducibility, we also make copies of the raw data available in the folders: `Raw_Price_Data.zip`, `Raw_Frequency_Deviation_Data_2015-2016.zip`, `Raw_Frequency_Deviation_Data_2017-2018.zip`, and `Raw_Frequency_Deviation_Data_2019.zip`.

The software used in this project was
1. Spyder 3.3.6 from the anaconda distribution with Python 3.6.9 64-bit and IPython 7.12.0,
2. JupyterLab 1.1.3,
3. Gurobi 9.0.0 with an academic license,
4. Gurobi's python interface `gurobipy`.

All numerical experiments were run on a workstation with 64GB RAM and an Intel i7-6700 CPU @ 3.40 GHz processor with a 64-bit operating system.

For inquiries please contact dirk.lauinger@epfl.ch

May 2020
