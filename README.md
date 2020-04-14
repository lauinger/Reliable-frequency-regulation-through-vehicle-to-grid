# Reliable frequency regulation through vehicle-to-grid

This code and data supplement the article "Reliable frequency regulation through vehicle-to-grid" by Dirk Lauinger, François Vuille, and Daniel Kuhn, available at ...

The code files are
1. `Main.py`,
2. `my_functions.py`.

The processed data files are
1. `ds.h5`: vehicle state and power for driving in kW,
2. `pa.h5`: availability price with half hourly resolution from 2015 through 2018 in EURO/kW/h,
3. `pb.h5`: utility price with half hourly resolution from 2015 through 2018 in EURO/kWh,
4. `pd.h5`: delivery price with half hourly resolution from 2015 through 2018 in EURO/kWh,
5. `pdd.h5`: delivery payment with half hourly resolution from 2015 through 2018 in EURO/kW/h,
6. `pr.h5`: total reserve price with half hourly resolution from 2015 through 2018 in EURO/kW/h.

The file `delta_10s.h5` takes about 200MB and is for space reasons not included in this repository. It can be constructed by running the jupyter notebook `Frequency.ipynb`.

The jupyter notebooks
1. `Availability_Price.ipynb`,
2. `Delivery_Price.ipynb`,
3. `Delivery_Payment.ipynb`,
4. `Driving.ipynb`,
5. `Frequency.ipynb`,
6. `Utility_Price.ipynb`

describe the creation of processed data from raw data.

The raw data stems from the French Transmission System Operator RTE and is linked to its source in the corresponding jupyter notebooks. For reproducibility, it is also included in this repository.
1. The availability price is in the excel files `Availability_2015` to `Availability_2019.
2. The delivery price is in the excel files `Delivery_2015` to `Delivery_2019`.
3. The frequency measurements are in the text files `RTE_Frequence_2015_01.txt` to `RTE_Frequence_2019_12.txt`.
4. The color of each day for the 'Tempo' pricing scheme is found in the excel files `Tempo_2014-2015.xlsx` to `Tempo_2019-2020.xlsx`.

The software used in this project was
1. Spyder 3.3.6 from the anaconda distribution with Python 3.6.9 64-bit and IPython 7.12.0,
2. JupyterLab 1.1.3,
3. Gurobi 9.0.0 with an academic license,
4. Gurobi's python interface `gurobipy`.

All numerical experiments were run on a workstation with 64GB RAM and an Intel i7-6700 CPU @ 3.40 GHz processor with a 64-bit operating system.

For inquiries please contact dirk.lauinger@epfl.ch

April 2020
