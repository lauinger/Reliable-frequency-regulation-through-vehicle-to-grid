# Reliable frequency regulation through vehicle-to-grid

The code and data supplement the article "Reliable frequency regulation through vehicle-to-grid" by Dirk Lauinger, Francois Vuille, and Daniel Kuhn, which appeared in journal in yyyy.

The code files are
1. `Main.py`,
2. `my_functions.py`.

The processed data files are
1. `ds.h5`: vehicle state (*s*) and power for driving (*d*) in kW,
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
6. `Utility_Price.ipynb`.
describe the data creation of processed data.

The raw data stems from the French Transmission System Operator RTE and is linked to in the corresponding jupyter notebooks. For reproducibility, it is also included in this repository.
1. The availability price is found in the excel files `Availability_2015` to `Availability_2018`.
2. The delivery price is found in the excel files `Delivery_2015` to `Delivery_2018`.
3. The frequency measurements are found in the text files `RTE_Frequence_2015_01.txt` to `RTE_Frequence_2018_12.txt`.
4. The color of each day for the 'Tempo' pricing scheme is found in the excel files `Tempo_2014-2015.xlsx` to `Tempo_2018-2019.xlsx`.

The software used in this project was
1. Spyder 3.3.3 from the anaconda distribution with Python 3.6.8 64-bit,
2. JupyterLab 0.35.4,
3. Gurobi 8.1.0 with an academic license,
4. Gurobi's python interface `gurobipy`.

All numerical experiments were run on a workstation with 64GB RAM and an Intel i7-6700 CPU @ 3.40 GHz processor with a 64-bit operating system.