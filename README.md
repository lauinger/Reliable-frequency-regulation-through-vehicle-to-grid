# Reliable Frequency Regulation through Vehicle-to-Grid: Encoding Legislation with Robust Constraints

This repository supplements the article "Reliable frequency regulation through vehicle-to-grid: Encoding legislation with robust constraints" by Dirk Lauinger, François Vuille, and Daniel Kuhn, available at https://pubsonline.informs.org/doi/10.1287/msom.2022.0154 and https://arxiv.org/pdf/2005.06042v4.pdf. This project was funded by the [Institut VEDECOM](http://www.vedecom.fr/). A recording of a talk based on this paper is available at https://www.youtube.com/watch?v=VTaooISx9yI and the slides are available at https://drive.switch.ch/index.php/s/81jvp4lDxaWi8HP.

The code files are in the folder `Experiments`, which includes the files:
1. `my_functions.py`: the functions called by the main files for each experiments,
2. `extra_plots.ipynb`: a jupyter notebook that plots reserve prices from the year 2015 to the year 2019 and frequency deviations (`regulation_price.pdf`), utility and reserve prices, driving needs, and the evolution of the state-of-charge with a 10s resolution on 9 August 2019 in the nominal scenario (`9aug.pdf`),

and the folders: 
1. `_base`: contains a main file to run the nominal scenario described in the paper as well as a `soc_analysis.ipynb` notebook that produces an empirical cumulative distribution plot (`soc.pdf`) of the vehicle state-of-charge with and without vehicle-to-grid. It also computes total energy throughput expressed in battery cycles with and without vehicle-to-grid.
2. `_data`: contains the file `vehicle_list.xlsx`. The files `delta_10s.h5`, `delta_Gmm_2pt5hours.h5`, `delta_Gmm_5hours.h5`, `ds.h5`, `pa.h5`, `pb.h5`, `pd.h5`, and `pr.h5` should be moved here. Some of these files exceed the 250MB maximum size allowed for by Github and are excluded for space reason. All the data files can be constructed with the scripts in the folder `Data_Processing_and_Analysis.zip`.
3. `Aggregation`: contains a special function file `my_functions_aggregator.py` and a main file `main_aggregator.py` to obtain the vehicle aggregation results in Appendix B "Vehicle Aggregator" in the online supplement of the paper. It also contains a pickled result file that can be analyzed with `mixed_fleet_analysis.ipynb`.
4. `Bid_at_midnight`: implements the multistage model referred to in Remark 6 of the paper.
5. `Crate_guarantee`: used to construct Figure 5.
6. `Driving_distance`: evaluates the value of vehicle-to-grid for driving distances of up to 30,000km per year for vehicles with uni- and bidirectional chargers. 
7. `Driving_time`: used to construct Figure 6.
8. `Overselling`: used to construct Figure 4.
9. `Scenarios`: used to construct Figure 3.
10. `Temporal_evolution`: evaluates the value of vehicle-to-grid for the years 2016, 2017, 2018, in addition to 2019, which is the standard for all other experiments.
11. `Tuning`: contains the main file needed to calculate the optimal tuning parameters for the nominal scenario for the years 2015 to 2019 reported in Section 5.2 "Backtesting Procedure and Baseline Strategy" of the paper.

Folders 4-10 contain a main file that is specific to each experiment. The results of the experiment are saved in .h5 files and visualized in jupyter notebooks, except for `Bid_at_midnight`. The runtimes are in the name of an otherwise empty textfile. When reproducing the experiments, all intermediate results will be saved in a `Results` folder. The folder `Overselling` contains several main files for various penalty parameters.

The folder `Processed_data_excluding_frequency_deviations.zip` contains time series produced with the data analysis scripts in `Data_Processing_and_Analysis.zip`. All time series have a resolution of 30 minutes and span the years from 2015 to 2019. These files are:
1. `ds.h5`: the vehicle's state and power consumption for driving in kW,
2. `pa.h5`: the availability price in €/kW/h,
3. `pb.h5`: the utility price in €/kWh,
4. `pd.h5`: the delivery price in €/kWh,
5. `pr.h5`: the estimated and true regulation price in €/kW/h.

The files `delta_10s.h5`, `delta_Gmm_2pt5hours.h5` and `delta_Gmm_5hours.h5` have a 10s resolution and are each 250MB large and excluded from this repository for space reasons. The file `delta_10s.h5` can be constructed from the raw frequency recordings by running the jupyter notebook `Frequency.ipynb`. The uncertainty set dependent frequency deviation signals `delta_Gmm_2pt5hours.h5` and `delta_Gmm_5hours.h5` are constructed based on the original deviation recordings in `delta_10s.h5` by running the notebook `Adjust_delta.ipynb`. Both notebooks are in the folder `Data_Processing_and_Analysis.zip`, which also includes: `Availability_Price.ipynb`, `Delivery_Price.ipynb`, `Delivery_Payment.ipynb`, `Driving.ipynb`, `Frequency.ipynb`, `Regulation_Price.ipynb`, and `Utility_Price.ipynb`. Each of these notebooks serves to convert raw data into processed data. The exact use of each notebook is described in its header. The notebooks provide links to the sources of the raw data. For convenience and reproducibility, we also make copies of the raw data available in the folders: `Raw_Price_Data.zip`, `Raw_Frequency_Deviation_Data_2015-2016.zip`, `Raw_Frequency_Deviation_Data_2017-2018.zip`, and `Raw_Frequency_Deviation_Data_2019.zip`. Finally, for the Aggregation Section, the vehicle usage profile of four different vehicles types is shown in `connected_vehicles.pdf` and the empirical cumulative distribution function of daily driving distances of the base vehicle with randon driving patterns on weekends is shown in `daily_driving_distance_base_random_weekends.pdf`.

The software used in this project was
1. Python 3.6.9 64-bit and IPython 7.12.0,
2. JupyterLab 3.1.7,
3. Gurobi 9.1.2 with an academic license,
4. Gurobi's python interface `gurobipy`.

All numerical experiments were run on a workstation with 64GB RAM and an Intel i7-6700 CPU @ 3.40 GHz processor with a 64-bit operating system.

For inquiries please contact lauinger(at)mit.edu

February 2024
