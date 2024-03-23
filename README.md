# Installation

```
cd Gmonte
pip install -r requirements.txt
pip install .
```

Install the module for the GNN model you want to use.

**Warning!**


In the current version, Gmonte **cannot** be used unless it is installed as a module of a GNN model class. In other words, Gmonte will not work with GNN models created with classes you have defined yourself.


# Required Files

**Structure file**


In ase-readable format (e.g. POSCAR).

**GNN models created in torch**


Place the model that predicts the energy and target properties. The classes used to create the model should be installed in the environment as modules.

**.py file for graphing**


Create a file that receives structural data and returns a Batch graphed by torch. This should be the same as the GNN model you created beforehand. Please refer to make_graph.py, which is placed as an example.

**config.yaml**


Enter the various configurations. Instructions are given in the file.

# How to use

Just type Gmonte in the terminal with the necessary files. The output files are montecarlo.csc, which shows the Monte Carlo sampling results, and GNN_result.json, which records the GNN predictions for the structure.

# How to see montecarlo.csv

The indexes show each iteration: energy_dash_TK, target_dash_TK show the GNN results for the candidate configurations, energy_TK, target_TK show the GNN results for the adopted configurations, E_target_TK shows the average of the properties adopted so far. E_target_TK shows the average value of the adopted properties so far. （The p indicates the probability of adoption by the Metropolis method; for indexes where RE is True, a replica exchange Monte Carlo method has been performed; for indexes where RE is True, a replica exchange Monte Carlo method has been performed.

