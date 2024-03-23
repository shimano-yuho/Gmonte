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

