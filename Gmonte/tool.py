#!/usr/bin/env python3

import numpy as np
import json
import io

#from Gmonte.structure import get_atoms_0, change, make_combo, search_data
#from Gmonte.probability  import probability, probability_RE, generate_output
#from Gmonte.GNN  import Machine_learning
#from Gmonte.make_graph  import make_torch

from config import g_config

def classification(omega):
    TH = g_config["Threshold"]

    if g_config["inequality"] == "==":
        if omega == TH:
            stable = True
        else:
            stable = False

    if g_config["inequality"] == ">":
        if omega > TH:
            stable = True
        else:
            stable = False

    if g_config["inequality"] == ">=":
        if omega >= TH:
            stable = True
        else:
            stable = False

    if g_config["inequality"] == "<":
        if omega < TH:
            stable = True
        else:
            stable = False

    if g_config["inequality"] == "<=":
        if omega <= TH:
            stable = True
        else:
            stable = False

    return stable

def atoms_2_json_str(atoms):
    with io.StringIO() as tmp_file:
        atoms.write(tmp_file,format='json')
        tmp_file.seek(0)
        return json.load(tmp_file)


def make_beta_0(mini, Maxu, M):
    beta = np.zeros(M)
    for i in range(M):
        beta[i] = (1/Maxu) +  (1/mini - 1/Maxu) * ((i+1)-1) / (M-1)
    beta = np.flip(beta)
    beta = 1/beta
    return beta



