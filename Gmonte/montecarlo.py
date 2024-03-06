#!/usr/bin/env python3

import pandas as pd
import numpy as np
import random
import time
import os
import json
import io
import copy
import math
import warnings

#from Gmonte.structure import get_atoms_0, change, make_combo, search_data
#from Gmonte.probability  import probability, probability_RE, generate_output
#from Gmonte.GNN  import Machine_learning
#from Gmonte.make_graph  import make_torch

from structure import get_atoms_0, search_data
from probability  import probability, probability_RE, generate_output
from GNN  import Machine_learning
from tool import atoms_2_json_str, make_beta_0
from config import g_config



def main():
    warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric")
    l = ["RE", "time"]
    kb = 8.6171*10**(-5)
    seed_value = g_config["seed_value"]
    e_f = g_config["Total_energy_before_synthesis/atom"]
    iteration = g_config["n_iteration"]
    n_alert = g_config["n_alert"]
    scalar = g_config["target_scalar"]

    structure = g_config["initial_structure"]
    atoms_0 = get_atoms_0(structure)

    start_time = time.time()

    if g_config["T_Auto"] == True:
        l_T = make_beta_0(g_config["T_min"], g_config["T_max"], g_config["n_T"])
    else:
        l_T = g_config["T_list"]

    if g_config["RE"] == True:
        n = g_config["RE_interval"]

    elif g_config["RE"] == False:
        n = iteration+1


    doped = g_config["atoms_doped"]
    replaced = g_config["atoms_replaced"]

    if len(replaced) > 1:
        raise ValueError("Only one type of atom should be substituted.")

    energy = np.zeros(len(l_T))
    target = np.zeros(len(l_T))
    energy_dash = np.zeros(len(l_T))
    target_dash = np.zeros(len(l_T))

    energy = [None] * len(l_T)
    target = [None] * len(l_T)
    energy_dash = [None] * len(l_T)
    target_dash = [None] * len(l_T)
    combos= [None]*len(l_T)


    omega = np.zeros(len(l_T))
    atoms_l = [atoms_0]*len(l_T)
     
    df = pd.DataFrame(np.zeros((iteration, 2)), columns=l) 

    data_path = g_config["data_path"]

    if data_path == False:
        data = []
    elif os.path.exists(data_path):
        print(f"Data file {data_path} loaded.")
        print("")
        with open(data_path, 'r') as file:
            data = json.load(file)
    else:
        print(f"{data_path}  does not exist")
        print("")
        data = []




    formula = atoms_0.get_chemical_formula()

    print(f"Material: {formula}")
    print("")
    print(f"atoms to be doped: " + " ".join(doped))
    print(f"atoms to be replaced: {replaced[0]}")
    print("")
    print(f"Doping concentration: {round(len([atom for atom in atoms_0 if atom.symbol in doped]) / len([atom for atom in atoms_0 if atom.symbol in doped + replaced]) * 100, 2)}%")
    print("")
    print("Temperature to be sampled: {}K".format("K ".join(map(str, [round(T, 2) for T in l_T]))))
    print("")

    if g_config["RE"] == True:
        print(f"Replica exchange every {n} sampling times Performs")
        print("")

    print("Start!")

    counter = 0

    for i in range(iteration):
        random.seed(seed_value+i)
        if (i + 1) % n_alert  == 0:
            print(f"Progress: {i + 1}/{iteration}")
        if (i + 1) % (n + 1) != 0:
            df.loc[i, "RE"] = False

            OKs = np.zeros(len(l_T))

            atoms_l_old = copy.deepcopy(atoms_l)

            while any(x != 1 for x in OKs):
               #print(f"OKs: {OKs}")
                for idx, j in enumerate(l_T):
               #    print(f"T = {j}K")
                    if OKs[idx] == 0:
                        combos[idx], energy_dash[idx], target_dash[idx], OKs[idx], atoms_l[idx]= search_data(data, atoms_l[idx].copy())
#                       print(combos[idx])
                    else:
                        pass

                for idx, ii in enumerate(OKs):
                    if ii == 0:
               #        print(l_T[idx])
               #        print("aaa")
                        omega[idx], energy_dash[idx], target_dash[idx] = Machine_learning(atoms_l[idx])
                        counter += 1
               #        print(target_dash[idx])


                
                for idx, j in enumerate(OKs):
                    d = {}
                    if j == 0:
                        d["combo"] = combos[idx]
                        d["structure"] = atoms_2_json_str(atoms_l[idx])

                        try:
                            d["energy"] = energy_dash[idx]
                        except ValueError:
                            d["energy"] = "unstable"

                        try:
                            d["target"] = target_dash[idx]
                        except ValueError:
                            d["target"] = "unstable"

                        if g_config["Classification"] == True:
                            d["omega"] = omega[idx]
                        data.append(d)

                       #print(type(d["target"]))
                        if type(d["target"]) == np.float64:
                            OKs[idx] = 1

                for idx, j in enumerate(OKs):
                    if j == -1:
                        OKs[idx] = 0
                    else:
                        pass

                for idx, j in enumerate(OKs):
                    if j == 0:
                        atoms_l[idx] = atoms_l_old[idx].copy()
                        pass
                    else:
                        pass

   #            print(OKs)

            for idx, j in enumerate(l_T):
#               print(f"T = {j}")
#               print(f"combo = {combos[idx]}")

                energy_dash[idx] = energy_dash[idx] - e_f

#               print(f"energy_dash: {energy_dash[idx]}")
#               print(f"target_dash: {target_dash[idx]}")

                df.loc[i, f"energy_dash_{j}K"] = energy_dash[idx]
                df.loc[i, f"target_dash_{j}K"] = target_dash[idx]

                if i == 0:
                    ad = True
                    p = 1
                else:
                    p = probability(energy[idx], energy_dash[idx], kb, j)
                    ad = generate_output(p)


                if ad == True:
#                   print(f"yes {p}")
                    energy[idx] = energy_dash[idx]
                    target[idx] = target_dash[idx]
                else:
#                   print(f"no {p}")
                    atoms_l[idx] = atoms_l_old[idx].copy()
                    pass
                
                df.loc[i, f"adoption_{j}K"] = ad

        elif (i + 1) % (n + 1) == 0:
            df.loc[i, "RE"] = True
            for idx in range(len(l_T) - 1):
                p = probability_RE(energy[-(1+idx)], energy[-(2+idx)], kb, l_T, idx)
                ad = generate_output(p)

                if idx == 0:
                    alpha = 1
                else:
                    alpha = 0

                if ad == True:
#                   print("RE: True")
                    df.loc[i, f"RE_{idx}"] = True
                    energy[-(1+idx)], energy[-(2+idx)] = energy[-(2+idx)], energy[-(1+idx)]
                    target[-(1+idx)], target[-(2+idx)] = target[-(2+idx)], target[-(1+idx)]
                    combos[-(1+idx)], combos[-(2+idx)] = combos[-(2+idx)], combos[-(1+idx)]
#                   print(combos)


                elif ad == False:
#                   print("RE: False")
                    df.loc[i, f"RE_{idx}"] = False
#                   print(combos)


        for idx, j in enumerate(l_T):
            df.loc[i, f"energy_{j}K"] = energy[idx]
            df.loc[i, f"target_{j}K"] = target[idx]

            if scalar == True:
                df.loc[i, f"E_target_{j}K"] = df[f"target_{j}K"].mean()
            else:
                pass

            df.loc[i, f"p_{j}K"] = p
            df.loc[i, "time"] = time.time() - start_time


        if (i + 1) % 10 == 0:
            df.to_csv("montecarlo.csv")
            json.dump(data,open(f"./data_new.json",'w'))

    df.to_csv("montecarlo.csv")
    json.dump(data,open(f"./GNN_result.json",'w'))

    print("Finish!")
    print(" ")

    print("Monte Carlo sampling results have been recorded in montecarlo.csv")
    print("GNN prediction results were recorded in GNN_result.json")

    if scalar == True:
        print(" ")
        print(f"Statistical expected values of the target from Monte Carlo sampling in {formula}")
        for idx, T in enumerate(l_T):
            print(f"{T}K: " , df[f"target_{T}K"].mean())

#   print(counter)


if __name__=='__main__':
    main()
 
