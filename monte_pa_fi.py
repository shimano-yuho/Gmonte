#!/usr/bin/env python3

import pandas as pd
import numpy as np
import random
import subprocess
import time
import sys
import os
import pymatgen.io.vasp
import ase.io.vasp
import shutil
import filecmp
import csv
import json
import io
import concurrent.futures
import copy
import asyncio
import grlearn.process
import grlearn.scripts.evaluate

oj=os.path.join
home = os.path.expanduser("~")

def shuffle_atoms(atoms, combo):
    for i, j in enumerate(combo):
        atoms[i], atoms[j] = atoms[j], atoms[i]
    return atoms

def get_POSCAR_0():
    with open(f"POSCAR_0", 'r') as file:
        lines = file.readlines()
    numbers = [int(num) for num in lines[6].split() if num.isdigit()]
    metals = [str(num) for num in lines[5].split()]
    coordinates_start = lines.index('Direct\n') + 2
    coordinates_end = coordinates_start + (sum(numbers) - numbers[-1] - 1)

    atom_lines = lines[coordinates_start:coordinates_end]
    atoms = [line.split() for line in atom_lines]

    pos = [lines[:coordinates_start], lines[coordinates_end:]]

    my_list =  range(numbers[0]-1+numbers[1]+numbers[2])
    
    n_dope = numbers[0]
    return atoms, my_list, n_dope, pos, numbers, metals

def change(combo, my_list, n_dope):
   #print(f"combo={combo}")
    new_list = [my_list[i] for i in range(len(my_list)) if i not in combo]
    n_1 = random.randint(0, len(combo)-1)
    n_2 = random.choice(new_list)
   #print(f"n_1={n_1}")
   #print(f"n_2={n_2}")
    combo[n_1] = n_2
    new_combo = sorted(combo[:n_dope-1]) + sorted(combo[n_dope-1:], reverse=False)
   #print(f"new_combo={new_combo}")
    return new_combo

def prepare_dir(beta):
    for i in range(len(beta)):

        dir_out=f'./RE_{i}'
        dir_data=f'./RE_{i}/data'

        if not os.path.isdir(dir_out):
            os.mkdir(dir_out)

        if not os.path.isdir(dir_data):
            os.mkdir(dir_data)

        cmd=f'cp config.yaml {dir_out}/'
        os.system(cmd)


def get_frame(atoms_0, combo, pos, data_dir):
    shuffled_atoms = shuffle_atoms(atoms_0, combo)

    with open(f"POSCAR_a", "w") as temp_file:
        temp_file.writelines(pos[0])  # 元のPOSCARのヘッダー部分を書き込み
        temp_file.writelines([' '.join(atom) + '\n' for atom in shuffled_atoms])  # シャッフルされた原子座標を書き込み
        temp_file.writelines(pos[1])  # 元のPOSCARの残りの部分を書き込み
    atoms=ase.io.vasp.read_vasp("POSCAR_a")

    with open(oj(data_dir, "frames.xyz"), "w") as fpw:
        atoms.info={"id":combo, "target":0,}
       #ase.io.extxyz.write_extxyz(fpw, atoms)
        ase.io.write(fpw, atoms, format="extxyz")
    return

def get_value(name, R_n):
   #print(name, R_n)
    with open(f'RE_{R_n}/{name}.csv', 'r', newline='') as file:
        csv_reader = csv.reader(file)

        first_row = next(csv_reader)
   #    print(first_row)
        try:
            value = float(first_row[0])
        except ValueError:
   #        print("No")
            value = "img"
    return value


def get_fe(metal, num):    
    
    with open("FE.json", "r") as file:
        data_fe = json.load(file)

    FE = 0
    for idx, m in enumerate(metal):
        if m == "O":
            pass
        else:
            print(num[idx])
            FE += next(item for item in data_fe if item["Metal"] == m)["Energy_per_meatal"] * num[idx]
    return FE


def search_data(data, combo_, my_list, n_dope, atoms_0, pos, R_n):
    combo = copy.deepcopy(combo_)
    energy = 0
    epsilon= 0
    OK = 0
    aaa = 0
    data_dir = f"./RE_{R_n}/data"
    omega_min = 0
    while aaa == 0:
        combo = change(combo, my_list, n_dope)
        for i, item in enumerate(data):
            if item["combo"]  == combo:
                if item["omega"]:
                    omega=item["omega"]
                    if omega > omega_min:
                        energy=item["energy"]
                        epsilon=item["epsilon"]
                        OK = 1
                        aaa = 1
                    else:
                        OK = -1
                        aaa = 1
        else:
            aaa = 1

    if OK == 0:
        get_frame(atoms_0, combo, pos, data_dir)

    return combo, energy, epsilon, OK



def Machine_learning(R_n):
    omega_min = 0
    OK = 0

    omega = grlearn.scripts.evaluate.main("./ML/omega.pt",f"./RE_{R_n}/data/", 10,)    

    if omega > omega_min:
        OK = 1
        energy = grlearn.scripts.evaluate.main("./ML/energy.pt",f"./RE_{R_n}/data/", 10,)    
        epsilon = grlearn.scripts.evaluate.main("./ML/eps.pt",f"./RE_{R_n}/data/", 10,)    
    else:
        print(f"RE_{R_n}: omega is unstableeeeeeeee!")
        energy = -1000
        epsilon = -1000
            
    return omega, energy, epsilon

def make_data(datas):
    for idx, i in enumerate((datas)):
        if idx == 0:
            merged_list=i
            continue
        for item2 in i:
            # "name"が重複している場合は無視
            if all(item1["name"] != item2["name"] for item1 in merged_list):
                merged_list.append(item2)

    return merged_list
        

def probability(energy, energy_dash, kb, T):
    p = np.exp((-1 * (energy_dash - energy)) / (kb * T))
    if p > 1:
        p = 1
    return p


def probability_RE(energy_1, energy_2, kb, l_T, idx):
    T_1 = l_T[-(1+idx)] 
    T_2 = l_T[-(2+idx)] 
    p = np.exp(((1/(kb*T_1)) - (1/(kb*T_2))) * (energy_1 - energy_2))
    if p > 1:
        p = 1
    return p

def generate_output(p):
    return random.random() < p


def make_beta_0(mini, Maxu, M):
    beta = np.zeros(M)
    for i in range(M):
        beta[i] = (1/Maxu) +  (1/mini - 1/Maxu) * ((i+1)-1) / (M-1)
    beta = np.flip(beta)
    beta = 1/beta
    return beta


def main():
    l = ["RE", "time"]
    kb = 8.6171*10**(-5)
    seed_value = 42
    random.seed(seed_value)

    atoms_0, my_list, n_dope, pos, numbers, metals = get_POSCAR_0()

    n_atoms = sum(numbers)

    e_f = get_fe(metals, numbers) / n_atoms
    print(f"Energy: {e_f}")

    iteration = 2000

    start_time = time.time()

    data_dir = "./data"
    l_T = make_beta_0(1,30,20)
    prepare_dir(l_T)
    n = 5
    energy = np.zeros(len(l_T))
    eps = np.zeros(len(l_T))

    energy_dash = np.zeros(len(l_T))
    eps_dash = np.zeros(len(l_T))
    omega = np.zeros(len(l_T))
     

    df = pd.DataFrame(np.zeros((iteration, 2)), columns=l) 


    file_path = 'data_new.json'

    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
        # 読み込んだデータを使って何か処理を行う
        print(data)
    else:
        print(f"{file_path} が存在しません。")
        data = []

    combos= [list(range(n_dope*2-1)) for _ in range(len(l_T))]

    for i in range(iteration):
        if (i + 1) % 1  == 0:
            print(f"進行状況: {i + 1}/{iteration}")
#       print("*" *30)
#       print(f"iteration {i}")
        if (i + 1) % (n + 1) != 0:
            df.loc[i, "RE"] = False

            OKs = np.zeros(len(l_T))

            while any(x != 1 for x in OKs):
                print(f"OKs: {OKs}")
                for idx, j in enumerate(l_T):
                    if OKs[idx] == 0:
                        combos[idx], energy_dash[idx], eps_dash[idx], OKs[idx] = search_data(data, combos[idx], my_list, n_dope, atoms_0, pos, idx)
                    else:
                        pass

                for idx, j in enumerate(l_T):
                    if OKs[idx] == 0:
                        try:
                            os.remove(f".RE_/{idx}/data/processed/data.pt")
                        except:
                            pass
                        grlearn.process.convert_atoms_to_graphs(f"./RE_{idx}/data")
                    else:
                        pass

                
                for idx, ii in enumerate(OKs):
                    if ii == 0:
                        omega[idx], energy_dash[idx], eps_dash[idx] = Machine_learning(idx)


                
                for idx, j in enumerate(OKs):
                    d = {}
                    if j == 0:
                        d["combo"] = combos[idx]
                        try:
                            d["energy"] = energy_dash[idx]
                        except ValueError:
                            d["energy"] = "img"

                        try:
                            d["epsilon"] = eps_dash[idx]
                        except ValueError:
                            d["epsilon"] = "img"
                        d["omega"] = omega[idx]
                        data.append(d)

                        if d["epsilon"] > 0:
                            OKs[idx] = 1

                for idx, j in enumerate(OKs):
                    if j == -1:
                        OKs[idx] = 0
                    else:
                        pass

   #            print(OKs)

            for idx, j in enumerate(l_T):
                print(f"T = {j}")
                print(f"combo = {combos[idx]}")

                energy_dash[idx] = energy_dash[idx] - e_f

                print(f"energy_dash: {energy_dash[idx]}")
                print(f"eps_dash: {eps_dash[idx]}")

                df.loc[i, f"energy_dash_{j}K"] = energy_dash[idx]
                df.loc[i, f"eps_dash_{j}K"] = eps_dash[idx]

                if i == 0:
                    ad = True
                    p = 1
                else:
                    p = probability(energy[idx], energy_dash[idx], kb, j)
                    ad = generate_output(p)

        #       print(f"energy: {energy}")
        #       print(f"energy_dash: {energy_dash}")
        #       print(f"p: {p}")
        #       print(f"ad: {ad}")

                if ad == True:
                    print(f"yes {p}")
                    energy[idx] = energy_dash[idx]
                    eps[idx] = eps_dash[idx]
                else:
                    print(f"no {p}")
                
                df.loc[i, f"energy_{j}K"] = energy[idx]
                df.loc[i, f"eps_{j}K"] = eps[idx]

                df.loc[i, f"E_eps_{j}K"] = df[f"eps_{j}K"].mean()
                df.loc[i, f"p_{j}K"] = p
                df.loc[i, f"adoption_{j}K"] = ad
                df.loc[i, "time"] = time.time() - start_time

        elif (i + 1) % (n + 1) == 0:
            df.loc[i, "RE"] = True
            for idx in range(len(l_T) - 1):
                p = probability_RE(df.loc[i-1, f"energy_{l_T[-(1+idx)]}K"], df.loc[i-1, f"energy_{l_T[-(2+idx)]}K"], kb, l_T, idx)
                ad = generate_output(p)

                if idx == 0:
                    alpha = 1
                else:
                    alpha = 0

                if ad == True:
                    print("RE: True")
                    df.loc[i, f"energy_{l_T[-(2+idx)]}K"] = df.loc[i-alpha, f"energy_{l_T[-(1+idx)]}K"]
                    df.loc[i, f"energy_{l_T[-(1+idx)]}K"] = df.loc[i-1, f"energy_{l_T[-(2+idx)]}K"]
                    df.loc[i, f"eps_{l_T[-(2+idx)]}K"] = df.loc[i-alpha, f"eps_{l_T[-(1+idx)]}K"]
                    df.loc[i, f"eps_{l_T[-(1+idx)]}K"] = df.loc[i-1, f"eps_{l_T[-(2+idx)]}K"]

                    df.loc[i, f"E_eps_{l_T[-(1+idx)]}K"] = df[f"eps_{l_T[-(1+idx)]}K"].mean()
                    df.loc[i, f"E_eps_{l_T[-(2+idx)]}K"] = df[f"eps_{l_T[-(2+idx)]}K"].mean()
                    df.loc[i, f"RE_{idx}"] = True

                    energy[-(1+idx)], energy[-(2+idx)] = energy[-(2+idx)], energy[-(1+idx)]
                    eps[-(1+idx)], eps[-(2+idx)] = eps[-(2+idx)], eps[-(1+idx)]
                    combos[-(1+idx)], combos[-(2+idx)] = combos[-(2+idx)], combos[-(1+idx)]
#                   print(combos)


                elif ad == False:
                    print("RE: False")
                    df.loc[i, f"energy_{l_T[-(1+idx)]}K"] = df.loc[i-alpha, f"energy_{l_T[-(1+idx)]}K"]
                    df.loc[i, f"energy_{l_T[-(2+idx)]}K"] = df.loc[i-1, f"energy_{l_T[-(2+idx)]}K"]
                    df.loc[i, f"eps_{l_T[-(1+idx)]}K"] = df.loc[i-alpha, f"eps_{l_T[-(1+idx)]}K"]
                    df.loc[i, f"eps_{l_T[-(2+idx)]}K"] = df.loc[i-1, f"eps_{l_T[-(2+idx)]}K"]

                    df.loc[i, f"E_eps_{l_T[-(1+idx)]}K"] = df[f"eps_{l_T[-(1+idx)]}K"].mean()
                    df.loc[i, f"E_eps_{l_T[-(2+idx)]}K"] = df[f"eps_{l_T[-(2+idx)]}K"].mean()
                    df.loc[i, f"RE_{idx}"] = False
#                   print(combos)


        if (i + 1) % 10 == 0:
            df.to_csv("montecarlo.csv")
            json.dump(data,open(f"./data_new.json",'w'))

    df.to_csv("montecarlo.csv")
    json.dump(data,open(f"./data_new.json",'w'))

if __name__=='__main__':
    main()

