#!/usr/bin/env python3

import numpy as np
import ase.io.vasp
import copy
import random

from tool import classification
from config import g_config


def get_atoms_0(structure):
#   atoms_0 = ase.io.vasp.read_vasp(structure)
    atoms_0 = ase.io.read(structure, format=g_config["structure_format"])
    return atoms_0

def change(atoms, doped, replaced):
    idx_doped = [atom.index for atom in atoms if atom.symbol in doped]
    idx_replaced = [atom.index for atom in atoms if atom.symbol in replaced]
    idx_1 = random.choice(idx_doped)
    idx_2 = random.choice(idx_replaced)
    atoms.positions[idx_1], atoms.positions[idx_2] = atoms.positions[idx_2].copy(), atoms.positions[idx_1].copy()

    return atoms

def make_combo(atoms, doped):
    l=[] 
    for idx, i in enumerate(doped):
        l.append(sorted([atom.position.tolist() for atom in atoms if atom.symbol == i], key=lambda x: np.sum(x)))
    return l

def search_data(data, atoms):
    #random.seed(g_config["seed_value"])
    doped = g_config["atoms_doped"]
    replaced = g_config["atoms_replaced"]
    energy=0
    target=0
    OK = 0
    atoms = change(atoms.copy(), doped, replaced)
    combo = make_combo(atoms, doped)
    for i, item in enumerate(data):
        if item["combo"] ==  combo:
            if g_config["Classification"] == True:
                omega=item["omega"]
                if classification(omega) == True:
                    energy=item["energy"]
                    target=item["target"]
                    OK = 1
                elif classification(omega) == False:
                    OK = -1

            elif g_config["Classification"] == False:
                energy=item["energy"]
                target=item["target"]
                OK = 1
#   print(OK)
    return combo, energy, target, OK, atoms

        

