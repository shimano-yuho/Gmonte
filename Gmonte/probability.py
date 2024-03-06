#!/usr/bin/env python3

import numpy as np
import random

from config import g_config


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
    random.seed(g_config["seed_value"])
    return random.random() < p


