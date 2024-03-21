#!/usr/bin/env python3

import numpy as np
import torch
import importlib.util

from Gmonte.tool import classification
from Gmonte.config import g_config


def predict(model, data):
    device_count=torch.cuda.device_count()
    if device_count==0:
        device='cpu'
    elif device_count==1:
        device='cuda'
    else:
        device='cuda'

    model = torch.load(model, map_location=torch.device(device))
    with torch.no_grad():
        data.to(device)
        out=model(data)
        _o=out.cpu().numpy()
        predictions=_o
    #   print(predictions)


    return predictions


def Machine_learning(atoms):

    graph_path = g_config["graph_path"]
    module_name = "main"

    spec = importlib.util.spec_from_file_location(module_name, graph_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    data = module.main(atoms)

    ML_energy = g_config["path_ML_energy"]
    ML_target = g_config["path_ML_target"]
    
    if g_config["Classification"]==True:
        ML_omega = g_config["path_ML_classification"]
        omega = predict(ML_omega, data)[0][0].astype(float)   

        if classification(omega) == True:
            energy = predict(ML_energy, data)[0][0].astype(float)
            target = predict(ML_target, data)    
            if g_config["target_scalar"]==True:
                target=target[0][0].astype(float)
        else:
    #       print(f"omega is unstableeeeeeeee!")
            energy = "unstable"
            target = "unstable"

    elif g_config["Classification"]==False:
        omega = 0
        energy = predict(ML_energy, data)[0][0].astype(float)
        target = predict(ML_target, data)
        if g_config["target_scalar"]==True:
            target=target[0][0].astype(float)
        
            
    return omega, energy, target

