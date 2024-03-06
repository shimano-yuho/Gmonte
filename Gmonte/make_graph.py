#!/usr/bin/env python3

import torch
from pymatgen.io.ase import AseAtomsAdaptor
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch


class GaussianSmearing(torch.nn.Module):
    def __init__(self,start=0.0,stop=5.0,num_gaussians=50):
        super(GaussianSmearing, self).__init__()
        offset=torch.linspace(start,stop,num_gaussians)
        self.gamma=-0.5/(offset[1]-offset[0]).item()**2
        self.register_buffer('offset',offset)

    def forward(self,r):
        r=r.view(-1,1)-self.offset.view(1,-1)
        return torch.exp(self.gamma*torch.pow(r,2))

def make_torch(atoms):
    adaptor=AseAtomsAdaptor

    distance_expansion=GaussianSmearing(0.0, 5, 50)

    # TODO: revert to torch.utils.data.Dataset?
    data=Data()
    structure=adaptor.get_structure(atoms)
    neis_all=structure.get_all_neighbors(r=5)
    nodes_i=[i for i,neis in enumerate(neis_all) for nei in neis]
    nodes_j=[nei.index for neis in neis_all for nei in neis]
    distances=[nei.nn_distance for neis in neis_all for nei in neis]
    nodes_i+=list(range(len(atoms)))
    nodes_j+=list(range(len(atoms)))
    distances+=[0.0]*len(atoms)
    distances=torch.tensor(distances,dtype=torch.float)
    edge_attr_full=distance_expansion(distances)

    data.edge_index=torch.LongTensor([nodes_i,nodes_j])
    ### setting all edge weights equal may degrade the performance slightly
    data.edge_weight=torch.zeros(len(nodes_i),dtype=torch.float)
    data.edge_attr=edge_attr_full
    data.structure_id=[[0]]*1
    data.Z=torch.tensor(atoms.get_atomic_numbers(),dtype=torch.long)
    data = Batch.from_data_list([data])
    return data

