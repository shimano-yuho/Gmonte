#!/usr/bin/env python3

import torch
import e3nn
import numpy 
from pymatgen.io.ase import AseAtomsAdaptor
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch

def _cell_vol(cell):
    V=abs(numpy.dot(cell[0],numpy.cross(cell[1],cell[2])))
    return V


class GaussianBasisProjection(torch.nn.Module):
    '''Basis of translated Gaussians
    '''
    def __init__(self,start=0.0,stop=5.0,num_gaussians=50):
        super().__init__()
        offset=torch.linspace(start,stop,num_gaussians)
        ## `register_buffer` registers a parameter that is
        ## saved and restored in the state_dict (is part of the persistent state),
        # but not trained by the optimizer
        # https://pytorch.org/docs/1.1.0/nn.html#torch.nn.Module.register_buffer
        self.register_buffer('offset',offset)
        self.gamma=-0.5/(offset[1]-offset[0]).item()**2
    #
    # projection onto the Gaussian basis exp(-ɣ(r-μ_i)^2)
    # of r_j, the lengths of the jth graph edge, one-hot encoded as δ(r-r_j)
    # the projection can be thought of as a generalized discrete Fourier decomposition in that basis
    # the Fourier coefficients (in the dual basis) are ∫dr exp(-ɣ(r-μ_i)^2)δ(r-r_j) = exp(-ɣ(r_j-μ_i)^2)
    #
    def forward(self,r):
        # view returns a new tensor with the same data as the `self` tensor but of a different shape
        # the size -1 is inferred from other dimensions
        # view(-1,1) -> column
        # view(1,-1) -> row
        #
        # for what happens when subtracting of arrays with different shapes,
        # see array broadcasting (https://numpy.org/doc/stable/user/basics.broadcasting.html) for general broadcasting rules
        # dimensions with size 1 are stretched or "copied" to match the other
        # thus,
        # column-row will result in a matrix with dimension NxM where N is column height and M is row length,
        # and in the resulting  matrix `r`, row number is edge index, column is distance
        r=r.view(-1,1)-self.offset.view(1,-1)
        return torch.exp(self.gamma*torch.pow(r,2))



def main(atoms):

    num_radial = 32
    graph_max_radius = 3
    edge_sh_lmax = 2

    distance_expansion=GaussianBasisProjection(
    0.0,
    graph_max_radius,
    num_radial
    )

    _irreps=e3nn.o3.Irreps.spherical_harmonics(edge_sh_lmax)
        # for multiplicity>1, spherical harmonics expansion will just be repeated
    sh=e3nn.o3.SphericalHarmonics(
        _irreps,
        normalize = True)

    data=Data()
    # distance_vec is a Na x Na x 3 array of interatomic distance vectors,
    # distance_vec_ij holds a vector pointing from i to j
    distance_vec=atoms.get_all_distances(mic=True,vector=True)
    # r is Na x Na symmetric matrix of interatomic distances
    # r_ij holds the distance between atoms i and j
    r=numpy.linalg.norm(distance_vec,axis=2)
    # numpy.where returns a tuple populated with idices into the array upon which it is called
    # it is a filter returning only those indices which satisfy the condition
    # zero dist. causing zero division, add r>0.1
    idx=numpy.where((r<graph_max_radius) & (r>0.1))
    # when indexed with indices returned by numpy.where, the array will be unrolled
    # thus, edge_attr_radial will be a 1D array
    edge_attr_radial=distance_expansion(torch.tensor(r[idx],dtype=torch.float))
    ### angular (tensor) features
    ### see also: class SphericalHarmonicEdgeAttrs in NequIP
    # array of bond directions corresponding to interatomic distances r[idx]
    # for division, arrays must have same dim, in this case 1<->3 mismatch will use broadcast rules
    # https://numpy.org/doc/stable/user/basics.broadcasting.html
    edge_attr_angular=sh(torch.tensor(distance_vec[idx]/r[idx].reshape(-1,1),dtype=torch.float))
    _combined_edge_attr=torch.einsum('bi,bj->bij',(edge_attr_radial,edge_attr_angular))
    _ts=[]
    for _sl in e3nn.o3.Irreps.spherical_harmonics(edge_sh_lmax).slices():
        _ts.append(torch.flatten(_combined_edge_attr[:,:,_sl],start_dim=-2))
    data.edge_attr=torch.cat(_ts,dim=-1)
    nodes_i=idx[0]
    nodes_j=idx[1]
    data.edge_index=torch.LongTensor(numpy.array([nodes_j,nodes_i],dtype=int))
    ## end Generate edge features

    data.structure_id=[[0]]
    data.Z=torch.tensor(atoms.get_atomic_numbers(),dtype=torch.long)
    data.vol=torch.tensor(_cell_vol(atoms.get_cell()))
    data = Batch.from_data_list([data])

    return data
