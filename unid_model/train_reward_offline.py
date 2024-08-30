import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from sklearn.cluster import KMeans

import glob,os
import os


import numpy as np

import torch
from torch import nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.distributions.one_hot_categorical import OneHotCategoricalStraightThrough

from utils import load_all_mode

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer




def load_types(modes_n = 20, clusteres = [],test=False):

    all_types = ['pedestrian','bicycle','car']
    all_trajs_modes = []
    all_clusterers = []

    for i,current_type in enumerate(all_types):

        if test:
            yx_all = np.load(f'unid_test_{current_type}.npy')
        else:        
            yx_all = np.load(f'unid_train_{current_type}.npy')


        first_8x = yx_all[:,24::15]
        next_12x = yx_all[:,:24:2]
        first_8y = yx_all[:,25::15]
        next_12y = yx_all[:,1:24:2]

        x0_ = np.hstack((first_8x,next_12x))
        y0_ = np.hstack((first_8y,next_12y))

        dxy0_ = np.dstack((np.diff(x0_),np.diff(y0_)))
        r0 = np.linalg.norm(dxy0_,axis=2)

        theta0 = np.arctan2(dxy0_[:,:,1],dxy0_[:,:,0])
        difftheta0 = np.diff(theta0,axis=1)

        rx,ry = np.cos(difftheta0)*r0[:,1:],np.sin(difftheta0)*r0[:,1:]



        rxy = np.vstack((rx.flatten(),ry.flatten())).T


        if len(clusteres):
            clusterer = clusteres[i]
        else:
            smoothed_pnts = np.unique(np.round(rxy[:,:2],2),axis=0)
            clusterer = KMeans(n_clusters=modes_n,random_state=42).fit(smoothed_pnts)#,n_init=10




        rxy_ = np.dstack((rx,ry))
        all_l = clusterer.predict(rxy_.reshape(-1,2)).reshape(rxy_.shape[0],-1)

        all_trajs_modes.append(all_l.copy())
        all_clusterers.append(clusterer)



    return all_clusterers, all_trajs_modes


def stat_model(divide_by = 5, min_reward=-3):

    N_modes = 20
    STEPS = 18
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    expert_states, expert_actions, expert_test_states, expert_test_actions, clusterers = load_all_mode(device,modes_n=N_modes,return_clusterers=True)

    clusterers, all_modes_z = load_types(modes_n=N_modes,clusteres=clusterers,test=False)


    full_model = []
    for type_i in range(3):
        full_model.append([])
        all_modes_4_type = all_modes_z[type_i]
        for step_i in range(STEPS+2):
            full_model[type_i].append([])
            mode_step_mask = (all_modes_4_type[:,0]==step_i)

            for local_i in range(STEPS-1):

                local_mode_step = all_modes_4_type[mode_step_mask].T[local_i+1]
                full_model[type_i][step_i].append([(local_mode_step == mode_i).mean() for mode_i in range(STEPS+2)])

        # save as numpy for every type

        full_model[type_i] = np.clip(np.log(np.array(full_model[type_i]))/divide_by,min_reward,0)

    # type,first_step_mode,order_of_step_within,[20 logged probabilties]
    
    return full_model



if __name__ == "__main__":
    res = stat_model()







