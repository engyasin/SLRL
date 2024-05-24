
import numpy as np
from utils import *
#from matplotlib import pyplot as plt

import cv2

import torch
from torch.distributions.normal import Normal


#from sklearn.cluster import KMeans

import glob,os
import os

import torch
from torch import nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.distributions.normal import Normal
from sklearn.cluster import KMeans


def load(type_=0):
    
    types_list = ['pedestrian','bicycle','car']
    
    yx_all = np.load(f'unid_train_{types_list[type_]}.npy')
    yx_test_all = np.load(f'unid_test_{types_list[type_]}.npy')

    # yx format
    # y: 24, x, y [0-11]
    # x: cx ,cy ,heading ,vx ,vy ,ax ,ay ,width ,length, vru5 ,vru10 ,vru20, nonvru5 ,nonvru10 ,nonvru20
    expert_states_all = []
    expert_test_states_all = []
    for step in reversed(range(8)):
        # first is the first step: 7
        speeds = np.linalg.norm(yx_all[:,27+(15*step):29+(15*step)],axis=1)[:,None]
        accel = np.linalg.norm(yx_all[:,29+(15*step):31+(15*step)],axis=1)[:,None]
        heading = (yx_all[:,26+(15*step):27+(15*step)]*np.pi/180) - (np.pi)

        nearby_objs = yx_all[:,[33+(15*step),34+(15*step),35+(15*step),36+(15*step),37+(15*step),38+(15*step)]]
        expert_states_all.append(np.hstack((speeds,heading,nearby_objs,np.zeros_like(speeds)+min(type_,2),accel)).copy())

    expert_states_all = np.hstack(expert_states_all)
    expert_actions = yx_all[:,[0,1,22,23]].copy()
    
    
    for step in reversed(range(8)):
        speeds = np.linalg.norm(yx_test_all[:,27+(15*step):29+(15*step)],axis=1)[:,None]
        heading = (yx_test_all[:,26+(15*step):27+(15*step)]*np.pi/180) #- (np.pi)
        accel = np.linalg.norm(yx_test_all[:,29+(15*step):31+(15*step)],axis=1)[:,None]
        
        nearby_objs = yx_test_all[:,[33+(15*step),34+(15*step),35+(15*step),36+(15*step),37+(15*step),38+(15*step)]]
        expert_test_states_all.append(np.hstack((speeds,heading,nearby_objs,np.zeros_like(speeds)+min(type_,2),accel)).copy())
    
    expert_test_states_all = np.hstack(expert_test_states_all)
    expert_test_actions = yx_test_all[:,[0,1,22,23]].copy()
    return expert_states_all,expert_actions,expert_test_states_all,expert_test_actions




def load_all_mode(device,modes_n = 5, return_clusterers=False):
    
    #val_in_set, val_out_set, train_discs = load_val(device)
    # env setup
    expert_states,expert_actions,expert_test_states,expert_test_actions = [], [], [], []
    clusterers = []
    for type_ in range(3): # Note no trucks
        expert_states_,expert_actions_,expert_test_states_,expert_test_actions_ = load(type_=type_)
        
        # NOTE degree of smoothing: first/last point
        smoothed_dp = np.unique(np.round(expert_actions_[:,:2],2),axis=0)

        clusterer = KMeans(n_clusters=modes_n,random_state=42).fit(smoothed_dp)#,n_init=10 # NOTE first step
        clusterer_labels_ = clusterer.predict(expert_actions_[:,:2])
        expert_actions_ = (np.hstack((expert_actions_,clusterer_labels_[:,None])))#.to(device)
        expert_test_actions_ = (np.hstack((expert_test_actions_,clusterer.predict(expert_test_actions_[:,:2])[:,None])))#.to(device)
        clusterers.append(clusterer)

        
        expert_states.append(expert_states_)
        expert_actions.append(expert_actions_)
        expert_test_states.append(expert_test_states_)
        expert_test_actions.append(expert_test_actions_)
        
    expert_states = torch.Tensor(np.vstack(expert_states)).to(device)
    expert_actions = torch.Tensor(np.vstack(expert_actions)).to(device)
    expert_test_states = torch.Tensor(np.vstack(expert_test_states)).to(device)
    expert_test_actions = torch.Tensor(np.vstack(expert_test_actions)).to(device)
    
    if return_clusterers:
        return expert_states,expert_actions[:,[0,1,4]],expert_test_states,expert_test_actions[:,[0,1,4]],clusterers
    return expert_states,expert_actions[:,[0,1,4]],expert_test_states,expert_test_actions[:,[0,1,4]]



if __name__ == '__main__':

    for i in range(3):
        _ = load(type_=i)