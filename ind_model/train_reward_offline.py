

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

class LongTermDiscriminator(nn.Module):

    def __init__(self, n_modes = 20, steps = 18):
        super().__init__()
        
        self.vector_state_root = nn.Sequential(
            layer_init(nn.Linear(steps,32)),
            nn.ReLU(),
            nn.Dropout(0.2),
            layer_init(nn.Linear(32,8)),
            nn.ReLU(),
            layer_init(nn.Linear(8,1))
        )

        
        self.n_modes = n_modes
        
    def forward(self,modes_vec):

        return self.vector_state_root(modes_vec)


class RewardModeSequance(nn.Module):
    def __init__(self, n_modes = 20, steps = 18):
        super().__init__()

        self.types = 3
        self.submodules = nn.ModuleList([LongTermDiscriminator(n_modes=n_modes,steps = steps)
                           for _ in range(self.types)])

        self.n_modes = n_modes
        
    def forward_all(self,modes_vec,types_vec):

        full_actions = torch.zeros_like(modes_vec[:,:1])
        for type_ in range(self.types):
            type_mask = (types_vec == type_)

            traj_modes = modes_vec[type_mask]
            score = self.submodules[type_](traj_modes)
            full_actions[type_mask] = score

        return full_actions
    
    def forward(self,modes_vec,type_n = 0):

        score = self.submodules[type_n](modes_vec)

        return score



def load_types(modes_n = 20, clusteres = [],test=False):

    all_types = ['ped','bi','car']
    all_trajs_modes = []
    all_clusterers = []

    for i,current_type in enumerate(all_types):

        if test:
            yx_all = np.load(f'ind_test_{current_type}.npy')
        else:        
            yx_all = np.load(f'ind_train_{current_type}.npy')


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
    #clusterers, all_modes_z_test = load_types(modes_n=N_modes,clusteres=clusterers,test=True)


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

def main():
    
    N_modes = 20
    STEPS = 18
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    expert_states, expert_actions, expert_test_states, expert_test_actions, clusterers = load_all_mode(device,modes_n=N_modes,return_clusterers=True)

    clusterers, all_modes_z = load_types(modes_n=N_modes,clusteres=clusterers,test=False)
    clusterers, all_modes_z_test = load_types(modes_n=N_modes,clusteres=clusterers,test=True)

    batch_size = 256
    lr = 0.0003
    epochs = 120

    full_ds_size = sum([m.shape[0] for m in all_modes_z])
    steps_per_epoch = full_ds_size//batch_size


    min_reward = -1e5

    breakpoint()
    reward_model = RewardModeSequance(n_modes=N_modes, steps=STEPS).to(device)
    optimizer = optim.Adam(params=reward_model.parameters(),lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    # form train and labels tensors
    all_modes_z_torch = []
    all_mode_z_traj_val = []
    all_mode_z_traj_test = []

    for m in all_modes_z:
        mode_z_traj_ = torch.as_tensor(m,dtype=torch.float,device=device)
        all_modes_z_torch.append(mode_z_traj_[:int(mode_z_traj_.shape[0]*0.85),:])
        all_mode_z_traj_val.append(mode_z_traj_[int(mode_z_traj_.shape[0]*0.85):,:])
        #all_mode_z_traj_test.append(mode_z_traj_[int(mode_z_traj_.shape[0]*0.9):,:])

    for m in all_modes_z_test:
        all_mode_z_traj_test.append(torch.as_tensor(m,dtype=torch.float,device=device))


    print(f"""
          Batch Size: {batch_size}
          Epochs: {epochs}
          Full training size: {full_ds_size}
          Steps per epoch: {steps_per_epoch}
          Learning rate: {lr}
        """)
    max_type_array = np.argmax([m.shape[0] for m in all_modes_z_torch])
    for epoch in range(epochs):
        all_losses = 0
        reward_model.train()

        full_inputs = []
        full_labels = []

        random_modes_all = torch.vstack(tuple([torch.randint_like(all_modes_z_torch[max_type_array],0,N_modes,device=device) for _ in range((epoch//10)+3)]))
        for type_i,mode_z_traj in enumerate(all_modes_z_torch):


            with torch.no_grad():
                out_random = reward_model(random_modes_all,type_n=type_i)

            #random_modes = random_modes_all[(out_random>(out_random.sort(0)[0][-(mode_z_traj.shape[0]+1)]))[:,0]]
            random_modes = random_modes_all[out_random.sort(0)[1][-(mode_z_traj.shape[0]):].flatten(),:]

            #random_modes = torch.randint_like(mode_z_traj,0,N_modes,device=device)

            full_inputs.append(torch.vstack((mode_z_traj,random_modes)))
            full_labels.append(torch.hstack((torch.ones(mode_z_traj.shape[0]),
                                             torch.zeros(random_modes.shape[0]))).float()[:,None].to(device=device))


        for step in range(steps_per_epoch):

            loss = 0
            for type_i in range(3): # loop over types
                out = reward_model(full_inputs[type_i],type_n=type_i)
                loss = criterion(out,full_labels[type_i])

                optimizer.zero_grad()

                loss.backward()
                optimizer.step()
                    
                all_losses +=  loss.item()

            #diffxy = (expert_actions[step*batch_size:(step+1)*batch_size,:2]-out)**2
            #loss_x, loss_y = (diffxy.mean(axis=0))
            #loss = loss_x + loss_y 
    
        print(f'Epoch {epoch}: {all_losses} Loss')

        # eval
        reward_model.eval()
        with torch.no_grad():
            out_item = 0
            for type_i in range(3):
                out_eval = reward_model(all_mode_z_traj_val[type_i],type_n=type_i)
                truepositive = (torch.sigmoid(out_eval)>0.5).float().mean().item()
                out_item += out_eval.sum().item()

            #loss_z = ((out_z.argmax(axis=1)) == (expert_test_actions[:,2])).sum()/expert_test_actions.shape[0]

        print(f'Eval Reward: {out_item}')
        print(f'Eval TP: {truepositive}')
        if truepositive > min_reward:
            min_reward = truepositive
            torch.save(reward_model,f'reward_{STEPS}_steps_ind_{0}_{reward_model.n_modes}.pth')
            print('Model saved')
        
    # test
    reward_model.eval()
    with torch.no_grad():
        out_item = 0
        for type_i in range(3):
            out_test = reward_model(all_mode_z_traj_test[type_i],type_n=type_i)
            truepositive = (torch.sigmoid(out_test)>0.5).float().mean().item()
            out_item += (out_test).sum().item()

    print(f'Test Loss: {out_item} ')
    print(f'Test TP: {truepositive} ')
            
    torch.save(reward_model,f'reward_{STEPS}_steps_ind_{epochs}_{reward_model.n_modes}.pth')



if __name__ == "__main__":
    stat_model()




