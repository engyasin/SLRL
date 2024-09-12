
from distutils.util import strtobool

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.distributions.one_hot_categorical import OneHotCategoricalStraightThrough
#from matplotlib import pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer



class RLAgent(nn.Module):

    def __init__(self, envs,img_size=[10,20],clusterers=[],n_modes=20):
        super().__init__()

        self.types_n = 3
        self.clusterers = clusterers
        self.n_modes = n_modes

        self.network = nn.Sequential(
            layer_init(nn.Conv2d(3*1, 32, 7, stride=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 8, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear((img_size[0]-8)*(img_size[1]-8) * 8, 16)),
            nn.ReLU(),
        )
        
        self.vector_state = nn.Sequential(
            layer_init(nn.Linear(10+envs.traj_mode_len,16)),
            nn.ReLU(),
        )
        
        self.actorS_z = nn.ModuleList([
            nn.Sequential(
                layer_init(nn.Linear(16+16,32)),
                nn.ReLU(),
                layer_init(nn.Linear(32,self.n_modes), std=1.0),
                ) for _ in range(self.types_n)])
        
        
        self.critic = nn.Sequential(
            layer_init(nn.Linear(16+16,8)),
            nn.ReLU(),
            layer_init(nn.Linear(8,8)),
            nn.Tanh(),
            layer_init(nn.Linear(8,1), std=1.0),
        )
        
        self.center = [int(img_size[0]/2),int(img_size[1]/5)]# NOTE ratio fixed


    def get_value(self, obs_vec, x):
        
        vec_state = self.vector_state(obs_vec)
        root = self.network(x / 255.0)
        
        critic_in = torch.hstack((root,vec_state))
        
        return self.critic(critic_in)
    
    def get_action(self,obs_vec, x,best=False):
        
        vec_state = self.vector_state(obs_vec)
        root = self.network(x / 255.0)
        
        critic_in = torch.hstack((root,vec_state))
        z_logits = torch.zeros((vec_state.shape[0],self.n_modes)).to(device)
        for i in range(self.types_n):
            z_logits[obs_vec[:,8]==i] = self.actorS_z[i](critic_in[obs_vec[:,8]==i])

        z_dist = Categorical(logits=z_logits)
        if best :
            return (z_dist.mode)
        return  z_dist.sample()#.unsqueeze(1) - self.max_step
    
    def get_action_and_value(self,obs_vec, x, action=None):
        
        vec_state = self.vector_state(obs_vec)
        root = self.network(x / 255.0)
        
        critic_in = torch.hstack((root,vec_state))

        z_logits_ = torch.zeros((vec_state.shape[0],self.n_modes)).to(device)
        for i in range(self.types_n):
            z_logits_[obs_vec[:,8]==i] = self.actorS_z[i](critic_in[obs_vec[:,8]==i])
            
        #z_logits *= self.types_masks.clone()[obs_vec[:,8].long(),:]
        z_logits = z_logits_ #+ self.modify_z(obs_vec,x)
        z_dist = Categorical(logits=z_logits)

        if action is None:
            action = (z_dist.sample())
        return  action, z_dist.log_prob(action),z_dist.entropy(),self.critic(critic_in)



class SLAgent(nn.Module):
    def __init__(self, envs, img_size=[10,20], n_modes = 16):
        super().__init__()
        
        self.types = 3
        self.vector_state_root = nn.Sequential(
            layer_init(nn.Linear(10,32)),
            nn.ReLU(),
        )
        self.actor_x = nn.ModuleList([nn.Sequential(
            layer_init(nn.Linear(32,8)),
            nn.ReLU(),
            layer_init(nn.Linear(8,2), std=2.0),
            ) for _ in range(n_modes)])
        
        self.actor_y = nn.ModuleList([nn.Sequential(
            layer_init(nn.Linear(32,8)),
            nn.ReLU(),
            layer_init(nn.Linear(8,2), std=2.0),
            ) for _ in range(n_modes)])
        
        self.z_model = nn.ModuleList([nn.Sequential(
            layer_init(nn.Linear(10,32)),
            nn.ReLU(),
            layer_init(nn.Linear(32,16)),
            nn.ReLU(),
            layer_init(nn.Linear(16,n_modes), std=1.0),
        ) for _ in range(self.types)])

        self.n_modes = n_modes
        self.center = [int(img_size[0]/2),int(img_size[1]/15)]# NOTE ratio fixed
        
    def get_z(self,obs_vec,best=False):
        
        z_logits = torch.zeros((obs_vec.shape[0],self.n_modes)).to(device)
        for i in range(self.types):
            mask = (obs_vec[:,8]==i)
            z_logits[mask] = self.z_model[i](obs_vec[mask])

        return z_logits

    def get_action(self,obs_vec,z_logits=None,best=False):
        
        

        if z_logits is None:
            z_logits = self.get_z(obs_vec)
        if len(z_logits.shape)>1:
            if best:
                best_modes = z_logits.argmax(axis=1)
            else:
                best_modes = Categorical(logits=z_logits).mode
        else:
            best_modes = z_logits.clone() # index vector
            
        actions = torch.zeros((z_logits.shape[0],2)).to(device)
        vec_state = self.vector_state_root(obs_vec)
        for mode in range(self.n_modes):
            mask = (best_modes==mode)
            logits_x = self.actor_x[mode](vec_state[mask])
            logits_y = self.actor_y[mode](vec_state[mask])
        
            loc = torch.hstack((logits_x[:,:1],logits_y[:,:1]))
            scale = torch.hstack((logits_x[:,1:],logits_y[:,1:]))
            probs_xy = Normal(loc=loc,scale=torch.relu(scale)+1e-8)
            
            if best:
                actions[mask] = probs_xy.mode
            else:
                actions[mask] = probs_xy.rsample()
                
        return  (actions),z_logits#.unsqueeze(1) - self.max_step



