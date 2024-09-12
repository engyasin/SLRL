
from distutils.util import strtobool

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.distributions.one_hot_categorical import OneHotCategoricalStraightThrough


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class AgentCNN_D(nn.Module):
    def __init__(self, envs,img_size=[10,20],clusterers=[],n_modes=20):
        super().__init__()

        self.types_n = 3
        self.clusterers = clusterers
        self.n_modes = n_modes

        self.network = nn.Sequential(
            layer_init(nn.Conv2d(3, 32, 7, stride=1)),
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
        

    def get_value(self, obs_vec, x):
        
        vec_state = self.vector_state(obs_vec)
        root = self.network(x / 255.0)
        
        critic_in = torch.hstack((root,vec_state))
        
        return self.critic(critic_in)
    
    def get_action(self,obs_vec, x,best=False):
        
        vec_state = self.vector_state(obs_vec)
        root = self.network(x / 255.0)
        
        critic_in = torch.hstack((root,vec_state))
        z_logits_ = torch.zeros((vec_state.shape[0],self.n_modes)).to(device)
        for i in range(self.types_n):
            z_logits_[obs_vec[:,8]==i] = self.actorS_z[i](critic_in[obs_vec[:,8]==i])
        
        z_dist = Categorical(logits=z_logits_)
        if best :
            return (z_dist.mode)
        return  z_dist.sample()
    
    def get_action_and_value(self,obs_vec, x, action=None):
        
        vec_state = self.vector_state(obs_vec)
        root = self.network(x / 255.0)
        
        critic_in = torch.hstack((root,vec_state))

        z_logits_ = torch.zeros((vec_state.shape[0],self.n_modes)).to(device)
        for i in range(self.types_n):
            z_logits_[obs_vec[:,8]==i] = self.actorS_z[i](critic_in[obs_vec[:,8]==i])
            
        z_dist = Categorical(logits=z_logits_)

        if action is None:
            action = (z_dist.sample())
        return  action, z_dist.log_prob(action),z_dist.entropy(),self.critic(critic_in)

