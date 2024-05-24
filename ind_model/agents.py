
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


class Discriminator(nn.Module):
    
    def __init__(self,steps=1,z=0):
        super().__init__()
        self.in_dim = int(10*steps+2+z)
        
        hidden_dim = int(2**np.ceil(np.log2(self.in_dim)))
        self.net = nn.Sequential(
            layer_init(nn.Linear(self.in_dim,hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim,8)),
            nn.ReLU(),
            layer_init(nn.Linear(8,1)),
        )
        
    
    def forward(self,x_in):
        #if more or less
        return self.net(x_in)


class made_up_clusterer():
    
    def __init__(self,expert_states,expert_actions,center=[10,6],img_size=[20,30],type_=0) -> None:
        
        self.n_modes = expert_actions[:,2].max().long()
        self.type_ = type_
        mask_type = (expert_states[:,8]==type_)

        prob_mask = np.zeros((img_size[0],img_size[1],self.n_modes+1))
        for i in range(self.n_modes+1):
            mask = mask_type*(expert_actions[:,2]==i)
            pnt_poses, probs = np.unique(np.round(expert_actions[mask][:,:2]),axis=0,return_counts=True)
            prob_mask[center[0]-pnt_poses[:,1].T.astype(int)+1,pnt_poses[:,0].T.astype(int)+center[1]+1,np.ones(pnt_poses.shape[0]).astype(int)*i] = probs/probs.sum()

        self.prob_mask = prob_mask.copy()
           
        
    def predict(self,img_obs,orgin):
        
        # img_obs (batch,20,30)
        res = (self.prob_mask[None,:,:,:]*img_obs[:,:,:,None]).sum(axis=1).sum(axis=1).copy()
        #breakpoint()
        return res


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer



class Agent_RL(nn.Module):
    def __init__(self, envs):
        super().__init__()
        
        self.max_step = 50
        self.critic = nn.Sequential(
            layer_init(nn.Linear(9, 32)),
            nn.ReLU(),
            layer_init(nn.Linear(32, 16)),
            nn.Tanh(),
            layer_init(nn.Linear(16,8)),
            nn.Tanh(),
            layer_init(nn.Linear(8, 1), std=1.0),
        )
        
        self.root = nn.Sequential(
            layer_init(nn.Linear(9, 32)),
            nn.ReLU(),
            layer_init(nn.Linear(32,64)),
            nn.ReLU(),
        )

        # min step 70*2 per axis (ped: 100)
        self.actor_x = nn.Sequential(
            layer_init(nn.Linear(64, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64,2), std=1.0),#2*self.max_step
        )
        
        self.actor_y = nn.Sequential(
            layer_init(nn.Linear(64, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64,2), std=1.0),#2*self.max_step
        )

    def get_value(self, x):
        return self.critic(x)
    
    def get_action(self, x,best=False):
        
        root = self.root(x)
        logits_x = self.actor_x(root)
        logits_y = self.actor_y(root)

        loc = torch.hstack((logits_x[:,:1],logits_y[:,:1]))
        scale = torch.hstack((logits_x[:,1:],logits_y[:,1:]))
        probs_xy = Normal(loc=loc,scale=torch.relu(scale)+1e-4)
        #action_onehot = None
        if best:
            return probs_xy.mode.clone()#.unsqueeze(1)# - self.max_step
        return  probs_xy.rsample()#.unsqueeze(1) - self.max_step
        

    def get_action_and_value(self, x, action=None):
        
        root = self.root(x)
        logits_x = self.actor_x(root)
        logits_y = self.actor_y(root)

        loc = torch.hstack((logits_x[:,:1],logits_y[:,:1]))
        scale = torch.hstack((logits_x[:,1:],logits_y[:,1:]))
        probs_xy = Normal(loc=loc,scale=torch.relu(scale)+1e-4)
        #action_onehot = None
        if action is None:
            action = probs_xy.rsample()#.unsqueeze(1)# - self.max_step
        return  action,probs_xy.log_prob(action).sum(axis=1),probs_xy.entropy(),self.critic(x)# - self.max_step


class AgentCNN_Z(nn.Module):
    def __init__(self, envs, img_size=[10,20],clusterers=[]):
        super().__init__()
        self.network_z = nn.Sequential(
            layer_init(nn.Conv2d(3, 32, 7, stride=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 8, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear((img_size[0]-8)*(img_size[1]-8) * 8, 16)),
            nn.ReLU(),
        )
        
        self.network_p = nn.Sequential(
            layer_init(nn.Conv2d(3, 32, 5, stride=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 8, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear((10-6)*(8-6)*8, 16)),
            nn.ReLU(),
        )
        
        self.vector_state = nn.Sequential(
            layer_init(nn.Linear(10,16)),
            nn.ReLU(),
        )
        self.actor_x = nn.Sequential(
            layer_init(nn.Linear(32+16,8)),
            nn.ReLU(),
            layer_init(nn.Linear(8,2), std=1.0),
        )
        
        self.actor_y = nn.Sequential(
            layer_init(nn.Linear(32+16,8)),
            nn.ReLU(),
            layer_init(nn.Linear(8,2), std=1.0),
        )
        
        self.critic = nn.Sequential(
            layer_init(nn.Linear(16+16,8)),
            nn.ReLU(),
            layer_init(nn.Linear(8,8)),
            nn.Tanh(),
            layer_init(nn.Linear(8,1), std=1.0),
        )
        
        self.types_masks = torch.Tensor([[1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1]]).to(device)
        
        self.center = [int(img_size[0]/2),int(img_size[1]/15)]# NOTE ratio fixed
        
        self.clusterers = clusterers
        self.n_modes = 5

    def get_value(self, obs_vec, x):
        vec_state = self.vector_state(obs_vec)
        root = self.network_z(x / 255.0)
        
        critic_in = torch.hstack((root,vec_state))
        
        return self.critic(critic_in)
    
    def modify_z(self,obs_vec, new_img_state):

        # type mask
        mask_type = torch.functional.F.one_hot(obs_vec[:,8].long(),num_classes=3).repeat_interleave(self.n_modes).reshape(obs_vec.shape[0],-1)
        new_arr = (new_img_state.bool().sum(axis=1)>1)
        possible_goals = torch.vstack(torch.where(new_arr)).T-torch.Tensor([0,self.center[1],self.center[0]]).to(device)
        all_z_labels = []
        for j,c_type in enumerate(obs_vec[:,8]):
            clusterer = self.clusterers[c_type.int()]
            if (possible_goals[:,0]==j).sum()==0:
                all_z_labels.append(mask_type[j,:])
                continue
            preds = torch.Tensor(clusterer.predict((possible_goals[possible_goals[:,0]==j][:,1:]).cpu())).to(device)
            z_labels = torch.unique(preds) + (c_type*self.n_modes)
            all_z_labels.append(torch.functional.F.one_hot(z_labels.long(),num_classes=self.n_modes*3).sum(axis=0))
        mask_road = torch.hstack((torch.vstack(all_z_labels),mask_type[:,-1:]))
        
        return torch.log(mask_road)

    def get_action(self,obs_vec, x,best=False):
        
        z_logits_ = self.network_z(x / 255.0)
        z_logits = z_logits_+self.modify_z(obs_vec,x)
        
        z_dist = OneHotCategoricalStraightThrough(logits=z_logits)
        if best:
            z = z_dist.mode.clone()
        else:
            z = z_dist.rsample()
        
        agent_img = x[:,:,self.center[1]-2:self.center[1]+8,self.center[0]-4:self.center[0]+4]
        root_ = self.network_p(agent_img / 255.0)
        
        vec_state = self.vector_state(obs_vec)
        root = torch.hstack((root_,z,vec_state))
        
        logits_x = self.actor_x(root)
        logits_y = self.actor_y(root)
        loc = torch.hstack((logits_x[:,:1],logits_y[:,:1]))
        scale = torch.hstack((logits_x[:,1:],logits_y[:,1:]))
        probs_xy = Normal(loc=loc,scale=torch.relu(scale)+1e-4)
        if best:
            return torch.hstack((probs_xy.mode,z))#.unsqueeze(1)# - self.max_step
        return  torch.hstack((probs_xy.rsample(),z))#.unsqueeze(1) - self.max_step
    
    def get_action_and_value(self,obs_vec, x, action=None):
        
        z_logits_ = self.network_z(x / 255.0)
        mask = torch.functional.F.one_hot(obs_vec[:,8].long(),num_classes=3).repeat_interleave(5).reshape(obs_vec.shape[0],-1)
        z_logits = z_logits_ + self.modify_z(obs_vec,x)
        
        
        z_dist = OneHotCategoricalStraightThrough(logits=z_logits)
        z = z_dist.rsample()

        agent_img = x[:,:,self.center[1]-2:self.center[1]+8,self.center[0]-4:self.center[0]+4]
        root_ = self.network_p(agent_img / 255.0)
        
        vec_state = self.vector_state(obs_vec)
        root = torch.hstack((root_,z,vec_state))
        critic_in = torch.hstack((z_logits_,vec_state))
        
        logits_x = self.actor_x(root)
        logits_y = self.actor_y(root)
        loc = torch.hstack((logits_x[:,:1],logits_y[:,:1]))
        scale = torch.hstack((logits_x[:,1:],logits_y[:,1:]))
        probs_xy = Normal(loc=loc,scale=torch.relu(scale)+1e-4)
        if action is None:
            action = torch.hstack((probs_xy.rsample(),z))
        return  action, probs_xy.log_prob(action[:,:2]).sum(axis=1)+z_dist.log_prob(action[:,2:]),probs_xy.entropy().sum(axis=1)+z_dist.entropy()*3,self.critic(critic_in)

    def set_clusterers(self,clusterers):
        self.clusterers = clusterers


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
            layer_init(nn.Linear(10,16)),
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
        z_logits_ = torch.zeros((vec_state.shape[0],self.n_modes)).to(device)
        for i in range(self.types_n):
            z_logits_[obs_vec[:,8]==i] = self.actorS_z[i](critic_in[obs_vec[:,8]==i])
        
        z_logits = z_logits_ #+ self.modify_z(obs_vec,x)
        z_dist = Categorical(logits=z_logits)
        if best :
            return (z_dist.mode)
        return  z_dist.sample()#.unsqueeze(1) - self.max_step
    

    def modify_z(self,obs_vec, new_img_state):

        # type mask
        #new_arr = (new_img_state.bool().sum(axis=1)>1).cpu().numpy()
        new_arr = (new_img_state.sum(axis=1)>255).cpu().numpy()
        all_z_labels = np.zeros((obs_vec.shape[0],self.n_modes))
        for t in range(3):
            type_mask = (obs_vec[:,8]==t)
            all_z_labels[type_mask] = self.clusterers[t].predict(new_arr[type_mask].transpose(0,2,1),new_img_state[type_mask].cpu().numpy())>0
        all_z_labels[np.logical_not(all_z_labels.any(axis=1))] += 1

        return torch.log(torch.Tensor(all_z_labels).to(device=device).float())

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




class AgentCNN(nn.Module):
    def __init__(self, envs,img_size=[10,20]):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(3, 32, 5, stride=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 8, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear((img_size[0]-6)*(img_size[1]-6)* 8, 16)),
            nn.ReLU(),
        )
        
        self.actor_x = nn.Sequential(
            layer_init(nn.Linear(16,8)),
            nn.ReLU(),
            layer_init(nn.Linear(8,2), std=1.0),
        )
        
        self.actor_y = nn.Sequential(
            layer_init(nn.Linear(16,8)),
            nn.ReLU(),
            layer_init(nn.Linear(8,2), std=1.0),
        )
        
        self.critic = nn.Sequential(
            layer_init(nn.Linear(16,8)),
            nn.ReLU(),
            layer_init(nn.Linear(8,8)),
            nn.Tanh(),
            layer_init(nn.Linear(8,1), std=1.0),
        )
        

    def get_value(self, dummy, x):
        return self.critic(self.network(x / 255.0))
    
    def get_action(self,dummy, x,best=False):
        
        root = self.network(x / 255.0)

        logits_x = self.actor_x(root)
        logits_y = self.actor_y(root)
        loc = torch.hstack((logits_x[:,:1],logits_y[:,:1]))
        scale = torch.hstack((logits_x[:,1:],logits_y[:,1:]))
        probs_xy = Normal(loc=loc,scale=torch.relu(scale)+1e-4)
        if best:
            return probs_xy.mode.clone()#.unsqueeze(1)# - self.max_step
        return  probs_xy.rsample()#.unsqueeze(1) - self.max_step
    
    def get_action_and_value(self,dummy, x, action=None):
        
        root = self.network(x / 255.0)
        logits_x = self.actor_x(root)
        logits_y = self.actor_y(root)
        loc = torch.hstack((logits_x[:,:1],logits_y[:,:1]))
        scale = torch.hstack((logits_x[:,1:],logits_y[:,1:]))
        probs_xy = Normal(loc=loc,scale=torch.relu(scale)+1e-4)
        if action is None:
            action = probs_xy.rsample()
        return  action, probs_xy.log_prob(action).sum(axis=1),probs_xy.entropy(),self.critic(root)



class Agent_lstm(Agent_RL):
    
    def __init__(self, envs):
        super().__init__(envs)
        
        self.lstm = nn.LSTM(64,64)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

        self.critic = nn.Sequential(
            layer_init(nn.Linear(64, 16)),
            nn.Tanh(),
            layer_init(nn.Linear(16,8)),
            nn.Tanh(),
            layer_init(nn.Linear(8, 1), std=1.0),
        )
        
    def get_states(self, x, lstm_state, done):
        hidden = self.root(x)

        # LSTM logic
        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state
    

    def get_value(self, x, lstm_state, done):
        hidden, _ = self.get_states(x, lstm_state, done)
        return self.critic(hidden)

    def get_action_and_value(self, x, lstm_state, done, action=None):
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        
        logits_x = self.actor_x(hidden)
        logits_y = self.actor_y(hidden)
        loc = torch.hstack((logits_x[:,:1],logits_y[:,:1]))
        scale = torch.hstack((logits_x[:,1:],logits_y[:,1:]))
        probs_xy = Normal(loc=loc,scale=torch.relu(scale)+1e-4)
        
        if action is None:
            action = probs_xy.rsample()#.unsqueeze(1)# - self.max_step
        return  action,probs_xy.log_prob(action).sum(axis=1),probs_xy.entropy(),self.critic(hidden), lstm_state# - self.max_step

    def get_action(self, x,lstm_state, done,best=False):
    
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        
        logits_x = self.actor_x(hidden)
        logits_y = self.actor_y(hidden)

        loc = torch.hstack((logits_x[:,:1],logits_y[:,:1]))
        scale = torch.hstack((logits_x[:,1:],logits_y[:,1:]))
        probs_xy = Normal(loc=loc,scale=torch.relu(scale)+1e-4)
        if best:
            return probs_xy.mode.clone()#.unsqueeze(1)# - self.max_step
        return  probs_xy.rsample()#.unsqueeze(1) - self.max_step