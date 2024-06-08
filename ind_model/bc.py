
import numpy as np

import torch
from torch import nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.distributions.one_hot_categorical import OneHotCategoricalStraightThrough

from utils import load_all_mode



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class AgentCNN_Z_BC_MB(nn.Module):
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
            layer_init(nn.Linear(8,2), std=1.0),
            ) for _ in range(n_modes)])
        
        self.actor_y = nn.ModuleList([nn.Sequential(
            layer_init(nn.Linear(32,8)),
            nn.ReLU(),
            layer_init(nn.Linear(8,2), std=1.0),
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
            best_modes = z_logits.argmax(axis=1)
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

class Agent_BC_MB(nn.Module):
    def __init__(self, envs, img_size=[10,20], n_modes = 16):
        super().__init__()
        
        self.vector_state_root = nn.Sequential(
            layer_init(nn.Linear(10,32)),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.actor_x = nn.ModuleList([nn.Sequential(
            layer_init(nn.Linear(32,16)),
            nn.ReLU(),
            layer_init(nn.Linear(16,2), std=1.0),
            ) for _ in range(n_modes)])
        
        self.actor_y = nn.ModuleList([nn.Sequential(
            layer_init(nn.Linear(32,16)),
            nn.ReLU(),
            layer_init(nn.Linear(16,2), std=1.0),
            ) for _ in range(n_modes)])
        
        self.n_modes = n_modes
        self.center = [int(img_size[0]/2),int(img_size[1]/15)]# NOTE ratio fixed
        
    def get_z(self,obs_vec,best=False):
        
        z_logits = torch.zeros((obs_vec.shape[0],self.n_modes)).to(device)
        return z_logits

    def get_action(self,obs_vec,z_logits=None,best=False):
        
        
        #best_modes = z_logits.argmax(axis=1)
        actions = torch.zeros((z_logits.shape[0],2)).to(device)
        vec_state = self.vector_state_root(obs_vec)
        for mode in range(self.n_modes):
            mask = (z_logits==mode)
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


class AgentClustererAll(nn.Module):
    def __init__(self, n_modes = 16, device=torch.cuda):
        super().__init__()

        self.types = 3
        self.submodules = nn.ModuleList([Agent_BC_MB(None,n_modes=n_modes)#.to(device=device) 
                           for _ in range(self.types)])

        self.n_modes = n_modes
        
    def get_action(self,obs_vec,z_logits=None,best=False):

        full_actions = torch.zeros_like(obs_vec[:,:2])
        #full_best_modes = full_actions[:,0].clone().long()
        for type_ in range(self.types):
            type_mask = (obs_vec[:,8] == type_)

            states = obs_vec[type_mask]
            z_logits_type = z_logits[type_mask]
            actions , _ = self.submodules[type_].get_action(states,z_logits=z_logits_type)
            full_actions[type_mask] = actions

        return full_actions, _

    

def main():
    
    N_modes = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    expert_states, expert_actions, expert_test_states, expert_test_actions, clusterers = load_all_mode(device,modes_n=N_modes,return_clusterers=True)

    batch_size = 256
    lr = 0.0001
    epochs = 100
    steps_per_epoch = expert_states.shape[0]//batch_size

            
    min_loss = 1e5
    val_split = 2000

    agent = AgentClustererAll(n_modes=N_modes).to(device)
    optimizer = optim.Adam(params=agent.parameters(),lr=lr)
    criterion = nn.MSELoss()
    #criterion_z = nn.BCEWithLogitsLoss()
    
    for epoch in range(epochs):
        all_losses = 0
        agent.train()
        for step in range(steps_per_epoch):

            z_label = torch.nn.functional.one_hot((expert_actions[step*batch_size:(step+1)*batch_size,2]).long(),
                                                  num_classes=agent.n_modes)

            out,_ = agent.get_action(expert_states[step*batch_size:(step+1)*batch_size,:10],z_logits=z_label.argmax(axis=1).clone(),best=False)

            #diffxy = (expert_actions[step*batch_size:(step+1)*batch_size,:2]-out)**2
            #loss_x, loss_y = (diffxy.mean(axis=0))
            #loss = loss_x + loss_y 
            loss = torch.sqrt(((expert_actions[step*batch_size:(step+1)*batch_size,:2]-out)**2).sum(axis=1)).mean()

            optimizer.zero_grad()
                
            (loss).backward()
                
            optimizer.step()
                
            all_losses +=  loss.item()
    
        print(f'Epoch {epoch}: {all_losses} Loss')

        # eval
        agent.eval()
        with torch.no_grad():
            z_label = torch.nn.functional.one_hot((expert_test_actions[:val_split,2]).long(),
                                                  num_classes=agent.n_modes)
            out,best_modes = agent.get_action(expert_test_states[:val_split,:10],best=True, z_logits=z_label.argmax(axis=1))
            loss_x, loss_y = abs(((expert_test_actions[:val_split,:2]-out))).mean(axis=0)
            loss_item = loss_x.item() + loss_y.item()
            #loss_z = ((out_z.argmax(axis=1)) == (expert_test_actions[:,2])).sum()/expert_test_actions.shape[0]

        print(f'Eval Lossx: {loss_x.item()}, Lossy: {loss_y.item()}')
        if loss_item < min_loss:
            min_loss = loss_item
            torch.save(agent,f'bc_agent_ind_{epochs}_{agent.n_modes}_kmeans.pth')
            print('Model saved')
        
    # test
    agent.eval()
    with torch.no_grad():
        z_label = torch.nn.functional.one_hot((expert_test_actions[val_split:,2]).long(),
                                                num_classes=agent.n_modes)
        out,best_modes = agent.get_action(expert_test_states[val_split:,:10],best=True, z_logits=z_label.argmax(axis=1))
        loss_ = torch.sqrt(((expert_test_actions[val_split:,:2]-out)**2).sum(axis=1))
        loss = loss_.mean()

    print(f'Test Loss: {loss.item()} ')
            
    torch.save(agent,f'bc_agent_ind_{epochs}_{agent.n_modes}_kmeans_last.pth')



if __name__ == "__main__":
    main()

