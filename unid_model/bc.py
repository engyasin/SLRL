
import numpy as np

import torch
from torch import nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.distributions.one_hot_categorical import OneHotCategoricalStraightThrough

from utils import load_all_mode


#val_in_set, val_out_set, train_discs = load_val(device)
# env setup

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
            best_modes = Categorical(logits=z_logits).sample()
            #best_modes = z_logits.argmax(axis=1)
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



def main():
    
    N_modes = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    expert_states, expert_actions, expert_test_states, expert_test_actions, clusterers = load_all_mode(device,modes_n=N_modes,return_clusterers=True)
    batch_size = 512
    lr = 0.0001
    epochs = 100
    steps_per_epoch = expert_states.shape[0]//batch_size

    val_split = 20000
    min_loss = 1e5
    agent = AgentCNN_Z_BC_MB(None,n_modes=N_modes).to(device)
    optimizer = optim.Adam(params=agent.parameters(),lr=lr)
    criterion_z = nn.BCEWithLogitsLoss()
    
    for epoch in range(epochs):
        all_losses = 0
        agent.train()
        for step in range(steps_per_epoch):
            #out_z = agent.get_z(expert_states[step*batch_size:(step+1)*batch_size,:10],best=False)
            z_label = torch.nn.functional.one_hot((expert_actions[step*batch_size:(step+1)*batch_size,2]).long(),
                                                  num_classes=agent.n_modes)
            out,_ = agent.get_action(expert_states[step*batch_size:(step+1)*batch_size,:10],z_logits=z_label.clone(),best=False)
            loss = torch.sqrt(((expert_actions[step*batch_size:(step+1)*batch_size,:2]-out)**2).sum(axis=1)).mean()

            #loss_z = criterion_z(out_z,z_label.float())

            optimizer.zero_grad()
                
            (loss).backward()
                
            optimizer.step()
                
            all_losses +=  loss.item()
    
        print(f'Epoch {epoch}: {all_losses} Loss')

        agent.eval()
        with torch.no_grad():
            out,_  = agent.get_action(expert_test_states[:val_split,:10],best=True)
            loss_ = torch.sqrt(((expert_test_actions[:val_split,:2]-out)**2).sum(axis=1))
            loss = loss_.mean()
            #loss_z = ((out_z.argmax(axis=1)) == (expert_test_actions[:,2])).sum()/expert_test_actions.shape[0]
        print(f'Eval Loss: {loss.item()} ')
        if min_loss > loss.item():
            min_loss = loss.item()
            torch.save(agent,f'bc_agent_unid_{epochs}_{agent.n_modes}_smoothed_one_step_new.pth')
            print('Model saved')
    agent.eval()
    with torch.no_grad():
        out,out_z = agent.get_action(expert_test_states[val_split:,:10],best=True)
        loss_ = torch.sqrt(((expert_test_actions[val_split:,:2]-out)**2).sum(axis=1))
        loss = loss_.mean()
        #loss_z = ((out_z.argmax(axis=1)) == (expert_test_actions[:,2])).sum()/expert_test_actions.shape[0]

    print(f'Test Loss: {loss.item()} ')
    #print(f'Test Z Acc: {loss_z} ')
            
    torch.save(agent,f'bc_agent_unid_{epochs}_{agent.n_modes}_smoothed_one_step_new_last.pth')



if __name__ == "__main__":
    main()

