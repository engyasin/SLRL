
import numpy as np

import torch
from torch import nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.distributions.one_hot_categorical import OneHotCategoricalStraightThrough
from agents import SLAgent

from utils import load_all_mode



def main():
    
    N_modes = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    expert_states, expert_actions, expert_test_states, expert_test_actions, clusterers = load_all_mode(device,modes_n=N_modes,return_clusterers=True)

    batch_size = 256
    lr = 0.0005
    epochs = 20
    steps_per_epoch = expert_states.shape[0]//batch_size

            
    min_loss = 1e5

    agent = SLAgent(None,n_modes=N_modes).to(device)
    optimizer = optim.Adam(params=agent.parameters(),lr=lr)
    criterion_z = nn.BCEWithLogitsLoss()
    
    for epoch in range(epochs):
        all_losses = 0
        agent.train()
        for step in range(steps_per_epoch):
            out_z = agent.get_z(expert_states[step*batch_size:(step+1)*batch_size,:10],best=False)
            z_label = torch.nn.functional.one_hot((expert_actions[step*batch_size:(step+1)*batch_size,2]).long(),
                                                  num_classes=agent.n_modes)

            out,_ = agent.get_action(expert_states[step*batch_size:(step+1)*batch_size,:10],z_logits=z_label.clone(),best=False)

            #diffxy = (expert_actions[step*batch_size:(step+1)*batch_size,:2]-out)**2
            #loss_x, loss_y = (diffxy.mean(axis=0))
            #loss = loss_x + loss_y 
            loss = torch.sqrt(((expert_actions[step*batch_size:(step+1)*batch_size,:2]-out)**2).sum(axis=1)).mean()
            loss_z = criterion_z(out_z,z_label.float())

            optimizer.zero_grad()
                
            (loss+loss_z*0.5).backward()
                
            optimizer.step()
                
            all_losses +=  loss.item()
    
        print(f'Epoch {epoch}: {all_losses} Loss')

        # eval
        agent.eval()
        with torch.no_grad():
            z_label = torch.nn.functional.one_hot((expert_test_actions[::2,2]).long(),
                                                  num_classes=agent.n_modes)
            out,_  = agent.get_action(expert_test_states[::2,:10],best=True)
            loss_ = torch.sqrt(((expert_test_actions[::2,:2]-out)**2).sum(axis=1))
            loss = loss_.mean()
            #loss_z = ((out_z.argmax(axis=1)) == (expert_test_actions[:,2])).sum()/expert_test_actions.shape[0]
        print(f'Eval Loss: {loss.item()} ')
        if min_loss > loss.item():
            min_loss = loss.item()
            torch.save(agent,f'bc_agent_ind_{epochs}_{agent.n_modes}_kmeans.pth')
            print('Model saved')
        
    # test
    agent.eval()
    with torch.no_grad():
        out,out_z = agent.get_action(expert_test_states[1::2,:10],best=True)
        loss_ = torch.sqrt(((expert_test_actions[1::2,:2]-out)**2).sum(axis=1))
        loss = loss_.mean()
        loss_z = ((out_z.argmax(axis=1)) == (expert_test_actions[1::2,2])).sum()/expert_test_actions[1::2,:].shape[0]

    print(f'Test Loss: {loss.item()} ')
    print(f'Test Z Acc: {loss_z} ')
            
    torch.save(agent,f'bc_agent_ind_{epochs}_{agent.n_modes}_kmeans_last.pth')



if __name__ == "__main__":
    main()

