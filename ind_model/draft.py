import numpy as np
import cv2
from utils import load
import torch
from sklearn.cluster import KMeans


def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #val_in_set, val_out_set, train_discs = load_val(device)
    # env setup
    expert_states,expert_actions,expert_test_states,expert_test_actions = [], [], [], []
    
    for type_ in range(4):
        expert_states_,expert_actions_,expert_test_states_,expert_test_actions_ = load(type_=type_)
        
        #remove zeros actions only from cars and trucks>1
        if type_>1:
            expert_states_ = expert_states_[(abs(expert_actions_)>2e-2).any(axis=1)]
            expert_actions_ = expert_actions_[(abs(expert_actions_)>2e-2).any(axis=1)]
            expert_test_states_ = expert_test_states_[(abs(expert_test_actions_)>2e-2).any(axis=1)]
            expert_test_actions_ = expert_test_actions_[(abs(expert_test_actions_)>2e-2).any(axis=1)]

        print(f'type: {type_}')
        print(expert_states_.shape[0])
        print(expert_test_states_.shape[0])
        
        modes_n = 5
        if type_!=3:

            clusterer = KMeans(n_clusters=modes_n,random_state=42).fit(expert_actions_[:,2:])#,n_init=10
            expert_actions_ = (np.hstack((expert_actions_,clusterer.labels_[:,None]+(modes_n*type_))))#.to(device)
            expert_test_actions_ = (np.hstack((expert_test_actions_,clusterer.predict(expert_test_actions_[:,2:])[:,None]+(modes_n*type_))))#.to(device)
        else:
            #truck has no modes
            clurster_labels = np.zeros(expert_actions.shape[0],1)+(modes_n*type_)
            expert_actions_ = (np.hstack((expert_actions_,clurster_labels)))#.to(device)
            expert_test_actions_ = (np.hstack((expert_test_actions_,clurster_labels)))#.to(device)
        
        expert_states.append(expert_states_)
        expert_actions.append(expert_actions_)
        expert_test_states.append(expert_test_states_)
        expert_test_actions.append(expert_test_actions_)
        
    expert_states = (np.vstack(expert_states))#.to(device)
    expert_actions = (np.vstack(expert_actions))#.to(device)
    expert_test_states = (np.vstack(expert_test_states))#.to(device)
    expert_test_actions = (np.vstack(expert_test_actions))#.to(device)
    

    




if __name__ == '__main__':
    
    
    main()