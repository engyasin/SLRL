
from numpy.random.mtrand import randint as randint
import pandas as pd
import numpy as np
from utils import *
from matplotlib import pyplot as plt

from agents import AgentCNN_D as Agent
from bc import AgentCNN_Z_BC_MB

from sklearn.cluster import KMeans

import glob,os
import os
import cv2

from utils import load_all_mode

import torch


from trafficenv_D import TrafficEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TrafficEnvMod(TrafficEnv):
    
    def __init__(self,w=182,h=114,num_agents = 15 + np.random.randint(15),max_steps=17,pixel2meter=None,make_img=False,img_size=[10,10],n_modes=20):
        self.speed_record = []
        super().__init__(w, h, num_agents, max_steps, pixel2meter, make_img, img_size,n_modes)
   
    def load_ds_scene(self):
        img_file = 'background_img.png'

        self.bg_img = cv2.imread(img_file)

        h,w,_ = self.bg_img.shape
        bg_img = cv2.resize(self.bg_img,dsize=[int(w),int(h)])
        self.road,self.sidewalk = bg_img.copy(),bg_img.copy()
        self.ports = np.array([[139,52],[113,73],[115,117],[127,106],[159,70],[82,124],[163,46]])
        self.w,self.h = bg_img.shape[1],bg_img.shape[0]
        
        self.wh = [self.w,self.h]
        
        self.time = 0
        self.heading_img = self.sidewalk.copy()/2
        
        

    def reset(self, num_agents = 15 + np.random.randint(15),max_steps=17,pixel2meter=None):

        

        if len(self.speed_record):
            self.all_statics = self.sum_statics()
        self.load_ds_scene()
        
        self.grid = np.zeros((self.h,self.w,3)) # 1 pixel = 1 meter
        
        self.num_agents = num_agents
        self.poses = (np.random.random((self.num_agents,2))*np.array([self.w,self.h]))#.astype(int)
        
        types_probs = np.random.rand(self.num_agents)
        self.types = (types_probs<0.18).astype(int)+(types_probs<0.25) # 0: Pedestrain, 1: Biker 2: Cars
        #### 
        vru_poses = np.vstack(np.where(self.sidewalk[:,:,1]>0)[::-1]).T
        nonvru_poses = np.vstack(np.where(self.road[:,:,1]>0)[::-1]).T
        np.random.shuffle(vru_poses)
        np.random.shuffle(nonvru_poses)
        
        self.poses[self.types<2] = vru_poses[:(self.types<2).sum()]
        self.poses[self.types>1] = nonvru_poses[:(self.types>1).sum()]
        self.zs = self.types.copy()
        
        self.headings = ((np.random.rand(self.num_agents)))*np.pi*2 # radian
        self.starting_port = np.linalg.norm((self.poses - self.ports[:,None,:]).reshape(-1,2),axis=1).reshape(len(self.ports),-1).argmin(axis=0)        

        self.speed_record = []
        self.colliding_record = np.zeros_like(self.headings)
        self.speeds = abs((np.random.rand(self.num_agents,1)-0.5)*2)
        self.speeds += (self.types[:,None])*5 # bikes and cars faster on y
        
        self.velocity = np.vstack((np.cos(self.headings),np.sin(self.headings))).T*self.speeds
        self.acceleration = np.zeros_like(self.speeds)
        self.old_velocity = self.velocity.copy()
        
        self.max_time = max_steps
        
        self.time = 0
        
        #self.max_time = 120
        
        self.rs = 0
        
        self.near_objs = self.update_near_objs()
        
        if self.make_img:
            self.make_image()
            
            self.imgs_states = np.array([self.get_agent_image(i) for i in range(self.num_agents)])
        
        self.states = np.hstack((abs(self.speeds*2.5),(self.headings[:,None]%(2*np.pi)),self.near_objs,self.zs[:,None],self.acceleration))
        self.history = []
        self.history.append(self.states.copy())
        self.history_poses = []
        self.history_poses.append(self.poses.copy())       
        self.history_z = []

        return [False for _ in range(self.num_agents)], self.states, self.imgs_states
    

    def rewards_test(self,actions,z):
        
        return np.zeros_like(self.speeds)
    
    
    def find_statics(self):
        
        
        self.speed_record.append(self.speeds.copy())
        distances = np.linalg.norm(self.poses - self.poses[:,None,:],axis=2)
        revert_self = np.logical_not(np.eye(self.num_agents).astype(bool))
        is_colliding = np.logical_not((((distances<0.5)*revert_self).sum(axis=1)[:,None]))
        self.colliding_record += (is_colliding.flatten() * (self.colliding_record == self.colliding_record.max()))
        

    def sum_statics(self):
        
        sucess_rate = (self.colliding_record[self.types>1] > (self.time-2)).mean()
        avg_survival = []

        all_speeds = np.array(self.speed_record).mean(axis=0)
        all_speeds_avgs = []
        for t in range(3):
            avg_survival.append(self.colliding_record[(self.types==t)].mean())
            all_speeds_avgs.append(all_speeds[(self.types==t)].mean())
            
        return sucess_rate,avg_survival,all_speeds_avgs
            
        
def main():
    N_MODES=20
    Espisode_length = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = TrafficEnvMod(make_img=True,num_agents=64,img_size=[20,30],n_modes=N_MODES)#[12,20]
    
    model = torch.load('ppo_agent_unid_image_d_last_step_1.pth',map_location=device)
    
    sucess_r, avg_surv, avg_speed = [],[],[]
    for scene in range(64):
        done,new_state,new_img_state = env.reset(num_agents=24)# TODO check n agents
        if type(done)==list:
            done = done[0]
        env.max_time = Espisode_length
        all_rewards = 0
        while not(done):
            with torch.no_grad():
                action = model.get_action(torch.Tensor(new_state).to(device),torch.Tensor(new_img_state).to(device),best=True).cpu().numpy()#*np.array([0.025,0])
                #action = env.bc_models.get_z(torch.Tensor(new_state).to(device)).cpu().numpy()#.argmax(axis=1)

            new_state, rewards, done,info,new_img_state = env.step(action)#-env.poses)
            env.find_statics()

            all_rewards += rewards
            if type(done)==list:
                done = done[0]
            #env.make_image()
            #env.render()
            #cv2.waitKey(100)
        #print(env.all_statics)#sum_statics())
        if env.all_statics[0] is not np.nan:
            sucess_r.append(np.nan_to_num(env.all_statics[0]))
        if np.nan not in env.all_statics[1][:]:
            avg_surv.append(np.nan_to_num(env.all_statics[1][:]))
        if np.nan not in env.all_statics[2][:]:
            #print(env.all_statics[2][:])
            avg_speed.append(np.nan_to_num(env.all_statics[2][:]))
        
        print(all_rewards.sum())
        print('GAME OVER')
    print(np.mean(sucess_r),np.std(sucess_r))
    print(np.mean(avg_surv,axis=0),np.std(avg_surv,axis=0))
    print(np.mean(avg_speed,axis=0),np.std(avg_speed,axis=0))
    breakpoint()

if __name__=='__main__':
    
    # 11: max ,46 mean 18
    main()