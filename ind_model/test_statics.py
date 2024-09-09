from numpy.random.mtrand import randint as randint

from utils import *
from matplotlib import pyplot as plt
from agents import RLAgent as Agent
from agents import SLAgent

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import glob,os
import os
import cv2

from agents import Discriminator
from utils import load_all_mode
import torch
from trafficenv_D import TrafficEnv


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import matplotlib
matplotlib.use('WebAgg') 
from matplotlib import pyplot as plt

class TrafficEnvMod(TrafficEnv):
    
    def __init__(self,w=182,h=114,num_agents = 15 + np.random.randint(15),max_steps=17,make_img=False,img_size=[10,10],n_modes=20):

        self.speed_record = []
        super().__init__(w, h, num_agents, max_steps,  make_img, img_size,n_modes)
   
    def load_ds_scene(self):
        img_file = 'img_extended_cut.png'
        self.scale = 0.00814635379575616 * 12 # fixed for scene 30-32

        bg_img = cv2.imread(img_file)
        h,w,_ = bg_img.shape

        self.w = int(w*self.scale)
        self.h = int(h*self.scale)
        self.bg_img = cv2.resize(bg_img,dsize=[int(self.w),int(self.h)])
        
        self.wh = [self.w,self.h]
        
        
        road = np.zeros_like(self.bg_img)
        road[(self.bg_img == 255).all(axis=2),:] = 255
        
        sidewalk = np.zeros_like(self.bg_img)
        sidewalk[(self.bg_img > 0).any(axis=2),:] = 255
        
        sidewalk[road>0] = 0
        sidewalk[:,:,[0]] = 0
        
        self.time = 0
        self.road,self.sidewalk,self.heading_img, self.ports = road.copy(),sidewalk.copy()/2,sidewalk.copy()/2,[2,self.h-2,4,self.w]
        
        

    def reset(self, num_agents = 15 + np.random.randint(15),max_steps=17,training_time=1):

        if len(self.speed_record):
            self.all_statics = self.sum_statics()

        self.load_ds_scene()
        self.num_agents = num_agents
        self.grid = np.zeros((self.h,self.w,3)) # 1 pixel = 1 meter

        self.initial_grid = cv2.resize((self.road+self.sidewalk),
                                       dsize=[self.w*self.default_scale,self.h*self.default_scale],
                                       interpolation=cv2.INTER_LINEAR)
        # state: speed,near,(type),lost
        self.poses = (np.random.random((self.num_agents,2))*np.array([self.w,self.h]))#.astype(int)
        
        types_probs = np.random.rand(self.num_agents)
        self.types = (types_probs<0.44).astype(int)+(types_probs<0.67) # 0: Pedestrain, 1: Biker 2: Cars
        
        #### 
        vru_poses = np.vstack(np.where(self.sidewalk[:,:,1]>0)[::-1]).T
        nonvru_poses = np.vstack(np.where(self.road[:,:,1]>0)[::-1]).T
        np.random.shuffle(vru_poses)
        np.random.shuffle(nonvru_poses)
        
        self.poses[self.types<2] = vru_poses[:(self.types<2).sum()]
        self.poses[self.types>1] = nonvru_poses[:(self.types>1).sum()]
        #breakpoint()
        self.zs = self.types.copy()#(np.random.rand(self.num_agents)*1).astype(int) # mode 20
        self.starting_poses = self.poses.copy()
        self.ports_poses = np.array([
            [0,self.ports[0]],[self.w-1,self.ports[1]],[self.ports[2],0],[self.ports[3],self.h-1]])

        self.headings = (((np.random.rand(self.num_agents)))-0.5)*np.pi*2*0 + (2.5*np.pi/2) # radian
        self.speed_record = []
        self.colliding_record = np.zeros_like(self.headings)


        #### init speed
        self.speeds = abs((np.random.rand(self.num_agents,1))*3) # [0,3]
        self.speeds += (self.types[:,None])*self.initial_speed_factor # bikes and cars faster on y
        
        self.velocity = np.vstack((np.cos(self.headings),np.sin(self.headings))).T*self.speeds
        self.acceleration = np.zeros_like(self.speeds)
        self.old_velocity = self.velocity.copy()
        
        self.max_time = max_steps
        self.time = 0
        self.rs = 0
        self.near_objs = self.update_near_objs()
        self.collsions = 0
        self.outOfRoad = 0
        
        if self.make_img:
            self.make_image_()
            self.imgs_states = np.array([self.get_agent_image(i) for i in range(self.num_agents)])
        
        self.states = np.hstack((abs(self.speeds*2.5),(self.headings[:,None]%(2*np.pi)),
                                 self.near_objs,self.zs[:,None],self.acceleration,
                                 np.zeros((self.num_agents,self.traj_mode_len))-1))
        self.history = []
        self.history.append(self.states.copy())
        self.history_poses = []
        self.history_poses.append(self.poses.copy())       
        self.history_z = []

        return [False for _ in range(self.num_agents)], self.states, self.imgs_states
    

    
    def find_statics(self):
        
        
        self.speed_record.append(self.speeds.copy())
        distances = np.linalg.norm(self.poses - self.poses[:,None,:],axis=2)
        revert_self = np.logical_not(np.eye(self.num_agents).astype(bool))
        is_colliding = np.logical_not((((distances<0.4)*revert_self).sum(axis=1)[:,None]))
        self.colliding_record += (is_colliding.flatten() * (self.colliding_record == self.colliding_record.max()))
        

    def sum_statics(self):
        
        sucess_rate = (self.colliding_record[self.types>1] > (self.time-2)).mean()
        all_speeds = np.array(self.speed_record).mean(axis=0)
        avg_survival = []
        all_speeds_avgs = []
        for t in range(3):
            avg_survival.append(self.colliding_record[(self.types==t)].mean())
            all_speeds_avgs.append(all_speeds[(self.types==t)].mean())
            
        return sucess_rate,avg_survival,all_speeds_avgs
            
        
def main():
    N_MODES=20
    episode_length = 64
    env = TrafficEnvMod(make_img=True,num_agents=64,img_size=[20,40],n_modes=N_MODES)#[12,20]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(f'ppo_agent_ind_image_d_smoothed_first_step_kmeans_with_reward_{N_MODES}_cl.pth',map_location=env.device)

    sucess_r, avg_surv, avg_speed, collisions, full_rewards, out_of_road = [],[],[],[],[],[]

    for scene in range(64):
        done,new_state,new_img_state = env.reset(num_agents=1*(scene+5),max_steps=episode_length)# TODO check n agents
        all_rewards = 0
        while sum(done)==0:
            with torch.no_grad():
                action = model.get_action(torch.Tensor(new_state).to(device),torch.Tensor(new_img_state).to(device),best=True).cpu().numpy()

            new_state, rewards, done, info, new_img_state = env.step(action)#-env.poses)
            env.find_statics()
            all_rewards += rewards

        collisions.append(info[0]['episode']['collisions'])
        out_of_road.append(info[0]['episode']['OutofRoad'])
        full_rewards.append(all_rewards.sum()) # for all agents
        sucess_r.append(env.all_statics[0]) 
        avg_surv.append(np.nan_to_num(env.all_statics[1][:]))
        avg_speed.append(np.nan_to_num(env.all_statics[2][:]))
        
        print(all_rewards.sum())
        print('GAME OVER')

    print(np.mean(sucess_r),np.std(sucess_r))
    print(np.mean(avg_surv,axis=0),np.std(avg_surv,axis=0))
    print(np.mean(avg_speed,axis=0),np.std(avg_speed,axis=0))


    agents_axis = [2*(x+1) for x in range(4)]
    avg_speed = np.array(avg_speed)
    avg_surv = np.array(avg_surv)

    plt.subplot(251)
    plt.title('Collisions')
    plt.plot(collisions)

    plt.subplot(252)
    plt.title('out_of_road')
    plt.plot(out_of_road)

    plt.subplot(253)
    plt.title('full_rewards')
    plt.plot(full_rewards)

    plt.subplot(254)
    plt.title('sucess_r')
    plt.plot(sucess_r)

    plt.subplot(255)
    plt.title('avg_surv_ped')
    plt.plot(avg_surv[:,0])

    plt.subplot(256)
    plt.title('avg_speed_ped')
    plt.plot(avg_speed[:,0])

    plt.subplot(257)
    plt.title('avg_surv_bike')
    plt.plot(avg_surv[:,1])

    plt.subplot(258)
    plt.title('avg_speed_bike')
    plt.plot(avg_speed[:,1])

    plt.subplot(259)
    plt.title('avg_surv_car')
    plt.plot(avg_surv[:,2])

    plt.subplot(2,5,10)
    plt.title('avg_speed_car')
    plt.plot(avg_speed[:,2])

    plt.show()

if __name__=='__main__':
    
    main()
