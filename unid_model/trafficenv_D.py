
import numpy as np
from utils import *

import cv2

import torch
from torch.distributions.normal import Normal

from bc import AgentCNN_Z_BC_MB

import random
import glob,os

from agents import AgentCNN_D as Agent
from train_reward_offline import stat_model

import matplotlib
matplotlib.use('WebAgg') 

import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")

class TrafficEnv():
    
    def __init__(self,w=182,h=114,num_agents = 15+np.random.randint(15),
                 max_steps=17,
                 make_img=False,
                 img_size=[10,10],
                 n_modes=20,
                 first_step=True,
                 random_scaling =True,
                 device=torch.device("cuda" if torch.cuda.is_available()  else "cpu")):
        
        self.wh = [w,h]
        self.w = w
        self.h = h
        self.initial_speed_factor = 1
        self.show_scale = 3
        
        self.rs = 0
        self.r1 = 5
        self.r2 = 10
        self.r3 = 20
        self.device = device
        
        self.fixed_reward_factor = 1
        self.training_progress = 1
        self.make_img = make_img
        self.imgs_states = None
        self.img_size = img_size
        self.traj_mode_len = 9 # was 17
        self.first_step = first_step
        self.n_modes = n_modes
        self.reward_traj_length = 18
        self.bg_img = cv2.imread('background_img.png') 

        self.full_reward_model = stat_model(divide_by=4,min_reward=-2)#4,-2
        
        _ = self.reset(num_agents=num_agents, max_steps=max_steps)

        if self.first_step: 
            self.bc_models = torch.load(f'bc_agent_unid_{40}_{self.n_modes}_smoothed_one_step_new.pth',map_location=device)
            #self.bc_models = torch.load(f'bc_agent_unid_{35}_{self.n_modes}_one_step.pth',map_location=self.device)

        else:
            self.bc_models = torch.load(f'bc_agent_unid_{35}_{self.n_modes}_last_step.pth',map_location=self.device)
            
        self.bc_models.eval()


        
    def update_near_objs(self):
        
        agents_near_objs = []
        for type_,agent_pos in zip(self.types,self.poses):
            distances_vru = np.linalg.norm(self.poses[self.types<2] - agent_pos,axis=1)
            distances_nonvru = np.linalg.norm(self.poses[self.types>1] - agent_pos,axis=1)
            bias = (type_<2)
            bias_ = (type_==2)
            agents_near_objs.append([(distances_vru<self.r1).sum()-bias,(distances_vru<self.r2).sum()-bias,(distances_vru<self.r3).sum()-bias,
                                     (distances_nonvru<self.r1).sum()-bias_,(distances_nonvru<self.r2).sum()-bias_,(distances_nonvru<self.r3).sum()-bias_])
            
        return np.array(agents_near_objs)
    
    def step(self,actions_d=None):
        
        if actions_d is None:
            with torch.no_grad():
                actions_d = self.bc_models.get_z(torch.Tensor(self.history[-1][:,:-self.traj_mode_len]).to(self.device)).cpu().numpy().argmax(axis=1)
        self.history_z.append(actions_d.copy())
        with torch.no_grad():
            actions, _ = self.bc_models.get_action(torch.Tensor(self.history[-1][:,:-self.traj_mode_len]).to(self.device),
                           z_logits=torch.Tensor(actions_d).to(self.device),best=True)
            actions = np.clip(actions.cpu().numpy(),-150,150)

        c_,s_ = np.cos(self.headings)[:,None],np.sin(self.headings)[:,None]
        self.velocity = (np.hstack((c_,-s_,s_,c_)).reshape(-1,2,2)@actions.T)[range(self.num_agents),:,range(self.num_agents)]
        self.headings = np.arctan2(self.velocity[:,1],self.velocity[:,0])
        self.speeds = np.linalg.norm(actions,axis=1)[:,None]
        self.poses += self.velocity
        self.poses = self.poses%np.array([self.w,self.h])
        
        out_of_road = self.road[self.poses.astype(int)[:,1],self.poses.astype(int)[:,0],np.zeros(len(self.poses)).astype(int)]==0
        self.out_of_road = out_of_road.copy()
        
        out_nonvru = (out_of_road*(self.zs>0))
        out_vru = (out_of_road*(self.zs==0)) 
        rewards_goal = np.zeros_like(self.speeds)
        n_nonvru = out_nonvru.sum()
        n_vru = out_vru.sum()
        if len(self.starting_port):
            if n_nonvru:
                self.poses[out_nonvru] = np.array(random.choices(self.ports[5:],k=n_nonvru)) + ((np.random.rand(n_nonvru,2)*6)-3).astype(int)
                self.starting_port[out_nonvru] = np.linalg.norm((self.poses[out_nonvru] - self.ports[:,None,:]).reshape(-1,2),axis=1).reshape(len(self.ports),-1).argmin(axis=0)   
                rewards_goal -= (1)*out_nonvru[:,None]*(self.time<14+4)*30
                self.headings[out_nonvru*(self.starting_port==5)] = -1*np.pi/4
                self.headings[out_nonvru*(self.starting_port==6)] = 3*np.pi/4
            if n_vru:
                self.poses[out_vru] = np.array(random.choices(self.ports[:5],k=n_vru)) + ((np.random.rand(n_vru,2)*6)-3).astype(int)
                self.starting_port[out_vru] = np.linalg.norm((self.poses[out_vru] - self.ports[:,None,:]).reshape(-1,2),axis=1).reshape(len(self.ports),-1).argmin(axis=0)  
                rewards_goal -= (1)**out_vru[:,None]*(self.time<12+3)*30

        self.acceleration = ((self.speeds*2.5)-(self.history[-1][:,0:1]))*2.5 #NOTE
        self.old_velocity = self.velocity.copy()
        # NOTE before clip to evaluate true action
        reward = self.rewards_test(actions,actions_d) + rewards_goal

        if self.make_img:
            self.make_image_()
            self.imgs_states = np.array([self.get_agent_image(i) for i in range(self.num_agents)])
            
        self.near_objs = self.update_near_objs()
        self.states = np.hstack((abs(self.speeds*2.5),(self.headings[:,None]%(2*np.pi)),
                                 self.near_objs,self.zs[:,None],abs(self.acceleration),
                                 np.zeros((self.num_agents,self.traj_mode_len))-1))
        trajs = np.vstack(self.history_z).T[:,-min(self.time+1,self.traj_mode_len):] 
        self.states[:,-trajs.shape[1]:] = trajs

        self.time += 1
        
        self.rs += reward
        info = []
        done = [(self.time>=self.max_time) for _ in range(self.num_agents)]
        
        for i in range(self.num_agents):
            if done[i]: 
                info.append({'episode':{'r':self.rs[i],'l':self.time,'collisions':self.collisions}})
        if all(done):
            _ = self.reset(num_agents = self.num_agents , max_steps= self.max_time)
        else:
            self.history.append(self.states.copy())
            self.history_poses.append(self.poses.copy())
            
        return self.states.copy(), np.array([r for r in reward]), done, info, self.imgs_states
    
    
    def rewards_test(self,actions,actions_d):
            
        # check collosion between vru and non-vru
        distances = np.linalg.norm(self.poses - self.poses[:,None,:],axis=2)+1e-6
        revert_self = np.logical_not(np.eye(self.num_agents).astype(bool))
        #revert_self = np.clip((((self.types[:,None]+1)*(self.types[None,:]+1))!=1),0.3,1)*revert_self

        rewards = ((-1/((distances*0.5)**2))*(distances<15)*revert_self).sum(axis=1)[:,None]*self.fixed_reward_factor*(abs(self.speeds)**1)
        self.collisions +=  (((distances<0.3)*revert_self).sum(axis=1)[:,None]).sum()/2
        rewards = np.clip(rewards,-100,0)
        
        ############## reward speeds

        rewards += np.log(np.clip((3-self.types)*abs(self.speeds.T[0]),0.1,1))[:,None]*self.fixed_reward_factor
        
        ############## Relastic Trajs
        if self.time>0 :
            trajs = np.vstack(self.history_z).T[:,-min(self.time+1,self.traj_mode_len):]
            for type_ii,traj,act_d,n_i in zip(self.types,trajs,actions_d,range(self.num_agents)):
                final_reward = np.zeros(self.n_modes)
                traj_len = (traj != -1 ).sum()
                for ii in range(traj_len):
                    final_reward += self.full_reward_model[type_ii][traj[ii],traj_len-ii-1]
                rewards[n_i] += final_reward[act_d]/(traj_len)

        return rewards

    def set_training_progress(self,training_progress):
        self.training_progress = training_progress
        
    def reset(self, num_agents = 20 + np.random.randint(15),max_steps=17):

        random_scale = (3*(1-self.training_progress)+2)#1.68
        #random_scale = (np.random.choice([1,0,-1])*flag+2)#1.68
        h,w,_ = self.bg_img.shape
        bg_img = cv2.resize(self.bg_img,dsize=np.round([w*random_scale,h*random_scale]).astype(int))
        self.road,self.sidewalk = bg_img.copy(),bg_img.copy()
        self.w,self.h = bg_img.shape[1],bg_img.shape[0]

        # make fixed grid
        self.initial_grid = cv2.resize((self.road+self.sidewalk),
                                       dsize=[self.w*self.show_scale,self.h*self.show_scale],
                                       interpolation=cv2.INTER_LINEAR)

        # [left,left,right,right,up, down]
        self.ports = np.round(np.array([[139,52],[113,73],[115,117],[127,106],[159,70],[82,124],[163,46]])*random_scale)

        
        self.grid = np.zeros((self.h,self.w,3)) # 1 pixel = 1 meter
        
        self.num_agents = num_agents
        # state: speed,near,(type),lost
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

        self.headings = ((np.random.rand(self.num_agents)-1))*np.pi*2 # radian


        self.starting_port = np.linalg.norm((self.poses - self.ports[:,None,:]).reshape(-1,2),axis=1).reshape(len(self.ports),-1).argmin(axis=0)        

        ### Init heading (change only for bicycle and cars)
        self.headings[(self.types>0)*(self.starting_port==5)] = -1*np.pi/4
        self.headings[(self.types>0)*(self.starting_port==6)] = 3*np.pi/4

        ### Init Speed
        self.speeds = abs((np.random.rand(self.num_agents,1))*2) # [0,3]
        self.speeds += (self.types[:,None])*self.initial_speed_factor # bikes and cars faster on y

        self.velocity = np.vstack((np.cos(self.headings),np.sin(self.headings))).T*self.speeds
        self.acceleration = np.zeros_like(self.speeds)
        self.old_velocity = self.velocity.copy()
        
        self.max_time = max_steps
        
        self.time = 0
        self.collisions = 0
        
        self.rs = np.zeros((num_agents,1))
        
        self.near_objs = self.update_near_objs()
        
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

    def get_agent_image(self,i):
        
        center = self.poses[i]
        heading = self.headings[i]
        R_mat = cv2.getRotationMatrix2D(center,heading*180/np.pi,1)
        img = cv2.warpAffine(self.grid, R_mat, (self.w*2,self.h*2), flags=cv2.INTER_LINEAR)
        out = img[max(int(center[1]-self.img_size[0]/2),0):int(center[1]+self.img_size[0]/2),
                  max(int(center[0]-self.img_size[1]*0.2),0):int(center[0]+self.img_size[1]*0.8),:].T
        place = np.zeros((3,self.img_size[1],self.img_size[0]))+128
        place[:out.shape[0],:out.shape[1],:out.shape[2]] = out
        return place
    
        
    def make_image_(self):
        scale = self.show_scale
        agents_draw_scale = 1
        self.grid = self.initial_grid.copy()
        colors = np.array([[0,255.0,0],[255.0,0,0],[0,0,255.0]])

        for i  in range(self.num_agents):
            if self.types[i]==1:
                # bicycle: length 3m
                p1 = ((self.poses[i]-np.array([np.cos(self.headings[i]),np.sin(self.headings[i])])*1.5*agents_draw_scale)*scale).astype(int)
                p2 = ((self.poses[i]+np.array([np.cos(self.headings[i]),np.sin(self.headings[i])])*1.5*agents_draw_scale)*scale).astype(int)
                self.grid = cv2.arrowedLine(self.grid,p1,p2,
                                    color=colors[self.types[i].astype(int)],thickness=2,tipLength=0.5)      

            elif self.types[i] == 0:
                #pedestrains
                #self.grid = cv2.circle(self.grid,(self.poses[i]*scale).astype(int),int(self.types[i]/2)+scale+1,
                #                       color=colors[self.types[i].astype(int)],thickness=-1)
                p1 = ((self.poses[i]-np.array([np.cos(self.headings[i]),np.sin(self.headings[i])])*1.0*agents_draw_scale)*scale).astype(int)
                p2 = ((self.poses[i]+np.array([np.cos(self.headings[i]),np.sin(self.headings[i])])*1.0*agents_draw_scale)*scale).astype(int)
                self.grid = cv2.arrowedLine(self.grid,p1,p2,
                                    color=colors[self.types[i].astype(int)],thickness=2,tipLength=0.5)    
            else:
                # cars
                p1 = (self.poses[i]-np.array([np.cos(self.headings[i]),np.sin(self.headings[i])])*2*agents_draw_scale) # 4=2+2 meters car length
                p2 = (self.poses[i]+np.array([np.cos(self.headings[i]),np.sin(self.headings[i])])*2*agents_draw_scale)

                p1a = ((p1+np.array([np.cos(self.headings[i]+(np.pi/2)),np.sin(self.headings[i]+(np.pi/2))])*1*agents_draw_scale)*scale) # 2=1+1 meters car width
                p1b = ((p1-np.array([np.cos(self.headings[i]+(np.pi/2)),np.sin(self.headings[i]+(np.pi/2))])*1*agents_draw_scale)*scale)
                p2a = ((p2+np.array([np.cos(self.headings[i]+(np.pi/2)),np.sin(self.headings[i]+(np.pi/2))])*0.3*agents_draw_scale)*scale)
                p2b = ((p2-np.array([np.cos(self.headings[i]+(np.pi/2)),np.sin(self.headings[i]+(np.pi/2))])*0.3*agents_draw_scale)*scale)

                points = np.array([p1a,p2a,p2b,p1b]).astype(np.int32)

                self.grid = cv2.fillPoly(self.grid,pts=[points],
                                          color=colors[self.types[i].astype(int)])

        self.im_to_show = self.grid.copy()
        self.grid = cv2.resize(self.grid,dsize=[self.w,self.h],interpolation=cv2.INTER_LINEAR)
      

    def render(self):
        cv2.imshow('MATS',self.im_to_show)

 
def main():
    N_MODES=20
    env = TrafficEnv(make_img=True,img_size=[20,40],n_modes=N_MODES,random_scaling=True,first_step=True)#[12,20]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(f'ppo_agent_unid_image_d_smoothed_{["last_step","first_step"][env.first_step]}_v3.pth',map_location=device)

    for scene in range(10):
        all_rewards = 0
        done,new_state,new_img_state = env.reset(num_agents=4*(scene+1),max_steps=16*8)#
        #env.set_training_progress(0)
        while not(done[0]):
            with torch.no_grad():
                action = model.get_action(torch.Tensor(new_state).to(device=device),torch.Tensor(new_img_state).to(device=device),best=True).cpu().numpy()

            new_state, rewards, done,info,new_img_state = env.step(action)
            env.render()
            if False:
                #plt.figure(figsize=(10,10))
                for i in range(9):
                    plt.subplot(3,3,i+1)
                    plt.imshow(new_img_state[i].T[:,:,::-1]/255.0)
                    plt.axis('off')
                plt.show()
            cv2.waitKey(10)
            all_rewards += rewards.sum()
        print(all_rewards/env.num_agents)
        print(info[0]['episode']['collisions'])
        print('GAME OVER')



if __name__=='__main__':
    
    main()
 


