
import numpy as np
from utils import *

import cv2

import torch
from torch.distributions.normal import Normal


import os

from agents import RLAgent 
from agents import SLAgent 

from make_roads import get_random_scene
from utils import load,load_all_mode
from train_reward_offline import RewardModeSequance,LongTermDiscriminator,stat_model

import matplotlib
matplotlib.use('WebAgg') 
from matplotlib import pyplot as plt


class TrafficEnv():
    
    def __init__(self, w=182, h=114,
                  num_agents = 15 + np.random.randint(15), 
                  max_steps=17,
                  make_img=False, 
                  img_size=[20,40],
                  n_modes=20,
                  train=False,
                  device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")):
        
        #self.max_time = 120
        self.wh = [w,h]
        self.w = w
        self.h = h
        
        self.rs = np.zeros((num_agents,1))
        self.r1 = 5
        self.r2 = 10
        self.r3 = 20
        
        self.fixed_reward_factor = 1.0
        self.device = device
        
        self.make_img = make_img
        self.default_scale = 3
        self.imgs_states = None
        self.img_size = img_size
        self.traj_mode_len = 9 # was 17
        self.train = train
        self.initial_speed_factor = 1 # avg of 1.5+[0,1,2] for all 3 types

        _ = self.reset(num_agents=num_agents, max_steps=max_steps)

        self.first_step = True
        self.n_modes = n_modes
        self.reward_traj_length = 18

        self.bc_models = torch.load(f'bc_agent_ind_{20}_{n_modes}_kmeans.pth',map_location=self.device)

        self.bc_models.eval()
        self.full_reward_model = stat_model(divide_by=4,min_reward=-2)
        
        
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
    
    def step(self,actions_d=None,training_time=1):
        
        if actions_d is None:
            with torch.no_grad():
                actions_d = self.bc_models.get_z(torch.Tensor(self.history[-1][:,:-self.traj_mode_len]).to(self.device)).cpu().numpy().argmax(axis=1)

        self.history_z.append(actions_d.copy())

        with torch.no_grad():

            actions, _ = self.bc_models.get_action(torch.Tensor(self.history[-1][:,:-self.traj_mode_len]).to(self.device),
                           best=True,z_logits=torch.Tensor(actions_d).to(self.device))
            actions = actions.cpu().numpy()
            
        #### Step
        c_,s_ = np.cos(self.headings)[:,None],np.sin(self.headings)[:,None]
        self.velocity = (np.hstack((c_,-s_,s_,c_)).reshape(-1,2,2)@actions.T)[range(self.num_agents),:,range(self.num_agents)]
        self.headings = np.arctan2(self.velocity[:,1],self.velocity[:,0])
        
        self.speeds = np.linalg.norm(actions,axis=1)[:,None]
        
        self.poses += self.velocity

        #### Find Reward
        rewards_goal = np.zeros_like(self.speeds)
        mask_out = np.logical_or(((self.poses[:,0]<=0)+(self.poses[:,0]>=self.w)), 
                                 ((self.poses[:,1]<=0)+(self.poses[:,1]>=self.h)))*(bool(len(self.starting_poses)))
        if mask_out.any():
            mask_1 = np.linalg.norm((self.starting_poses[mask_out] - self.ports_poses[:,None,:]).reshape(-1,2),axis=1).reshape(len(self.ports_poses),-1)>(self.w//3)
            mask_2 = np.linalg.norm((self.poses[mask_out] - self.ports_poses[:,None,:]).reshape(-1,2),axis=1).reshape(len(self.ports_poses),-1)<15
            rewards_goal[mask_out] += (mask_1*mask_2).any(axis=0)[:,None]*20

        #### Teleport
        self.poses[:,1][self.poses[:,0]>=self.w] += (self.ports[0]-self.ports[1])
        self.poses[:,1][self.poses[:,0]<0] += (self.ports[1]-self.ports[0])

        self.poses[:,0][self.poses[:,1]>=self.h] += (self.ports[2]-self.ports[3])
        self.poses[:,0][self.poses[:,1]<0] += (self.ports[3]-self.ports[2])
        
        self.poses = self.poses%np.array([self.w,self.h])
        
        if mask_out.any() :
            self.starting_poses[mask_out] = self.poses[mask_out].copy()
        
        #### Find New State
        self.acceleration = ((self.speeds*2.5)-(self.history[-1][:,0:1]))*2.5 
        self.old_velocity = self.velocity.copy()
        # NOTE before clip to evaluate true action
        reward = self.rewards_test(actions,actions_d) + rewards_goal
        if not(self.train):
            self.teleport()

        if self.make_img:
            self.make_image_()
            self.past_imgs_states = self.imgs_states.copy()
            self.imgs_states = np.array([self.get_agent_image(i) for i in range(self.num_agents)])
            
        self.near_objs = self.update_near_objs()

        self.states = np.hstack((abs(self.speeds*2.5),(self.headings[:,None]%(2*np.pi)),
                                 self.near_objs,self.zs[:,None],abs(self.acceleration),
                                 np.zeros((self.num_agents,self.traj_mode_len))-1))
        
        trajs = np.vstack(self.history_z).T[:,-min(self.time+1,self.traj_mode_len):] 
        self.states[:,-trajs.shape[1]:] = trajs

        self.time += 1
        
        # update global velocity average

        self.rs += reward
        info = []
        done = [(self.time>=self.max_time) for _ in range(self.num_agents)]
        
        for i in range(self.num_agents):
            if done[i]: 
                info.append({'episode':{'r':self.rs[i],'l':self.time,'collisions':self.collisions,'OutofRoad':self.outOfRoad}})
        if all(done):
            _ = self.reset(num_agents = self.num_agents , max_steps= self.max_time,training_time=training_time)
        else:
            self.history.append(self.states.copy())
            self.history_poses.append(self.poses.copy())
            
        return self.states.copy(), np.array([r for r in reward]), done, info, self.imgs_states
        
    
    
    def rewards_test(self,actions,actions_d):
        

        #### check collosion between vru and non-vru
        distances = np.linalg.norm(self.poses - self.poses[:,None,:],axis=2)
        revert_self = np.logical_not(np.eye(self.num_agents).astype(bool))
        types_mat_1 = (self.types>1)[:,None]

        rewards = -0.25*((((distances<2.0)*revert_self).sum(axis=1))[:,None])*self.fixed_reward_factor
        rewards = -0.9*((((distances<1.0)*revert_self).sum(axis=1))[:,None])*self.fixed_reward_factor
        rewards = -1.5*((((distances<0.4)*revert_self).sum(axis=1)*abs(self.speeds.T[0]))[:,None])*self.fixed_reward_factor
        self.collisions += (((distances<0.3)*revert_self).sum(axis=1)[:,None]).sum()/2

        #### check vrus on side walk or road TODO (quick on roads)
        vru_poses = (self.poses[(self.types<2)].T-0).astype(int)
        rewards[(self.types<2)] -= (4*np.logical_not(
            (self.sidewalk[vru_poses[1],vru_poses[0],1]+self.road[vru_poses[1],vru_poses[0],1]).astype(bool)))[:,None]*self.fixed_reward_factor
        rewards[(self.types<2)] -= (1*np.logical_not((self.sidewalk[vru_poses[1],vru_poses[0],1]).astype(bool)))[:,None]*self.fixed_reward_factor
        
        #### check nonvru on road only
        nonvru_poses = (self.poses[types_mat_1.T[0]].T-0).astype(int)
        rewards[types_mat_1.T[0]] -= (5*np.logical_not(self.road[nonvru_poses[1],nonvru_poses[0],1].astype(bool)))[:,None]*self.fixed_reward_factor
        
        #### speed should be big for cars and none zero of vrus NOTE (added) (sligtly encourge high speeds)
        rewards[(self.types==0)] -= (0.2*((abs(self.speeds.T[0])[(self.types==0)])<0.5))[:,None] # pedestrains
        rewards[(self.types==1)] -= (0.2*((abs(self.speeds.T[0])[(self.types==1)])<0.9))[:,None] # bikes
        rewards[(self.types==2)] -= (0.2*((abs(self.speeds.T[0])[(self.types==2)])<2))[:,None] # cars

        #### Relastic Trajs
        if self.time>0 :
            trajs = np.vstack(self.history_z).T[:,-min(self.time+1,self.traj_mode_len):]
            for type_ii,traj,act_d,n_i in zip(self.types,trajs,actions_d,range(self.num_agents)):
                final_reward = np.zeros(self.n_modes)
                traj_len = (traj != -1).sum()
                for ii in range(traj_len):
                    final_reward += self.full_reward_model[type_ii][traj[ii],traj_len-ii-1]
                rewards[n_i] += final_reward[act_d]/(traj_len)

        return rewards

        
    def teleport(self):

        cars_mask = (self.types>1)[:,None]
        vru_mask = np.logical_not(cars_mask)

        nonvru_poses = (self.poses[cars_mask.T[0]].T-0).astype(int)
        mask_nonvru = np.logical_not((self.road[nonvru_poses[1],nonvru_poses[0],1]+
                                      self.sidewalk[nonvru_poses[1],nonvru_poses[0],1]).astype(bool))
        if mask_nonvru.any():
            cars_mask.T[0][cars_mask.T[0]] = mask_nonvru
            self.poses[cars_mask.T[0].argmax()] = self.ports_poses[np.random.choice(range(len(self.ports))),:]+np.random.choice([0,3,-3])
            self.starting_poses[cars_mask.T[0].argmax()] = self.poses[cars_mask.T[0].argmax()]

        vru_poses = (self.poses[(self.types<2)].T-0).astype(int)
        mask_vru = np.logical_not((self.sidewalk[vru_poses[1],vru_poses[0],1]+self.road[vru_poses[1],vru_poses[0],1]).astype(bool))
        if mask_vru.any():
            vru_mask.T[0][vru_mask.T[0]] = mask_vru
            self.poses[vru_mask.T[0].argmax()] = self.ports_poses[np.random.choice(range(len(self.ports))),:]+np.random.choice([0,3,-3])
            self.starting_poses[vru_mask.T[0].argmax()] = self.poses[vru_mask.T[0].argmax()]

        self.outOfRoad += mask_vru.sum()+mask_nonvru.sum()

    def reset(self, num_agents = 15 + np.random.randint(15), max_steps=17, training_time=1):

        # training_time is factor from 0 to 1 (percentage of done steps/total steps)
        #### Initilize
        if self.train:
            ease_factor = (1-training_time)
            self.w = int(self.wh[0]*(2*ease_factor+1))#0+2.5#2,1
            self.h = int(self.wh[1]*(2*ease_factor+1))
            lane_width = int(np.round(2 + 2.5*ease_factor))
        else:
            self.w = int(self.wh[0]*2.5)#0+2.5#2,1
            self.h = int(self.wh[1]*2.5)
            lane_width = 2 
        
        self.grid = np.zeros((self.h,self.w,3)) # 1 pixel = 1 meter
        self.full_grid = np.zeros((self.h*3,self.w*3,3))

        self.road,self.sidewalk,self.heading_img, self.ports = get_random_scene(self.w,self.h,lane_width=lane_width)
        
        # make fixed grid
        self.initial_grid = cv2.resize((self.road+self.sidewalk),
                                       dsize=[self.w*self.default_scale,self.h*self.default_scale],
                                       interpolation=cv2.INTER_LINEAR)
        self.num_agents = num_agents
        # state: speed,near,(type),lost
        self.poses = (np.random.random((self.num_agents,2))*np.array([self.w,self.h]))#.astype(int)
        types_probs = np.random.rand(self.num_agents)
        self.types = (types_probs<0.44).astype(int)+(types_probs<0.67) # 0: Pedestrain, 1: Biker 2: Cars
        
        #### initilize poses
        vru_poses = np.vstack(np.where(self.sidewalk[:,:,1]>0)[::-1]).T
        nonvru_poses = np.vstack(np.where(self.road[:,:,1]>0)[::-1]).T
        np.random.shuffle(vru_poses)
        np.random.shuffle(nonvru_poses)
        
        self.poses[self.types<2] = vru_poses[:(self.types<2).sum()]
        self.poses[self.types>1] = nonvru_poses[:(self.types>1).sum()]

        self.zs = self.types.copy()
        self.starting_poses = self.poses.copy()
        self.ports_poses = np.array([
            [0,self.ports[0]],[self.w-1,self.ports[1]],[self.ports[2],0],[self.ports[3],self.h-1]])


        #### init headings
        self.headings = (((np.random.rand(self.num_agents)))-0.5)*np.pi*4*(training_time*self.train) # radian
        true_headings = self.heading_img[self.poses.T[1].astype(int),self.poses.T[0].astype(int),0]
        step=0
        # search for the true headings from image
        while not(true_headings.astype(bool).all()):
            step += 1
            for a,b in zip([step,-step,step,-step],[step,step,-step,-step]):
                true_headings += self.heading_img[np.clip(self.poses.T[1].astype(int)+a,0,self.h-1),
                                    np.clip(self.poses.T[0].astype(int)+b,0,self.w-1),0]*(np.logical_not(true_headings.astype(bool)))
        self.headings += true_headings

        #### init speed
        self.speeds = abs((np.random.rand(self.num_agents,1))*3) # [0,3]
        self.speeds += (self.types[:,None])*self.initial_speed_factor # bikes and cars faster on y
        
        self.velocity = np.vstack((np.cos(self.headings),np.sin(self.headings))).T*self.speeds
        self.acceleration = np.zeros_like(self.speeds)
        self.old_velocity = self.velocity.copy()
        
        self.max_time = max_steps
        
        self.time = 0
        self.collisions = 0
        self.outOfRoad = 0
        
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
        self.past_imgs_states = np.zeros_like(self.imgs_states)-1

        return [False for _ in range(self.num_agents)], self.states, self.imgs_states

    def get_agent_image(self,i):
        

        center = self.poses[i] + np.array([self.w,self.h])*abs(self.train-1)

        heading = self.headings[i]
        R_mat = cv2.getRotationMatrix2D(center,heading*180/np.pi,1)
        if self.train:
            img = cv2.warpAffine(self.grid, R_mat, (int(self.w*1.5),int(self.h*1.5)), flags=cv2.INTER_LINEAR,borderValue=1)
        else:
            img = cv2.warpAffine(self.full_grid, R_mat, (self.w*3,self.h*3), flags=cv2.INTER_LINEAR,borderValue=1)

        out = img[max(int(center[1]-self.img_size[0]/2),0):int(center[1]+self.img_size[0]/2),
                  max(int(center[0]-self.img_size[1]*0.2),0):int(center[0]+self.img_size[1]*0.8),:].T
        place = np.zeros((3,self.img_size[1],self.img_size[0]))+128
        place[:out.shape[0],:out.shape[1],:out.shape[2]] = out
        return place
    
    def make_full_grid(self):


        cd_ = self.ports[2] -self.ports[3]
        ba_ = self.ports[1] -self.ports[0]

        self.full_grid[self.h:self.h*2,self.w:self.w*2,:] = self.grid.copy()

        # up
        self.full_grid[:self.h,self.w+cd_:self.w*2+cd_,:] = self.grid.copy()
        #down
        self.full_grid[self.h*2:,self.w-cd_:self.w*2-cd_,:] = self.grid.copy()
        #left
        self.full_grid[self.h-ba_:self.h*2-ba_,:self.w,:] = self.grid.copy()
        #right
        self.full_grid[self.h+ba_:self.h*2+ba_,self.w*2:,:] = self.grid.copy()



    def make_image_(self):


        scale = self.default_scale
        self.grid = self.initial_grid.copy()
        colors = np.array([[0,255.0,0],[255.0,0,0],[0,0,255.0]])

        for i  in range(self.num_agents):
            if self.types[i] == 1:

                p1 = (self.poses[i]-np.array([np.cos(self.headings[i]),np.sin(self.headings[i])])*1.7).astype(int)*scale
                p2 = (self.poses[i]+np.array([np.cos(self.headings[i]),np.sin(self.headings[i])])*1.7).astype(int)*scale
                self.grid = cv2.arrowedLine(self.grid,p1,p2,
                                    color=colors[self.types[i].astype(int)],thickness=2)      

            elif self.types[i] == 0:
                #pedestrains

                p1 = ((self.poses[i]-np.array([np.cos(self.headings[i]),np.sin(self.headings[i])])*1.0)*scale).astype(int)
                p2 = ((self.poses[i]+np.array([np.cos(self.headings[i]),np.sin(self.headings[i])])*1.0)*scale).astype(int)
                self.grid = cv2.arrowedLine(self.grid,p1,p2,
                                    color=colors[self.types[i].astype(int)],thickness=2,tipLength=0.5)

            else:
                p1 = (self.poses[i]-np.array([np.cos(self.headings[i]),np.sin(self.headings[i])])*2) # 4=2+1 meters car length
                p2 = (self.poses[i]+np.array([np.cos(self.headings[i]),np.sin(self.headings[i])])*2)

                p1a = ((p1+np.array([np.cos(self.headings[i]+(np.pi/2)),np.sin(self.headings[i]+(np.pi/2))])*1)*scale) # 2=1+1 meters car width
                p1b = ((p1-np.array([np.cos(self.headings[i]+(np.pi/2)),np.sin(self.headings[i]+(np.pi/2))])*1)*scale)

                p2a = ((p2+np.array([np.cos(self.headings[i]+(np.pi/2)),np.sin(self.headings[i]+(np.pi/2))])*0.7)*scale)
                p2b = ((p2-np.array([np.cos(self.headings[i]+(np.pi/2)),np.sin(self.headings[i]+(np.pi/2))])*0.7)*scale)


                points = np.array([p1a,p2a,p2b,p1b]).astype(np.int32)

                self.grid = cv2.fillPoly(self.grid,pts=[points],
                                          color=colors[self.types[i].astype(int)])


        self.im_to_show = self.grid.copy()
        self.grid = cv2.resize(self.grid,dsize=[self.w,self.h],interpolation=cv2.INTER_LINEAR)
        if not(self.train):
            self.make_full_grid()

    def render(self):
        
        cv2.imshow('MATS',self.im_to_show)

 
def main():

    N_MODES=20
    
    env = TrafficEnv(make_img=True,num_agents=64,img_size=[20,40],n_modes=N_MODES,train=False)
    model = torch.load(f'ppo_agent_ind_with_reward_{N_MODES}_cl_not_learned.pth',map_location=env.device)

    ########################### LOOP

    for scene in [0,1,2,3,4,5]:
        done,new_state,new_img_state = env.reset(num_agents=16*(scene+1),max_steps=int(16*8))

        all_rewards = 0
        while sum(done)==0:
            with torch.no_grad():
                action = model.get_action(torch.Tensor(new_state).to(device=env.device),
                                          torch.Tensor(new_img_state).to(device=env.device),best=True).cpu().numpy()
            new_state, rewards, done,info,new_img_state = env.step(action) # empty input to get bc baseline
            all_rewards += rewards
            #env.make_image_()
            env.render()
            if False:
                plt.imshow(env.full_grid[:,:,::-1])
                plt.show()
            cv2.waitKey(1)
        print(all_rewards.mean())
        print('GAME OVER')


if __name__=='__main__':
    
    main()
 


