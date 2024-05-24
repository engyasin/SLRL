
import numpy as np
from utils import *

import cv2

import torch
from torch.distributions.normal import Normal

from bc import AgentCNN_Z_BC_MB


import glob,os
import os

from agents import AgentCNN_D as Agent
from agents import Discriminator,made_up_clusterer

from make_roads import get_random_scene
from utils import load,load_all_mode



device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")

class TrafficEnv():
    
    def __init__(self,w=182,h=114,num_agents = 15 + np.random.randint(15),max_steps=17,pixel2meter=None,make_img=False,img_size=[10,10],n_modes=20):
        
        #self.max_time = 120
        self.wh = [w,h]
        self.w = w
        self.h = h
        
        self.rs = 0
        self.r1 = 5
        self.r2 = 10
        self.r3 = 20
        
        self.fixed_reward_factor = 1
        
        self.make_img = make_img
        self.imgs_states = None
        self.img_size = img_size
        
        _ = self.reset(num_agents=num_agents, max_steps=max_steps, pixel2meter=pixel2meter)
        self.first_step = False
        self.n_modes = n_modes
        self.bc_models = torch.load(f'bc_agent_ind_{75}_{n_modes}_one_step.pth',map_location=torch.device('cpu'))
        self.bc_models.to(device)
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
                actions_d = self.bc_models.get_z(torch.Tensor(self.history[-1]).to(device)).cpu().numpy().argmax(axis=1)
        self.history_z.append(actions_d.copy())
        with torch.no_grad():
            actions, _ = self.bc_models.get_action(torch.Tensor(self.history[-1]).to(device),
                           z_logits=torch.Tensor(actions_d).to(device),best=True)
            actions = actions.cpu().numpy()
            
        c_,s_ = np.cos(self.headings)[:,None],np.sin(self.headings)[:,None]
        self.velocity = (np.hstack((c_,-s_,s_,c_)).reshape(-1,2,2)@actions.T)[range(self.num_agents),:,range(self.num_agents)]
        self.headings = np.arctan2(self.velocity[:,1],self.velocity[:,0])
        
        self.speeds = np.linalg.norm(actions,axis=1)[:,None]
        
        self.poses += self.velocity

        rewards_goal = np.zeros_like(self.speeds)
        mask_out = np.logical_or(((self.poses[:,0]<0)*(self.poses[:,0]>=self.w)), 
                                 ((self.poses[:,1]<0)*(self.poses[:,1]>=self.h)))
        if mask_out.any():
            mask_1 = np.linalg.norm((self.starting_poses[mask_out] - self.ports_poses).reshape(-1,2),axis=1).reshape(len(self.ports_poses),-1)>(self.w//3)
            mask_2 = np.linalg.norm((self.poses[mask_out] - self.ports_poses).reshape(-1,2),axis=1).reshape(len(self.ports_poses),-1)<6
            rewards_goal[mask_out] += (mask_1*mask_2).any(axis=1)[:,None]*4

        # or
        self.poses[:,1][self.poses[:,0]>=self.w] = self.ports[0]+np.random.choice([3,-3])
        self.poses[:,1][self.poses[:,0]<0] = self.ports[1]+np.random.choice([3,-3])

        self.poses[:,0][self.poses[:,1]>=self.h] = self.ports[2]+np.random.choice([3,-3])
        self.poses[:,0][self.poses[:,1]<0] = self.ports[3]+np.random.choice([3,-3])
        
        self.poses = self.poses%np.array([self.w,self.h])
        
        if mask_out.any():
            self.starting_poses[mask_out] = self.poses[mask_out].copy()
        
        self.acceleration = ((self.speeds*2.5)-(self.history[-1][:,0:1]))*2.5 # NOTE
        self.old_velocity = self.velocity.copy()
        # NOTE before clip to evaluate true action
        reward = self.rewards_test(actions,actions_d) + rewards_goal
        
        if self.make_img:
            self.make_image()
            self.imgs_states = np.array([self.get_agent_image(i) for i in range(self.num_agents)])
            
        self.near_objs = self.update_near_objs()
        self.states = np.hstack((abs(self.speeds*2.5),(self.headings[:,None]%(2*np.pi)),self.near_objs,self.zs[:,None],abs(self.acceleration)))

        self.time += 1
        
        # update global velocity average

        self.rs += reward
        info = []
        done = [(self.time>=self.max_time) for _ in range(self.num_agents)]
        
        for i in range(self.num_agents):
            if done[i]: 
                info.append({'episode':{'r':self.rs[i],'l':self.time}})
        if all(done):
            _ = self.reset(num_agents = self.num_agents , max_steps= self.max_time)
        else:
            self.history.append(self.states.copy())
            self.history_poses.append(self.poses.copy())
            
        return self.states.copy(), np.array([r for r in reward]), done, info, self.imgs_states
        
    
    
    def rewards_test(self,actions,actions_d):
        
        
            
        # check collosion between vru and non-vru
        distances = np.linalg.norm(self.poses - self.poses[:,None,:],axis=2)
        revert_self = np.logical_not(np.eye(self.num_agents).astype(bool))
        types_mat_1 = (self.types>1)[:,None]
        rewards = -0.5*(((distances<1.0)*revert_self).sum(axis=1)[:,None])*self.fixed_reward_factor
        rewards = -1.0*(((distances<0.5)*revert_self).sum(axis=1)[:,None])*self.fixed_reward_factor
        
        ################
        # check vrus on side walk or road TODO (quick on roads)
        vru_poses = (self.poses[(self.types<2)].T-0).astype(int)
        #try:
        rewards[(self.types<2)] -= (4*np.logical_not(
            (self.sidewalk[vru_poses[1],vru_poses[0],1]+self.road[vru_poses[1],vru_poses[0],1]).astype(bool)))[:,None]*self.fixed_reward_factor
        rewards[(self.types<2)] -= (0.3*np.logical_not((self.sidewalk[vru_poses[1],vru_poses[0],1]).astype(bool)))[:,None]*self.fixed_reward_factor
        
        ############
        # check nonvru on road only
        nonvru_poses = (self.poses[types_mat_1.T[0]].T-0).astype(int)
        rewards[types_mat_1.T[0]] -= (4.5*np.logical_not(self.road[nonvru_poses[1],nonvru_poses[0],1].astype(bool)))[:,None]*self.fixed_reward_factor
        #except IndexError:
        
        
        ### speed should be big for cars NOTE (added) (sligtly encourge high speeds)
        rewards += abs(0.02*(abs(self.speeds.T[0])>1))[:,None]

        
        return rewards

        
    def reset(self, num_agents = 15 + np.random.randint(15),max_steps=17,pixel2meter=None):

        self.w = int(self.wh[0]*(np.random.rand()+0.5))
        self.h = int(self.wh[1]*(np.random.rand()+0.5))
        
        self.grid = np.zeros((self.h,self.w,3)) # 1 pixel = 1 meter
        self.road,self.sidewalk,self.heading_img, self.ports = get_random_scene(self.w,self.h)
        
        self.num_agents = num_agents
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


        self.headings = (((np.random.rand(self.num_agents)))-0.5)*np.pi*2 # radian

        
        if True:
            self.headings *= 0.25#6
            new_headings = self.heading_img[self.poses.T[1].astype(int),self.poses.T[0].astype(int),0]
            step=0
            while not(new_headings.astype(bool).all()):
                step += 1
                for a,b in zip([step,-step,step,-step],[step,step,-step,-step]):
                    new_headings += self.heading_img[np.clip(self.poses.T[1].astype(int)+a,0,self.h-1),
                                        np.clip(self.poses.T[0].astype(int)+b,0,self.w-1),0]*(np.logical_not(new_headings.astype(bool)))
        
            self.headings += new_headings

        self.speeds = abs((np.random.rand(self.num_agents,1)-0.5)*2)
        self.speeds += (self.types[:,None])*5 # bikes and cars faster on y
        
        self.velocity = np.vstack((np.cos(self.headings),np.sin(self.headings))).T*self.speeds
        self.acceleration = np.zeros_like(self.speeds)
        self.old_velocity = self.velocity.copy()
        
        self.max_time = max_steps
        
        self.time = 0
        
        
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

    def get_agent_image(self,i):
        
        center = self.poses[i]
        heading = self.headings[i]
        R_mat = cv2.getRotationMatrix2D(center,heading*180/np.pi,1)
        img = cv2.warpAffine(self.grid, R_mat, (self.w*2,self.h*2), flags=cv2.INTER_LINEAR)
        out = img[max(int(center[1]-self.img_size[0]/2),0):int(center[1]+self.img_size[0]/2),
                  max(int(center[0]-self.img_size[1]*0.2),0):int(center[0]+self.img_size[1]*0.8),:].T
        place = np.zeros((3,self.img_size[1],self.img_size[0]))+255
        place[:out.shape[0],:out.shape[1],:out.shape[2]] = out
        return place
    
    
    def make_image(self):
        self.grid = (self.road + self.sidewalk).copy()#np.zeros((int(self.w/size),int(self.h/size),3))
        colors = np.array([[0,255.0,0],[255.0,0,0],[0,0,255.0]])
        for i  in range(self.num_agents):
            p1 = (self.poses[i]-np.array([np.cos(self.headings[i]),np.sin(self.headings[i])])*1).astype(int)
            p2 = (self.poses[i]+np.array([np.cos(self.headings[i]),np.sin(self.headings[i])])*1).astype(int)
            self.grid = cv2.arrowedLine(self.grid,p1,p2,
                                   color=colors[self.types[i].astype(int)],thickness=1)            

        
    def make_image_(self):
        self.grid = (self.road + self.sidewalk).copy()#np.zeros((int(self.w/size),int(self.h/size),3))
        colors = np.array([[0,255.0,0],[255.0,0,0],[0,0,255.0]])
        for i  in range(self.num_agents):
            if self.types[i]==1:
                p1 = (self.poses[i]-np.array([np.cos(self.headings[i]),np.sin(self.headings[i])])*1).astype(int)
                p2 = (self.poses[i]+np.array([np.cos(self.headings[i]),np.sin(self.headings[i])])*1).astype(int)
                self.grid = cv2.arrowedLine(self.grid,p1,p2,
                                    color=colors[self.types[i].astype(int)],thickness=1)    

            elif self.types[i] in [0,2]:
                #pedestrains
                self.grid = cv2.circle(self.grid,self.poses[i].astype(int),int(self.types[i]/2)+1,
                                       color=colors[self.types[i].astype(int)],thickness=-1)
            else:
                p1 = (self.poses[i]-np.array([np.cos(self.headings[i]),np.sin(self.headings[i])])*2)
                p2 = (self.poses[i]+np.array([np.cos(self.headings[i]),np.sin(self.headings[i])])*2)
                p1a = (p1+np.array([np.sin(self.headings[i]+(np.pi/4)),-np.cos(self.headings[i]+(np.pi/4))])*1).astype(int)
                p2a = (p2-np.array([np.sin(self.headings[i]+(np.pi/4)),-np.cos(self.headings[i]+(np.pi/4))])*1).astype(int)
                self.grid = cv2.rectangle(self.grid,p1a,p2a,color=colors[self.types[i].astype(int)],thickness=-1)

        
    def render(self,scale=3):
        
        self.im_to_show = cv2.resize(self.grid,dsize=[self.w*scale,self.h*scale])
        cv2.imshow('GAME',self.im_to_show)

 
def main():
    N_MODES=20
    
    env = TrafficEnv(make_img=True,num_agents=64,img_size=[20,30],n_modes=N_MODES)#[12,20]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = torch.load('ppo_agent_ind_image_d_last_step_5.pth',map_location=torch.device('cpu'))
    model = torch.load('ppo_agent_ind_image_d_smoothed_last_step_35.pth')#,map_location=torch.device('cpu'))
    model.center = [10,6]
    init_vid = True
    for scene in [0]:
        done,new_state,new_img_state = env.reset(num_agents=32)
        if init_vid:
            fourcc = cv2.VideoWriter_fourcc(*'XVID') 
            out = cv2.VideoWriter('output1.avi', fourcc, 5.0, (env.w*3, env.h*3))
            env.max_steps = 32*50
            init_vid = False
        if type(done)==list:
            done = done[0]
        env.max_time = int(16*50)
        all_rewards = 0
        while not(done):
            with torch.no_grad():
                action = model.get_action(torch.Tensor(new_state).to(device=device),torch.Tensor(new_img_state).to(device=device),best=False).cpu().numpy()#*np.array([0.025,0])
                action_bc = env.bc_models.get_z(torch.Tensor(new_state).to(device=device)).argmax(axis=1).cpu().numpy()

            new_state, rewards, done,info,new_img_state = env.step(action)#-env.poses)
            all_rewards += rewards
            if type(done)==list:
                done = done[0]
                
            env.render()
            out.write(env.im_to_show.copy().astype(np.uint8))
            cv2.waitKey(10)#000)
        print(all_rewards.sum())
        print('GAME OVER')
    out.release()


if __name__=='__main__':
    
    main()
 


