
import numpy as np
from utils import *

import cv2

import torch
from torch.distributions.normal import Normal

from bc import AgentClustererAll,Agent_BC_MB,AgentCNN_Z_BC_MB


import glob,os
import os

from agents import AgentCNN_D as Agent
from agents import Discriminator,made_up_clusterer

from make_roads import get_random_scene
from utils import load,load_all_mode
#from bc_ml import AgentClustererAll,AgentClusterer
from train_reward_offline import RewardModeSequance,LongTermDiscriminator,stat_model

device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")

class TrafficEnv():
    
    def __init__(self, w=182, h=114, num_agents = 15 + np.random.randint(15), max_steps=17, pixel2meter=None, make_img=False, img_size=[10,10],n_modes=20,train=False):
        
        #self.max_time = 120
        self.wh = [w,h]
        self.w = w
        self.h = h
        
        self.rs = np.zeros((num_agents,1))
        self.r1 = 5
        self.r2 = 10
        self.r3 = 20
        
        self.fixed_reward_factor = 1.5
        
        self.make_img = make_img
        self.default_scale = 3
        self.imgs_states = None
        self.img_size = img_size
        self.traj_mode_len = 9
        self.train = train
        _ = self.reset(num_agents=num_agents, max_steps=max_steps, pixel2meter=pixel2meter)
        self.first_step = True
        self.n_modes = n_modes
        self.bc_models = torch.load(f'bc_agent_ind_{75}_{n_modes}_one_step.pth',map_location=device)
        #self.bc_models = torch.load(f'bc_agent_ind_{35}_{n_modes}_last_step.pth',map_location=device)

        ##self.bc_models = torch.load(f'bc_agent_ind_{100}_{n_modes}_ml_type_{4}_.pth')
        ##self.bc_models = torch.load(f'bc_agent_ind_{100}_{n_modes}_kmeans.pth')

        self.reward_traj_length = 18
        self.trained_reward = torch.load(f'reward_{self.reward_traj_length}_steps_ind_120_{n_modes}.pth',map_location=device).to(device)
        self.bc_models.to(device)

        self.bc_models.eval()
        self.trained_reward.eval()

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
                # TODO
                actions_d = self.bc_models.get_z(torch.Tensor(self.history[-1][:,:-self.traj_mode_len]).to(device)).cpu().numpy().argmax(axis=1)
        self.history_z.append(actions_d.copy())
        with torch.no_grad():
            actions, _ = self.bc_models.get_action(torch.Tensor(self.history[-1][:,:-self.traj_mode_len]).to(device),
                           best=True,z_logits=torch.Tensor(actions_d).to(device))
            actions = actions.cpu().numpy()
            
        c_,s_ = np.cos(self.headings)[:,None],np.sin(self.headings)[:,None]
        self.velocity = (np.hstack((c_,-s_,s_,c_)).reshape(-1,2,2)@actions.T)[range(self.num_agents),:,range(self.num_agents)]
        self.headings = np.arctan2(self.velocity[:,1],self.velocity[:,0])
        
        self.speeds = np.linalg.norm(actions,axis=1)[:,None]
        
        self.poses += self.velocity

        rewards_goal = np.zeros_like(self.speeds)
        mask_out = np.logical_or(((self.poses[:,0]<=0)+(self.poses[:,0]>=self.w)), 
                                 ((self.poses[:,1]<=0)+(self.poses[:,1]>=self.h)))*(bool(len(self.starting_poses)))
        if mask_out.any():
            mask_1 = np.linalg.norm((self.starting_poses[mask_out] - self.ports_poses[:,None,:]).reshape(-1,2),axis=1).reshape(len(self.ports_poses),-1)>(self.w//3)
            mask_2 = np.linalg.norm((self.poses[mask_out] - self.ports_poses[:,None,:]).reshape(-1,2),axis=1).reshape(len(self.ports_poses),-1)<8
            rewards_goal[mask_out] += (mask_1*mask_2).any(axis=0)[:,None]*20

        # or
        self.poses[:,1][self.poses[:,0]>=self.w] = self.ports[0]+np.random.choice([3,-3])
        self.poses[:,1][self.poses[:,0]<0] = self.ports[1]+np.random.choice([3,-3])

        self.poses[:,0][self.poses[:,1]>=self.h] = self.ports[2]+np.random.choice([3,-3])
        self.poses[:,0][self.poses[:,1]<0] = self.ports[3]+np.random.choice([3,-3])
        
        self.poses = self.poses%np.array([self.w,self.h])
        
        if mask_out.any() :
            self.starting_poses[mask_out] = self.poses[mask_out].copy()
        
        self.acceleration = ((self.speeds*2.5)-(self.history[-1][:,0:1]))*2.5 # NOTE
        self.old_velocity = self.velocity.copy()
        # NOTE before clip to evaluate true action
        reward = self.rewards_test(actions,actions_d) + rewards_goal
        
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
                info.append({'episode':{'r':self.rs[i],'l':self.time}})
        if all(done):
            _ = self.reset(num_agents = self.num_agents , max_steps= self.max_time,training_time=training_time)
        else:
            self.history.append(self.states.copy())
            self.history_poses.append(self.poses.copy())
            
        return self.states.copy(), np.array([r for r in reward]), done, info, self.imgs_states
        
    
    
    def rewards_test(self,actions,actions_d):
        
        # check collosion between vru and non-vru
        #distances = np.linalg.norm(self.poses - self.poses[:,None,:],axis=2)+1e-6
        #revert_self = np.logical_not(np.eye(self.num_agents).astype(bool))

        #rewards = ((-1/((distances*2)**2))*(distances<18)*revert_self).sum(axis=1)[:,None]*self.fixed_reward_factor*abs(self.speeds/5)
        #self.collsions +=  (((distances<0.3)*revert_self).sum(axis=1)[:,None]).sum() 
        #rewards = np.clip(rewards,-1e2,0)

        # check collosion between vru and non-vru
        distances = np.linalg.norm(self.poses - self.poses[:,None,:],axis=2)
        revert_self = np.logical_not(np.eye(self.num_agents).astype(bool))
        types_mat_1 = (self.types>1)[:,None]
        rewards = -0.8*(((distances<1.0)*revert_self).sum(axis=1)[:,None])*self.fixed_reward_factor
        rewards = -2.0*(((distances<0.4)*revert_self).sum(axis=1)[:,None])*self.fixed_reward_factor
        self.collsions += (((distances<0.3)*revert_self).sum(axis=1)[:,None]).sum()/2
        ################
        # check vrus on side walk or road TODO (quick on roads)
        vru_poses = (self.poses[(self.types<2)].T-0).astype(int)
        #try:
        rewards[(self.types<2)] -= (4*np.logical_not(
            (self.sidewalk[vru_poses[1],vru_poses[0],1]+self.road[vru_poses[1],vru_poses[0],1]).astype(bool)))[:,None]*self.fixed_reward_factor
        rewards[(self.types<2)] -= (0.30*np.logical_not((self.sidewalk[vru_poses[1],vru_poses[0],1]).astype(bool)))[:,None]*self.fixed_reward_factor
        
        ############
        # check nonvru on road only
        nonvru_poses = (self.poses[types_mat_1.T[0]].T-0).astype(int)
        rewards[types_mat_1.T[0]] -= (4.5*np.logical_not(self.road[nonvru_poses[1],nonvru_poses[0],1].astype(bool)))[:,None]*self.fixed_reward_factor
        # except IndexError: 
        
        ### speed should be big for cars and no stop of vrus NOTE (added) (sligtly encourge high speeds)
        #rewards += abs(0.03*(abs(self.speeds.T[0])>1))[:,None]
        # pedestrains
        rewards[(self.types==0)] += (0.05*((abs(self.speeds.T[0])[(self.types==0)])>0.5))[:,None]
        rewards[(self.types==1)] += (0.05*((abs(self.speeds.T[0])[(self.types==1)])>0.9))[:,None] # bikes
        rewards[(self.types==2)] += (0.05*((abs(self.speeds.T[0])[(self.types==2)])>2))[:,None] # cars

        #rewards += (actions[:,0]/100)[:,None]
        if False:
            avg_speed_ped = np.linalg.norm((self.starting_poses[(self.types==0)] - self.poses[(self.types==0)]),axis=1)/(self.time+1)
            rewards[(self.types==0)] -= 0.25*(avg_speed_ped<0.4)[:,None]

            avg_speed_bike = np.linalg.norm((self.starting_poses[(self.types==1)] - self.poses[(self.types==1)]),axis=1)/(self.time+1)
            rewards[(self.types==1)] -= 0.25*(avg_speed_bike<0.6)[:,None]

            avg_speed_car = np.linalg.norm((self.starting_poses[(self.types==2)] - self.poses[(self.types==2)]),axis=1)/(self.time+1)
            rewards[(self.types==2)] -= 0.1*(avg_speed_car<2)[:,None]*(self.rs[(self.types==2)]<5) # not out yet.

        #rewards[(self.types==0)] += (0.1*((actions[:,0][(self.types==0)])))[:,None]
        #rewards[(self.types==1)] += (0.02*((actions[:,0][(self.types==1)])))[:,None] # bikes
        #rewards[(self.types==2)] += (0.000002*((actions[:,0][(self.types==2)])))[:,None] # cars
        #rewards -= abs(0.05*(actions[:,0]<2))[:,None]
        #rewards[types_mat_1.T[0]] += abs(0.04*((actions[:,0][types_mat_1.T[0]])>3))[:,None]

        #if False:#len(self.history_z)>(self.reward_traj_length-1) and 

            modes_vec = torch.as_tensor(np.vstack(self.history_z[-(self.reward_traj_length):]).T,dtype=torch.float,device=device)

            with torch.no_grad():
                for ii in range(3):
                    traj_reward = self.trained_reward(modes_vec[self.types==ii],type_n=ii).cpu().numpy()
                    rewards[(self.types==ii)] += (np.clip(traj_reward,-5,0)/2.0)+(np.clip(traj_reward,0,5)/30)

        if self.time>0:

            trajs = np.vstack(self.history_z).T[:,-min(self.time+1,self.traj_mode_len):] #17

            #traj_len = trajs.shape[1]
            for type_ii,traj,act_d,n_i in zip(self.types,trajs,actions_d,range(self.num_agents)):
                final_reward = np.zeros(self.n_modes)
                traj_len = (traj != -1 ).sum()
                for ii in range(traj_len):
                    final_reward += self.full_reward_model[type_ii][traj[ii],traj_len-ii-1]

                rewards[n_i] += final_reward[act_d]/(traj_len)
            #final_reward = np.zeros(20)
            #for i,mode in enumerate(self.history_z[-min(self.time,17):]):
            #    final_reward *= self.full_model[type_i][mode,min(self.time,17)-i,:]

        return rewards

        
    def reset(self, num_agents = 15 + np.random.randint(15),max_steps=17,pixel2meter=None,training_time=1):

        # training_time is factor from 0 to 1 (percentage of done steps/total steps)
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
            self.headings *= (training_time*self.train)#4#6
            new_headings = self.heading_img[self.poses.T[1].astype(int),self.poses.T[0].astype(int),0]
            step=0
            while not(new_headings.astype(bool).all()):
                step += 1
                for a,b in zip([step,-step,step,-step],[step,step,-step,-step]):
                    new_headings += self.heading_img[np.clip(self.poses.T[1].astype(int)+a,0,self.h-1),
                                        np.clip(self.poses.T[0].astype(int)+b,0,self.w-1),0]*(np.logical_not(new_headings.astype(bool)))
        
            self.headings += new_headings

        self.speeds = abs((np.random.rand(self.num_agents,1)-0.5)*2)
        self.speeds += (self.types[:,None])*4 # bikes and cars faster on y
        
        self.velocity = np.vstack((np.cos(self.headings),np.sin(self.headings))).T*self.speeds
        self.acceleration = np.zeros_like(self.speeds)
        self.old_velocity = self.velocity.copy()
        
        self.max_time = max_steps
        
        self.time = 0
        self.collsions = 0
        
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

        
    def make_image_(self,scale=0):


        scale = self.default_scale
        # TODO delete scale: Now it is global parameter
        #self.grid = (self.road + self.sidewalk).copy()#np.zeros((int(self.w/size),int(self.h/size),3))
        #self.grid = cv2.resize(self.grid,dsize=[self.w*scale,self.h*scale])
        self.grid = self.initial_grid.copy()
        colors = np.array([[0,255.0,0],[255.0,0,0],[0,0,255.0]])
        #tringle_pnts = np.array([[0,1],[1,0],[0,-1]])*0.1

        for i  in range(self.num_agents):
            if self.types[i]==1:
                #breakpoint()
                #tringle_pnts *= np.array([[np.cos(self.headings[i]),np.sin(self.headings[i])]]*3)
                #tringle_pnts += self.poses[i]
                #for j in range(3):
                    
                #    self.grid = cv2.line(self.grid,tringle_pnts[j-1].astype(int),tringle_pnts[j].astype(int),
                #                    color=colors[self.types[i].astype(int)],thickness=1) 
                p1 = (self.poses[i]-np.array([np.cos(self.headings[i]),np.sin(self.headings[i])])*2).astype(int)*scale
                p2 = (self.poses[i]+np.array([np.cos(self.headings[i]),np.sin(self.headings[i])])*2).astype(int)*scale
                self.grid = cv2.arrowedLine(self.grid,p1,p2,
                                    color=colors[self.types[i].astype(int)],thickness=2)      


            elif self.types[i] == 0:
                #pedestrains
                self.grid = cv2.circle(self.grid,self.poses[i].astype(int)*scale,int(self.types[i]/2)+scale+1,
                                       color=colors[self.types[i].astype(int)],thickness=-1)
            else:
                p1 = (self.poses[i]-np.array([np.cos(self.headings[i]),np.sin(self.headings[i])])*(scale-1))
                p2 = (self.poses[i]+np.array([np.cos(self.headings[i]),np.sin(self.headings[i])])*(scale-1))
                p1a = (p1+np.array([np.sin(self.headings[i]+(np.pi/4)),-np.cos(self.headings[i]+(np.pi/4))])*1).astype(int)*scale
                p2a = (p2-np.array([np.sin(self.headings[i]+(np.pi/4)),-np.cos(self.headings[i]+(np.pi/4))])*1).astype(int)*scale
                #self.grid = cv2.arrowedLine(self.grid,(p1*scale).astype(int),(p2*scale).astype(int),
                #                    color=colors[self.types[i].astype(int)],thickness=2)     
                self.grid = cv2.rectangle(self.grid,p1a,p2a,color=colors[self.types[i].astype(int)],thickness=-1)

        self.grid_full = self.grid.copy()
        self.grid = cv2.resize(self.grid,dsize=[self.w,self.h],interpolation=cv2.INTER_LINEAR)
        
    def render(self,scale=3):
        
        self.im_to_show = cv2.resize(self.grid_full,dsize=[self.w*scale,self.h*scale])
        cv2.imshow('GAME',self.im_to_show)

 
def main():
    N_MODES=20
    
    env = TrafficEnv(make_img=True,num_agents=64,img_size=[20,40],n_modes=N_MODES,train=False)#[12,20]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(f'ppo_agent_ind_image_d_smoothed_first_step_kmeans_{N_MODES}.pth',map_location=device)
    model = torch.load(f'ppo_agent_ind_image_d_smoothed_first_step_kmeans_with_reward_{N_MODES}.pth',map_location=device)
    #model = torch.load('ppo_agent_ind_image_d_last_step_5.pth',map_location=device)
    model.center = [10,6]
    init_vid = True

    ###########################

    for scene in [0,1,2,3,4,5]:
        done,new_state,new_img_state = env.reset(num_agents=16*(scene+1))
        if init_vid:
            #fourcc = cv2.VideoWriter_fourcc(*'XVID') 
            #out = cv2.VideoWriter('output1.avi', fourcc, fps, frame_size)
            env.max_steps = 32*50
            init_vid = False
        if type(done)==list:
            done = done[0]
        env.max_time = int(16*8)
        all_rewards = 0
        while not(done):
            #new_img_state = np.concatenate((env.past_imgs_states,new_img_state),axis=1)
            with torch.no_grad():
                action = model.get_action(torch.Tensor(new_state).to(device=device),torch.Tensor(new_img_state).to(device=device),best=True).cpu().numpy()#*np.array([0.025,0])
                #action_bc = env.bc_models.get_z(torch.Tensor(new_state).to(device=device)).argmax(axis=1).cpu().numpy()
            new_state, rewards, done,info,new_img_state = env.step(action)#-env.poses)
            all_rewards += rewards
            if type(done)==list:
                done = done[0]
            env.make_image_()
            env.render(scale=3)

            cv2.waitKey(1)#000)
        print(all_rewards.mean())
        print('GAME OVER')


if __name__=='__main__':
    
    main()
 


