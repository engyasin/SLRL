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

import torch
from trafficenv_D import TrafficEnv


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


SCENE_ID = 11# 11:2590,12:2240

class TrafficEnvMod(TrafficEnv):
    
    def __init__(self, w=182, h=114, num_agents=..., max_steps=17, 
                 pixel2meter=None, make_img=False, img_size=[10,10],
                 n_modes=20,first_step=True):
        
        img_file = 'background_img.png'

        self.bg_img = cv2.imread(img_file)

        h,w,_ = self.bg_img.shape
        bg_img = cv2.resize(self.bg_img,dsize=[int(w),int(h)])
        self.road,self.sidewalk = bg_img.copy(),bg_img.copy()
        self.ports = np.array([[139,52],[113,73],[82,124],[115,117],[127,106],[159,70]])
        self.w,self.h = bg_img.shape[1],bg_img.shape[0]
        
        self.wh = [self.w,self.h]
        
        self.time = 0
        #[[0,0],[self.w,3],[3,0],[self.w,self.h-3]]
        
        self.rs = 0
        self.r1 = 5
        self.r2 = 10
        self.r3 = 20
        
        self.make_img = make_img
        self.imgs_states = None
        self.img_size = img_size
        
        self.first_step = first_step
        self.n_modes = n_modes
        

        if self.first_step:
            self.bc_models = torch.load(f'bc_agent_unid_{35}_{self.n_modes}_one_step.pth',map_location=device)
        else:
            self.bc_models = torch.load(f'bc_agent_unid_{35}_{self.n_modes}_last_step.pth',map_location=device)
            
        self.bc_models.eval()
        
        
        _ = self.reset(num_agents=num_agents, max_steps=max_steps, pixel2meter=pixel2meter)
        
        
    def get_demo_data(self,scene_ids=[30],step_id=0):
        
        if step_id==0:
            files_ids = scene_ids
            root = './unids/'
            self.all_files = []
            self.all_meta_and_statics = []
            for x in files_ids:
                idx = '{0:02d}'.format(x)
                meta_info_file = os.path.join(root,f'{idx}_recordingMeta.csv')
                file_ = os.path.join(root,f'{idx}_tracks.csv')
                static_file = os.path.join(root,f'{idx}_tracksMeta.csv')

                meta_info = pd.read_csv(meta_info_file)
                static_ = pd.read_csv(static_file)

                self.all_files.append(pd.read_csv(file_))
                self.all_meta_and_statics.append((meta_info,static_))
            self.all_types_dict = {}

            for id_ in scene_ids:
                self.all_types_dict.update({id_:{}})
                for agent_type,rows_ in self.all_meta_and_statics[0][1].groupby('class'):
                    self.all_types_dict[id_].update({agent_type:rows_['trackId'].to_numpy()}) 
            
            self.parked_cars = []
            for trackId,rows_ in self.all_files[0].groupby('trackId'):
                #if all_meta_and_statics[30][1]['class'] == 'car':
                if trackId in self.all_types_dict[scene_ids[0]]['car']:
                    #print(trackId)
                    if (np.round(np.diff(rows_['xCenter'].to_numpy()),2).any()) or (np.round(np.diff(rows_['yCenter'].to_numpy()),2).any()):
                        pass
                    else:
                        self.parked_cars.append(trackId)
                        
        poses = []
        tracks_ids = []
        velocity = []
        acceleration = []
        heading = []
        types_ = []  
                    
        for frame,rows_ in self.all_files[0].groupby('frame'):
            if (int(step_id+(step_id>1))*10)==frame:
                poses = []
                tracks_ids = []
                velocity = []
                acceleration = []
                heading = []
                types_ = []
                for trackId,datapnt in rows_.groupby('trackId'):
                    if trackId in self.parked_cars:
                        continue
                    poses.append([datapnt['xCenter'].to_numpy()[0],-datapnt['yCenter'].to_numpy()[0]])
                    tracks_ids.append(trackId)
                    acceleration.append([datapnt['xAcceleration'].to_numpy()[0],datapnt['yAcceleration'].to_numpy()[0]])
                    velocity.append([datapnt['xVelocity'].to_numpy()[0],datapnt['yVelocity'].to_numpy()[0]])
                    heading.append(2*np.pi-(datapnt['heading'].to_numpy()[0]*np.pi/180))#-np.pi
                    
                    if trackId in self.all_types_dict[scene_ids[0]]['car']:
                        types_.append(2)
                    elif trackId in self.all_types_dict[scene_ids[0]]['pedestrian']:
                        types_.append(0)
                    elif trackId in self.all_types_dict[scene_ids[0]]['bicycle']:
                        types_.append(1)
                    else:
                        types_.append(2)
                
                break
            else:
                continue
        if len(velocity):
            vel_ = np.linalg.norm(np.array(velocity),axis=1)[:,None]
            acc_ = np.linalg.norm(np.array(acceleration),axis=1)[:,None]
        else:
            vel_ = []
            acc_ = []

        
        return np.array(poses),np.array(types_),vel_,acc_,np.array(heading),(tracks_ids)

        
    def update(self,poses,types,speeds,acceleration,headings,trackIds):
        to_delete = np.lib.setdiff1d(self.trackIds,trackIds)
        to_add = np.lib.setdiff1d(trackIds,self.trackIds)
        
        for track in to_add:
            j = trackIds.index(track)
            self.poses = np.append(self.poses,poses[j:j+1,:],axis=0)
            self.types = np.append(self.types,types[j])
            self.speeds = np.append(self.speeds,speeds[j:j+1,:],axis=0)
            self.acceleration = np.append(self.acceleration,acceleration[j:j+1,:],axis=0)
            self.headings = np.append(self.headings,headings[j])
            self.trackIds.append(track)
            self.step_orders.update({track:0})
            
        for track in to_delete:
            j = self.trackIds.index(track)
            self.poses = np.vstack((self.poses[:j],self.poses[j+1:,:]))
            self.types = np.hstack((self.types[:j],self.types[j+1:]))
            self.speeds = np.vstack((self.speeds[:j],self.speeds[j+1:]))
            self.acceleration = np.vstack((self.acceleration[:j],self.acceleration[j+1:]))
            self.headings = np.hstack((self.headings[:j],self.headings[j+1:]))
            self.trackIds.pop(j)
            
        self.num_agents = self.poses.shape[0]
        self.zs = self.types.copy()
        
        self.rs = 0
        
        self.near_objs = self.update_near_objs()
        
        
        
        if self.make_img:
            self.make_image()
            self.imgs_states = np.array([self.get_agent_image(i) for i in range(self.num_agents)])
        try:
            self.states = np.hstack((abs(self.speeds*2.5),(self.headings[:,None]%(2*np.pi)),self.near_objs,self.zs[:,None],abs(self.acceleration)))
        except ValueError:
            breakpoint()
            
        self.history = []
        self.history.append(self.states.copy())
        self.history_poses = []
        self.history_poses.append(self.poses.copy())
        
        return self.states.copy(),self.imgs_states

    
    def reset(self, num_agents=..., max_steps=17, pixel2meter=None):
        
        
        self.grid = np.zeros((self.h,self.w,3)) # 1 pixel = 1 meter
        self.poses = []
        
        while True:
            self.poses,self.types,self.speeds,self.acceleration,self.headings,self.trackIds = self.get_demo_data(scene_ids=[SCENE_ID], step_id=self.time - (self.time!=0))
            
            if len(self.poses)==0:
                self.time += 1
            else:
                break
        self.speeds /= 2.5
        self.step_orders = {track:0 for track in self.trackIds}

        self.velocity = np.vstack((np.cos(self.headings),np.sin(self.headings))).T*self.speeds
        
        self.old_velocity = self.velocity.copy()
        
        self.num_agents = self.poses.shape[0]
        # state: speed,near,(type),lost
        self.zs = self.types.copy()
        
        self.starting_port = []#np.linalg.norm((self.poses - self.ports[:,None,:]).reshape(-1,2),axis=1).reshape(len(self.ports),-1).argmin(axis=0)        

        #self.headings[(self.types>0)*(self.starting_port==5)] = -1*np.pi/4
        #self.headings[(self.types>0)*(self.starting_port==6)] = 3*np.pi/4
        
        self.max_time = self.time+max_steps
    
        
        self.rs = 0
    
        self.near_objs = self.update_near_objs()
        
        if self.make_img:
            self.make_image()
            
            self.imgs_states = np.array([self.get_agent_image(i) for i in range(self.num_agents)])
        
        self.states = np.hstack((abs(self.speeds)*2.5,(self.headings[:,None]%(2*np.pi)),self.near_objs,self.zs[:,None],self.acceleration))
        self.history = []
        self.history.append(self.states.copy())
        self.history_poses = []
        self.history_poses.append(self.poses.copy())       
        self.history_z = []

        return [False for _ in range(self.num_agents)], self.states, self.imgs_states

    def rewards_test(self,actions,z):
        
        return np.zeros_like(self.speeds)




if __name__ == '__main__':
    
    N_modes = 20 
    max_steps = 32
    
    
    env = TrafficEnvMod(make_img=True,img_size=[20,30],first_step=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_bc = torch.load(f'bc_agent_unid_{35}_{N_modes}_last_step.pth',map_location=device)
    #model = torch.load('ppo_agent_unid_image_d_last_step_1.pth',map_location=device)
    model = torch.load(f'ppo_agent_unid_image_d_smoothed_{["last_step","first_step"][env.first_step]}_new.pth',map_location=device)
    
    full_errors = []
    errors_dict_ped = {i:[] for i in range(max_steps)}
    errors_dict_bi = {i:[] for i in range(max_steps)}
    errors_dict_car = {i:[] for i in range(max_steps)}
    errors_dicts = [errors_dict_ped,errors_dict_bi,errors_dict_car]
    num_agents_minmax = []
    while env.time<2590:
        done,new_state,new_img_state = env.reset(max_steps=max_steps)

        step_order = 1

        set_rule = False
        while not(done[0]):
            action_ = np.random.random((env.num_agents,2+N_modes))
            with torch.no_grad():
                action_t,z_logits = env.bc_models.get_action(torch.Tensor(new_state).to(device),best=True)
                action_ = torch.hstack((action_t,torch.sigmoid(z_logits))).cpu().numpy()

            while True:
                poses,types,speeds,acceleration,headings,tracks = env.get_demo_data(scene_ids=[SCENE_ID], step_id=env.time + (env.time==0))
                if len(poses)==0:
                    env.time += 1
                else:
                    break
            
            if set_rule :
                common_agents = np.lib.intersect1d(env.trackIds,tracks)
                vel = poses[[tracks.index(t) for t in common_agents],:] - env.poses[[env.trackIds.index(t) for t in common_agents],:]
                action_gt = vel.copy()
                for m in range(vel.shape[0]):
                    c,s = np.cos(-env.headings[m]),np.sin(-env.headings[m])
                    R_mat = np.array([[c, -s], [s, c]])
                    action_gt[m] = R_mat@ vel[m].T
                set_rule = False
                
            new_state, rewards, done,info,new_img_state = env.step((z_logits.cpu().numpy()))
            if all(done): continue
            new_state, new_img_state = env.update(poses,types,speeds,acceleration,headings,tracks)
            
            vel = poses[[tracks.index(t) for t in env.trackIds],:]-env.poses
            num_agents_minmax.append(len(env.trackIds))
            errors = np.linalg.norm(vel,axis=1)[:,None]
            print(f'Steps: {[env.step_orders[t] for t in env.trackIds]}, ADE: {errors.T[0].mean()}')
            for t,err,type_ in zip(env.trackIds,errors.T[0],env.types):
                errors_dicts[type_][env.step_orders[t]].append(err)
            step_order += 1
            for k,v in env.step_orders.items():
                if k in env.trackIds:
                    env.step_orders.update({k:v+1})
                
            full_errors.append(errors.T[0].mean())

        print(f'time: {env.time}')
        
        print(f'final error: {np.mean(full_errors)}')
    print(f'max agents: {max(num_agents_minmax)}, mean: {np.mean(num_agents_minmax)}')
    for t in [0,1,2]:
        print(f'type: {t}')
        for k,v in errors_dicts[t].items():
            if len(v):
                print(f'step: {k}: error: {np.mean(v)}, size: {len(v)}')

