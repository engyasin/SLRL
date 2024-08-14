#import pandas as pd
import numpy as np
from utils import *
#from matplotlib import pyplot as plt


#from sklearn.cluster import KMeans

import glob,os
import os
import cv2


def get_random_scene(w,h,lane_width = 2):
    
    img = np.zeros((h,w,3))
    
    #main road
    a = np.random.randint(0,h//2),0
    b = np.random.randint(h//2,h),(w-1)
    main_lanes = np.random.randint(1,5)*lane_width
    
    dy_step = abs(a[0]-b[0])/w
    center_line = (a[0]+(np.arange(w)*dy_step)).astype(int)
    
    cv2.line(img,np.array([a[1],a[0]]),np.array([b[1],b[0]]),color=[255,255,255],thickness=main_lanes)
    heading_img = np.zeros((h,w,3))
    if True:
        template_img =  np.zeros((h,w,3))
        half_lane = main_lanes//2
        cv2.line(template_img,np.array([a[1],a[0]-half_lane]),np.array([b[1],b[0]-half_lane]),color=[255,255,255],thickness=half_lane+1)

        heading_img[template_img>0] = np.arctan2(a[0]-b[0],a[1]-b[1])#+np.pi#/2#y/x (for reverse without pi)
        template_img =  np.zeros((h,w,3))
        cv2.line(template_img,np.array([a[1],a[0]+half_lane]),np.array([b[1],b[0]+half_lane]),color=[255,255,255],thickness=half_lane+1)
        heading_img[template_img>0] = np.arctan2(a[0]-b[0],a[1]-b[1])+np.pi#/2#y/x (for reverse without pi)
    
    # cutting roads
    c = 0,np.random.randint(0,w)
    d = (h-1),np.random.randint(0,w)
    sub_lanes = np.random.randint(1,3)*lane_width
    
    line_pnts = np.hstack((np.arange(w)[:,None],center_line[:,None]))
    pnts_c_idx = np.linalg.norm(line_pnts-np.array([c[1],c[0]]),axis=1).argmin()
    c_ops = np.array([pnts_c_idx,center_line[pnts_c_idx]])
    pnts_d_idx = np.linalg.norm(line_pnts-np.array([d[1],d[0]]),axis=1).argmin()
    d_ops = np.array([pnts_d_idx,center_line[pnts_d_idx]])
    
    cv2.line(img,np.array([c[1],c[0]]),c_ops,color=[255,255,255],thickness=sub_lanes)
    cv2.line(img,np.array([d[1],d[0]]),d_ops,color=[255,255,255],thickness=sub_lanes)
    cv2.circle(img,[c_ops,d_ops][np.random.randint(0,2)],color=[255,255,255],
               radius=np.random.randint(2,6)*lane_width,
               thickness=np.random.choice([-1,2]))
    
    if True:
        template_img = np.zeros((h,w,3))
        cv2.line(template_img,np.array([c[1],c[0]]),c_ops,color=[255,255,255],thickness=sub_lanes)
        heading_img[template_img>0] = np.arctan2(c[0]-c_ops[1],c[1]-c_ops[0])-(np.pi*np.random.choice([0,1]))#-np.pi/2#y/x (for reverse without pi)
        
        template_img = np.zeros((h,w,3))
        cv2.line(template_img,np.array([d[1],d[0]]),d_ops,color=[255,255,255],thickness=sub_lanes)
        heading_img[template_img>0] = np.arctan2(d[0]-d_ops[1],d[1]-d_ops[0])-(np.pi*np.random.choice([0,1]))#/2#y/x (for reverse without pi)
        
        
    
    img = cv2.dilate(img,cv2.getStructuringElement(2,[2,2]),iterations=1) # circle
    road = img.copy()
    
    sidewalk = cv2.dilate(img,cv2.getStructuringElement(2,[max(main_lanes,5),max(main_lanes,5)]),iterations=1) # circle
    sidewalk[road>0] = 0
    sidewalk[:,:,[0]] = 0
    
    return road,sidewalk/2,heading_img,[a[0],b[0],c[1],d[1]]


def main():
    
    w,h = 182,114
    img = np.zeros((h,w,3))
    
    
    lane_width = 3
    #main road
    a = np.random.randint(0,h//2),0
    b = np.random.randint(h//2,h),(w-1)
    main_lanes = np.random.randint(1,5)*lane_width
    
    dy_step = abs(a[0]-b[0])/w
    center_line = (a[0]+(np.arange(w)*dy_step)).astype(int)
    
    cv2.line(img,np.array([a[1],a[0]]),np.array([b[1],b[0]]),color=[255,255,255],thickness=main_lanes)
    

    
    # cutting roads
    c = 0,np.random.randint(0,w)
    d = (h-1),np.random.randint(0,w)
    sub_lanes = np.random.randint(1,3)*lane_width
    
    line_pnts = np.hstack((np.arange(w)[:,None],center_line[:,None]))
    pnts_c_idx = np.linalg.norm(line_pnts-np.array([c[1],c[0]]),axis=1).argmin()
    c_ops = np.array([pnts_c_idx,center_line[pnts_c_idx]])
    pnts_d_idx = np.linalg.norm(line_pnts-np.array([d[1],d[0]]),axis=1).argmin()
    d_ops = np.array([pnts_d_idx,center_line[pnts_d_idx]])
    
    cv2.line(img,np.array([c[1],c[0]]),c_ops,color=[255,255,255],thickness=sub_lanes)
    cv2.line(img,np.array([d[1],d[0]]),d_ops,color=[255,255,255],thickness=sub_lanes)
    cv2.circle(img,[c_ops,d_ops][np.random.randint(0,2)],color=[255,255,255],
               radius=np.random.randint(2,5)*lane_width,
               thickness=np.random.choice([-1,2]))
    

    
    img = cv2.dilate(img,cv2.getStructuringElement(2,[2,2]),iterations=1) # circle
    road = img.copy()
    
    sidewalk = cv2.dilate(img,cv2.getStructuringElement(2,[max(main_lanes,4),max(main_lanes,4)]),iterations=1) # circle
    sidewalk[road>0] = 0
    sidewalk[:,:,[0,2]] = 0
    breakpoint()
    cv2.imshow('result',sidewalk[:,:,1])
    cv2.waitKey()
    
    

if __name__=='__main__':
    
    main()
