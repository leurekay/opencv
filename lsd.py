#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 15:03:05 2018

@author: dirac
"""

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import solve


image_path='data/bank5.jpg'
n_display=5
eps_rho=20
eps_theta=0.05
threshold_filter=30

img_raw = cv2.imread(image_path)
img_hsv = cv2.cvtColor(img_raw,cv2.COLOR_BGR2HSV)

img = cv2.cvtColor(img_raw,cv2.COLOR_BGR2GRAY)



def clockwise_sort(X):
    """
    X: (N,2) numpyArray
    """
    X=np.array(X)
    center=np.mean(X,axis=0)
    center_X=X-center
    center_X_norm=center_X/np.linalg.norm(center_X,axis=1).reshape([-1,1])
#    xAxis=np.array([1,0])
    def get_complex(a):
        return complex(a[0],a[1])
    center_X_complex=np.apply_along_axis(get_complex,1,center_X_norm)
    angle=np.angle(center_X_complex)*180/np.pi
    order=np.argsort(angle)
    
    X=X[order]
    return center,X


def line_coef(line):
    line=np.array(line).reshape([4,])
    x1,y1,x2,y2=line
    if x1==x2:
        rho=x1
        if x1>=0:
            theta=0
            return rho,theta 
        else:
            theta=np.pi
            return abs(rho),theta 
    if y1==y2:
        rho=y1
        if y1>=0:
            theta=0.5*np.pi
            return rho,theta 
        else:
            theta=1.5*np.pi      
            return abs(rho),theta 
    
    k=(y1-y2)/float(x1-x2)
    b=y1-k*x1
    rho=abs(b)/np.sqrt(1+k**2)
    
    
    A = np.array([[x1,y1],[x2,y2]])
    b = np.array([rho,rho])
    try:
        x = solve(A,b)
    except :
        print ('777777777777777')
    
    cos,sin=x
    theta1=np.arccos(cos)
#    print (cos,sin,theta1)
    theta2=2*np.pi-theta1
    if abs(np.sin(theta1)-sin)<0.000001:
        theta=theta1
    if abs(np.sin(theta2)-sin)<0.000001:
        theta=theta2


    return rho,theta 

def line_coef2(line):
    pass


def points_project_onto_line(line_coef,points):
    """
    line_coef:[rho,theta] identify a unique line
    points: shape[N,2]
    
    return :coord on 1 dim project-axis
    """
    
    rho,theta=line_coef
    R=np.array([[np.cos(0.5*np.pi-theta),-np.sin(0.5*np.pi-theta)],
                 [np.sin(0.5*np.pi-theta),np.cos(0.5*np.pi-theta)]])
    
    new_xy=np.dot(points,R.T)
    return new_xy[:,0]





def merge_interval(intervals):
    """
    :type intervals: numpyArray shape:[N,2]
    :rtype: intervals after merged. [k,2] where k<=N
    """
    
    
    intervals=np.apply_along_axis(lambda x:[min(x),max(x)],1,intervals)
    
    n=len(intervals)
    if n==0:
        return []
    intervals=sorted(intervals,key=lambda x : x[0])
    box=[intervals[0]]
    for i in range(1,n):
        pre_s,pre_e=box[-1]
        cur_s,cur_e=intervals[i]
        if cur_s>pre_e:
            box.append(intervals[i])
        else:
            box[-1]=([pre_s,max(pre_e,cur_e)])
    return np.array(box)



def total_project_length(line_coef,lines):
    """
    project every line-segment onto the fix straight line which 
    described by rho and theta.
    then, compute the total length of projected line segments
    """
    
    
    left=points_project_onto_line(line_coef,lines[:,:2]).reshape([-1,1])
    right=points_project_onto_line(line_coef,lines[:,2:]).reshape([-1,1])
    intervals=np.concatenate([left,right],axis=1)
    merges=merge_interval(intervals)
    length=np.apply_along_axis(lambda x:x[1]-x[0],1,merges)
    tot_length=np.sum(length)
    return tot_length



def farthest2points(points):
    """
    points:[N,2]
    """
    points=np.array(points)
    points=points.reshape([-1,2])
    ret=0
    N=points.shape[0]
    for i in range(N):
        for j in range(i+1,N):
            delta=points[i]-points[j]
            dist=np.sqrt(delta[0]**2+delta[1]**2)
            if dist>ret:
                ret=dist
    return ret


def cluster(lines_info):
    """
    lines_info:[N,7] 
        e.g. [[x1,y1.x2,y2,rho,theta,length],
              [............................]]
        
    return : dict {(rho,theta):[index1,index2...],...}
    """
   
    n_line,n_feature=lines_info.shape
    assert n_feature==7
    dic={tuple(lines_info[0][4:6]):[0]}
    for i in range(1,n_line):
        line=lines_info[i][:4]
        rho,theta,length=lines_info[i][4:7]
        
        flag=0
        for key in dic:
            if abs(rho-key[0])<eps_rho and abs(theta-key[1])<eps_theta:
                dic[key].append(i)
                flag=1
                sub_lines_info=lines_info[dic[key]]
                total_len=np.sum(sub_lines_info[:,6])
                ratio_coef=np.apply_along_axis(lambda x : x[4:6]*x[6]/float(total_len),1,sub_lines_info)
                new_key=ratio_coef.sum(axis=0)
                dic[tuple(new_key)]=dic[key]
                dic.pop(key)
                break
        if flag==0:
            dic[(rho,theta)]=[i]
    return dic

def mass_center_of_lines(lines):
    masses=np.apply_along_axis(lambda x:np.sqrt((x[0]-x[2])**2+(x[1]-x[3])**2),1,lines).reshape([-1,1])
    centers=np.apply_along_axis(lambda x:[(x[0]+x[2])/2.0,(x[1]+x[3])/2.0],1,lines)
    mass=np.sum(masses)
    center=np.dot(masses.T,centers)/mass
    return center.reshape([1,-1])


def render(img,lines,color=None,thick=None):
    drawn_img=np.copy(img)    
    lines=np.array(lines)
    shape=lines.shape
    if len(shape)==1:
        if shape[0]==0:
            return drawn_img
        lines=lines.reshape([1,-1])
    shape=lines.shape
    assert shape[1]==2 or shape[1]==4
    if shape[1]==2:
        for line in lines:
            rho,theta=line
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 3000*(-b))
            y1 = int(y0 + 3000*(a))
            x2 = int(x0 - 3000*(-b))
            y2 = int(y0 - 3000*(a))
            if color and thick:
                cv2.line(drawn_img,(x1,y1),(x2,y2),color,thick)
            else:
                cv2.line(drawn_img,(x1,y1),(x2,y2),(0,255,0),4)       
    if shape[1]==4:
        if color and thick:
            for x1,y1,x2,y2 in lines:
                cv2.line(drawn_img,(x1,y1),(x2,y2),color,thick)
        else:
            drawn_img = lsd.drawSegments(drawn_img, lines)
    return drawn_img
  
def combination(n, k):
    """
    :type n: int
    :type k: int
    :rtype: List[List[int]]
    """
    box=[]
    def dfs(a):
        if len(a)==k:
            box.append(a)
        else:
            if len(a)==0:
                s=1
            else:
                s=a[-1]+1
            for v in range(s,n+1):
                dfs(a+[v])
    
    dfs([])
    return box

def permutation(n,k):
    box=[]
    def dfs(a):
        if len(a)==k:
            box.append(a)
        else:
            for v in range(1,n+1):
                if v not in a:
                    dfs(a+[v])
    
    dfs([])
    return box
        

def crosspoint_matrix(lines):
    n=len(lines)
    M=np.zeros([n,n,2])
    for i in range(n):
        for j in range(i+1,n):
            rho1,theta1=lines[i]
            rho2,theta2=lines[j]
            a1,b1 = np.cos(theta1),np.sin(theta1)
            a2,b2 = np.cos(theta2),np.sin(theta2)
    
            a = np.array([[a1,b1],[a2,b2]])
            b = np.array([rho1,rho2])
            
            try:
                x = solve(a, b)
            except :
                M[i,j,:]=M[j,i,:]=-1
#                print ('================\n',a,b,'\n==============')
            else:
                M[i,j,:]=M[j,i,:]=x        
    return M


class PolarLine():
    def f(self):
        return 6
    def __init__(self,img,base_coef,indexs,lines_info):
        self.img=img
        self.shape=img.shape
        self.base_coef=np.array(base_coef)
        self.indexs=indexs
        self.lines=lines_info[indexs,:4]
        self.coefs=lines_info[indexs,4:6]
        self.lengths=lines_info[indexs,6]
        self.len_project=total_project_length(self.base_coef,self.lines)
        self.center=mass_center_of_lines(self.lines)
    def sample(self,n):
        w,h,_=self.shape
        box=[]
        count=0
        while count<n:
            point=np.random.normal(loc=list(self.center),scale=[10,10],size=[1,2])
            if point[0][0]>0 and point[0][0]<w and point[0][1]>0 and point[0][1]<h:
                box.append(point)
                count+=1
        return np.concatenate(box,axis=0)
    
    def hsv(self,n):
        pass
    
    def get(self):
        return self.f()



    

#detect all lines by LSD 
lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_NONE)
dlines = lsd.detect(img)
lines = lsd.detect(img)[0]  # Position 0 of the returned tuple are the detected lines
lines=lines.reshape([-1,4])

#filter  shorter lines
filter_index=np.apply_along_axis(lambda x:(x[0]-x[2])**2+(x[1]-x[3])**2>threshold_filter**2,1,lines)
lines=lines[filter_index]

#polar coeffient of the lines
coefs=np.apply_along_axis(lambda x : np.array(list(line_coef(x))),1,lines)

#corresponding length of each line-segement
lengths=np.apply_along_axis(lambda x:np.sqrt((x[0]-x[2])**2+(x[1]-x[3])**2),1,lines).reshape([-1,1])

#combine the line coordinates ,coefs ,lengths into a matrix
lines_info=np.concatenate([lines,coefs,lengths],axis=1)

#group lines into several clusters by their rho and theta
dic=cluster(lines_info)

base_coef_set=[PolarLine(img_hsv,key,dic[key],lines_info) for key in dic.keys()]
base_coef_set=sorted(base_coef_set,key=lambda x:x.len_project,reverse=True)
base_coef_set=base_coef_set[:n_display]

straights=list(map(lambda x: x.base_coef,base_coef_set))




drawn_img=render(img_raw,lines)
drawn_img=render(drawn_img,straights)
#drawn_img=render(drawn_img,[200,400,700,500],(100,100,100),5)

#colors=np.random.randint(40,255,[1000,3])
#drawn_img=img_raw
#for i,line in enumerate(lines):
#    color=(int(colors[i][0]),int(colors[i][1]),int(colors[i][2]))
#    drawn_img=render(drawn_img,line,color=color,thick=2)
#drawn_img=render(drawn_img,straights)


#p=PolarLine(zippo[3][1],dic[zippo[3][1]],lines_info)
#p.mass_center()
#cc=p.center
#cv2.circle(drawn_img,(int(cc[0][0]),int(cc[0][1])),25,(14,124,155),-1)



fig=plt.figure(figsize=[20,15])
plt.imshow(drawn_img)

pp=permutation(8,4)
cc=combination(8,4)
M=crosspoint_matrix(straights)