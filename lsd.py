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




img = cv2.imread('data/bank7.jpg')
img_=np.copy(img)

grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#img=cv2.GaussianBlur(grey,(5,5),4)
img=grey

def hough_line(img,threh_hough):
    
    height,width=img.shape[0],img.shape[1]
    edges = cv2.Canny(img,90,150)
    lines = cv2.HoughLines(edges,1,np.pi/180,threh_hough)
    lines=lines.reshape([-1,2])
    n,_=lines.shape
    
    remove_list=[]
    
    box=[]
    for i in range(n):
        if i in remove_list:
            continue
        for j in range(i+1,n):
            if j in remove_list:
                continue
            rho1,theta1=lines[i]
            rho2,theta2=lines[j]
            if abs(rho1-rho2)<50 and abs(theta1-theta2)<0.1:
                remove_list.append(j)
                continue
            
            a1,b1 = np.cos(theta1),np.sin(theta1)
            a2,b2 = np.cos(theta2),np.sin(theta2)
    
            a = np.array([[a1,b1],[a2,b2]])
            b = np.array([rho1,rho2])
            
            try:
                x = solve(a, b)
            except :
                pass
#                print ('================\n',a,b,'\n==============')
            else:
                if (x[0]>0 and x[0]<width) and (x[1]>0 and x[1]<height):
                    box.append(x)
    box=np.array(box)
#    print (box)
    
    index_line=[x for x in range(n) if x not in remove_list]
    return lines[index_line],box


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
            return rho,theta 
    if y1==y2:
        rho=y1
        if y1>=0:
            theta=0.5*np.pi
            return rho,theta 
        else:
            theta=1.5*np.pi      
            return rho,theta 
    
    k=(y1-y2)/float(x1-x2)
    b=y1-k*x1
    rho=abs(b)/np.sqrt(1+k**2)
    
    
    A = np.array([[x1,y1],[x2,y2]])
    b = np.array([rho,rho])
    try:
        x = solve(A,b)
    except :
        print ('777777777777777')
    
    theta=np.arccos(x[0])
            
    return rho,theta 

def line_coef2(line):
    pass


def points_project_onto_line(line_coef,points):
    """
    line:[rho,theta] identify a unique line
    points: shape[N,2]
    
    return :coord on 1 dim project-axis
    """
    
    rho,theta=line_coef
    R=np.array([[np.cos(0.5*np.pi-theta),-np.sin(0.5*np.pi-theta)],
                 [np.sin(0.5*np.pi-theta),np.cos(0.5*np.pi-theta)]])
    
    new_xy=np.dot(points,R.T)
    return new_xy[:,0]

def union_length1d():
    pass



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


def cluster(lines,coefs):
    assert lines.shape[0]==coefs.shape[0]
    dic={tuple(coefs[0]):[0]}
    n_line=lines.shape[0]
    for i in range(1,n_line):
        rho,theta=coefs[i]
        
        flag=0
        for key in dic:
            if abs(rho-key[0])<4 and abs(theta-key[1])<0.01:
                dic[key].append(i)
                flag=1
                new_key=coefs[dic[key]].mean(axis=0)
                dic[tuple(new_key)]=dic[key]
                dic.pop(key)
                break
        if flag==0:
            dic[(rho,theta)]=[i]
    return dic

lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_NONE)

# Detect lines in the image
dlines = lsd.detect(img)#TODO 返回什么？
lines = lsd.detect(img)[0]  # Position 0 of the returned tuple are the detected lines
lines=lines.reshape([-1,4])


#filter  very short line
filter_index=np.apply_along_axis(lambda x:(x[0]-x[2])**2+(x[1]-x[3])**2>30**2,1,lines)
lines=lines[filter_index]

#polar coeffient of the lines
coefs=np.apply_along_axis(lambda x : np.array(list(line_coef(x))),1,lines)

length=np.apply_along_axis(lambda x:np.sqrt((x[0]-x[2])**2+(x[1]-x[3])**2),1,lines)

#group lines into several clusters 
dic=cluster(lines,coefs)

dic2={}
for key in dic:
    dic2[key]=length[dic[key]].sum()

zippo = zip(dic2.values(),dic2.keys())
zippo=sorted(zippo,reverse=True)

straights=zippo[:0]
straights=list(map(lambda x: x[1],straights))


# Draw detected lines in the image
drawn_img = lsd.drawSegments(img, lines)



for line in straights:
    rho,theta=line
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(drawn_img,(x1,y1),(x2,y2),(0,255,0),6)
    
#cv2.imshow('LSD',drawn_img)
#cv2.waitKey(0)
    
#fig=plt.figure(figsize=[20,15])
#plt.imshow(drawn_img)