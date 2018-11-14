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


n_first=8
n_second=6
eps_rho=20
eps_theta=0.05
threshold_filter=30





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
        lines=lines.astype('int')
        if color and thick:
            for x1,y1,x2,y2 in lines:
                cv2.line(drawn_img,(x1,y1),(x2,y2),color,thick)
        else:
            lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_NONE)
            drawn_img = lsd.drawSegments(drawn_img, lines)
    return drawn_img


def render_point(img,points,color=None,thick=None):
    drawn_img=np.copy(img)    
    points=np.array(points,'int')
    shape=points.shape
    if len(shape)==1:
        if shape[0]==0:
            return drawn_img
        points=points.reshape([1,-1])
    shape=points.shape
    assert shape[1]==2
   
    for point in points:
        x,y=point
        if color and thick:
            cv2.circle(drawn_img,(x,y),10,color,thick)
        else:
            cv2.circle(drawn_img,(x,y),10,(145,154,55),10)       
    
    return drawn_img
    
def sample_line_sides(line_segement,n_sample,margin):
    """
    sample symmetry pair points along the line,without consider confinement
    """
    x1,y1,x2,y2=np.array(line_segement).reshape([-1])
    rho,theta=line_coef(line_segement)
    orthogonal_vector=np.array([np.cos(theta),np.sin(theta)])*margin
    xx=np.linspace(x1,x2,n_sample).reshape([-1,1])
    yy=np.linspace(y1,y2,n_sample).reshape([-1,1])
    online_samples=np.concatenate([xx,yy],axis=1)
    samples_pairs=np.concatenate([online_samples-orthogonal_vector,online_samples+orthogonal_vector],axis=1)
    return samples_pairs


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
                s=0
            else:
                s=a[-1]+1
            for v in range(s,n):
                dfs(a+[v])
    
    dfs([])
    return box

def permutation(n,k):
    box=[]
    def dfs(a):
        if len(a)==k:
            box.append(a)
        else:
            for v in range(0,n):
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



def isConvexPolygon(sorted_points):
    """
    sorted_points: shape [N,2]
    draw N points in turn,judge whether a convexPolygon with N sides
    """
    sorted_points=np.array(sorted_points)
    shape=sorted_points.shape
    assert shape[0]>=3 and shape[1]==2
    n=shape[0]
    vectors=[]
    for i in range(n):
        v=sorted_points[(i+1)%n]-sorted_points[i%n]
        vectors.append(v)
    dets=[]
    for i in range(n):
        v1=vectors[i%n]
        v2=vectors[(i+1)%n]
        V=np.array([v1,v2])
        det=np.linalg.det(V)
        if abs(det-0)<1e-9:
            return False
        dets.append(det>0)
    sets=set(dets)
    if len(sets)==1:
        return True
    else:
        return False

        



def find_kSide_from_nLine(lines,k,confinement=None):
    """
    return:all kSide,shape:[x,k]  clockwise!!
    [[point11,point12,...point1k],
     [point21,point22,...point2k],
     ............................]
    """
    lines=np.array(lines)
    assert lines.shape[1]==2
    n=lines.shape[0]
    M=crosspoint_matrix(lines)
    select_indexs=combination(n,k)
    
    box=[]
    for select in select_indexs:
        first_ind=select[0]
        permus=permutation(k-1,k-1)
        permus=map(lambda x: [first_ind]+[select[ele+1] for ele in x],permus)
        permus=list(permus)
        
        for permu in permus:
            points=[M[permu[i%k],permu[(i+1)%k],:] for i in range(k)]
            points=np.array(points)
            if isConvexPolygon(points):
                box.append(points)
                break
    box2=[]
    if confinement:
        xmin,ymin,xmax,ymax=confinement
        for npoints in box:
            if (npoints[:,0]>xmin).all() and (npoints[:,0]<xmax).all() and (npoints[:,1]>ymin).all() and (npoints[:,1]<ymax).all():
                box2.append(npoints)
        
    return box2


def points2lineSegements(points):
    points=np.array(points)
    assert points.shape[0]>=3 and points.shape[1]==2
    points2=np.copy(points)
    points2=np.concatenate([points2,points2[0,:].reshape([1,-1])])
    points2=points2[1:,:]
    ret=np.concatenate([points,points2],axis=1)
    return ret



def get_calibration(img):
    straights=find_more_edge(img)
    drawn_img=img
    drawn_img=render(drawn_img,straights)
    return drawn_img


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
            point=np.random.normal(loc=list(self.center),scale=[25,25],size=[1,2])
            if point[0][0]>0 and point[0][0]<w and point[0][1]>0 and point[0][1]<h:
                box.append(point)
                count+=1
        return np.concatenate(box,axis=0)
    
    def sample_select(self,n):
        rho,theta=self.base_coef
        f=lambda x,y:np.cos(theta)*x+np.sin(theta)*y
        box_pos=[]
        box_neg=[]
        points=self.sample(n)
        for i,point in enumerate(points):
            if f(point[0],point[1])<rho:
                box_neg.append(point)
            else:
                box_pos.append(point)
        return [np.array(box_neg,'int'),np.array(box_pos,'int')]
      
    def symmetry_sample(self,n):
        
        h,w,_=self.shape
        scale=min(w,h)
        rho,theta=self.base_coef
        line_vector=np.array(np.cos(theta-0.5*np.pi),np.sin(theta-0.5*np.pi))
        
        start=self.center-line_vector*scale/1
        end=self.center+line_vector*scale/1
        line=np.concatenate([start,end],axis=1)
#        x1,y1,x2,y2=start[0][0],start[0][1],end[0][0],end[0][1]
        samples=[]
        for margin in [5,10]:
            samples.append(sample_line_sides(line,50+margin,margin))
        samples=np.concatenate(samples,axis=0)
        
        valid_samples=[]
        for xn,yn,xp,yp in samples:
#            print (xn,yn,xp,yp)
            if xn<w and yn<h and xp<w and yp<h and xn>0 and yn>0 and xp>0 and yp>0:
                valid_samples.append([xn,yn,xp,yp])
        valid_samples=np.array(valid_samples,'int')
     
        neg_fatures=self.img[valid_samples[:,1],valid_samples[:,0]].reshape([-1]).astype('int')
        pos_fatures=self.img[valid_samples[:,3],valid_samples[:,2]].reshape([-1]).astype('int')
        
        cos=np.dot(neg_fatures,pos_fatures)/(np.linalg.norm(neg_fatures)*np.linalg.norm(pos_fatures))
        self.similarity=cos
        return cos
        
        
        
        
    
    def hsv(self,n):
        points_neg,points_pos=self.sample_select(n)
        hsv_neg=self.img[points_neg[:,0],points_neg[:,1]]
        hsv_pos=self.img[points_pos[:,0],points_pos[:,1]]
        mean_neg=np.mean(hsv_neg,axis=0)
        mean_pos=np.mean(hsv_pos,axis=0)
        return mean_neg,mean_pos

    
def find_more_edge(img_raw):
    img_hsv = cv2.cvtColor(img_raw,cv2.COLOR_BGR2HSV)
    img = cv2.cvtColor(img_raw,cv2.COLOR_BGR2GRAY)    
    

    #detect all lines by LSD 
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_NONE)
#    dlines = lsd.detect(img)
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
    base_coef_set=base_coef_set[:n_first]
    
    straights=list(map(lambda x: x.base_coef,base_coef_set))
    return straights


if __name__=='__main__':
    image_path='data/bank2.jpg'

    img_raw = cv2.imread(image_path)
    img_hsv = cv2.cvtColor(img_raw,cv2.COLOR_BGR2HSV)
    img = cv2.cvtColor(img_raw,cv2.COLOR_BGR2GRAY)
    
    straights=find_more_edge(img_raw)

    

    
    cc=combination(8,4)
    pp=permutation(8,4)
    M=crosspoint_matrix(straights)

    crosspoints=M.reshape([M.shape[0]*M.shape[1],2])    


    

    h,w,_=img_raw.shape
    nsides_box=find_kSide_from_nLine(straights,4,[0,0,w,h])
    nsides_box2=list(map(lambda x:points2lineSegements(x),nsides_box))
    
    
    vv=isConvexPolygon([[10,10],[30,10],[10,30],[30,30]])
    
    
    
    
    drawn_img=img_raw
    for ooxx in nsides_box2:
        drawn_img=render(drawn_img,ooxx,(np.random.randint(0,1),np.random.randint(200,201),np.random.randint(0,1)),5)
#    drawn_img=render_point(drawn_img,crosspoints)
    
#    drawn_img=render_point(drawn_img,nsides_box[0])
    
    fig=plt.figure(figsize=[20,15])
    plt.imshow(drawn_img)
    