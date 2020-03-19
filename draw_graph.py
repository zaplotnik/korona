#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 23:12:46 2020

@author: ziga
"""

import networkx as nx           # For making graphs,manipulation etc
import matplotlib.pyplot as plt # For plotting the graphs
import numpy as np # Matrix manipulation


# # Generating sample data
# G = nx.florentine_families_graph()
# adjacency_matrix = nx.adjacency_matrix(G)

# # The actual work
# # You may prefer `nx.from_numpy_matrix`.
# G2 = nx.from_scipy_sparse_matrix(adjacency_matrix)
# nx.draw_circular(G2)
# plt.axis('equal')

N=100
A=np.zeros((N,N))

h1 = 269898 # 1 person
h2 = 209573 # 2 person
h3 = 152959 # 3 person
h4 = 122195 # 4 person
h5 =  43327 # 5 person
h6 =  17398 # 6 person
h7 =   6073 # 7 person
h8 =   3195 # 8 person
Nall= (h1+2*h2+3*h3+4*h4+5*h5+6*h6+7*h7+8*h8)

h1 = int(1.*h1*N/Nall)
h2 = int(1.*h2*N/Nall)
h3 = int(1.*h3*N/Nall)
h4 = int(1.*h4*N/Nall)
h5 = int(1.*h5*N/Nall)
h6 = int(1.*h6*N/Nall)
h7 = int(1.*h7*N/Nall)
h8 = int(1.*h8*N/Nall)

i = h1

# generate h2
ps = 2
end = i + ps*h2
print(i)
while i < end:
    for j in range(0,ps):
        l = 0
        for k in range(0,ps):
            if j is not k:
                A[i+j,i+k] = 1
                A[i+k,i+j] = 1
                l += 1
    i += ps

# generate h3
ps = 3
end = i + ps*h3 
print(i)
while i < end:
    for j in range(0,ps):
        l = 0
        for k in range(0,ps):
            if j is not k:
                A[i+j,i+k] = 1
                A[i+k,i+j] = 1
                l += 1
    i += ps
    
# generate h4
ps = 4
end = i + ps*h4 
print(i)
while i < end:
    for j in range(0,ps):
        l = 0
        for k in range(0,ps):
            if j is not k:
                A[i+j,i+k] = 1
                A[i+k,i+j] = 1
                l += 1
    i += ps 
    
    
# generate h5
ps = 5
end = i + ps*h5 
print(i)
while i < end:
    for j in range(0,ps):
        l = 0
        for k in range(0,ps):
            if j is not k:
                A[i+j,i+k] = 1
                A[i+k,i+j] = 1
                l += 1
    i += ps
print(i)
# write
N = i
B = np.copy(A)

#%%
cells_num = h1 + h2 +h3 +h4 + h5

positions = []
r=5
dr = 1

i = 0
dtheta = 360./(cells_num)
for ind in range(h1):
    x0,y0 = (r*np.cos(i*dtheta*np.pi/180.),r*np.sin(i*dtheta*np.pi/180.))
    print (x0,y0)
    positions.append([x0,y0])
    i += 1
    
for ind in range(h2):
    x0,y0 = (r*np.cos(i*dtheta*np.pi/180.),r*np.sin(i*dtheta*np.pi/180.))
    for j in range(2):
        positions.append([x0+dr*np.random.uniform(-1,1),y0+dr*np.random.uniform(-1,1)]) 
        i += 1

for ind in range(h3):
    x0,y0 = (r*np.cos(i*dtheta*np.pi/180.),r*np.sin(i*dtheta*np.pi/180.))
    for j in range(3):
        positions.append([x0+dr*np.random.uniform(-1,1),y0+dr*np.random.uniform(-1,1)]) 
        i += 1

for ind in range(h4):
    x0,y0 = (r*np.cos(i*dtheta*np.pi/180.),r*np.sin(i*dtheta*np.pi/180.))
    for j in range(4):
        positions.append([x0+dr*np.random.uniform(-1,1),y0+dr*np.random.uniform(-1,1)]) 
        i += 1
        
for ind in range(h5):
    x0,y0 = (r*np.cos(i*dtheta*np.pi/180.),r*np.sin(i*dtheta*np.pi/180.))
    for j in range(5):
        positions.append([x0+dr*np.random.uniform(-1,1),y0+dr*np.random.uniform(-1,1)]) 
        i += 1


#%%
mu = 0.2;
sigma = 0.9
rands = np.random.lognormal(mu,sigma,N)-0.2
rands = rands/2. # each connection represents two nodes

# futher divide
rands = rands/3.

rands[rands>10.] = 10.
rands = np.round(rands,0)
rands_int = rands.astype(int)

A = np.copy(B)
for i in range(N):
    pn = rands_int[i] # number of extra connections for node i
    for k in range(pn):
        j = np.random.randint(0,N)
        A[i,j] = 1
        A[j,i] = 1
# # G=nx.from_numpy_matrix(A)






fig = plt.figure(figsize=(7,7))
positions = np.array(positions)
for i in range(positions.shape[0]):
    for j in range(positions.shape[0]):
        if A[i,j] == 1:
            # plot edge
            plt.plot([positions[i,0],positions[j,0]],[positions[i,1],positions[j,1]],"k-",lw=0.5)

for i in range(positions.shape[0]):
    plt.plot(positions[i,0],positions[i,1],'ro')
plt.savefig("graph_03.png",dpi=300)
# fixed_positions = dict(zip(nodes, positions))
# # fixed_positions = {1:(0,0),2:(-1,2)}
# fixed_nodes = fixed_positions.keys()
# pos = nx.spring_layout(G,pos=fixed_positions, fixed = fixed_nodes)
# nx.draw_networkx(G,pos)


# nx.draw_random(G,node_size=50)
