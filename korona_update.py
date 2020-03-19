#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 01:34:41 2020

@author: ziga
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy
import datetime


run = 4

### QUESTIONS ###

# How long is it going to last without overshooting the capacity?
# What measures are needed to contain the virus


# https://www.worldometers.info/coronavirus/coronavirus-symptoms/

dr = 0.025 # death ratio without overshooting (China)
cr = 0.047 # critical ratio, need of intensive care (China) 
sr = 0.138 # severe ratio of all (China), oxygen
mr = 0.809 # mild ratio - recover at home (China)

# dr/cr = death rate of critically ill on intensive care
dr_wc = 0.8 # death rate of critically ill without intensive care


# healthcare capacity
res_num = 130

#  number of people
N= int(2.045795*10**6)

# currently infected (12.marec)
Ni = 96

# ratio proposed by A. Ihan (5-10x): https://www.rtvslo.si/slovenija/varesnici-je-okuzenih-pri-nas-od-petkrat-do-desetkrat-vec/517219
Ni = int(Ni * 7.5)

# 0 susceptible, 1 to Tc - sick and contagious, -1 immune(recovered), -2  dead
status = np.zeros(N,dtype=np.int8)


# closed clusteres - not accounted yet
# Metlika:20

# contagious period: https://www.sciencenews.org/article/coronavirus-most-contagious-before-during-first-week-symptoms
Tc = 12 # days # ASK SPECIALISTS !! should be modelled smootly.... @RJerala https://t.co/9HxIDCQvUJ?amp=1
Td = 22 # days when he goes dead, median hospital stay Td-Tc = 10 days

rand_p = np.random.randint(0,N,Ni)
days_contagious = np.random.exponential(3,Ni)+1
days_contagious[days_contagious>Tc]=Tc
days_contagious = np.round(days_contagious,0)

status[rand_p] = days_contagious.astype(np.int8)

# transmission efficiency during close contact
alpha = 0.1 # ASK SPECIALISTS!!


# households in Slovenia: https://www.stat.si/StatWeb/News/Index/7725
h1 = 269898 # 1 person
h2 = 209573 # 2 person
h3 = 152959 # 3 person
h4 = 122195 # 4 person
h5 =  43327 # 5 person
h6 =  17398 # 6 person
h7 =   6073 # 7 person
h8 =   3195 # 8 person

# elderly care
ec_size = 20000 # persons in elderly care centers
ec_centers = 100 # elderly care centers (100 for simplicity, actually 102)
pp_center = int(ec_size/ec_centers) # people per center
group_size=25 # number of poeple in one group
gp_center = int(pp_center/group_size)

np.save("status{:02d}".format(run),status)



#%% TEST

# con=np.zeros((6,2))
# con_max=np.zeros(6)
# i=0
# while i < 6:
#     for j in range(0,3):
#         l = 0
#         for k in range(0,3):
#             if j is not k:
#                 con[i+j,l] = i+k
#                 con_max[i+j] +=1
#                 l += 1
#     i+=3

#%%

# GENERATE NETWORK

# first add households as clusters where disease can spread infinitely
maxc = 100
connections = np.zeros((N,150),dtype=np.int32)
connection_max = np.zeros(N,dtype=np.int32) # number of connections for each person


print("Generating network...")
print("Family/care clusters...")
# generate h1
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
                connections[i+j,l] = i+k
                connection_max[i+j] += 1
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
                connections[i+j,l] = i+k
                connection_max[i+j] += 1
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
                connections[i+j,l] = i+k
                connection_max[i+j] += 1
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
                connections[i+j,l] = i+k
                connection_max[i+j] += 1
                l += 1
    i += ps
    
# generate h6
ps=6
end = i + ps*h6 
print(i)
while i < end:
    for j in range(0,ps):
        l = 0
        for k in range(0,ps):
            if j is not k:
                connections[i+j,l] = i+k
                connection_max[i+j] += 1
                l += 1
    i += ps
    
# generate h7
ps = 7
end = i + ps*h7
print(i)
while i < end:
    for j in range(0,ps):
        l = 0
        for k in range(0,ps):
            if j is not k:
                connections[i+j,l] = i+k
                connection_max[i+j] += 1
                l += 1
    i += ps  
    
# generate h8
ps=8
end = i + 8*h8 
print(i)
while i < end:
    for j in range(0,ps):
        l = 0
        for k in range(0,ps):
            if j is not k:
                connections[i+j,l] = i+k
                connection_max[i+j] += 1
                l += 1
    i += ps   

print(i)    

# elderly centers - groups of 25 in a center of 200
end = i + ec_size
while i < end:
    for a in range(ec_centers):
        for b in range(gp_center):
            for j in range(0,group_size):
                l = 0
                for k in range(0,group_size):
                    if j is not k:
                        connections[i+j,l] = i+k
                        connection_max[i+j] += 1
                        l += 1
                
            i += group_size 
        
print(i)

family_connections = (connections>0).sum()/2.
print(family_connections)
#%% ADD OTHER CONNECTIONS

# x= np.arange(0.2,100,1)
mu = 0.2;
sigma = 0.9

# mean = np.exp(mu+sigma**2/2.)
# print("Mean number of social contacts/human/day:", mean)
# pdf = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2)) / (x * sigma * np.sqrt(2 * np.pi)))
# plt.loglog(x-0.2+1,pdf*N)
# plt.xlim([0,50])
# plt.ylim([1,2*10**6])
# plt.xlabel("stevilo njihovih kontaktov")
# plt.ylabel("stevilo ljudi")
# plt.xticks([i+1 for i in range(11)] + [i+1 for i in range(15,50,5)],\
#            [str(i) for i in range(11)]  +[i for i in range(15,50,5)])
# plt.grid()

# number = pdf*N
# plt.title("Stevilo kontaktov izven druzinskih/skrbniskih clustrov")
# plt.text(2,1250,"{0} ljudi ima 0 kontaktov dnevno".format(int(number[0])))
# plt.text(2,250,"{0} ljudi ima 1 kontakt dnevno".format(int(number[1])))
# plt.text(2,50,"{0} ljudi ima 2 kontakta dnevno".format(int(number[2])))
# plt.text(2,10,"{0} ljudi ima 5 kontaktov dnevno".format(int(number[5])))
# plt.text(2,2,"{0} ljudi ima 20 kontaktov dnevno".format(int(number[20])))
# # print(number)
# plt.savefig("povezave.png",dpi=300)

# we generate random numbers according to that distribution
rands = np.random.lognormal(mu,sigma,N)-0.2
# rands_plot=np.round(np.copy(rands),0)
# rands_plot_int = rands_plot.astype(int)

rands = rands/2. # each connection represents two nodes

# further divide
rands = rands/2.5

rands[rands>50.] = 50.
rands = np.round(rands,0)
rands_int = rands.astype(int)



tp = np.zeros(101)
for i in range(101):
    tp[i] = (np.sum((rands_int)==i))
plt.loglog(range(1,102),tp)
plt.xlim([0,50])
plt.ylim([1,2*10**6])
plt.xlabel("stevilo njihovih kontaktov")
plt.ylabel("stevilo ljudi")
plt.xticks([i+1 for i in range(11)] + [i+1 for i in range(15,50,5)],\
           [str(i) for i in range(11)]  +[i for i in range(15,50,5)])
plt.grid()

plt.title("Stevilo kontaktov izven druzinskih/skrbniskih clustrov")
props = dict(boxstyle='round', facecolor='red', alpha=0.3)  
plt.text(4.2,10050,"Povprecno stevilo kontaktov/osebo/dan\nv druzini/skrbnistvu: {0}\nPovprecno stevilo kontaktov/osebo/dan\nizven druzin/skrbnistev [RIZICNO]: {1}".format(\
   np.round(1.*family_connections/N,2),np.round(1.*rands_int.sum()/N,2)),bbox=props)

plt.text(1.2,10,"{0} ljudi ima 0 RIZICNIH kontaktov dnevno\n{1} ljudi ima 1 RIZICNI kontakt dnevno\n{2} ljudi ima 5 RIZICNIH kontaktov dnevno".format(\
        int(tp[0]),int(tp[1]),int(tp[5])),bbox=props)  


plt.savefig("povezave{:02d}.png".format(run),dpi=300)

#%%
# now we add connections to existing connections
print("Other connections...")
for i in range(N):
    pn = rands_int[i] # number of extra connections for node i
    for j in range(pn):
        iad = np.random.randint(0,N)
        connections[i,connection_max[i]] = iad
        connections[iad,connection_max[iad]] = i
        connection_max[i] += 1
        connection_max[iad] += 1
    
long_connections = (connections>0).sum()/2.-family_connections
print("long connections")
#%% SIMULATE VIRUS SPREAD
status = np.load("status{:02d}.npy".format(run))        
        
        
Nt = 500
days = np.zeros(Nt+1)
immune = np.zeros(Nt+1)        # -1
dead = np.zeros(Nt+1)          # -2
susceptible = np.zeros(Nt+1)   # 0
contagious = np.zeros(Nt+1)    # 1 to Tc
critical=np.zeros(Nt+1)  # Tc+1 to Td

# start
day = 0 
immune[day] = np.sum(status==-1) 
dead[day] = np.sum(status==-2)
susceptible[day] = np.sum(status==0)
contagious[day] = np.sum(np.logical_and(status>0,status<=Tc))
critical[day] = np.sum(np.logical_and(status>Tc,status <= Td))

date0 = datetime.datetime(2020,3,12)
print("Day: {0}".format(date0.strftime("%B %d")))
print("Dead: ", dead[day])
print("Critical: ", critical[day])
print("Infected + contagious: ",contagious[day])
print("Infected [published]: ",contagious[day]/7.5)
print("Susceptible: ", susceptible[day])
print("Immune: ",immune[day])



print("Simulate virus spread over network")

while day < Nt:
    # print(day)
    status_old = np.copy(status)
    
    # go over all contagious indices
    contagious_ind = (np.where(np.logical_and(status_old>0,status_old<=Tc)))[0]
    print(critical[day])
    for i in contagious_ind:
        # go through all his susceptible connections
        con_i = connections[i,:connection_max[i]]
        for j in con_i:
            # infect susceptible connection with probability alpha
            if status_old[j] == 0 and np.random.random() < alpha:
                status[j] = 1 
        
        # if patient is infected for Tc days, he is not contagious anymore. 
        # he can become immune and will get quick over sickness or or is still heavily sick 
        # (requires respirator) with probabiliy cr
        if status_old[i] == Tc:
            if np.random.random() > cr:
                status[i] = -1
                status_old[i] = -1
        
    
    # go again over all sick indices and update number of deaths
    contagious_ind = (np.where(np.logical_and(status_old>0,status_old<=Td)))[0] # Td not Tc

    for i in contagious_ind:
         
        if status_old[i] >= Td:

            # after 10 days on intensive care 53% critically ill die
            # 90% critically ill die (here assummed after 10 days) withou treatment
            if (critical[day] < res_num and np.random.random()<dr/cr) or \
                (critical[day]> res_num and np.random.random()<dr_wc): # 
                status[i] = -2 # dead
                status_old[i] = -2
            else: 
                status[i] = -1 # gets over sickness, immune
                status_old[i] = -1
                
        
    
    # go again over all sick indices and update number of days sick, also Tc --> Tc+1 (intensive care)
    contagious_ind = (np.where(np.logical_and(status_old>0,status_old<=Td)))[0] # Td not Tc
    for i in contagious_ind:
        status[i] += 1
        
        
    # impose additional measure on day
    #
    #
    #
    #
    #
    #
    
    
    day += 1
    # compute statistics
    days[day] = day
    immune[day] = np.sum(status==-1) 
    dead[day] = np.sum(status==-2)
    susceptible[day] = np.sum(status==0)
    contagious[day] = np.sum(np.logical_and(status>0,status<=Tc))
    critical[day] = np.sum(np.logical_and(status>Tc,status <= Td))
    
    date = date0 + datetime.timedelta(day)
    print("\nDay: {0} (+{1})".format(date.strftime("%B %d"),day))
    print("Dead: ", dead[day])
    print("Critical (requires intensive care): ", critical[day])
    print("Infected + contagious: ",contagious[day])
    print("Infected [published]: ",int(contagious[day]/7.5))
    print("Susceptible: ", susceptible[day])
    print("Immune: ",immune[day])
    


#%% PLOT FIELDS
real_data= [96,141,181,219,253]
days_real = np.arange(0,len(real_data))


fig = plt.figure(figsize=(12,6))

plt.title("COVID-19 Pandemy in Slovenia")
plt.plot(days,contagious,label="Contagious [real=7.5x positive tested]",color="blue",lw=2)
plt.plot(days,contagious/7.5,'b--',label="Contagious [positive tested]")
plt.plot(days,immune,label="Immune",color="green")
plt.plot(days,critical,label="Critical (requires intensive care)",color="red",lw=2)
plt.plot(days,dead,'k-',label="Dead")
plt.plot(days,[res_num]*len(days),'r--',label="Future healthcare capacity (130 respirators)")
plt.plot(days_real,real_data,'ko',label="Real data (change of testing from March 14 on)")
plt.yscale('log')

xticks_lbls = []
for i in range(Nt+1):
    date = date0+datetime.timedelta(i)
    xticks_lbls.append(date.strftime("%B %d"))
plt.xticks(range(0,Nt,20),xticks_lbls[::20],rotation=60)
plt.fill_between(days, res_num ,0,alpha=0.2,color='r')
# plt.fill_between(days,critical,res_num)
plt.grid()
plt.legend()
fig.savefig("potek_pandemije{:02d}.png".format(run),dpi=300)
    
# plt.plot(days)
np.save("contagious{:02d}".format(run),contagious)
np.save("immune{:02d}".format(run),immune)
np.save("critical{:02d}".format(run),critical)
np.save("dead{:02d}".format(run),dead)

#%%
# # Generating sample data
# G = nx.florentine_families_graph()
# adjacency_matrix = nx.adjacency_matrix(G)

# # The actual work
# # You may prefer `nx.from_numpy_matrix`.
# G2 = nx.from_scipy_sparse_matrix(adjacency_matrix)
# nx.draw_circular(G2)
# plt.axis('equal')