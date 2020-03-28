#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 22:14:39 2020

@author: ziga
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import datetime

run = 6


#%% READ REAL DATA
# https://github.com/slo-covid-19/data/blob/master/csv/stats.csv
data_stats = pd.read_csv(r"https://raw.githubusercontent.com/slo-covid-19/data/master/csv/stats.csv",\
                         index_col="date", usecols=["date", "tests.positive.todate","state.in_hospital", "state.icu","state.deceased.todate"], parse_dates=["date"])


# day 0 = March 12
data_hospitalised = data_stats["state.in_hospital"][12:]
data_icu = data_stats["state.icu"][12:]
data_cases = data_stats["tests.positive.todate"][12:]
data_dead = data_stats["state.deceased.todate"][12:]
data_days = np.arange(0,data_hospitalised.shape[0])

#%% READ SIMULATED DATA
days = np.loadtxt("./save/tab_days_001.txt")
Nt =days.shape[0]

N = 1000
active = np.zeros((N,days.shape[0]))
infectious = np.zeros((N,days.shape[0]))
incubation = np.zeros((N,days.shape[0]))
symptoms = np.zeros((N,days.shape[0]))
hospitalized = np.zeros((N,days.shape[0]))
icu = np.zeros((N,days.shape[0]))
dead = np.zeros((N,days.shape[0]))
immune = np.zeros((N,days.shape[0]))
susceptible = np.zeros((N,days.shape[0]))



j = 0
for i in range(N):
    print(i)
    if os.path.exists("./save2/tab_active_{:03d}.txt".format(i)):
        
        active[j,:] = np.loadtxt("./save2/tab_active_{:03d}.txt".format(i))
        infectious[j,:] = np.loadtxt("./save2/tab_infectious_{:03d}.txt".format(i))
        incubation[j,:] = np.loadtxt("./save2/tab_incubation_{:03d}.txt".format(i))
        symptoms[j,:] = np.loadtxt("./save2/tab_symptoms_{:03d}.txt".format(i))
        hospitalized[j,:] = np.loadtxt("./save2/tab_hospitalized_{:03d}.txt".format(i))
        icu[j,:] = np.loadtxt("./save2/tab_icu_{:03d}.txt".format(i))
        dead[j,:] = np.loadtxt("./save2/tab_dead_{:03d}.txt".format(i))
        immune[j,:] = np.loadtxt("./save2/tab_immune_{:03d}.txt".format(i))
        susceptible[j,:] = np.loadtxt("./save2/tab_susceptible_{:03d}.txt".format(i))
        
        j += 1
    else:
        continue


#%% COMPUTE PARAMETERS
active_median = np.median(active[:j],axis=0)
active_01 = np.percentile(active[:j],1,axis=0)
active_05 = np.percentile(active[:j],5,axis=0)
active_10 = np.percentile(active[:j],10,axis=0)
active_25 = np.percentile(active[:j],25,axis=0)
active_75 = np.percentile(active[:j],75,axis=0)
active_90 = np.percentile(active[:j],90,axis=0)
active_95 = np.percentile(active[:j],95,axis=0)
active_99 = np.percentile(active[:j],99,axis=0)

infectious_median = np.median(infectious[:j],axis=0)
infectious_01 = np.percentile(infectious[:j],1,axis=0)
infectious_05 = np.percentile(infectious[:j],5,axis=0)
infectious_10 = np.percentile(infectious[:j],10,axis=0)
infectious_25 = np.percentile(infectious[:j],25,axis=0)
infectious_75 = np.percentile(infectious[:j],75,axis=0)
infectious_90 = np.percentile(infectious[:j],90,axis=0)
infectious_95 = np.percentile(infectious[:j],95,axis=0)
infectious_99 = np.percentile(infectious[:j],99,axis=0)

symptoms_median = np.median(symptoms[:j],axis=0)
symptoms_01 = np.percentile(symptoms[:j],1,axis=0)
symptoms_05 = np.percentile(symptoms[:j],5,axis=0)
symptoms_10 = np.percentile(symptoms[:j],10,axis=0)
symptoms_25 = np.percentile(symptoms[:j],25,axis=0)
symptoms_75 = np.percentile(symptoms[:j],75,axis=0)
symptoms_90 = np.percentile(symptoms[:j],90,axis=0)
symptoms_95 = np.percentile(symptoms[:j],95,axis=0)
symptoms_99 = np.percentile(symptoms[:j],99,axis=0)

hospitalized_median = np.median(hospitalized[:j],axis=0)
hospitalized_01 = np.percentile(hospitalized[:j],1,axis=0)
hospitalized_05 = np.percentile(hospitalized[:j],5,axis=0)
hospitalized_10 = np.percentile(hospitalized[:j],10,axis=0)
hospitalized_25 = np.percentile(hospitalized[:j],25,axis=0)
hospitalized_75 = np.percentile(hospitalized[:j],75,axis=0)
hospitalized_90 = np.percentile(hospitalized[:j],90,axis=0)
hospitalized_95 = np.percentile(hospitalized[:j],95,axis=0)
hospitalized_99 = np.percentile(hospitalized[:j],99,axis=0)

icu_median = np.median(icu[:j],axis=0)
icu_01 = np.percentile(icu[:j],1,axis=0)
icu_05 = np.percentile(icu[:j],5,axis=0)
icu_10 = np.percentile(icu[:j],10,axis=0)
icu_25 = np.percentile(icu[:j],25,axis=0)
icu_75 = np.percentile(icu[:j],75,axis=0)
icu_90 = np.percentile(icu[:j],90,axis=0)
icu_95 = np.percentile(icu[:j],95,axis=0)
icu_99 = np.percentile(icu[:j],99,axis=0)

dead_median = np.median(dead[:j],axis=0)
dead_01 = np.percentile(dead[:j],1,axis=0)
dead_05 = np.percentile(dead[:j],5,axis=0)
dead_10 = np.percentile(dead[:j],10,axis=0)
dead_25 = np.percentile(dead[:j],25,axis=0)
dead_75 = np.percentile(dead[:j],75,axis=0)
dead_90 = np.percentile(dead[:j],90,axis=0)
dead_95 = np.percentile(dead[:j],95,axis=0)
dead_99 = np.percentile(dead[:j],99,axis=0)

immune_median = np.median(immune[:j],axis=0)
immune_01 = np.percentile(immune[:j],1,axis=0)
immune_05 = np.percentile(immune[:j],5,axis=0)
immune_10 = np.percentile(immune[:j],10,axis=0)
immune_25 = np.percentile(immune[:j],25,axis=0)
immune_75 = np.percentile(immune[:j],75,axis=0)
immune_90 = np.percentile(immune[:j],90,axis=0)
immune_95 = np.percentile(immune[:j],95,axis=0)
immune_99 = np.percentile(immune[:j],99,axis=0)


#%% PLOT FIELDS
fig = plt.figure(figsize=(12,6))

plt.title("COVID-19 Pandemy in Slovenia")

# simulated values
# plt.fill_between(days,active_01,active_99,color="blue",alpha=0.1)
#plt.fill_between(days,active_05,active_95,color="blue",alpha=0.2)
#plt.plot(days,active_median,label="Active",color="blue",lw=3)

# plt.fill_between(days,symptoms_01,symptoms_99,color="green",alpha=0.1)
plt.fill_between(days,symptoms_05,symptoms_95,color="green",alpha=0.2)
plt.plot(days,symptoms_median,label="Symptoms ~ positive cases",color="green",lw=3)

# plt.fill_between(days,symptoms_01,symptoms_99,color="green",alpha=0.1)
plt.fill_between(days,hospitalized_05,hospitalized_95,color="orange",alpha=0.2)
plt.plot(days,hospitalized_median,label="Hospitalised",color="orange",lw=3)


plt.fill_between(days,icu_05,icu_95,color="brown",alpha=0.2)
plt.plot(days,icu_median,label="ICU",color="brown",lw=3)
 
plt.fill_between(days,dead_05,dead_95,color="black",alpha=0.4)
plt.plot(days,dead_median,label="Fatal",color="black",lw=3)

# plt.plot(days[:20],contagious[:20]/7.5,'b--',label="Active cases")
# plt.plot(days,immune,label="Immune",color="green")
# plt.plot(days,critical,label="Critical (ICU)",color="red",lw=2)
# plt.plot(days,dead,'k-',label="Dead")
# plt.plot(days,[res_num]*len(days),'r--',label="Future healthcare capacity (125 respirators)")


# real data
plt.plot(data_days,data_cases,'go',label="Positive cases (data)")
plt.plot(data_days,data_hospitalised,'o',color='orange',label="Hospitalized (data)")
plt.plot(data_days,data_icu,'o',color='brown',label="ICU (data)")
plt.plot(data_days,data_dead,'ko',label="Fatal (data)")


plt.yscale('log')

xticks_lbls = []
date0 = datetime.datetime(2020,3,12)
for i in range(Nt):
    date = date0+datetime.timedelta(i)
    xticks_lbls.append(date.strftime("%B %d"))
plt.xticks(range(0,Nt,10),xticks_lbls[::10],rotation=60)
# plt.fill_between(days, res_num ,0,alpha=0.2,color='r')
# plt.fill_between(days,critical,res_num)
plt.ylabel("Number of nodes")
plt.grid()
plt.ylim([1,10**6])
plt.legend()
fig.savefig("potek_pandemije{:02d}.png".format(run),dpi=250)
    
