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

run = 19

save_data = True
#%% READ REAL DATA
# https://github.com/slo-covid-19/data/blob/master/csv/stats.csv
data_stats = pd.read_csv(r"https://raw.githubusercontent.com/slo-covid-19/data/master/csv/stats.csv",\
                         index_col="date", usecols=["date", "tests.positive.todate","state.in_hospital", "state.icu","state.deceased.todate"], parse_dates=["date"])


# day 0 = March 12
data_hospitalised = data_stats["state.in_hospital"][17:]
data_icu = data_stats["state.icu"][17:]
data_cases = data_stats["tests.positive.todate"][17:]
data_dead = data_stats["state.deceased.todate"][17:]
#data_cases_new = data_stats["tests.positive"][17:]
data_days = np.arange(0,data_hospitalised.shape[0])
Ndata = len(data_dead)


#%% READ SIMULATED DATA
days = np.loadtxt("./2020_04_03/tab_days_002.txt")
Nt =days.shape[0]

N = 5000
active = np.zeros((N,days.shape[0]))
infectious = np.zeros((N,days.shape[0]))
incubation = np.zeros((N,days.shape[0]))
symptoms = np.zeros((N,days.shape[0]))
hospitalized = np.zeros((N,days.shape[0]))
icu = np.zeros((N,days.shape[0]))
dead = np.zeros((N,days.shape[0]))
immune = np.zeros((N,days.shape[0]))
susceptible = np.zeros((N,days.shape[0]))

inc_dead_data = data_dead.values>0
inc_icu_data = data_icu.values>0

blah = np.zeros((2,N))

j = 0
for i in range(N):
    print(i)
    if os.path.exists("./2020_04_03/tab_active_{:03d}.txt".format(i)):
        
        active[j,:] = np.loadtxt("./2020_04_03/tab_active_{:03d}.txt".format(i))
        infectious[j,:] = np.loadtxt("./2020_04_03/tab_infectious_{:03d}.txt".format(i))
        incubation[j,:] = np.loadtxt("./2020_04_03/tab_incubation_{:03d}.txt".format(i))
        symptoms[j,:] = np.loadtxt("./2020_04_03/tab_symptoms_{:03d}.txt".format(i))
        hospitalized[j,:] = np.loadtxt("./2020_04_03/tab_hospitalized_{:03d}.txt".format(i))
        icu[j,:] = np.loadtxt("./2020_04_03/tab_icu_{:03d}.txt".format(i))
        dead[j,:] = np.loadtxt("./2020_04_03/tab_dead_{:03d}.txt".format(i))
        immune[j,:] = np.loadtxt("./2020_04_03/tab_immune_{:03d}.txt".format(i))
        susceptible[j,:] = np.loadtxt("./2020_04_03/tab_susceptible_{:03d}.txt".format(i))
        
        
        dead_model = dead[j,:Ndata] > 0
        icu_model = icu[j,:Ndata] > 0       

        inc_dead_all = inc_dead_data & dead_model
        inc_icu_all = inc_icu_data & icu_model
        
        #print dead[j,:Ndata][inc_dead_all]
        #print (data_dead.values)[inc_dead_all]   

        #print icu[j,:Ndata][inc_icu_all]
        #print (data_icu.values)[inc_icu_all]      

        sum1 = (np.abs(np.log((data_dead.values)[inc_dead_all]) - np.log(dead[j,:Ndata][inc_dead_all]))).sum()
        sum2 = (np.abs((np.log(data_icu.values)[inc_icu_all]) - np.log(icu[j,:Ndata][inc_icu_all]))).sum()       
        print sum1, sum2
        #print ""
        blah[0,j] = i
        blah[1,j] = sum1 + sum2 
        j += 1
    else:
        continue

print j
a =  np.percentile(blah[1,:j],5)
print a
inds = blah[0,:j][blah[1,:j] < a]
inds = inds.astype(np.int)
print inds


#%% COMPUTE PARAMETERS
active_median = np.median(active[inds],axis=0)
active_01 = np.percentile(active[inds],1,axis=0)
active_05 = np.percentile(active[inds],5,axis=0)
active_10 = np.percentile(active[inds],10,axis=0)
active_25 = np.percentile(active[inds],25,axis=0)
active_75 = np.percentile(active[inds],75,axis=0)
active_90 = np.percentile(active[inds],90,axis=0)
active_95 = np.percentile(active[inds],95,axis=0)
active_99 = np.percentile(active[inds],99,axis=0)

infectious_median = np.median(infectious[inds],axis=0)
infectious_01 = np.percentile(infectious[inds],1,axis=0)
infectious_05 = np.percentile(infectious[inds],5,axis=0)
infectious_10 = np.percentile(infectious[inds],10,axis=0)
infectious_25 = np.percentile(infectious[inds],25,axis=0)
infectious_75 = np.percentile(infectious[inds],75,axis=0)
infectious_90 = np.percentile(infectious[inds],90,axis=0)
infectious_95 = np.percentile(infectious[inds],95,axis=0)
infectious_99 = np.percentile(infectious[inds],99,axis=0)

symptoms_median = np.median(symptoms[inds],axis=0)
symptoms_01 = np.percentile(symptoms[inds],1,axis=0)
symptoms_05 = np.percentile(symptoms[inds],5,axis=0)
symptoms_10 = np.percentile(symptoms[inds],10,axis=0)
symptoms_25 = np.percentile(symptoms[inds],25,axis=0)
symptoms_75 = np.percentile(symptoms[inds],75,axis=0)
symptoms_90 = np.percentile(symptoms[inds],90,axis=0)
symptoms_95 = np.percentile(symptoms[inds],95,axis=0)
symptoms_99 = np.percentile(symptoms[inds],99,axis=0)

hospitalized_median = np.median(hospitalized[inds],axis=0)
hospitalized_01 = np.percentile(hospitalized[inds],1,axis=0)
hospitalized_05 = np.percentile(hospitalized[inds],5,axis=0)
hospitalized_10 = np.percentile(hospitalized[inds],10,axis=0)
hospitalized_25 = np.percentile(hospitalized[inds],25,axis=0)
hospitalized_75 = np.percentile(hospitalized[inds],75,axis=0)
hospitalized_90 = np.percentile(hospitalized[inds],90,axis=0)
hospitalized_95 = np.percentile(hospitalized[inds],95,axis=0)
hospitalized_99 = np.percentile(hospitalized[inds],99,axis=0)

icu_median = np.median(icu[inds],axis=0)
icu_01 = np.percentile(icu[inds],1,axis=0)
icu_05 = np.percentile(icu[inds],5,axis=0)
icu_10 = np.percentile(icu[inds],10,axis=0)
icu_25 = np.percentile(icu[inds],25,axis=0)
icu_75 = np.percentile(icu[inds],75,axis=0)
icu_90 = np.percentile(icu[inds],90,axis=0)
icu_95 = np.percentile(icu[inds],95,axis=0)
icu_99 = np.percentile(icu[inds],99,axis=0)

dead_median = np.median(dead[inds],axis=0)
dead_01 = np.percentile(dead[inds],1,axis=0)
dead_05 = np.percentile(dead[inds],5,axis=0)
dead_10 = np.percentile(dead[inds],10,axis=0)
dead_25 = np.percentile(dead[inds],25,axis=0)
dead_75 = np.percentile(dead[inds],75,axis=0)
dead_90 = np.percentile(dead[inds],90,axis=0)
dead_95 = np.percentile(dead[inds],95,axis=0)
dead_99 = np.percentile(dead[inds],99,axis=0)

immune_median = np.median(immune[inds],axis=0)
immune_01 = np.percentile(immune[inds],1,axis=0)
immune_05 = np.percentile(immune[inds],5,axis=0)
immune_10 = np.percentile(immune[inds],10,axis=0)
immune_25 = np.percentile(immune[inds],25,axis=0)
immune_75 = np.percentile(immune[inds],75,axis=0)
immune_90 = np.percentile(immune[inds],90,axis=0)
immune_95 = np.percentile(immune[inds],95,axis=0)
immune_99 = np.percentile(immune[inds],99,axis=0)

if save_data: 
	np.savetxt("./data/active_median.txt",active_median)
	np.savetxt("./data/active_05.txt",active_05)
	np.savetxt("./data/active_10.txt",active_10)
	np.savetxt("./data/active_25.txt",active_25)
	np.savetxt("./data/active_75.txt",active_75)
	np.savetxt("./data/active_90.txt",active_90)
	np.savetxt("./data/active_95.txt",active_95)

	np.savetxt("./data/infectious_median.txt",infectious_median)
	np.savetxt("./data/infectious_05.txt",infectious_05)
	np.savetxt("./data/infectious_10.txt",infectious_10)
	np.savetxt("./data/infectious_25.txt",infectious_25)
	np.savetxt("./data/infectious_75.txt",infectious_75)
	np.savetxt("./data/infectious_90.txt",infectious_90)
	np.savetxt("./data/infectious_95.txt",infectious_95)

	np.savetxt("./data/symptoms_median.txt",symptoms_median)
	np.savetxt("./data/symptoms_05.txt",symptoms_05)
	np.savetxt("./data/symptoms_10.txt",symptoms_10)
	np.savetxt("./data/symptoms_25.txt",symptoms_25)
	np.savetxt("./data/symptoms_75.txt",symptoms_75)
	np.savetxt("./data/symptoms_90.txt",symptoms_90)
	np.savetxt("./data/symptoms_95.txt",symptoms_95)

	np.savetxt("./data/hospitalized_median.txt",hospitalized_median)
	np.savetxt("./data/hospitalized_05.txt",hospitalized_05)
	np.savetxt("./data/hospitalized_10.txt",hospitalized_10)
	np.savetxt("./data/hospitalized_25.txt",hospitalized_25)
	np.savetxt("./data/hospitalized_75.txt",hospitalized_75)
	np.savetxt("./data/hospitalized_90.txt",hospitalized_90)
	np.savetxt("./data/hospitalized_95.txt",hospitalized_95)

	np.savetxt("./data/icu_median.txt",icu_median)
	np.savetxt("./data/icu_05.txt",icu_05)
	np.savetxt("./data/icu_10.txt",icu_10)
	np.savetxt("./data/icu_25.txt",icu_25)
	np.savetxt("./data/icu_75.txt",icu_75)
	np.savetxt("./data/icu_90.txt",icu_90)
	np.savetxt("./data/icu_95.txt",icu_95)

	np.savetxt("./data/dead_median.txt",dead_median)
	np.savetxt("./data/dead_05.txt",dead_05)
	np.savetxt("./data/dead_10.txt",dead_10)
	np.savetxt("./data/dead_25.txt",dead_25)
	np.savetxt("./data/dead_75.txt",dead_75)
	np.savetxt("./data/dead_90.txt",dead_90)
	np.savetxt("./data/dead_95.txt",dead_95)

	np.savetxt("./data/immune_median.txt",immune_median)
	np.savetxt("./data/immune_05.txt",immune_05)
	np.savetxt("./data/immune_10.txt",immune_10)
	np.savetxt("./data/immune_25.txt",immune_25)
	np.savetxt("./data/immune_75.txt",immune_75)
	np.savetxt("./data/immune_90.txt",immune_90)
	np.savetxt("./data/immune_95.txt",immune_95)
    
	dates = [(datetime.datetime(2020,3,12) + datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(0,Nt)]
    
	raw_data = {
        'date' : dates,
        'active_median': active_median, 
        'active_05': active_05,
        'active_25': active_25, 
        'active_75': active_75,
        'active_95': active_95,
        'infectious_median': infectious_median, 
        'infectious_05': infectious_05,
        'infectious_25': infectious_25, 
        'infectious_75': infectious_75,
        'infectious_95': infectious_95, 
        'symptoms_median': symptoms_median, 
        'symptoms_05': symptoms_05,
        'symptoms_25': symptoms_25, 
        'symptoms_75': symptoms_75,
        'symptoms_95': symptoms_95, 
        'hospitalized_median': hospitalized_median, 
        'hospitalized_05': hospitalized_05,
        'hospitalized_25': hospitalized_25, 
        'hospitalized_75': hospitalized_75,
        'hospitalized_95': hospitalized_95, 
        'icu_median': icu_median, 
        'icu_05': icu_05,
        'icu_25': icu_25, 
        'icu_75': icu_75,
        'icu_95': icu_95, 
        'dead_median': dead_median, 
        'dead_05': dead_05,
        'dead_25': dead_25, 
        'dead_75': dead_75,
        'dead_95': dead_95,
        'immune_median': immune_median, 
        'immune_05': immune_05,
        'immune_25': immune_25, 
        'immune_75': immune_75,
        'immune_95': immune_95
        }
	df = pd.DataFrame(raw_data, columns = list(raw_data.keys()))
	df.to_csv('slo_pandemic_2020_04_03.csv')


#%% PLOT FIELDS
fig = plt.figure(figsize=(12,6))

plt.title("COVID-19 Pandemic in Slovenia")

# simulated values
# plt.fill_between(days,active_01,active_99,color="blue",alpha=0.1)
plt.fill_between(days,active_25,active_75,color="blue",alpha=0.2)
plt.plot(days,active_median,label="Active",color="blue",lw=3)

# plt.fill_between(days,symptoms_01,symptoms_99,color="green",alpha=0.1)
plt.fill_between(days,symptoms_25,symptoms_75,color="green",alpha=0.2)
plt.plot(days,symptoms_median,label="Symptomatic (cumulative)",color="green",lw=3)

plt.fill_between(days,infectious_25,infectious_75,color="green",alpha=0.2)
plt.plot(days,infectious_median,label="Infectious",color="cyan",lw=3)

# plt.fill_between(days,symptoms_01,symptoms_99,color="green",alpha=0.1)
plt.fill_between(days,hospitalized_25,hospitalized_75,color="orange",alpha=0.2)
plt.plot(days,hospitalized_median,label="Hospitalised",color="orange",lw=3)


plt.fill_between(days,icu_25,icu_75,color="brown",alpha=0.2)
plt.plot(days,icu_median,label="ICU",color="brown",lw=3)
 
plt.fill_between(days,dead_25,dead_75,color="black",alpha=0.4)
plt.plot(days,dead_median,label="Dead (cumulative)",color="black",lw=3)

# plt.plot(days[:20],contagious[:20]/7.5,'b--',label="Active cases")
# plt.plot(days,immune,label="Immune",color="green")
# plt.plot(days,critical,label="Critical (ICU)",color="red",lw=2)
# plt.plot(days,dead,'k-',label="Dead")
# plt.plot(days,[res_num]*len(days),'r--',label="Future healthcare capacity (125 respirators)")


# real data
plt.plot(data_days,data_cases,'go',label="Positive cases (data)")
plt.plot(data_days,data_hospitalised,'o',color='orange',label="Hospitalized (data)")
plt.plot(data_days,data_icu,'o',color='brown',label="ICU (data)")
plt.plot(data_days,data_dead,'ko',label="Dead (data)")


plt.yscale('log')

xticks_lbls = []
date0 = datetime.datetime(2020,3,12)
for i in range(Nt):
    date = date0+datetime.timedelta(i)
    xticks_lbls.append(date.strftime("%B %d"))
plt.xticks(range(0,Nt,10),xticks_lbls[::10],rotation=40)
# plt.fill_between(days, res_num ,0,alpha=0.2,color='r')
# plt.fill_between(days,critical,res_num)
plt.ylabel("Number of nodes")
plt.grid(b=True, which='major', color='grey', linestyle='-')
plt.grid(b=True, which='minor', color='grey', linestyle='--')
plt.ylim([1,10**5])
plt.xlim([-1,Nt])
plt.legend()
fig.savefig("potek_pandemije{:02d}.png".format(run),dpi=250)
    
