#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 01:34:41 2020

@author: ziga
"""

import matplotlib.pyplot as plt
# import networkx as nx
import numpy as np
from scipy import stats
import datetime
import os
import sys


ens = True
run = int(sys.argv[1])

### QUESTIONS ###

# How long is it going to last without overshooting the capacity?
# What measures are needed to contain the virus?
# Does eliminating the superspreaders slow the virus spread
# How long does it take for superspreaders to ge


### PARAMETERS ###

# healthcare capacity
res_num = 171
beds = 590

#  number of nodes
N= int(2.045795*10**6)

# currently infected (12.marec)
Ni = 200#127*2


if not ens:
    doubling_time = 3.5 #days
    
    hr = 0.0637 # hospitalization ratio
    # infection fatality ratio mean
    dr = 0.0116 # death ratio, need of intensive care dr/(dr+cr)=0.53
    cr = 0.88679*dr # critical ratio, need of intensive care, 
    sr = hr-dr-cr # severe ratio
  
    # dr/cr = death rate of critically ill on intensive care
    dr_cr = dr/(dr+cr)
    dr_cr_wc = 0.9 # death rate of critically ill without intensive care
    dr_sr_wc = 0.1 # death rate of severly ill without oxygen
    
    # incubation period distribution
    ipp1,ipp2,ipp3 = 0.54351624  ,-0.09672665 ,4.396788397
    # inc = stats.lognorm.rvs(ipp1,ipp2,ipp3)
    
    # home to recovery 
    ihh1, ihh2, ihh3 = 0.60720317, 0., 6.
    
    # illness onset to hospitalisation distribution
    ioh1,ioh2,ioh3 = 1.3991675382957414, 0.05548227771906829, 1.44454597312056
    
    # hospital admission to death distribution
    ihd1,ihd2,ihd3 = 0.71678881, 0.21484379, 6.485309
    
    # hospital admission to leave (severe) distribution
    ihls1, ihls2, ihls3 = 0.60720317, 0., 12.99999999
    
    # hospital admission to leave (non-severe) distribution
    ihln1, ihln2, ihln3 = 0.60720317, 0., 11.00000008
    
    # icu_length_of_stay = 8.
    # hospital_length_of_stay = 11.
    
    
    start_of_inf = 3. # after infection
    end_of_inf =  2. # after illness onset
    mean_incubation_period = stats.lognorm.mean(ipp1,ipp2,ipp3)
    infectious_period = mean_incubation_period - start_of_inf + end_of_inf # 10 days 
    
    R0 = 2.68
    sar_family = 0.35
    sar_family_daily = 1.-np.exp(np.log(1-sar_family)/infectious_period)
    sar_other  = 0.1596
    sar_other_daily = 1.-np.exp(np.log(1-sar_other)/infectious_period)
    
    asymptomatic_ratio = 0.4
    factor = 2.**(stats.lognorm.mean(ipp1,ipp2,ipp3)/doubling_time)/(1.-asymptomatic_ratio)
    Ni = int(Ni * factor)
    
else:
    doubling_time =np.random.normal(3.5,0.5)
    
    hr = stats.lognorm.rvs(0.57615709, 2.17310146, 4.19689622)/100.
    dr = stats.lognorm.rvs(0.44779353, 0.15813357)/100.
    cr = 0.88679*dr
    sr = hr-dr-cr # severe ratio
    
    # dr/cr = death rate of critically ill on intensive care
    dr_cr = dr/(dr+cr)
    dr_cr_wc = np.random.normal(0.9,0.05) # death rate of critically ill without intensive care
    dr_sr_wc = np.random.normal(0.1,0.05) # death rate of severly ill without oxygen
    
    # incubation period distribution
    ipp1,ipp2,ipp3 = 0.54351624  ,-0.09672665 ,4.396788397
    # inc = stats.lognorm.rvs(ipp1,ipp2,ipp3)
    
    # illness onset to hospitalisation distribution
    ioh1,ioh2,ioh3 = 1.3991675382957414, 0.05548227771906829, 1.44454597312056
    
    # home to recovery 
    ihh1, ihh2, ihh3 = 0.60720317, 0., 6.
    
    # hospital admission to death distribution
    ihd1,ihd2,ihd3 = 0.71678881, 0.21484379, 6.485309
    
    # hospital admission to leave (critical) distribution
    ihls1, ihls2, ihls3 = 0.60720317, 0., stats.lognorm.rvs(0.42140269,8.99999444, 4.00000571)
    
    # hospital admission to leave (severe) distribution
    ihln1, ihln2, ihln3 = 0.60720317, 0., stats.lognorm.rvs(0.42142559, 9.00007222, 1.99992887)
    
    # icu_length_of_stay = np.random.normal(8.,2.)
    # hospital_length_of_stay = np.random.normal(11,1.75)
    
    start_of_inf = np.random.normal(3.,0.5)  # after infection
    end_of_inf =  np.random.normal(2.,1.) # after illness onset
    mean_incubation_period = stats.lognorm.mean(ipp1,ipp2,ipp3)
    infectious_period = mean_incubation_period - start_of_inf + end_of_inf
    
    R0 = stats.lognorm.rvs(0.35535982, 1.14371067, 1.53628849)
    sar_family = np.random.normal(0.35,0.0425)
    sar_family_daily = 1.-np.exp(np.log(1-sar_family)/infectious_period)
    sar_other = stats.lognorm.rvs(0.3460311057053344, 0.04198595886371728, 0.11765118249339841)
    sar_other_daily = 1.-np.exp(np.log(1-sar_other)/infectious_period)

    asymptomatic_ratio = np.random.normal(0.4,0.10)
    factor = 2.**(stats.lognorm.mean(ipp1,ipp2,ipp3)/doubling_time)/(1.-asymptomatic_ratio)
    Ni = int(Ni * factor)

print "INITIAL CONDITIONS (model data)"
print "Number of positively tested as of March 12: ",Ni
print "Percentage of asymptomatic: ",asymptomatic_ratio*100.
print ""

print "TRANSMISSION DYNAMICS PARAMETERS"

print "Basic reproduction number R0: ", R0
print "Doubling time in the initial uncontrolled phase: ",doubling_time
print "Multiplication factor for initial number of infected: ",factor
print "Secondary attack rate - household contacts: ", sar_family
print "Secondary attack rate - household contacts,daily: ",sar_family_daily
print "Secondary attack rate - outer contacts: ", sar_other
print "Secondary attack rate - outer contacts,daily: ",sar_other_daily
print "Mean/median/std incubation period: ", stats.lognorm.mean(ipp1,ipp2,ipp3),\
    stats.lognorm.median(ipp1,ipp2,ipp3), stats.lognorm.std(ipp1,ipp2,ipp3)

print "Infectious period: ",infectious_period


print ""
print "CLINICAL PARAMETERS"
print "Hospitalization ratio: ", hr
print "Includes"
print "Fatality ratio (if all are hospitalised): ", dr
print "Critical ratio without fatal: ", cr, " and with fatal: ", cr+dr
print "Severe ratio: ", sr

print "Fatality ratio of critically ill: ", dr_cr
print "Fatality ratio of critically ill without intensive care: ", dr_cr_wc
print "Fatality ratio of severely ill without hospitalisation: ", dr_sr_wc

print "Mean/median/std illness onset to hospitalisation: ", stats.lognorm.mean(ioh1,ioh2,ioh3),\
    stats.lognorm.median(ioh1,ioh2,ioh3), stats.lognorm.std(ioh1,ioh2,ioh3) 

print "Mean/median/std hospitalisation admission to death: ", stats.lognorm.mean(ihd1,ihd2,ihd3),\
    stats.lognorm.median(ihd1,ihd2,ihd3), stats.lognorm.std(ihd1,ihd2,ihd3)
print "Mean/median/std hospitalisation admission to leave (severe): ", stats.lognorm.mean(ihls1, ihls2, ihls3),\
    stats.lognorm.median(ihls1, ihls2, ihls3), stats.lognorm.std(ihls1, ihls2, ihls3)
print "Mean/median/std hospitalisation admission to leave (non-severe): ", stats.lognorm.mean(ihln1, ihln2, ihln3),\
    stats.lognorm.median(ihln1, ihln2, ihln3), stats.lognorm.std(ihln1, ihln2, ihln3)


status_active = np.zeros(N,dtype=np.bool)
status_immune = np.zeros(N,dtype=np.bool) #immune and recovered
status_susceptible = np.zeros(N,dtype=np.bool); status_susceptible[:] = 1
status_infectious = np.zeros(N,dtype=np.bool)
status_till_eof = np.zeros(N,dtype=np.bool)
status_incubation = np.zeros(N,dtype=np.bool)
status_onset = np.zeros(N,dtype=np.bool)
status_home = np.zeros(N,dtype=np.bool)
status_hospitalized_severe = np.zeros(N,dtype=np.bool)
status_hospitalized_critical = np.zeros(N,dtype=np.bool)
status_hospitalized_dead = np.zeros(N,dtype=np.bool)
status_dead = np.zeros(N,dtype=np.bool)

incubation_period = np.zeros(N,dtype=np.float16) + 20000.
infectious_period_start = np.zeros(N,dtype=np.float16) + 20000.
infectious_period_end = np.zeros(N,dtype=np.float16) + 20000.
onset_period = np.zeros(N,dtype=np.float16) + 20000.
home_recovery_period = np.zeros(N,dtype=np.float16) + 20000.
hospitalization_period_severe = np.zeros(N,dtype=np.float16) + 20000.
hospitalization_period_critical = np.zeros(N,dtype=np.float16) + 20000.
hospitalization_period_dead = np.zeros(N,dtype=np.float16) + 20000.

#### DEFINE INITIAL CONDITION ####
# randomly generate initial condition
rand_p = np.random.randint(0,N,Ni)
days_infected = np.random.exponential(4.9,Ni)
status_active[rand_p] = 1
status_susceptible[rand_p] = 0

# assign incubation
inc = stats.lognorm.rvs(ipp1,ipp2,ipp3,size=Ni)
Linc = (days_infected - inc < 0.5)
incubation_period[rand_p] = Linc *days_infected
status_incubation[rand_p[Linc]] = 1

# assign infectious
Linfs = (days_infected - start_of_inf < 0.5)
infectious_period_start[rand_p] = Linfs *days_infected

Linfe =  (days_infected - (inc + end_of_inf) < 0.5)
infectious_period_end[rand_p] = Linfe *days_infected

status_infectious[rand_p[Linfe & ~Linfs]] = 1

status_till_eof[rand_p[Linfe]] = 1
status_immune[rand_p[~Linfe]] = 1

# assign illness onset to hospitalization
ons = stats.lognorm.rvs(ioh1,ioh2,ioh3,size=Ni)
Lons = (days_infected - (inc + ons) < 0.5)
onset_period[rand_p] = (Lons & ~Linc)*days_infected
status_onset[rand_p[Lons & ~Linc]] = 1 

inc_ons = rand_p[Lons & ~Linc]
Nincons = inc_ons.shape[0]

# choose which go to hospital, which stay home
boze = np.random.choice(2,Nincons,p=[(1-hr),hr],replace=True)
    
#---> stay at home or asymptomatic
inc_home = inc_ons[boze==0]
home_recovery_period[inc_home] = stats.lognorm.rvs(ihh1,ihh2,ihh3,size=len(inc_home))
status_home[inc_home] = 1
    
#---> go to hospital
inc_hosp = inc_ons[boze==1]
n_hosp = inc_hosp.shape[0]


# seperate hospitalized into 
pd = dr/hr
pc = cr/hr
ps = sr/hr 
bog = np.random.choice(3,n_hosp,p=[pd,pc,ps],replace=True)

hosp_p_dead = inc_hosp[bog==0]
hospitalization_period_dead[hosp_p_dead] = stats.lognorm.rvs(ihd1,ihd2,ihd3,size=len(hosp_p_dead)) # starting the death count
status_hospitalized_dead[hosp_p_dead] = 1

hosp_p_critical = inc_hosp[bog==1]
hospitalization_period_critical[hosp_p_critical] = stats.lognorm.rvs(ihls1, ihls2, ihls3,size=len(hosp_p_critical)) # starting critical hospitalization count
status_hospitalized_critical[hosp_p_critical] = 1

hosp_p_severe = inc_hosp[bog==2]
hospitalization_period_severe[hosp_p_severe] = stats.lognorm.rvs(ihln1, ihln2, ihln3,size=len(hosp_p_severe)) # starting severe hospitalization count
status_hospitalized_severe[hosp_p_severe] = 1 


Nactive   = np.sum(status_active)
Nincubation =  np.sum(status_incubation)
Ninfectious = np.sum(status_infectious)
Ntill_eof = np.sum(status_till_eof)
Nhospitalized = np.sum(status_hospitalized_dead+status_hospitalized_critical+status_hospitalized_severe)
Nicu = np.sum(status_hospitalized_dead+status_hospitalized_critical)
Nimmune = np.sum(status_immune)
Nsusceptible = np.sum(status_susceptible) 
Ndead = np.sum(status_dead)

Nsymptoms = int((Nactive-Nincubation)*(1.-asymptomatic_ratio))

print ""
print "INITIAL STATE"
date0 = datetime.datetime(2020,3,12)
print "Day: {0}".format(date0.strftime("%B %d"))
print "Number of active (from infection to recovery): ",Nactive
print "Number of infectious people: ", Ninfectious
print "Number of people in incubation phase: ", Nincubation
print "Number of people with symptoms to date (proxy for tested cases): ", Nsymptoms
print "Number of hospitalized: ", Nhospitalized
print "Number of people in intensive care: ", Nicu
print "Number of dead: ",Ndead

print "Number of immune: ", Nimmune
print "Number of susceptible: ", Nsusceptible
print "Nall", Ntill_eof + Nimmune + Nsusceptible


# print "Number"

# households in Slovenia: https://www.stat.si/StatWeb/News/Index/7725
# these numbers are hardcoded in fortran
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


#%%

# GENERATE NETWORK

# first add households as clusters where disease can spread infinitely
maxc_family = 25
maxc_other = 450
connections_family = np.zeros((N,maxc_family),dtype=np.int32,order='F')
connections_other = np.zeros((N,maxc_other),dtype=np.int32,order='F')
connection_family_max = np.zeros(N,dtype=np.int32)

#%%

print "Generating social network..."
print "Family/care clusters..."

import generate_connections2

# call fotran function
connections_family,connection_family_max = generate_connections2.household(connections_family)

print "Outer contacts..."

#%%
# we assume Gamma probability distribution

k = 0.3
theta=22.5 ## 22.5=normal
rands = np.random.gamma(k,theta,N)
random_sample = np.random.random(N)
rands_input = (rands - rands.astype(np.int32)>random_sample) + rands.astype(np.int32)
rands_input_sorted = np.argsort(rands_input)

connections_other,connection_other_max = generate_connections2.others(connections_other,\
                                                    rands_input,rands_input_sorted,k+1.,theta)

#%%
# tp = np.zeros(450,dtype=np.int32)
# for i in range(450):
#     tp[i] = (connection_other_max==i).sum()

# plt.figure(1)
# plt.loglog(np.arange(0,450),tp,'ko-')
# plt.xscale('log')
# plt.xlim([0.9,450])
# plt.ylim([1,2*10**6])
# plt.xlabel(r"Number of contacts $x$")
# plt.ylabel(r"Number of people with $x$ contacts")
# ticks_num= [1,2,3,4,5,6,8,11,16,21,31,51,71,101]
# str_ticks_num= [str(tck-1) for tck in ticks_num]
# plt.xticks(ticks_num,str_ticks_num)
# plt.grid()
# plt.title("Contacts outside quarantined household/care clusters")
# props = dict(boxstyle='round', facecolor='red', alpha=0.4)  
# plt.text(1.1,2000,"Average number of outer contacts\n per person per day: {:5.2f}".format(\
#     np.sum(connection_other_max)/N),bbox=props)
# plt.text(1.1,100,"Average number of family contacts\n per person per day: {:5.2f}".format(\
#     np.sum(1.*connection_family_max)/N-1),bbox=props)
# # plt.text(0.34,2,"{0} persons have 0 contact per day\n{1} persons have 1 contact per day\n{2} persons have 5-6 contacts per day".format(\
# #         int(tp[0]),int(tp[1]),int(tp[5])),bbox=props)  
# plt.savefig("kontakti_new{:02d}.png".format(run),dpi=300)


#%% SIMULATE VIRUS SPREAD       
        
Nt = 120
tab_days = np.zeros(Nt+1)
tab_active = np.zeros(Nt+1)
tab_infectious = np.zeros(Nt+1)
tab_incubation = np.zeros(Nt+1)
tab_symptoms = np.zeros(Nt+1)
tab_hospitalized = np.zeros(Nt+1)
tab_icu = np.zeros(Nt+1)
tab_dead = np.zeros(Nt+1)
tab_immune = np.zeros(Nt+1)
tab_susceptible = np.zeros(Nt+1) 

r0_save = np.zeros(Nt+1)

# intersting statistics
day_infected = np.zeros(N,dtype=np.int16)
r0_table = np.zeros(N,dtype=np.float16)    

# start simulation
print "Simulate virus spread over network"

day = 0 

tab_active[day] = Nactive
tab_infectious[day] = Ninfectious
tab_incubation[day] = Nincubation
tab_symptoms[day] = int((Nactive-Nincubation)*(1.-asymptomatic_ratio))
tab_hospitalized[day] = Nhospitalized
tab_icu[day] = Nicu
tab_dead[day] = Ndead
tab_immune[day] = Nimmune
tab_susceptible[day] = Nsusceptible


while day < Nt:
    
    status_susceptible_old = np.copy(status_susceptible)
    status_infectious_old = np.copy(status_infectious)
    
    # remove 1 day from incubation period, infectious_period start/end
    incubation_period[incubation_period>-0.5] -= 1
    infectious_period_start[infectious_period_start>-0.5] -= 1
    infectious_period_end[infectious_period_end>-0.5] -= 1
    onset_period[onset_period>-0.5] -= 1
    home_recovery_period[home_recovery_period>-0.5] -= 1
    hospitalization_period_dead[hospitalization_period_dead>-0.5] -= 1
    hospitalization_period_critical[hospitalization_period_critical>-0.5] -= 1
    hospitalization_period_severe[hospitalization_period_severe>-0.5] -= 1
    
    
    # spread the virus
    print "spread the virus"
    # go over all infectious nodes indices
    infectious_ind = (np.where(status_infectious_old == 1))[0]

    # CRITICALLY SLOW PART WITH PYTHON FOR LOOPS !!!! IMPROVE
    for i in infectious_ind:
        # go through all his susceptible connections
        con_other = connections_other[i,:connection_other_max[i]]
        con_family = connections_family[i,:connection_family_max[i]]
 
        for j in con_other:
            # infect susceptible connection with probability sar_other
            if status_susceptible_old[j] == 1 and np.random.random() < sar_other_daily:
                incubation_period[j] = stats.lognorm.rvs(ipp1,ipp2,ipp3)
                infectious_period_start[j] = start_of_inf
                infectious_period_end[j] = incubation_period[j] + end_of_inf
                onset_period[j] = incubation_period[j] + stats.lognorm.rvs(ioh1,ioh2,ioh3)
                
                # update status
                status_active[j] = 1
                status_till_eof[j] = 1
                status_susceptible[j] = 0
                status_incubation[j] = 1
                
                # compute other statistics
                day_infected[j] = day
                r0_table[i] += 1 # compute r0
                
        for j in con_family:
            # infect susceptible connection with probability sar_family
            if status_susceptible_old[j] == 1 and np.random.random() < sar_family_daily:
                incubation_period[j] = stats.lognorm.rvs(ipp1,ipp2,ipp3)
                infectious_period_start[j] = start_of_inf
                infectious_period_end[j] = incubation_period[j] + end_of_inf
                onset_period[j] = incubation_period[j] + stats.lognorm.rvs(ioh1,ioh2,ioh3)
                
                # update status
                status_active[j] = 1
                status_till_eof[j] = 1
                status_susceptible[j] = 0
                status_incubation[j] = 1
                
                # compute other statistics
                day_infected[j] = day
                r0_table[i] += 1 # compute r0
        
    print "check illness development"    
    # check the illness development
    # where incubation period < 0.5 --> illness onset   
    inc_ind = np.where((-0.5 < incubation_period) & (incubation_period < 0.5))[0]
    status_incubation[inc_ind] = 0
    status_onset[inc_ind] = 1
                
    # where infectiousnees period start < 0.5 --> status = infectious
    inc_inf_start = np.where((-0.5 < infectious_period_start) & (infectious_period_start < 0.5))[0]
    status_infectious[inc_inf_start] = 1
    
    # where infectiousness period end < 0.5 -->   status=immune/recovered, no longer infectious
    inc_inf_end = np.where((-0.5 < infectious_period_end) & (infectious_period_end < 0.5))[0]
    status_infectious[inc_inf_end] = 0
    status_till_eof[inc_inf_end] = 0 
    status_immune[inc_inf_end] = 1
    
    # when onset < 0.5 --> status = hospitalised, no longer onset          
    inc_ons = np.where((-0.5 < onset_period) & (onset_period < 0.5))[0]
    status_onset[inc_ons] = 0 
    Nincons = inc_ons.shape[0]
    
    # choose which go to hospital, which stay home
    boze = np.random.choice(2,Nincons,p=[(1-hr),hr],replace=True)
    
    #---> stay at home or asymptomatic
    inc_home = inc_ons[boze==0]
    home_recovery_period[inc_home] = stats.lognorm.rvs(ihh1, ihh2, ihh3,len(inc_home))
    status_home[inc_home] = 1
    
    #---> go to hospital
    inc_hosp = inc_ons[boze==1]
    n_hosp = inc_hosp.shape[0]
    
    # # assign hospitalisation from illness onset
    # n_hosp = int(np.round((hr*Nincons),0))
     
    # # indices of all hospitalised nodes
    # inc_hosp = np.random.choice(inc_ons, size = n_hosp, replace=False )

    # seperate hospitalized into 
    pd = dr/hr #death
    pc = cr/hr #critical
    ps = sr/hr #severe
    bog = np.random.choice(3,n_hosp,p=[pd,pc,ps],replace=True)

    inc_hosp_dead = inc_hosp[bog==0]
    hospitalization_period_dead[inc_hosp_dead] = stats.lognorm.rvs(ihd1,ihd2,ihd3,size=len(inc_hosp_dead)) # starting the death count
    status_hospitalized_dead[inc_hosp_dead] = 1
    
    inc_hosp_critical = inc_hosp[bog==1]
    hospitalization_period_critical[inc_hosp_critical] = stats.lognorm.rvs(ihls1, ihls2, ihls3,size=len(inc_hosp_critical)) # starting critical hospitalization count
    status_hospitalized_critical[inc_hosp_critical] = 1
    
    inc_hosp_severe = inc_hosp[bog==2]
    hospitalization_period_severe[inc_hosp_severe] = stats.lognorm.rvs(ihln1, ihln2, ihln3,size=len(inc_hosp_severe)) # starting severe hospitalization count
    status_hospitalized_severe[inc_hosp_severe] = 1 

    # now remove people from home recovery
    inc_home = np.where((-0.5 < home_recovery_period) & (home_recovery_period < 0.5))[0]
    status_home[inc_home] = 0
    status_active[inc_home] = 0

    # now remove living/deceased from hospitals
    inc_hosp_dead = np.where((-0.5 < hospitalization_period_dead) & (hospitalization_period_dead < 0.5))[0]
    status_hospitalized_dead[inc_hosp_dead] = 0
    status_dead[inc_hosp_dead] = 1
    status_active[inc_hosp_dead] = 0
    
    inc_hosp_critical = np.where((-0.5 < hospitalization_period_critical) & (hospitalization_period_critical < 0.5))[0]
    status_hospitalized_critical[inc_hosp_critical] = 0
    status_immune[inc_hosp_critical] = 1
    status_active[inc_hosp_critical] = 0
    
    inc_hosp_severe = np.where((-0.5 < hospitalization_period_severe) & (hospitalization_period_severe < 0.5))[0]
    status_hospitalized_severe[inc_hosp_severe] = 0 
    status_immune[inc_hosp_severe] = 1
    status_active[inc_hosp_severe] = 0
    

    ###############################################    
    # here, one can impose additional measures, disconnect some modes
    # 
    # to remove all contacts of node[i], type rands[i] = 0
    #
    #
    #
    #
    #
    ###############################################
    
    day += 1
    date = date0 + datetime.timedelta(day)
    
    # compute statistics
    Nactive   = np.sum(status_active)
    Nincubation =  np.sum(status_incubation)
    Ninfectious = np.sum(status_infectious)
    Ntill_eof = np.sum(status_till_eof)
    Nhospitalized = np.sum(status_hospitalized_dead+status_hospitalized_critical+status_hospitalized_severe)
    Nicu = np.sum(status_hospitalized_dead+status_hospitalized_critical)
    Nimmune = np.sum(status_immune)
    Nsusceptible = np.sum(status_susceptible)
    Ndead = np.sum(status_dead)
    
    Nsymptoms = Nsymptoms + int(inc_ind.shape[0]*(1.-asymptomatic_ratio))
    
    print "\nDay: {0} (+{1})".format(date.strftime("%B %d"),day)
    print "Number of active (from infection to recovery): ",Nactive
    print "Number of infectious people: ", Ninfectious
    print "Number of people in incubation phase: ", Nincubation
    print "Number of people with symptoms to date (should be proxy for tested cases): ",Nsymptoms
    print "Number of hospitalized: ", Nhospitalized
    print "Number of people in intensive care: ", Nicu
    print "Number of dead: ",Ndead
    
    print "Number of immune/recovered: ", Nimmune
    print "Number of susceptible: ", Nsusceptible
    print "Nall", Ntill_eof + Nimmune  + Nsusceptible
    

    # add statistics to table
    tab_days[day] = day
    tab_active[day] = Nactive
    tab_infectious[day] = Ninfectious
    tab_incubation[day] = Nincubation
    tab_symptoms[day] = Nsymptoms
    tab_hospitalized[day] = Nhospitalized
    tab_icu[day] = Nicu
    tab_dead[day] = Ndead
    tab_immune[day] = Nimmune
    tab_susceptible[day] = Nsusceptible
    
    # compute other statistics
    r0_save[day] = (r0_table[r0_table>0]).mean()
    # print "r0: ",r0_save[day]

    # rewire outer contacts
    print "Rewiring outer contacts"
    
    # nasvet 14. marca
    if day == 2:
        rands = rands/np.random.uniform(1.5,2.5) # /2.
    
    # ukrepi 16. marca
    if day == 4:
        rands = rands/np.random.uniform(3.0,5.5) # /8.
    
    # ukrepi 18.marca
    if day == 8:
        rands = rands/np.random.uniform(1.2,4.5) # /2.
    
    random_sample = np.random.random(N)
    rands_input = (rands - rands.astype(np.int32)>random_sample) + rands.astype(np.int32)
    rands_input_sorted = np.argsort(rands_input)
    
    connections_other,connection_other_max = generate_connections2.others(connections_other,\
                                                        rands_input,rands_input_sorted,k+1.,theta)
    print ""


print "Simulation finished"
print ""
print "Saving fields"
# save fields
np.savetxt("./save2/tab_days_{:03d}.txt".format(run),tab_days,fmt='%8d')
np.savetxt("./save2/tab_active_{:03d}.txt".format(run),tab_active,fmt='%8d')
np.savetxt("./save2/tab_infectious_{:03d}.txt".format(run),tab_infectious,fmt='%8d')
np.savetxt("./save2/tab_incubation_{:03d}.txt".format(run),tab_incubation,fmt='%8d')
np.savetxt("./save2/tab_symptoms_{:03d}.txt".format(run),tab_symptoms,fmt='%8d')
np.savetxt("./save2/tab_hospitalized_{:03d}.txt".format(run),tab_hospitalized,fmt='%8d')
np.savetxt("./save2/tab_icu_{:03d}.txt".format(run), tab_icu,fmt='%8d')
np.savetxt("./save2/tab_dead_{:03d}.txt".format(run), tab_dead,fmt='%8d')
np.savetxt("./save2/tab_immune_{:03d}.txt".format(run), tab_immune,fmt='%8d')
np.savetxt("./save2/tab_susceptible_{:03d}.txt".format(run), tab_susceptible,fmt='%8d')

# np.savetxt("./save/day_infected_{:03d}.txt".format(run),day_infected) 
# np.savetxt("./save/rands_input_{:03d}.txt".format(run),rands)  
