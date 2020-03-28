#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 12:35:57 2020

@author: ziga
"""
import numpy as np
from scipy.special import *
from scipy.stats import *


# Nall = 2078938

# nmb = np.array([212011,193037,211211,\
#                 290227,303945,302099,281171,\
#                     172426,112527])
# rate = np.array([0.0016,0.007,\
#                  0.031,0.084,\
#                  0.16,0.60,1.9,\
#                      4.3,7.8])
# rate_low = np.array([0.000185,0.0015,\
#                 0.014,0.041,\
#                  0.076,0.34,1.1,\
#                      2.5,3.8])
# rate_high = np.array([0.0249,0.050,\
#                  0.092,0.185,\
#                  0.32,1.3,3.9,\
#                     8.4,13.3])
# print (1.*nmb/Nall*rate_high).sum()



# hosp_rate = np.array([0.,0.04,\
#                  1.1,3.4,\
#                  4.3,8.2,11.8,\
#                      16.6,18.4])
# hosp_rate_low = np.array([0.,0.02,\
#                 0.62,2.1,\
#                  2.5,4.9,7.0,\
#                      9.9,11.0])
# hosp_rate_high = np.array([0.,0.08,\
#                  2.1,7.0,\
#                  8.7,16.7,24.0,\
#                     33.8,37.6])

# print (1.*nmb/Nall*hosp_rate).sum()
# print (1.*nmb/Nall*hosp_rate_low).sum()    
# print (1.*nmb/Nall*hosp_rate_high).sum()


dr = 0.0116

lambdaa=1.89
k=0.6

from scipy.optimize import fsolve,minimize


def mineq(x):
    mu,sigma = x
    return (lognorm.cdf(0.63,mu,sigma)-0.05)**2 +\
         (lognorm.cdf(2.22,mu,sigma)-0.95)**2 \
         + (lognorm.median(mu,sigma) - 1.16)**2

x0 = [0.4,0.2]
res = minimize(mineq,x0) 

mu,sigma = res.x[:]
print res.x
print lognorm.cdf(2.22,mu,sigma)
print lognorm.cdf(0.63,mu,sigma)
print lognorm.median(mu,sigma)


def mineq2(x):
    sigma,pos,mu = x
    return (lognorm.cdf(3.8,sigma,pos,mu)-0.05)**2 +\
         (lognorm.cdf(13.,sigma,pos,mu)-0.95)**2 \
         + (lognorm.median(sigma,pos,mu) - 6.37)**2

x0 = [1.,5.,6]
res2 = minimize(mineq2,x0) 

mu2,sigma2,pos = res2.x[:]
print res2.x
print lognorm.cdf(3.8,mu2,sigma2,pos)
print lognorm.cdf(13.,mu2,sigma2,pos)
print lognorm.median(mu2,sigma2,pos)


plt.figure(1)
x= np.arange(0.,15.,0.01)
plt.plot(x,lognorm.pdf(x,mu,sigma),'k-',label="infection fatality ratio")
plt.plot(x,lognorm.pdf(x,mu2,sigma2,pos),'r-',label="infection hospitalisation ratio")
plt.xlabel("percentage [%]")
plt.ylabel("Probabaility density")
plt.legend()
plt.savefig("IFR_IHR.png",dpi=300)


#%%
reproduction_numbers=np.array([3.0,2.2,2.68,1.95,2.25,4.5,\
                               3.11,6.47,2.7,2.35,6.49,2.9,\
                                   2.55,2.24,3.58,2.5,2.2])
reproduction_numbers_low = np.array([1.5,1.4,2.47,1.4,2.0,4.4,\
                                2.39,5.71,2.2,1.15,6.31,2.32,\
                                    2.0,1.96,2.89,1.5,1.4])
reproduction_numbers_high = np.array([4.5,3.9,2.86,2.5,2.5,4.6,\
                                      4.13,7.23,3.7,4.77,6.66,3.63,\
                                          3.1,2.55,4.39,3.5,3.9])
ylabels = ["This study"," ",
           "Kucharski et al.: Wuhan", \
           "Li, Leung and Leung: Wuhan",\
            "Wu et al.: Greater Wuhan",\
            "WHO Initial Estimate: Hubei",\
            "WHO-China Joint Mission: Hubei",\
            "Liu et al.: Guangdong",\
            "Read et al.: Wuhan",\
            "Tang et al.: China",\
            "Wu et al.: Hongkong ES",\
            "Kucharski et al.: Wuhan",\
            "Shen et al.: Hubei" ,\
            "Liu et al.: China and overseas",\
            "Majumder et al.: Wuhan",\
            "Zhao et al.: China",\
            "Zhao et al.: China",\
            "Imai: Wuhan",\
            "Qun Li et al.: China"]
    
plt.figure(2)
for i in range(2,19):
    plt.plot([reproduction_numbers_low[i-2],reproduction_numbers_high[i-2]],\
             [i,i],'kx-')
    plt.plot(reproduction_numbers[i-2],i,'ko')
plt.plot([np.median(reproduction_numbers_low),np.median(reproduction_numbers_high)],\
             [0,0],'rx-')
plt.plot(np.median(reproduction_numbers),0,'ro')
plt.yticks(range(0,19),ylabels)
plt.xlabel(r"Reproduction number $R_0$")
plt.tight_layout()
plt.savefig("r0_distribution.png",dpi=300)
   

#%%
def mineq3(x):
    sigma,pos,mu = x
    return (lognorm.cdf(2.0,sigma,pos,mu)-0.05)**2 +\
         (lognorm.cdf(3.9,sigma,pos,mu)-0.95)**2 \
         + (lognorm.median(sigma,pos,mu) - 2.68)**2

x0 = [0.5,2.,1]
res3 = minimize(mineq3,x0) 

mu3,sigma3,pos3 = res3.x[:]
print res3.x
print lognorm.cdf(2.,mu3,sigma3,pos3)
print lognorm.cdf(3.9,mu3,sigma3,pos3)
print lognorm.median(mu3,sigma3,pos3)


plt.figure(3)
x= np.arange(0.,6.,0.01)
plt.plot(x,lognorm.pdf(x,mu3,sigma3,pos3),'k-')
plt.xlabel(r"$R_0$")
plt.ylabel("Probabaility density")
# plt.legend()
plt.savefig("pdf_r0.png",dpi=300)

#%% ESTIMATING SARC
R0 = scipy.stats.lognorm.rvs(0.35535982, 1.14371067, 1.53628849,size=100000)
SARh = np.random.normal(0.35,0.0425,size=100000)
Nh = 1.5
Nc = 13.5
SARc = 1.*(R0-SARh*Nh)/Nc

a=plt.hist(SARc, bins = 50)
vals = a[0]
cntr = (a[1][1:]+a[1][:-1])/2.
b=scipy.stats.lognorm.fit(SARc, loc=0)
plt.plot(cntr,scipy.stats.lognorm.pdf(cntr,0.3442885157369917, 0.041072786026378566, 0.11874846218805696)*1000.)
print(b)

plt.figure(4)
x= np.arange(0.,0.6,0.001)
plt.plot(x,lognorm.pdf(x,b[0],b[1],b[2]),'k-',label="outer contacts")
plt.plot(x,norm.pdf(x,0.35,0.0425),'r-',label="household contacts")
plt.xlabel(r"Secondary attack rate (SAR)")
plt.ylabel("Probability distribution")
plt.legend()
plt.savefig("sar.png",dpi=300)


#%%
data_day_infected=np.genfromtxt("day_infected.txt")
data_num_con = np.genfromtxt("rands_input.txt")


#%%
nn=data_num_con.shape[0]
cons = np.arange(0,300)
days_con = np.zeros(300)
days_con_num = np.zeros(300)

for i in range(nn):
    nc = int(data_num_con[i])
    days_con[nc] += data_day_infected[i]
    days_con_num[nc] += 1
    
days_con = days_con/days_con_num
#%%
plt.figure(5)
plt.plot(cons,days_con)
plt.xlabel("Number of contacts")
plt.ylabel("Day of infection")
plt.xlim([-1,75])
plt.grid()
plt.savefig("day_of_infection.png",dpi=300)


#%% incubation period
def mineq4(x):
    sigma,pos,mu = x
    return (lognorm.cdf(1.7,sigma,pos,mu)-0.05)**2 +\
         (lognorm.cdf(10.6,sigma,pos,mu)-0.95)**2 \
            + (lognorm.cdf(15.4,sigma,pos,mu)-0.99)**2 \
         + (lognorm.mean(sigma,pos,mu) - 5.0)**2 \
        + (lognorm.median(sigma,pos,mu) - 4.3)**2

def mineq5(x):
    sigma,pos,mu = x
    return (lognorm.cdf(1.2,sigma,pos,mu)-0.05)**2 +\
         (lognorm.cdf(8.5,sigma,pos,mu)-0.95)**2 \
            + (lognorm.cdf(11.7,sigma,pos,mu)-0.99)**2 \
         + (lognorm.mean(sigma,pos,mu) - 4.2)**2 \
        + (lognorm.median(sigma,pos,mu) - 3.5)**2

def mineq6(x):
    sigma,pos,mu = x
    return (lognorm.cdf(2.3,sigma,pos,mu)-0.05)**2 +\
         (lognorm.cdf(14.1,sigma,pos,mu)-0.95)**2 \
            + (lognorm.cdf(22.5,sigma,pos,mu)-0.99)**2 \
         + (lognorm.mean(sigma,pos,mu) - 6.0)**2 \
        + (lognorm.median(sigma,pos,mu) - 5.1)**2

x0 = [0.5,2.,1]
res4 = minimize(mineq4,x0) 
sigma4,pos4,mu4 = res4.x[:]
print res4.x
print lognorm.cdf(1.7,sigma4,pos4,mu4)
print lognorm.cdf(10.6,sigma4,pos4,mu4)
print lognorm.median(sigma4,pos4,mu4)

x0 = [0.56,0.3,3.]
res5 = minimize(mineq5,x0) 
sigma5,pos5,mu5 = res5.x[:]
print res5.x
print lognorm.cdf(1.2,sigma5,pos5,mu5)
print lognorm.cdf(8.5,sigma5,pos5,mu5)
print lognorm.median(sigma5,pos5,mu5)

x0 = [sigma4,pos4,mu4]
res6 = minimize(mineq6,x0) 
sigma6,pos6,mu6 = res6.x[:]
print res6.x
print lognorm.cdf(2.3,sigma6,pos6,mu6)
print lognorm.cdf(14.1,sigma6,pos6,mu6)
print lognorm.median(sigma6,pos6,mu6)


#%%
def mineq7(x):
    sigma,pos,mu = x
    return (lognorm.cdf(0.2,sigma,pos,mu)-0.05)**2 +\
         (lognorm.cdf(14.0,sigma,pos,mu)-0.95)**2 \
            + (lognorm.cdf(35.0,sigma,pos,mu)-0.99)**2 \
         + (lognorm.mean(sigma,pos,mu) - 3.9)**2 \
        + (lognorm.median(sigma,pos,mu) - 1.5)**2

x0 = [sigma4,pos4,mu4]
res7 = minimize(mineq7,x0) 
sigma7,pos7,mu7 = res7.x[:]
print res7.x
print lognorm.cdf(0.2,sigma7,pos7,mu7)
print lognorm.cdf(14.0,sigma7,pos7,mu7)
print lognorm.median(sigma7,pos7,mu7)
print ""

def mineq8(x):
    sigma,pos,mu = x
    return (lognorm.cdf(6.5,sigma,pos,mu)-0.05)**2 +\
         (lognorm.cdf(26.8,sigma,pos,mu)-0.95)**2 \
            + (lognorm.cdf(36.0,sigma,pos,mu)-0.99)**2 \
         + (lognorm.mean(sigma,pos,mu) - 14.5)**2 \
        + (lognorm.median(sigma,pos,mu) - 13.2)**2

x0 = [sigma6,pos6,mu6]
res8 = minimize(mineq8,x0) 
sigma8,pos8,mu8 = res8.x[:]
print res8.x
print lognorm.cdf(6.5,sigma8,pos8,mu8)
print lognorm.cdf(26.8,sigma8,pos8,mu8)
print lognorm.median(sigma8,pos8,mu8)


plt.figure(6)
x= np.arange(0.,25.,0.01)
plt.plot(x,lognorm.pdf(x,sigma4,pos4,mu4),'g-',label="Incubation period (mean = 5.0 days)")
plt.plot(x,lognorm.pdf(x,sigma7,pos7,mu7),'r-',label="Illness onset to hospital admission (mean = 3.9 days)")
plt.plot(x,lognorm.pdf(x,sigma8,pos8,mu8),'k-',label="Illness onset do death (mean = 8.6 days)")
# plt.plot(x,lognorm.pdf(x,sigma6,pos6,mu6),'k-.',label="mean = 6.0, 95% CI HIGH")

plt.xlabel(r"days")
plt.ylabel("Probability density")
plt.legend()
plt.savefig("pdf_incubation.png",dpi=300)

#%% hospitalization and ICU length

def mineq9(x):
    sigma,pos,mu = x
    return (lognorm.cdf(0.4,sigma,pos,mu)-0.05)**2 +\
       +  (lognorm.cdf(12,sigma,pos,mu)-0.95)**2 \
        + (lognorm.mean(sigma,pos,mu) - 11)**2

x0 = [sigma4,pos4,mu4]
res7 = minimize(mineq9,x0) 
sigma7,pos7,mu7 = res7.x[:]
print res7.x


#%% Initial distribution of infection length
distexp = np.random.exponential(4.8,Ni)

#%%
plt.hist(distexp,bins=50,color="k")
plt.xlabel(r"Days infected $t$")
plt.ylabel(r"Number of patients infected for $t$ days")
plt.xlim([0,25])
plt.savefig("IC.png",dpi=300)


#%%hospital admission to death
def mineq8(x):
    sigma,pos,mu = x
    return (lognorm.cdf(2.2,sigma,pos,mu)-0.05)**2 +\
         (lognorm.cdf(20.5,sigma,pos,mu)-0.95)**2 \
            + (lognorm.cdf(32.6,sigma,pos,mu)-0.99)**2 \
         + (lognorm.mean(sigma,pos,mu) - 8.6)**2 \
        + (lognorm.median(sigma,pos,mu) - 6.7)**2

x0 = [sigma6,pos6,mu6]
res8 = minimize(mineq8,x0) 
sigma8,pos8,mu8 = res8.x[:]
print res8.x
print lognorm.cdf(2.2,sigma8,pos8,mu8)
print lognorm.cdf(20.5,sigma8,pos8,mu8)
print lognorm.median(sigma8,pos8,mu8)

# hospital admission to leave - severe
def mineq1(x):
    sigma,mu = x
    return (lognorm.median(sigma,0,mu) - 13.)**2

x0 = [sigma6,mu6]
res1 = minimize(mineq1,x0) 
sigma1,mu1 = res1.x[:]
print res1.x
print lognorm.median(sigma1,0,mu1)


# hospital admission to leave - non-sever
def mineq2(x):
    sigma,mu = x
    return  (lognorm.median(sigma,0,mu) - 11.)**2

x0 = [sigma6,mu6]
res2 = minimize(mineq2,x0) 
sigma2,mu2 = res2.x[:]
print res2.x
print lognorm.median(sigma2,0,mu2)


plt.figure(8)
x= np.arange(0.,30.,0.01)
plt.plot(x,lognorm.pdf(x,sigma8,pos8,mu8),'k-',label="Hospital admission to death")
plt.plot(x,lognorm.pdf(x,sigma1,0,mu1),'r-',label="Hospital admission to hospital leave (severe)")
plt.plot(x,lognorm.pdf(x,sigma2,0,mu2),'g-',label="Hospital admission to hospital leave (non-severe)")
# plt.plot(x,lognorm.pdf(x,sigma6,pos6,mu6),'k-.',label="mean = 6.0, 95% CI HIGH")

plt.xlabel(r"days")
plt.ylabel("Probability density")
plt.legend()
plt.savefig("hospital_admission_to_outcome.png",dpi=300)


#%%
def mineq8(x):
    sigma,pos,mu = x
    return (lognorm.cdf(10,sigma,pos,mu)-0.05)**2 +\
         (lognorm.cdf(13,sigma,pos,mu)-0.95)**2 \
        + (lognorm.median(sigma,pos,mu) - 11.)**2

x0 = [0.42140269, 8.99999444, 4.00000571]
res8 = minimize(mineq8,x0) 
sigma8,pos8,mu8 = res8.x[:]
print res8.x
print lognorm.cdf(10,sigma8,pos8,mu8)
print lognorm.cdf(13,sigma8,pos8,mu8)
print lognorm.median(sigma8,pos8,mu8)

