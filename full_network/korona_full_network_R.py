#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: ziga
@modified: Luka M

### QUESTIONS ###########################################################
#                                                                       #
#   How long is it going to last without overshooting the capacity?     #
#   What measures are needed to contain the virus?                      #
#   Does eliminating the superspreaders slow the virus spread           #
#   How long does it take for superspreaders to ge                      #
#                                                                       #
#########################################################################
"""

import os
import sys
import numpy as np
from scipy import stats
import datetime
import matplotlib.pyplot as plt

import ages_module
import generate_connections2    # compiled Fortran 90 code for Python2 (.so file)

class Covid19():
    def __init__(self, N, N_init):
        self.population = N
        self.N_init = N_init

        self.status_susceptible = np.ones(N, dtype=np.bool)
        self.status_incubation = np.zeros(N, dtype=np.bool)
        self.status_infectious = np.zeros(N, dtype=np.bool)
        self.status_active = np.zeros(N, dtype=np.bool)         # infection -> recovery/death
        self.status_till_eof = np.zeros(N, dtype=np.bool)       # infection -> end of infectious
        self.status_onset = np.zeros(N, dtype=np.bool)          # end of incubation -> hospitalization (minus home recovery & deceased)
        self.status_home = np.zeros(N, dtype=np.bool)
        self.status_hosp_severe = np.zeros(N, dtype=np.bool)    # hospitalised (severe)
        self.status_hosp_critic = np.zeros(N, dtype=np.bool)    # hospitalised (critical)
        self.status_hosp_deceas = np.zeros(N, dtype=np.bool)    # hospitalised (deceased)
        self.status_deceased = np.zeros(N, dtype=np.bool)
        self.status_immune = np.zeros(N, dtype=np.bool)         # immune & recovered
        self.status_isolated = np.zeros(N, dtype=np.bool)

        self.N_susceptible = N
        self.N_incubation = 0
        self.N_infectious = 0
        self.N_active = 0           # infection -> recovery/death
        self.N_till_eof = 0         # infection -> end of infectious
        self.N_onset = 0            # end of incubation -> hospitalization (minus home recovery & deceased)
        self.N_home = 0
        self.N_hosp = 0             # hospitalised (severe+critic+deceas)
        self.N_icu = 0              # hospitalised (critic+deceas)
        self.N_deceased = 0
        self.N_immune = 0           # immune & recovered
        self.N_symptoms = 0         # (active-incubation)*(1-ratio_asymptomatic)

        self.probability_female = 0.5
        self.probability_male = 1.-self.probability_female

        self.property_gender = np.zeros(N, dtype=np.bool)       # 0=female, 1=male
        self.property_age = np.zeros(N, dtype=np.int)
        self.property_age_group = np.zeros(N, dtype=np.int)
        self.age_groups = [(0,33),(33,66),(66,100)]

        self.period_incubation = np.zeros(N, dtype=np.float16) + 20000.
        self.period_infectious_start = np.zeros(N, dtype=np.float16) + 20000.
        self.period_infectious_end = np.zeros(N, dtype=np.float16) + 20000.
        self.period_onset = np.zeros(N, dtype=np.float16) + 20000.
        self.period_home_recovery = np.zeros(N, dtype=np.float16) + 20000.
        self.period_hosp_severe = np.zeros(N, dtype=np.float16) + 20000.
        self.period_hosp_critical = np.zeros(N, dtype=np.float16) + 20000.
        self.period_hosp_deceased = np.zeros(N, dtype=np.float16) + 20000.

    def set_transmission_clinical_parameters(self):
        self.doubling_time = np.random.normal(3.5, 0.5)
        self.R0 = stats.lognorm.rvs(0.35535982, 1.14371067, 1.53628849)

        # Distribution parameters by age groups
        self.param_infectious_start = [np.random.normal(3.,0.5), np.random.normal(3.,0.5), np.random.normal(3.,0.5)]      # after infection
        self.param_infectious_end = [np.random.normal(2.5,0.5), np.random.normal(2.5,0.5), np.random.normal(2.5,0.5)]     # after illness onset

        temp_param_hosp = stats.lognorm.rvs(0.57615709, 2.17310146, 4.19689622) / 100.
        temp_param_deceas = stats.lognorm.rvs(0.44779353, 0.15813357) / 100.
        temp_param_critic = 0.88679 * temp_param_deceas
        temp_param_severe = temp_param_hosp - temp_param_deceas - temp_param_critic
        self.ratio_hosp = [temp_param_hosp, temp_param_hosp, temp_param_hosp]
        self.ratio_deceas = [temp_param_deceas, temp_param_deceas, temp_param_deceas]
        self.ratio_critic = [temp_param_critic, temp_param_critic, temp_param_critic]
        self.ratio_severe = [temp_param_severe, temp_param_severe, temp_param_severe]

        self.ratio_critical_to_deceased = [d/(c+d)for c, d in zip(self.ratio_critic, self.ratio_deceas)]
        self.ratio_severe_to_deceased_wc = [np.random.normal(0.1,0.05), np.random.normal(0.1,0.05), np.random.normal(0.1,0.05)]      # wc=without (appropriate medical) care
        self.ratio_critic_to_deceased_wc = [np.random.normal(0.9,0.05), np.random.normal(0.9,0.05), np.random.normal(0.9,0.05)]      # wc=without (appropriate medical) care

        temp_param_s = stats.lognorm.rvs(0.42142559, 9.00007222, 1.99992887)        # temp_param=temporary parameter
        temp_param_c = stats.lognorm.rvs(0.42140269, 8.99999444, 4.00000571)
        self.distparams_incubation = [(0.54351624, -0.09672665, 4.396788397), (0.54351624, -0.09672665, 4.396788397), (0.54351624, -0.09672665, 4.396788397)]   # incubation period
        self.distparams_onset2hosp = [(1.39916754,  0.05548228, 1.444545973), (1.39916754,  0.05548228, 1.444545973), (1.39916754,  0.05548228, 1.444545973)]   # illness onset -> hospitalisation
        self.distparams_home2recov = [(0.60720317,  0.00000000, 6.000000000), (0.60720317,  0.00000000, 6.000000000), (0.60720317,  0.00000000, 6.000000000)]   # home -> recovery
        self.distparams_hosp2recov_s=[(0.60720317,  0.00000000,temp_param_s), (0.60720317,  0.00000000,temp_param_s), (0.60720317,  0.00000000,temp_param_s)]   # hospital admission (severe) -> leave
        self.distparams_hosp2recov_c=[(0.60720317,  0.00000000,temp_param_c), (0.60720317,  0.00000000,temp_param_c), (0.60720317,  0.00000000,temp_param_c)]   # hospital admission (critical) -> leave
        self.distparams_hosp2decea = [(0.71678881,  0.21484379, 6.485309000), (0.71678881,  0.21484379, 6.485309000), (0.71678881,  0.21484379, 6.485309000)]   # hospital admission -> decease

        temp_param_mean_incubation_period = [stats.lognorm.mean(*params) for params in self.distparams_incubation]
        temp_param_infectious_period = [m-s+e for m, s, e in zip(temp_param_mean_incubation_period, self.param_infectious_start, self.param_infectious_end)]
        temp_param_sar_others = stats.lognorm.rvs(0.3460311057053344, 0.04198595886371728, 0.11765118249339841)

        temp_param_sar_family = np.random.normal(0.35, 0.0425)
        temp_param_sar_others = stats.lognorm.rvs(0.3460311057053344, 0.04198595886371728, 0.11765118249339841)
        self.sar_family = [temp_param_sar_family, temp_param_sar_family, temp_param_sar_family]
        self.sar_others = [temp_param_sar_others, temp_param_sar_others, temp_param_sar_others]
        self.sar_family_daily = [1. - np.exp(np.log(1-sar)/period) for sar, period in zip(self.sar_family, temp_param_infectious_period)]
        self.sar_others_daily = [1. - np.exp(np.log(1-sar)/period) for sar, period in zip(self.sar_others, temp_param_infectious_period)]

        temp_param_asympt = np.random.normal(0.4, 0.10)
        self.ratio_asympt = [temp_param_asympt, temp_param_asympt, temp_param_asympt]
        factors = [2.**(stats.lognorm.mean(*params)/self.doubling_time) / (1. - ratio) for params, ratio in zip(self.distparams_incubation, self.ratio_asympt)]
        self.N_init = int(self.N_init * sum(factors)/len(factors))

        # Print:
        print "INITIAL CONDITIONS (model data)"
        print "Number of positively tested as of March 12: ", self.N_init
        print "Percentage of asymptomatic: ", np.array(self.ratio_asympt) * 100.
        print ""

        print "TRANSMISSION DYNAMICS PARAMETERS"

        print "Basic reproduction number R0: ", self.R0
        print "Doubling time in the initial uncontrolled phase: ", self.doubling_time
        print "Multiplication factor for initial number of infected: ", factors
        print "Secondary attack rate - household contacts: ", self.sar_family
        print "Secondary attack rate - household contacts,daily: ", self.sar_family_daily
        print "Secondary attack rate - outer contacts: ", self.sar_others
        print "Secondary attack rate - outer contacts,daily: ", self.sar_others_daily
        print "Mean/median/std incubation period: ", [(stats.lognorm.mean(*params), stats.lognorm.median(*params), stats.lognorm.std(*params))
                                                      for params in self.distparams_incubation]

        print "Infectious period: ", temp_param_infectious_period

        print ""
        print "CLINICAL PARAMETERS"
        print "Hospitalization ratio: ", self.ratio_hosp
        print "Includes"
        print "Fatality ratio (if all are hospitalised): ", self.ratio_deceas
        print "Critical ratio without fatal: ", self.ratio_critic, " and with fatal: ", [c+d for c, d in zip(self.ratio_critic, self.ratio_deceas)]
        print "Severe ratio: ", self.ratio_severe

        print "Fatality ratio of critically ill: ", self.ratio_critical_to_deceased
        print "Fatality ratio of critically ill without intensive care: ", self.ratio_critic_to_deceased_wc
        print "Fatality ratio of severely ill without hospitalisation: ", self.ratio_severe_to_deceased_wc

        print "Mean/median/std illness onset to hospitalisation: ", [(stats.lognorm.mean(*params), stats.lognorm.median(*params), stats.lognorm.std(*params))
                                                                     for params in self.distparams_onset2hosp]
        print "Mean/median/std hospitalisation admission to death: ", [(stats.lognorm.mean(*params), stats.lognorm.median(*params), stats.lognorm.std(*params))
                                                                       for params in self.distparams_hosp2decea]
        print "Mean/median/std hospitalisation admission to leave (critical): ", [(stats.lognorm.mean(*params), stats.lognorm.median(*params), stats.lognorm.std(*params))
                                                                                for params in self.distparams_hosp2recov_c]
        print "Mean/median/std hospitalisation admission to leave (severe): ", [(stats.lognorm.mean(*params), stats.lognorm.median(*params), stats.lognorm.std(*params))
                                                                                for params in self.distparams_hosp2recov_s]

    def set_initial_condition(self, N_init):
        ''' Randomly generates initial condition. '''
        self.day = 0
        self.date0 = datetime.datetime(2020, 3, 12)

        random_people = np.random.choice(self.population, N_init, replace=False)
        N_init_female = int(self.probability_female*N_init)
        N_init_male = N_init-N_init_female
        random_female = random_people[:N_init_female],
        random_male = random_people[N_init_female:]

        self.property_gender[random_male] = 1
        self.property_age[random_female] = ages_module.random_ages(ages_module.female_age_demograpy_distribution,
                                                                   0, ages_module.max_age, N_init_female)
        self.property_age[random_male] = ages_module.random_ages(ages_module.male_age_demograpy_distribution,
                                                                 0, ages_module.max_age, N_init_male)
        self.property_age_group[random_people] = [ages_module.assign_agegroup(age, self.age_groups)for age in self.property_age[random_people]]

        time_infected = np.random.exponential(4.9, N_init)
        self.status_active[random_people] = 1
        self.status_susceptible[random_people] = 0

        # assign incubation
        time_incubation = np.array([stats.lognorm.rvs(*self.distparams_incubation[i]) for i in self.property_age_group[random_people]])
        identify_incubation = (time_infected - time_incubation < 0.5)
        self.status_incubation[random_people[identify_incubation]] = 1
        self.period_incubation[random_people] = identify_incubation * time_infected

        # assign infectious
        identify_infectious_start = (time_infected - np.array([self.param_infectious_start[i] for i in self.property_age_group[random_people]]) < 0.5)
        identify_infectious_end = (time_infected - np.array([self.param_infectious_end[i] for i in self.property_age_group[random_people]]) < 0.5)
        self.period_infectious_start[random_people] = identify_infectious_start * time_infected
        self.period_infectious_end[random_people] = identify_infectious_end * time_infected

        self.status_infectious[random_people[identify_infectious_end & ~identify_infectious_start]] = 1
        self.status_till_eof[random_people[identify_infectious_end]] = 1
        self.status_immune[random_people[~identify_infectious_end]] = 1

        # assign illness onset to hospitalization
        time_onset = np.array([stats.lognorm.rvs(*self.distparams_onset2hosp[i]) for i in self.property_age_group[random_people]])
        identify_onset = (time_infected - (time_incubation + time_onset) < 0.5)
        self.period_onset[random_people] = (identify_onset & ~identify_incubation) * time_infected
        self.status_onset[random_people[identify_onset & ~identify_incubation]] = 1

        # choose which go to hospital, which stay home
        identify_inc_ons = random_people[identify_onset & ~identify_incubation]
        identify_inc_ons_home = np.array([np.random.choice(2, p=[(1 - self.ratio_hosp[i]), self.ratio_hosp[i]]) for i in self.property_age_group[identify_inc_ons]])
        # ---> stay at home or asymptomatic
        identify_home = identify_inc_ons[identify_inc_ons_home==0]
        self.status_home[identify_home] = 1
        self.period_home_recovery[identify_home] = np.array([stats.lognorm.rvs(*self.distparams_home2recov[i]) for i in self.property_age_group[identify_home]])
        # ---> go to hospital
        identify_hosp = identify_inc_ons[identify_inc_ons_home == 1]
        identify_hosp_state = np.array([np.random.choice(3, p=[self.ratio_deceas[i]/self.ratio_hosp[i], self.ratio_critic[i]/self.ratio_hosp[i], self.ratio_severe[i]/self.ratio_hosp[i]]) for i in self.property_age_group[identify_hosp]])
        identify_hosp_deceas = identify_hosp[identify_hosp_state == 0]
        identify_hosp_critic = identify_hosp[identify_hosp_state == 1]
        identify_hosp_severe = identify_hosp[identify_hosp_state == 2]
        self.status_hosp_deceas[identify_hosp_deceas] = 1
        self.status_hosp_critic[identify_hosp_critic] = 1
        self.status_hosp_severe[identify_hosp_severe] = 1
        self.period_hosp_deceased[identify_hosp_deceas] = np.array([stats.lognorm.rvs(*self.distparams_hosp2decea[i]) for i in self.property_age_group[identify_hosp_deceas]])
        self.period_hosp_deceased[identify_hosp_critic] = np.array([stats.lognorm.rvs(*self.distparams_hosp2recov_c[i]) for i in self.property_age_group[identify_hosp_critic]])
        self.period_hosp_deceased[identify_hosp_severe] = np.array([stats.lognorm.rvs(*self.distparams_hosp2recov_s[i]) for i in self.property_age_group[identify_hosp_severe]])

        # now remove people from home recovery
        identify_home = np.where(self.period_home_recovery < 0.5)[0]
        self.status_home[identify_home] = 0
        self.status_active[identify_home] = 0

        # now remove living/deceased from hospitals
        identify_hosp_deceas = np.where(self.period_hosp_deceased < 0.5)[0]
        identify_hosp_critic = np.where(self.period_hosp_critical < 0.5)[0]
        identify_hosp_severe = np.where(self.period_hosp_severe < 0.5)[0]
        self.status_hosp_deceas[identify_hosp_deceas] = 0
        self.status_hosp_critic[identify_hosp_critic] = 0
        self.status_hosp_severe[identify_hosp_severe] = 0
        self.status_deceased[identify_hosp_deceas] = 1
        self.status_immune[identify_hosp_critic] = 1
        self.status_immune[identify_hosp_severe] = 1
        self.status_active[identify_hosp_deceas] = 0
        self.status_active[identify_hosp_critic] = 0
        self.status_active[identify_hosp_severe] = 0

        self.calculate_statistics()

        print ""
        print "INITIAL STATE"
        print "Day: {0}".format(self.date0.strftime("%B %d"))
        print "Number of active (from infection to recovery): ", self.N_active
        print "Number of infectious people: ", self.N_infectious
        print "Number of people in incubation phase: ", self.N_incubation
        print "Number of people with symptoms to date (proxy for tested cases): ", self.N_symptoms
        print "Number of hospitalized: ", self.N_hosp
        print "Number of people in intensive care: ", self.N_icu
        print "Number of dead: ", self.N_deceased

        print "Number of immune: ", self.N_immune
        print "Number of susceptible: ", self.N_susceptible
        print "Nall", self.N_till_eof + self.N_immune + self.N_susceptible

    def generate_initial_network(self):
        # # these numbers are hardcoded in FORTRAN code
        # h1, h2, h3, h4, h5, h6, h7, h8 = 269898, 209573, 152959, 122195, 43327, 17398, 6073, 3195   # 1/2/3/4/5/6/7/8 persons per household
        # # elderly care
        # ec_size = 20000                         # persons in elderly care centers
        # ec_centers = 100                        # elderly care centers (100 for simplicity, actually 102)
        # pp_center = int(ec_size / ec_centers)   # people per center
        # group_size = 25                         # number of poeple in one group
        # gp_center = int(pp_center / group_size)

        # first add households as clusters where disease can spread infinitely
        maxc_family = 25
        maxc_others = 450
        self.connections_family = np.zeros((N, maxc_family), dtype=np.int32, order='F')
        self.connections_others = np.zeros((N, maxc_others), dtype=np.int32, order='F')
        self.connection_family_max = np.zeros(N, dtype=np.int32)
        self.connection_others_max = np.zeros(N, dtype=np.int32)

        print "Generating social network..."
        print "Family/care clusters..."
        self.connections_family, self.connection_family_max = generate_connections2.household(self.connections_family)      # call fotran function

        print "Outer contacts..."
        # we assume Gamma probability distribution
        k, theta = 0.3, 22.5    # 22.5=normal
        rands = np.random.gamma(k, theta, N)
        random_sample = np.random.random(N)
        rands_input = (rands - rands.astype(np.int32) > random_sample) + rands.astype(np.int32)
        rands_input_sorted = np.argsort(rands_input)
        self.connections_others, self.connection_others_max = generate_connections2.others(self.connections_others, rands_input, rands_input_sorted, k + 1., theta)

    def calculate_statistics(self):
        self.N_active = np.sum(self.status_active)
        self.N_incubation = np.sum(self.status_incubation)
        self.N_infectious = np.sum(self.status_infectious)
        self.N_till_eof = np.sum(self.status_till_eof)
        self.N_hosp = np.sum(self.status_hosp_severe + self.status_hosp_critic + self.status_deceased)
        self.N_icu = np.sum(self.status_hosp_critic + self.status_deceased)
        self.N_immune = np.sum(self.status_immune)
        self.N_susceptible = np.sum(self.status_susceptible)
        self.N_deceased = np.sum(self.status_deceased)
        self.N_symptoms = int(np.sum([(self.status_active[self.property_age_group==i] - self.status_incubation[self.property_age_group==i]) * (1. - self.ratio_asympt[i]) for i in self.property_age_group]))

        return self.N_susceptible, self.N_incubation, self.N_infectious, self.N_active, self.N_symptoms,\
               self.N_till_eof, self.N_hosp, self.N_icu, self.N_deceased, self.N_immune

    def virus_spread_step(self):
        status_susceptible_old = np.copy(self.status_susceptible)
        status_infectious_old = np.copy(self.status_infectious)

        # remove 1 day from incubation period, infectious_period start/end
        self.period_incubation[self.period_incubation > -0.5] -= 1
        self.period_infectious_start[self.period_infectious_start > -0.5] -= 1
        self.period_infectious_end[self.period_infectious_end > -0.5] -= 1
        self.period_onset[self.period_onset > -0.5] -= 1
        self.period_home_recovery[self.period_home_recovery > -0.5] -= 1
        self.period_hosp_deceased[self.period_hosp_deceased > -0.5] -= 1
        self.period_hosp_critical[self.period_hosp_critical > -0.5] -= 1
        self.period_hosp_severe[self.period_hosp_severe > -0.5] -= 1

        # spread the virus
        print "spread the virus"
        # go over all infectious nodes indices
        identify_infectious = (np.where(status_infectious_old == 1))[0]

        # CRITICALLY SLOW PART WITH PYTHON FOR LOOPS !!!! IMPROVE
        for i in identify_infectious:
            # go through all his susceptible connections
            con_other = self.connections_other[i, :self.connection_other_max[i]]
            con_family = self.connections_family[i, :self.connection_family_max[i]]

            for j in con_other:
                # infect susceptible connection with probability sar_other
                if status_susceptible_old[j] == 1 and np.random.random() < self.sar_others_daily:
                    self.period_incubation[j] = stats.lognorm.rvs(*self.distparams_incubation[j])
                    self.period_infectious_start[j] = self.param_infectious_start[j]
                    self.period_infectious_end[j] = self.incubation_period[j] + self.param_infectious_end[j]
                    self.period_onset[j] = incubation_period[j] + stats.lognorm.rvs(ioh1, ioh2, ioh3)

                    # update status
                    status_active[j] = 1
                    status_till_eof[j] = 1
                    status_susceptible[j] = 0
                    status_incubation[j] = 1

                    # compute other statistics
                    day_infected[j] = day
                    r0_table[i] += 1  # compute r0

            for j in con_family:
                # infect susceptible connection with probability sar_family
                if status_susceptible_old[j] == 1 and np.random.random() < sar_family_daily:
                    incubation_period[j] = stats.lognorm.rvs(ipp1, ipp2, ipp3)
                    infectious_period_start[j] = start_of_inf
                    infectious_period_end[j] = incubation_period[j] + end_of_inf
                    onset_period[j] = incubation_period[j] + stats.lognorm.rvs(ioh1, ioh2, ioh3)

                    # update status
                    status_active[j] = 1
                    status_till_eof[j] = 1
                    status_susceptible[j] = 0
                    status_incubation[j] = 1

                    # compute other statistics
                    day_infected[j] = day
                    r0_table[i] += 1  # compute r0

        print "check illness development"
        # check the illness development
        # where incubation period < 0.5 --> illness onset
        inc_ind = np.where((-0.5 < incubation_period) & (incubation_period < 0.5))[0]
        status_incubation[inc_ind] = 0
        status_onset[inc_ind] = 1

        if isolate_from_family:
            if day >= day_isolate:
                inc_choice = np.random.choice(inc_ind, int((1 - asymptomatic_ratio) * len(inc_ind)), replace=False)
                # print inc_choice
                all_isolated = np.concatenate((all_isolated, inc_choice))
                connection_family_max[all_isolated] = 0
                connection_other_max[all_isolated] = 0

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
        boze = np.random.choice(2, Nincons, p=[(1 - hr), hr], replace=True)

        # ---> stay at home or asymptomatic
        inc_home = inc_ons[boze == 0]
        home_recovery_period[inc_home] = stats.lognorm.rvs(ihh1, ihh2, ihh3, len(inc_home))
        status_home[inc_home] = 1

        # ---> go to hospital
        inc_hosp = inc_ons[boze == 1]
        n_hosp = inc_hosp.shape[0]

        # # assign hospitalisation from illness onset
        # n_hosp = int(np.round((hr*Nincons),0))

        # # indices of all hospitalised nodes
        # inc_hosp = np.random.choice(inc_ons, size = n_hosp, replace=False )

        # seperate hospitalized into
        pd = dr / hr  # death
        pc = cr / hr  # critical
        ps = sr / hr  # severe
        bog = np.random.choice(3, n_hosp, p=[pd, pc, ps], replace=True)

        inc_hosp_dead = inc_hosp[bog == 0]
        hospitalization_period_dead[inc_hosp_dead] = stats.lognorm.rvs(ihd1, ihd2, ihd3, size=len(
            inc_hosp_dead))  # starting the death count
        status_hospitalized_dead[inc_hosp_dead] = 1

        inc_hosp_critical = inc_hosp[bog == 1]
        hospitalization_period_critical[inc_hosp_critical] = stats.lognorm.rvs(ihls1, ihls2, ihls3, size=len(
            inc_hosp_critical))  # starting critical hospitalization count
        status_hospitalized_critical[inc_hosp_critical] = 1

        inc_hosp_severe = inc_hosp[bog == 2]
        hospitalization_period_severe[inc_hosp_severe] = stats.lognorm.rvs(ihln1, ihln2, ihln3, size=len(
            inc_hosp_severe))  # starting severe hospitalization count
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

        inc_hosp_critical = \
        np.where((-0.5 < hospitalization_period_critical) & (hospitalization_period_critical < 0.5))[0]
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
        Nactive = np.sum(status_active)
        Nincubation = np.sum(status_incubation)
        Ninfectious = np.sum(status_infectious)
        Ntill_eof = np.sum(status_till_eof)
        Nhospitalized = np.sum(status_hospitalized_dead + status_hospitalized_critical + status_hospitalized_severe)
        Nicu = np.sum(status_hospitalized_dead + status_hospitalized_critical)
        Nimmune = np.sum(status_immune)
        Nsusceptible = np.sum(status_susceptible)
        Ndead = np.sum(status_dead)

        Nsymptoms = Nsymptoms + int(inc_ind.shape[0] * (1. - asymptomatic_ratio))

        print "\nDay: {0} (+{1})".format(date.strftime("%B %d"), day)
        print "Number of active (from infection to recovery): ", Nactive
        print "Number of infectious people: ", Ninfectious
        print "Number of people in incubation phase: ", Nincubation
        print "Number of people with symptoms to date (should be proxy for tested cases): ", Nsymptoms
        print "Number of hospitalized: ", Nhospitalized
        print "Number of people in intensive care: ", Nicu
        print "Number of dead: ", Ndead

        print "Number of immune/recovered: ", Nimmune
        print "Number of susceptible: ", Nsusceptible
        print "Nall", Ntill_eof + Nimmune + Nsusceptible

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
        r0_save[day] = (r0_table[r0_table > 0]).mean()
        # print "r0: ",r0_save[day]

        # rewire outer contacts
        print "Rewiring outer contacts"

        # nasvet 14. marca
        if day == 2:
            rands = rands / np.random.uniform(1.5, 2.5)  # /2.

        # ukrepi 16. marca
        if day == 4:
            rands = rands / np.random.uniform(3.5, 4.5)  # /8.

        # ukrepi 20.marca
        if day == 8:
            rands = rands / np.random.uniform(1.5, 3.)  # /2.

        random_sample = np.random.random(N)
        rands_input = (rands - rands.astype(np.int32) > random_sample) + rands.astype(np.int32)
        rands_input_sorted = np.argsort(rands_input)

        connections_other, connection_other_max = generate_connections2.others(connections_other, \
                                                                               rands_input, rands_input_sorted, k + 1.,
                                                                               theta)
        if isolate_from_family:
            if day >= day_isolate + 1:
                connection_other_max[all_isolated] = 0

        print ""


        pass

if __name__ == "__main__":
    # run = int(sys.argv[1])

    isolate_from_family = False
    day_isolate = 20  # 20 = 1.april
    # healthcare capacity
    res_num = 75  # respirators
    beds = 590

    N = int(2.045795 * 10 ** 6)     # number of nodes
    Ni = 200  # 127*2               # currently infected (12.marec)

    covid19 = Covid19(N, Ni)
    covid19.set_transmission_clinical_parameters()
    covid19.set_initial_condition(Ni)

    Nt = 20  # 120
    tab_days = np.zeros(Nt+1)

    tab_susceptible = np.zeros(Nt+1)
    tab_incubation = np.zeros(Nt+1)
    tab_infectious = np.zeros(Nt+1)
    tab_active = np.zeros(Nt+1)
    tab_symptoms = np.zeros(Nt+1)
    tab_till_eof = np.zeros(Nt+1)
    tab_hosp = np.zeros(Nt+1)
    tab_icu = np.zeros(Nt+1)
    tab_deceased = np.zeros(Nt+1)
    tab_immune = np.zeros(Nt+1)
    tabs_stats1 = [tab_susceptible, tab_incubation, tab_infectious, tab_active, tab_symptoms, tab_till_eof, tab_hosp, tab_icu, tab_deceased, tab_immune]

    # intersting statistics
    r0_save = np.zeros(Nt+1)
    day_infected = np.zeros(N, dtype=np.int16)
    r0_table = np.zeros(N, dtype=np.float16)

    # calculate statistics
    statistics_results = covid19.calculate_statistics() # susceptible, incubation, infectious, active, symptoms, till_eof, hosp, icu, deceased, immune
    for i, tab in enumerate(tabs_stats1):
        tab[covid19.day] = statistics_results[i]

    # start simulation
    print "Simulate virus spread over network"

    while covid19.day < Nt:
        covid19.virus_spread_step()


    print "Simulation finished"
    print ""
    print "Saving fields..."
    # save fields
    # np.savetxt("./2020_04_03/tab_days_{:03d}.txt".format(run),tab_days,fmt='%8d')
    # np.savetxt("./2020_04_03/tab_active_{:03d}.txt".format(run),tab_active,fmt='%8d')
    # np.savetxt("./2020_04_03/tab_infectious_{:03d}.txt".format(run),tab_infectious,fmt='%8d')
    # np.savetxt("./2020_04_03/tab_incubation_{:03d}.txt".format(run),tab_incubation,fmt='%8d')
    # np.savetxt("./2020_04_03/tab_symptoms_{:03d}.txt".format(run),tab_symptoms,fmt='%8d')
    # np.savetxt("./2020_04_03/tab_hospitalized_{:03d}.txt".format(run),tab_hosp,fmt='%8d')
    # np.savetxt("./2020_04_03/tab_icu_{:03d}.txt".format(run), tab_icu,fmt='%8d')
    # np.savetxt("./2020_04_03/tab_dead_{:03d}.txt".format(run), tab_dead,fmt='%8d')
    # np.savetxt("./2020_04_03/tab_immune_{:03d}.txt".format(run), tab_immune,fmt='%8d')
    # np.savetxt("./2020_04_03/tab_susceptible_{:03d}.txt".format(run), tab_susceptible,fmt='%8d')

    # np.savetxt("./save/day_infected_{:03d}.txt".format(run),day_infected)
    # np.savetxt("./save/rands_input_{:03d}.txt".format(run),rands)
    print "Fields saved"
    print ""
    print "* End of Main *"


