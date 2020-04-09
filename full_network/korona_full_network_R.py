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
#   To do list:
#       - PRIPISI STAROSTI IN SPOL NOVIM OKUZENIM!!
#       - DEBUG IT ALL!! (maybe solved)
"""

import os
import sys
from itertools import izip
import numpy as np
from scipy import stats
import datetime
import matplotlib.pyplot as plt

import ages_module
import covid19_parameters2
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
        # self.age_groups = [(0,30),(30,60),(60,ages_module.max_age)]
        self.age_groups = [(10*i+0,10*i+9 +1) for i in range(8)]+[(80, ages_module.max_age)]
        self.number_of_age_groups = len(self.age_groups)
        self.period_incubation = np.zeros(N, dtype=np.float16) + 20000.
        self.period_infectious_start = np.zeros(N, dtype=np.float16) + 20000.
        self.period_infectious_end = np.zeros(N, dtype=np.float16) + 20000.
        self.period_onset = np.zeros(N, dtype=np.float16) + 20000.
        self.period_home_recovery = np.zeros(N, dtype=np.float16) + 20000.
        self.period_hosp_severe = np.zeros(N, dtype=np.float16) + 20000.
        self.period_hosp_critical = np.zeros(N, dtype=np.float16) + 20000.
        self.period_hosp_deceased = np.zeros(N, dtype=np.float16) + 20000.

        self.init_other_stats()

    def set_transmission_clinical_parameters(self):
        # self.R0 = stats.lognorm.rvs(0.35535982, 1.14371067, 1.53628849)
        parameters = covid19_parameters2.read_parameters()

        self.doubling_time = parameters[0]
        self.param_infectious_start, self.param_infectious_end = parameters[1], parameters[2]
        self.ratio_asympt = parameters[3]
        self.ratio_hosp, self.ratio_severe, self.ratio_critic, self.ratio_deceas = parameters[4], parameters[5], parameters[6], parameters[7]
        self.ratio_critic_to_deceas = parameters[8]
        self.ratio_severe_to_deceas_wc, self.ratio_critic_to_deceas_wc = parameters[9], parameters[10]
        self.distparams_incubation, self.distparams_onset2hosp, self.distparams_home2recov = parameters[11], parameters[12], parameters[13]
        self.distparams_hosp2recov_s, self.distparams_hosp2recov_c, self.distparams_hosp2decea = parameters[14], parameters[15], parameters[16]
        self.sar_family, self.sar_others = parameters[17], parameters[18]
        self.sar_family_daily, self.sar_others_daily = parameters[19], parameters[20]

        factors = [2.**(stats.lognorm.mean(*params)/self.doubling_time) / (1. - ratio) for params, ratio in izip(self.distparams_incubation, self.ratio_asympt)]
        self.N_init = int(self.N_init * sum(factors)/len(factors))

        print "INITIAL CONDITIONS (model data)"
        print "Number of positively tested as of March 12: ", self.N_init
        print "Percentage of asymptomatic: ", np.array(self.ratio_asympt) * 100.
        print ""

        print "TRANSMISSION DYNAMICS PARAMETERS"

        # print "Basic reproduction number R0: ", self.R0
        print "Doubling time in the initial uncontrolled phase: ", self.doubling_time
        print "Multiplication factor for initial number of infected: ", factors
        print "Secondary attack rate - household contacts: ", self.sar_family
        print "Secondary attack rate - household contacts,daily: ", self.sar_family_daily
        print "Secondary attack rate - outer contacts: ", self.sar_others
        print "Secondary attack rate - outer contacts,daily: ", self.sar_others_daily
        print "Mean/median/std incubation period: ", [(stats.lognorm.mean(*params), stats.lognorm.median(*params), stats.lognorm.std(*params))
                                                      for params in self.distparams_incubation]

        # print "Infectious period: ", temp_param_infectious_period

        print ""
        print "CLINICAL PARAMETERS"
        print "Hospitalization ratio: ", self.ratio_hosp
        print "Includes"
        print "Fatality ratio (if all are hospitalised): ", self.ratio_deceas
        print "Critical ratio without fatal: ", self.ratio_critic, " and with fatal: ", [c+d for c, d in izip(self.ratio_critic, self.ratio_deceas)]
        print "Severe ratio: ", self.ratio_severe

        # print "Fatality ratio of critically ill: ", self.ratio_critic_to_deceas
        print "Fatality ratio of critically ill without intensive care: ", self.ratio_critic_to_deceas_wc
        print "Fatality ratio of severely ill without hospitalisation: ", self.ratio_severe_to_deceas_wc

        print "Mean/median/std illness onset to hospitalisation: ", [(stats.lognorm.mean(*params), stats.lognorm.median(*params), stats.lognorm.std(*params))
                                                                     for params in self.distparams_onset2hosp]
        print "Mean/median/std hospitalisation admission to death: ", [(stats.lognorm.mean(*params), stats.lognorm.median(*params), stats.lognorm.std(*params))
                                                                       for params in self.distparams_hosp2decea]
        print "Mean/median/std hospitalisation admission to leave (critical): ", [(stats.lognorm.mean(*params), stats.lognorm.median(*params), stats.lognorm.std(*params))
                                                                                for params in self.distparams_hosp2recov_c]
        print "Mean/median/std hospitalisation admission to leave (severe): ", [(stats.lognorm.mean(*params), stats.lognorm.median(*params), stats.lognorm.std(*params))
                                                                                for params in self.distparams_hosp2recov_s]

    def set_initial_condition(self, date0=datetime.datetime(2020, 3, 12)):
        ''' Randomly generates initial condition. '''
        self.day = 0
        self.date = date0

        self.isolate_from_family = False
        self.day_isolate = 20000.

        random_people = np.random.choice(self.population, self.N_init, replace=True)
        N_init_female = int(self.probability_female*self.N_init)
        N_init_male = self.N_init-N_init_female
        random_female = random_people[:N_init_female],
        random_male = random_people[N_init_female:]

        self.property_gender[random_male] = 1
        self.property_age[random_female] = ages_module.random_ages(ages_module.female_age_demograpy_distribution,
                                                                   0, ages_module.max_age, N_init_female)
        self.property_age[random_male] = ages_module.random_ages(ages_module.male_age_demograpy_distribution,
                                                                 0, ages_module.max_age, N_init_male)
        self.property_age_group[random_people] = np.array([ages_module.assign_agegroup(age, self.age_groups) for age in self.property_age[random_people]])

        time_infected = np.random.exponential(4.9, self.N_init)
        self.status_active[random_people] = 1
        self.status_susceptible[random_people] = 0

        # assign incubation
        time_incubation = np.array([stats.lognorm.rvs(*self.distparams_incubation[i]) for i in self.property_age_group[random_people]])
        identify_incubation = (time_infected - time_incubation < 0.5)
        self.status_incubation[random_people[identify_incubation]] = 1
        self.period_incubation[random_people] = identify_incubation * time_infected

        # assign infectious
        identify_infectious_start = (time_infected - np.array([self.param_infectious_start[i] for i in self.property_age_group[random_people]]) < 0.5)
        identify_infectious_end = (time_infected-time_incubation - np.array([self.param_infectious_end[i] for i in self.property_age_group[random_people]]) < 0.5)
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
        identify_onset = random_people[identify_onset & ~identify_incubation]
        self.separate_home_hosp(identify_onset)

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

        self.identify_incubation = np.where((self.status_active==1) & (self.status_incubation==0))[0]    # not active -> not incubation
        self.calculate_stats()

        print ""
        print "INITIAL STATE"
        print "Day: {0}".format(self.date.strftime("%B %d"))
        self.print_state_info()
        print ""

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
        self.k, self.theta = 0.3, 22.5    # 22.5=normal
        self.rands = np.random.gamma(self.k, self.theta, N)
        self.rands_int = self.rands.astype(np.int32)
        rands_input = self.rands_int + (self.rands - self.rands_int > np.random.random(N))
        rands_input_sorted = np.argsort(rands_input)
        self.connections_others, self.connection_others_max = generate_connections2.others(self.connections_others, rands_input, rands_input_sorted, self.k+1., self.theta)

    def calculate_stats(self):
        self.N_active = np.sum(self.status_active)
        self.N_incubation = np.sum(self.status_incubation)
        self.N_infectious = np.sum(self.status_infectious)
        self.N_till_eof = np.sum(self.status_till_eof)
        self.N_hosp = np.sum(self.status_hosp_severe + self.status_hosp_critic + self.status_deceased)
        self.N_icu = np.sum(self.status_hosp_critic + self.status_deceased)
        self.N_immune = np.sum(self.status_immune)
        self.N_susceptible = np.sum(self.status_susceptible)
        self.N_deceased = np.sum(self.status_deceased)
        self.N_symptoms += int(np.sum([np.sum(self.property_age_group[self.identify_incubation]==i) * (1.-self.ratio_asympt[i]) for i in range(self.number_of_age_groups)]))

        return self.N_susceptible, self.N_incubation, self.N_infectious, self.N_active, self.N_symptoms,\
               self.N_till_eof, self.N_hosp, self.N_icu, self.N_deceased, self.N_immune

    def init_other_stats(self):
        self.stats_day_infected = np.zeros(N, dtype=np.int16)
        self.r0_table = np.zeros(N, dtype=np.float16)

    def calculate_other_stats(self, i, identify_infected):
        self.stats_day_infected[identify_infected] = self.day
        self.r0_table[i] += identify_infected.size  # compute r0

    def separate_home_hosp(self, identify_onset):
        '''
        Separates those who are onset into two groups of those that stay home(0) and those who need to be hospitalised(1).
        People from hosiptalised group(1) are further split into those who will stay severe(0), get critical(1) or die(2).
        '''
        identify_onset_home = np.array([np.random.choice(2, p=[(1 - self.ratio_hosp[i]), self.ratio_hosp[i]]) for i in self.property_age_group[identify_onset]])
        # stay at home or asymptomatic
        identify_home = identify_onset[identify_onset_home==0]
        self.status_home[identify_home] = 1
        self.period_home_recovery[identify_home] = np.array([stats.lognorm.rvs(*self.distparams_home2recov[i]) for i in self.property_age_group[identify_home]])
        # go to hospital
        identify_hosp = identify_onset[identify_onset_home == 1]
        identify_hosp_state = np.array([np.random.choice(3, p=[self.ratio_severe[i]/self.ratio_hosp[i], self.ratio_critic[i]/self.ratio_hosp[i], self.ratio_deceas[i]/self.ratio_hosp[i]]) for i in self.property_age_group[identify_hosp]])
        identify_hosp_severe = identify_hosp[identify_hosp_state == 0]
        identify_hosp_critic = identify_hosp[identify_hosp_state == 1]
        identify_hosp_deceas = identify_hosp[identify_hosp_state == 2]
        self.status_hosp_deceas[identify_hosp_deceas] = 1
        self.status_hosp_critic[identify_hosp_critic] = 1
        self.status_hosp_severe[identify_hosp_severe] = 1
        self.period_hosp_deceased[identify_hosp_deceas] = np.array([stats.lognorm.rvs(*self.distparams_hosp2decea[i]) for i in self.property_age_group[identify_hosp_deceas]])
        self.period_hosp_deceased[identify_hosp_critic] = np.array([stats.lognorm.rvs(*self.distparams_hosp2recov_c[i]) for i in self.property_age_group[identify_hosp_critic]])
        self.period_hosp_deceased[identify_hosp_severe] = np.array([stats.lognorm.rvs(*self.distparams_hosp2recov_s[i]) for i in self.property_age_group[identify_hosp_severe]])

    def print_state_info(self):
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

    def infect_connected_nodes(self, i, susceptible):
        # go through all i's susceptible connections
        con_family = self.connections_family[i, :self.connection_family_max[i]]
        con_others = self.connections_others[i, :self.connection_others_max[i]]

        # infect susceptible connection with probabilities sar_family and sar_others
        for con, prob in izip([con_family, con_others], [self.sar_family_daily, self.sar_others_daily]):
            if con.size == 0:
                continue
            identify_infected = np.array(con[(susceptible[con] == 1) & (np.random.random(size=con.size) < prob[self.property_age_group[con]])])
            if identify_infected.size == 0:
                continue
            self.period_incubation[identify_infected] = np.array(
                [stats.lognorm.rvs(*self.distparams_incubation[j]) for j in self.property_age_group[identify_infected]])
            self.period_infectious_start[identify_infected] = self.param_infectious_start[
                self.property_age_group[identify_infected]]
            self.period_infectious_end[identify_infected] = self.period_incubation[identify_infected] + self.param_infectious_end[self.property_age_group[identify_infected]]
            self.period_onset[identify_infected] = self.period_incubation[identify_infected] + np.array(
                [stats.lognorm.rvs(*self.distparams_onset2hosp[j]) for j in self.property_age_group[identify_infected]])
            self.status_susceptible[identify_infected] = 0
            self.status_active[identify_infected] = 1
            self.status_incubation[identify_infected] = 1
            self.status_till_eof[identify_infected] = 1

            self.calculate_other_stats(i, identify_infected)

    def virus_spread_step(self):
        # remove 1 day from incubation period, infectious_period start/end
        self.period_incubation -= 1
        self.period_infectious_start -= 1
        self.period_infectious_end -= 1
        self.period_onset -= 1
        self.period_home_recovery -= 1
        self.period_hosp_deceased -= 1
        self.period_hosp_critical -= 1
        self.period_hosp_severe -= 1

        print "spread the virus"
        status_susceptible_old = np.copy(self.status_susceptible)
        status_infectious_old = np.copy(self.status_infectious)

        # CRITICALLY SLOW PART WITH PYTHON FOR LOOPS !!!! IMPROVE <--- is it better now? hope so, there's one loop less
        for i in np.where(status_infectious_old == 1)[0]:       # go over all infectious nodes indices
            self.infect_connected_nodes(i, status_susceptible_old)

    def check_illness_development(self):
        print "check illness development"
        # where incubation period < 0.5 --> illness onset
        identify_incubation = np.where((-0.5 <= self.period_incubation) & (self.period_incubation < 0.5))[0]
        self.identify_incubation = identify_incubation
        self.status_incubation[identify_incubation] = 0
        self.status_onset[identify_incubation] = 1

        if self.isolate_from_family and self.day >= self.day_isolate:
            self.identify_isolated = np.random.choice(identify_incubation, int((1 - self.ratio_asympt)*identify_incubation.size), replace=False)
            self.status_isolated[self.identify_isolated] = 1
            self.connection_family_max[self.identify_isolated] = 0

        # where infectiousness period start < 0.5 --> status = infectious
        identify_infectious_start = np.where(self.period_infectious_start < 0.5)[0]
        self.status_infectious[identify_infectious_start] = 1

        # where infectiousness period end < 0.5 --> status=immune/recovered, no longer infectious
        identify_infectious_end = np.where(self.period_infectious_end < 0.5)[0]
        self.status_infectious[identify_infectious_end] = 0
        self.status_till_eof[identify_infectious_end] = 0
        self.status_immune[identify_infectious_end] = 1

        # when onset < 0.5 --> status = hospitalised, no longer onset
        identify_onset = np.where(self.period_onset < 0.5)[0]
        self.separate_home_hosp(identify_onset)

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

    def additional_measures(self):
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
        pass

    def network_rewiring(self):
        self.day += 1
        self.date += datetime.timedelta(self.day)

        print "\nDay: {0} (+{1})".format(self.date.strftime("%B %d"), self.day)
        self.print_state_info()

        print "Rewiring outer contacts"
        print ""

        if self.day in [2, 4, 8]:
            if self.day == 2:                               # nasvet 14. marca
                self.rands /= np.random.uniform(1.5, 2.5)   # divide by 2.
            elif self.day == 4:                             # ukrepi 16. marca
                self.rands /= np.random.uniform(3.5, 4.5)   # divide by 8.
            elif self.day == 8:                             # ukrepi 20. marca
                self.rands /= np.random.uniform(1.5, 3.)    # divide by 2.
            self.rands_int = self.rands.astype(np.int32)

        rands_input = self.rands_int + (self.rands - self.rands_int > np.random.random(N))
        rands_input_sorted = np.argsort(rands_input)

        self.connections_others, self.connection_others_max = generate_connections2.others(self.connections_others, rands_input, rands_input_sorted, self.k+1., self.theta)
        if self.isolate_from_family and self.day >= self.day_isolate + 1:
            self.connection_others_max[self.identify_isolated] = 0

if __name__ == "__main__":
    try:
        run = int(sys.argv[1])
    except:
        run = None

    # healthcare capacity
    # res_num = 75  # respirators
    # beds = 590

    N = int(2.045795 * 10**6)       # number of nodes
    Ni = 200  # 127*2               # currently infected (12.marec)

    covid19 = Covid19(N, Ni)
    covid19.set_transmission_clinical_parameters()
    covid19.set_initial_condition()
    covid19.generate_initial_network()

    covid19.isolate_from_family = False
    covid19.day_isolate = 20        # 20 = 1. april

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

    tabs_stats = [tab_susceptible, tab_incubation, tab_infectious, tab_active, tab_symptoms, tab_till_eof, tab_hosp, tab_icu, tab_deceased, tab_immune]

    # intersting statistics
    r0_save = np.zeros(Nt+1)

    # calculate statistics
    statistics_results = covid19.calculate_stats()      # susceptible, incubation, infectious, active, symptoms, till_eof, hosp, icu, deceased, immune
    for i, tab in enumerate(tabs_stats):
        tab[covid19.day] = statistics_results[i]
    tab_symptoms[0] = covid19.N_symptoms

    # start simulation
    print "Simulate virus spread over network"

    while covid19.day < Nt:
        covid19.virus_spread_step()
        covid19.check_illness_development()
        covid19.additional_measures()                   # here, one can impose additional measures, disconnect some modes
        covid19.network_rewiring()

        statistics_results = covid19.calculate_stats()  # susceptible, incubation, infectious, active, symptoms, till_eof, hosp, icu, deceased, immune
        for i, tab in enumerate(tabs_stats):
            tab[covid19.day] = statistics_results[i]

        # compute other statistics
        # print covid19.r0_table
        # r0_save[covid19.day] = covid19.r0_table[covid19.r0_table > 0].mean()

    print "Simulation finished"

    # save fields
    if run != None:
        print ""
        print "Saving fields..."
        np.savetxt("./2020_04_03/tab_days_{:03d}.txt".format(run),tab_days,fmt='%8d')
        np.savetxt("./2020_04_03/tab_active_{:03d}.txt".format(run),tab_active,fmt='%8d')
        np.savetxt("./2020_04_03/tab_infectious_{:03d}.txt".format(run),tab_infectious,fmt='%8d')
        np.savetxt("./2020_04_03/tab_incubation_{:03d}.txt".format(run),tab_incubation,fmt='%8d')
        np.savetxt("./2020_04_03/tab_symptoms_{:03d}.txt".format(run),tab_symptoms,fmt='%8d')
        np.savetxt("./2020_04_03/tab_hospitalized_{:03d}.txt".format(run),tab_hosp,fmt='%8d')
        np.savetxt("./2020_04_03/tab_icu_{:03d}.txt".format(run), tab_icu,fmt='%8d')
        np.savetxt("./2020_04_03/tab_dead_{:03d}.txt".format(run), tab_deceased,fmt='%8d')
        np.savetxt("./2020_04_03/tab_immune_{:03d}.txt".format(run), tab_immune,fmt='%8d')
        np.savetxt("./2020_04_03/tab_susceptible_{:03d}.txt".format(run), tab_susceptible,fmt='%8d')

        np.savetxt("./save/day_infected_{:03d}.txt".format(run),covid19.stats_day_infected)
        # np.savetxt("./save/rands_input_{:03d}.txt".format(run),covid19.rands)   saves only last set of rands (why do you want that?)
        print "Fields saved"

    print ""
    print "* End of Main *"


