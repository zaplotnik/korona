import numpy as np
from scipy import stats
from itertools import izip

def read_parameters():
    # temporary parameters
    temp_param_infectious_start = np.random.normal(3., 0.5)
    temp_param_infectious_end = np.random.normal(2.5, 0.5)
    temp_param_s = stats.lognorm.rvs(0.42142559, 9.00007222, 1.99992887)
    temp_param_c = stats.lognorm.rvs(0.42140269, 8.99999444, 4.00000571)
    temp_param_asympt1 = np.random.normal(0.5, 0.1)
    temp_param_asympt2 = np.random.normal(0.3, 0.1)

    doubling_time = np.random.normal(3.5, 0.5)                                  # for whole population
    # R0 = stats.lognorm.rvs(0.35535982, 1.14371067, 1.53628849)                # this parameter is never used (why do we need it?)

    # Distribution parameters by age groups
    param_infectious_start = np.array([temp_param_infectious_start,             # after infection
                                       temp_param_infectious_start,
                                       temp_param_infectious_start,
                                       temp_param_infectious_start,
                                       temp_param_infectious_start,
                                       temp_param_infectious_start,
                                       temp_param_infectious_start,
                                       temp_param_infectious_start,
                                       temp_param_infectious_start])
    param_infectious_end = np.array([temp_param_infectious_end,                 # after illness onset
                                     temp_param_infectious_end,
                                     temp_param_infectious_end,
                                     temp_param_infectious_end,
                                     temp_param_infectious_end,
                                     temp_param_infectious_end,
                                     temp_param_infectious_end,
                                     temp_param_infectious_end,
                                     temp_param_infectious_end])

    ratio_asympt = [temp_param_asympt1,                                         # ratio_asympt = asymptomatic / all_infected(i.e. known+unknown)
                    temp_param_asympt1,
                    temp_param_asympt1,
                    temp_param_asympt1,
                    temp_param_asympt1,
                    temp_param_asympt2,
                    temp_param_asympt2,
                    temp_param_asympt2,
                    temp_param_asympt2]
    ratio_severe = [stats.norm.rvs(16.09999999, 1.67365310) /1000000,           # ratio_severe = severe / all_infected(i.e. known+unknown)
                    stats.norm.rvs(4.079997660, 0.84185997) /10000,
                    stats.norm.rvs(10.39999943, 2.13262174) /1000,
                    stats.norm.rvs(3.429999990, 0.21143857) /100,
                    stats.norm.rvs(4.250001620, 0.87760372) /100,
                    stats.norm.rvs(8.160003800, 1.68370209) /100,
                    stats.norm.rvs(11.79999819, 2.44263192) /100,
                    stats.norm.rvs(16.59999901, 3.43586235) /100,
                    stats.norm.rvs(18.40000124, 3.77561592) /100]
    ratio_deceas = [stats.norm.rvs(16.0999999, 1.67365310) /1000000,            # ratio_critical = critical / all_infected(i.e. known+unknown)
                    stats.norm.rvs(6.94999864, 1.27959154) /100000,
                    stats.norm.rvs(3.08999842, 0.87240868) /10000,
                    stats.norm.rvs(8.44000088, 2.22481098) /10000,
                    stats.norm.rvs(1.61006021, 0.43275679) /1000,
                    stats.norm.rvs(5.95000235, 1.28064016) /1000,
                    stats.norm.rvs(9.64999838, 2.09200771) /500,
                    stats.norm.rvs(4.28000624, 0.93405922) /100,
                    stats.norm.rvs(7.80032793, 2.13982958) /100]
    # deceased/(deceased+critical)=0.53
    ratio_critic = [(1-0.53)/0.53*d for d in ratio_deceas],                     # ratio_critical = deceased / all_infected(i.e. known+unknown)
    ratio_hosp = [s+c+d for s,c,d in izip(ratio_severe,ratio_critic,ratio_deceas)]      # ratio_hosp = hospitalised / all_infected(i.e. known+unknown)

    ratio_critic_to_deceas = [d/(c+d) for c, d in izip(ratio_critic, ratio_deceas)]

    ratio_severe_to_deceas_wc = [np.random.normal(0.1, 0.05),                   # wc=without (appropriate medical) care
                                 np.random.normal(0.1, 0.05),
                                 np.random.normal(0.1, 0.05),
                                 np.random.normal(0.1, 0.05),
                                 np.random.normal(0.1, 0.05),
                                 np.random.normal(0.1, 0.05),
                                 np.random.normal(0.1, 0.05),
                                 np.random.normal(0.1, 0.05),
                                 np.random.normal(0.1, 0.05)]
    ratio_critic_to_deceas_wc = [np.random.normal(0.9, 0.05),
                                 np.random.normal(0.9, 0.05),
                                 np.random.normal(0.9, 0.05),
                                 np.random.normal(0.9, 0.05),
                                 np.random.normal(0.9, 0.05),
                                 np.random.normal(0.9, 0.05),
                                 np.random.normal(0.9, 0.05),
                                 np.random.normal(0.9, 0.05),
                                 np.random.normal(0.9, 0.05)]

    distparams_incubation = [(0.54351624, -0.09672665, 4.396788397),    # incubation period
                             (0.54351624, -0.09672665, 4.396788397),
                             (0.54351624, -0.09672665, 4.396788397),
                             (0.54351624, -0.09672665, 4.396788397),
                             (0.54351624, -0.09672665, 4.396788397),
                             (0.54351624, -0.09672665, 4.396788397),
                             (0.54351624, -0.09672665, 4.396788397),
                             (0.54351624, -0.09672665, 4.396788397),
                             (0.54351624, -0.09672665, 4.396788397)]
    distparams_onset2hosp = [(1.39916754, 0.05548228, 1.444545973),     # illness onset -> hospitalisation
                             (1.39916754, 0.05548228, 1.444545973),
                             (1.39916754, 0.05548228, 1.444545973),
                             (1.39916754, 0.05548228, 1.444545973),
                             (1.39916754, 0.05548228, 1.444545973),
                             (1.39916754, 0.05548228, 1.444545973),
                             (1.39916754, 0.05548228, 1.444545973),
                             (1.39916754, 0.05548228, 1.444545973),
                             (1.39916754, 0.05548228, 1.444545973)]
    distparams_home2recov = [(0.60720317, 0.00000000, 6.000000000),     # home -> recovery
                             (0.60720317, 0.00000000, 6.000000000),
                             (0.60720317, 0.00000000, 6.000000000),
                             (0.60720317, 0.00000000, 6.000000000),
                             (0.60720317, 0.00000000, 6.000000000),
                             (0.60720317, 0.00000000, 6.000000000),
                             (0.60720317, 0.00000000, 6.000000000),
                             (0.60720317, 0.00000000, 6.000000000),
                             (0.60720317, 0.00000000, 6.000000000)]
    distparams_hosp2recov_s = [(0.60720317, 0.00000000, temp_param_s),  # hospital admission (severe) -> leave
                               (0.60720317, 0.00000000, temp_param_s),
                               (0.60720317, 0.00000000, temp_param_s),
                               (0.60720317, 0.00000000, temp_param_s),
                               (0.60720317, 0.00000000, temp_param_s),
                               (0.60720317, 0.00000000, temp_param_s),
                               (0.60720317, 0.00000000, temp_param_s),
                               (0.60720317, 0.00000000, temp_param_s),
                               (0.60720317, 0.00000000, temp_param_s)]
    distparams_hosp2recov_c = [(0.60720317, 0.00000000, temp_param_c),  # hospital admission (critical) -> leave
                               (0.60720317, 0.00000000, temp_param_c),
                               (0.60720317, 0.00000000, temp_param_c),
                               (0.60720317, 0.00000000, temp_param_c),
                               (0.60720317, 0.00000000, temp_param_c),
                               (0.60720317, 0.00000000, temp_param_c),
                               (0.60720317, 0.00000000, temp_param_c),
                               (0.60720317, 0.00000000, temp_param_c),
                               (0.60720317, 0.00000000, temp_param_c)]
    distparams_hosp2decea = [(0.71678881, 0.21484379, 6.485309000),     # hospital admission -> decease
                             (0.71678881, 0.21484379, 6.485309000),
                             (0.71678881, 0.21484379, 6.485309000),
                             (0.71678881, 0.21484379, 6.485309000),
                             (0.71678881, 0.21484379, 6.485309000),
                             (0.71678881, 0.21484379, 6.485309000),
                             (0.71678881, 0.21484379, 6.485309000),
                             (0.71678881, 0.21484379, 6.485309000),
                             (0.71678881, 0.21484379, 6.485309000)]

    temp_param_mean_incubation_period = [stats.lognorm.mean(*params) for params in distparams_incubation]
    temp_param_infectious_period = [m - s + e for m, s, e in izip(temp_param_mean_incubation_period, param_infectious_start, param_infectious_end)]

    temp_param_sar_family = np.random.normal(0.35, 0.0425)
    temp_param_sar_others = stats.lognorm.rvs(0.3460311057053344, 0.04198595886371728, 0.11765118249339841)

    sar_family = np.array([temp_param_sar_family,
                           temp_param_sar_family,
                           temp_param_sar_family,
                           temp_param_sar_family,
                           temp_param_sar_family,
                           temp_param_sar_family,
                           temp_param_sar_family,
                           temp_param_sar_family,
                           temp_param_sar_family])
    sar_others = np.array([temp_param_sar_others,
                           temp_param_sar_others,
                           temp_param_sar_others,
                           temp_param_sar_others,
                           temp_param_sar_others,
                           temp_param_sar_others,
                           temp_param_sar_others,
                           temp_param_sar_others,
                           temp_param_sar_others])

    sar_family_daily = np.array([1. - np.exp(np.log(1 - sar) / period) for sar, period in izip(sar_family, temp_param_infectious_period)])
    sar_others_daily = np.array([1. - np.exp(np.log(1 - sar) / period) for sar, period in izip(sar_others, temp_param_infectious_period)])

    return doubling_time,\
           param_infectious_start,param_infectious_end,\
           ratio_asympt,\
           ratio_hosp,ratio_severe,ratio_critic,ratio_deceas,\
           ratio_critic_to_deceas,\
           ratio_severe_to_deceas_wc,ratio_critic_to_deceas_wc,\
           distparams_incubation,distparams_onset2hosp,distparams_home2recov,\
           distparams_hosp2recov_s,distparams_hosp2recov_c,distparams_hosp2decea,\
           sar_family,sar_others,\
           sar_family_daily,sar_others_daily

