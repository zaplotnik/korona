import numpy as np
from scipy import stats
from itertools import izip

def read_parameters():
    # temporary parameters
    temp_param_hosp = stats.lognorm.rvs(0.57615709, 2.17310146, 4.19689622) / 100.
    temp_param_deceas = stats.lognorm.rvs(0.44779353, 0.15813357) / 100.
    temp_param_critic = (1-0.53)/0.53 * temp_param_deceas               # deceased/(deceased+critical)=0.53
    temp_param_severe = temp_param_hosp - temp_param_deceas - temp_param_critic
    temp_param_s = stats.lognorm.rvs(0.42142559, 9.00007222, 1.99992887)
    temp_param_c = stats.lognorm.rvs(0.42140269, 8.99999444, 4.00000571)
    temp_param_asympt = np.random.normal(0.40, 0.10)

    doubling_time = np.random.normal(3.5, 0.5)                          # for whole population
    # R0 = stats.lognorm.rvs(0.35535982, 1.14371067, 1.53628849)        # this parameter is never used (why do we need it?)

    # Distribution parameters by age groups
    param_infectious_start = np.array([np.random.normal(3., 0.5),       # after infection
                                       np.random.normal(3., 0.5),
                                       np.random.normal(3., 0.5),
                                       np.random.normal(3., 0.5),
                                       np.random.normal(3., 0.5),
                                       np.random.normal(3., 0.5),
                                       np.random.normal(3., 0.5),
                                       np.random.normal(3., 0.5),
                                       np.random.normal(3., 0.5)])
    param_infectious_end = np.array([np.random.normal(2.5, 0.5),        # after illness onset
                                     np.random.normal(2.5, 0.5),
                                     np.random.normal(2.5, 0.5),
                                     np.random.normal(2.5, 0.5),
                                     np.random.normal(2.5, 0.5),
                                     np.random.normal(2.5, 0.5),
                                     np.random.normal(2.5, 0.5),
                                     np.random.normal(2.5, 0.5),
                                     np.random.normal(2.5, 0.5)])

    ratio_asympt = [temp_param_asympt,                                  # ratio_asympt = asymptomatic / all_infected(i.e. known+unknown)
                    temp_param_asympt,
                    temp_param_asympt,
                    temp_param_asympt,
                    temp_param_asympt,
                    temp_param_asympt,
                    temp_param_asympt,
                    temp_param_asympt,
                    temp_param_asympt]
    ratio_hosp = [temp_param_hosp,                                      # ratio_hosp = hospitalised / all_infected(i.e. known+unknown)
                  temp_param_hosp,
                  temp_param_hosp,
                  temp_param_hosp,
                  temp_param_hosp,
                  temp_param_hosp,
                  temp_param_hosp,
                  temp_param_hosp,
                  temp_param_hosp]
    ratio_severe = [temp_param_severe,                                  # ratio_severe = severe / all_infected(i.e. known+unknown)
                    temp_param_severe,
                    temp_param_severe,
                    temp_param_severe,
                    temp_param_severe,
                    temp_param_severe,
                    temp_param_severe,
                    temp_param_severe,
                    temp_param_severe]
    ratio_critic = [temp_param_critic,                                  # ratio_critical = critical / all_infected(i.e. known+unknown)
                    temp_param_critic,
                    temp_param_critic,
                    temp_param_critic,
                    temp_param_critic,
                    temp_param_critic,
                    temp_param_critic,
                    temp_param_critic,
                    temp_param_critic]
    ratio_deceas = [temp_param_deceas,                                  # ratio_critical = deceased / all_infected(i.e. known+unknown)
                    temp_param_deceas,
                    temp_param_deceas,
                    temp_param_deceas,
                    temp_param_deceas,
                    temp_param_deceas,
                    temp_param_deceas,
                    temp_param_deceas,
                    temp_param_deceas]
    ratio_critical_to_deceased = [d / (c + d) for c, d in izip(ratio_critic, ratio_deceas)]

    ratio_severe_to_deceased_wc = [np.random.normal(0.1, 0.05),         # wc=without (appropriate medical) care
                                   np.random.normal(0.1, 0.05),
                                   np.random.normal(0.1, 0.05),
                                   np.random.normal(0.1, 0.05),
                                   np.random.normal(0.1, 0.05),
                                   np.random.normal(0.1, 0.05),
                                   np.random.normal(0.1, 0.05),
                                   np.random.normal(0.1, 0.05),
                                   np.random.normal(0.1, 0.05)]
    ratio_critic_to_deceased_wc = [np.random.normal(0.9, 0.05),
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
           ratio_critical_to_deceased,\
           ratio_severe_to_deceased_wc,ratio_critic_to_deceased_wc,\
           distparams_incubation,distparams_onset2hosp,distparams_home2recov,\
           distparams_hosp2recov_s,distparams_hosp2recov_c,distparams_hosp2decea,\
           sar_family,sar_others,\
           sar_family_daily,sar_others_daily

