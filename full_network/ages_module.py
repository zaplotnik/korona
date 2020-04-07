"""
@author: Luka M
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

max_age = 100
age_groups1 = [(0,4),(5,14),(15,24),(25,34),(35,44),(45,54),(55,64),(65,74),(75,84),(85,max_age)]
age_groups2 = [(0,14),(15,34),(35,54),(55,64),(65,max_age)]
age_groups3 = [(0,15),(16,29),(30,49),(50,59),(60,max_age)]

def parse_age_groups(age_groups, max_age=max_age):
    '''
    Input:  age groups; format : ["??-??", "??-??", ..., "??+"]
    Output: age groups; format : [(??,??),(??,??), ..., (??,max_age)]
    '''
    result = []
    for string in age_groups:
        if '-' in string:
            result.append(tuple(map(int, string.split('-'))))
        if '+' in string:
            result.append((int(string[:string.index('+')]), max_age))
    return result

def parse_age_groups_i(age_groups):
    '''
    Input:  age groups; format : [(??,??),(??,??), ..., (??,max_age)]
    Output: age groups; format : ["??-??", "??-??", ..., "??+"]
    '''
    result = []
    for a,b in age_groups[:-1]:
        result += [str(a)+'-'+str(b)]
    result += [str(age_groups[-1][0])+'+']
    return result

def generate_age_group_distribution(name, age_groups, probabilities):
    '''
    Generates discrete distribution from age demography data.
    '''

    xk = np.arange(age_groups[-1][-1] + 1)  # max age
    pk = []
    for group, prob in zip(age_groups_demography, probabilities):
        a, b = group
        n = b - a + 1
        pk += n * [prob / n]
    pk = np.array(pk, dtype=np.float32)/np.sum(pk)

    return stats.rv_discrete(name=name, values=(xk, pk))

# Load age demography and Create age demography distributions for female and male
age_demograpy = pd.read_csv(r'data/age_demography.csv')
age_groups_demography = parse_age_groups(age_demograpy['age.group'], max_age)
male_age_demograpy_distribution = generate_age_group_distribution('male_age_demograpy_distribution',age_groups_demography, age_demograpy['male'])
female_age_demograpy_distribution = generate_age_group_distribution('female_age_demograpy_distribution', age_groups_demography, age_demograpy['female'])

# Load daily acquired date for patients' age groups and Calculate differences between days
cols = ["age.female."+str(age_group[0])+"-"+str(age_group[1]) for age_group in age_groups1[:-1]]+["age.female."+str(age_groups1[-1][0])+"+"]+\
       ["age.male."+str(age_group[0])+"-"+str(age_group[1]) for age_group in age_groups1[:-1]]+["age.male."+str(age_groups1[-1][0])+"+"]

data_stats = pd.read_csv(r"https://raw.githubusercontent.com/slo-covid-19/data/master/csv/stats.csv",
                          index_col="date",usecols=["date"]+[col+".todate" for col in cols],parse_dates=["date"])
data_stats.iloc[-1,:].fillna(data_stats.iloc[-2,:], inplace=True)
data_stats.fillna(0., inplace=True)

for col in cols:
    data_stats[col+".new"] = np.zeros_like(data_stats[col+".todate"])
    data_stats[col+".new"][0] = data_stats[col + ".todate"][0]  # First acquired day
    data_stats[col+".new"][1:] = np.diff(data_stats[col+".todate"])

day_from_first_case = -8    # first confirmed case on day 1
num_days = len(data_stats[cols[0]+".todate"])

def random_ages(distribution, a=0, b=max_age, size=1):
    '''
    Returns list of random ages between a and b ( a <= age < b ) according to chosen distribution.
    '''
    result = []
    while len(result) != size:
        age = distribution.rvs()
        if a <= age < b:
            result.append(age)
    return result

def generate_sample():
    '''
    Generates random sample of ages for new daily patients.
    '''

    sample = {'age.female.new' : [], 'age.male.new' : []}

    for day in range(num_days):
        female_new_day = []
        male_new_day = []
        for age_group, col in zip(2*age_groups1, cols):
            if 'female' in col:
                female_new_day += random_ages(male_age_demograpy_distribution, age_group[0], age_group[1]+1, int(data_stats[col+".new"][day]))
            else:   # male
                male_new_day += random_ages(female_age_demograpy_distribution, age_group[0], age_group[1]+1, int(data_stats[col+".new"][day]))
        sample['age.female.new'].append(female_new_day)
        sample['age.male.new'].append(male_new_day)

    sample['date'] = data_stats.index
    sample['day'] = day_from_first_case + np.arange(0,len(sample['date']))
    sample = pd.DataFrame(data=sample)
    sample = sample[['day', 'date', 'age.female.new', 'age.male.new']]

    return sample

def assign_agegroup(age, groups):
    for i, group in enumerate(groups):
        if group[0] <= age < group[1]:
            return i

def groupby_age_groups(sample, age_groups):
    '''
    Groups new daily patients by age groups, creates new columns "age.female.new.groups" and "age.male.new.groups" in
    sample (pandas.DataFrame) and for both returns numpy.array.
    '''
    for sex in ['female', 'male']:
        sample['age.' + sex + '.new.groups'] = [[] for i in range(len(sample['age.' + sex + '.new']))]
        for i, row in enumerate(sample['age.' + sex + '.new']):
            new_day_groups = []
            for age_group in age_groups:
                new_day_groups += [np.sum((np.array(row) >= age_group[0]) & (np.array(row) <= age_group[1]))]
            sample['age.' + sex + '.new.groups'][i] = new_day_groups

    return np.array([sample['age.female.new.groups'].tolist(), np.array(sample['age.male.new.groups'].tolist())])

def plot_age_groups(y, age_group_labels=False, date_labels=False, linewidth=1., figsize=(8,6),
                    title='Casovni potek novo okuzenih po starostnih skupinah', filename=False, dpi=100):
    flag = True
    if age_group_labels is False:
        age_group_labels = range(y.shape[1])
        flag = False
    if date_labels is False:
        date_labels = range(y.shape[0])
        rot = 'horizontal'
    else:
        rot = 'vertical'

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    x = np.zeros_like(y, dtype=np.float)
    x[:, np.arange(y.shape[1])] = np.transpose([range(y.shape[0])])
    x[np.arange(y.shape[0])] += np.linspace(0, 0.5, y.shape[1])

    fig = plt.figure(figsize=figsize)
    for x_col, y_col, color, label in zip(x.transpose(), y.transpose(), colors[:y.shape[1]], age_group_labels):
        plt.vlines(x_col, np.zeros_like(y_col), y_col, color=color, label=label, lw=linewidth)

    xtick_labels = [str(date)[:10] for date in date_labels]
    plt.xticks(np.arange(0, y.shape[0]) + 0.25, xtick_labels, rotation=rot, size='small')
    plt.yticks(size='small')
    plt.ylim(ymin=0)
    plt.grid(axis='y', linestyle='--')
    if flag:
        plt.legend(loc='upper left', fontsize='small')

    plt.title(title, size=14)
    plt.ylabel('st. novo okuzenih v starostni skupini', size=10)

    if filename:
        fig.savefig(filename, dpi=dpi)

    return 0

#%%
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', None)
#
# print(generate_sample())

#%% Plot
# import matplotlib.pyplot as plt
#
# plt.figure(1)
# x = np.array([0]+[el[1] for el in age_groups_demography])
# ym = np.array(age_demograpy['male']) / np.sum(age_demograpy['male'])
# yf = np.array(age_demograpy['female']) / np.sum(age_demograpy['female'])
# ym = np.append([ym[0]], ym)
# yf = np.append([yf[0]], yf)
#
# hm = male_age_demograpy_distribution.rvs(size=10000)
# hf = female_age_demograpy_distribution.rvs(size=10000)
# plt.hist(-hm, bins=np.arange(-101,2), density=True, color='b', alpha=0.5, orientation='vertical')
# h2 = plt.hist(hf, bins=np.arange(0,101), density=True, color='r', alpha=0.5, orientation='vertical')
# rescale = yf[0] / h2[0][np.nonzero(h2[0])[0][0]]
# plt.step(-x, ym/rescale, 'b', x, yf/rescale, 'r')

#%%
# import matplotlib.pyplot as plt
#
# plt.figure(2)
# x = np.array([0]+[el[1] for el in age_groups_demography])
# ym = np.array(age_demograpy['male']) / np.sum(age_demograpy['male'])
# yf = np.array(age_demograpy['female']) / np.sum(age_demograpy['female'])
# ym = np.append([ym[0]], ym)
# yf = np.append([yf[0]], yf)
#
#
# hm = np.array(random_ages(male_age_demograpy_distribution, 10, 50, 10000))
# hf = np.array(random_ages(female_age_demograpy_distribution, 10, 50, 10000))
# plt.hist(-hm, bins=np.arange(-101,2), density=True, color='b', alpha=0.5, orientation='vertical')
# h2 = plt.hist(hf, bins=np.arange(0,101), density=True, color='r', alpha=0.5, orientation='vertical')
# rescale = yf[0] / h2[0][np.nonzero(h2[0])[0][0]]
# plt.step(-x, ym/rescale, 'b', x, yf/rescale, 'r')
#
# plt.show()