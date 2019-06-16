# -*- coding: UTF-8 -*-
# Copyright (C) 2018 Antoine Murry
""" Main lib for papillon Project
"""
import os
import math
import random

import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np

import numpy.random as rn

from papillon.data import DATA_SOURCE

pd.set_option('display.width', 200)

RADIUS = 10


def read_data(data_file):
    """
    reads input vector, computes all angles
    :input:
        data_file: path to file
    """
    disk = pd.read_csv(data_file)
    disk.loc[:, 'bl_angle'] = disk.blade * 360/disk.shape[0]
    disk.loc[:, 'bl_angle'] = disk.bl_angle.map(math.radians)
    return disk


def cpt_blades_imbalance(disk):
    """
    computes blades imbalance based
    on vector of blades mass and
    on vector of blades angles
    :input:
        :disk: pd.DataFrame read from read_data
    """
    fx = disk.w * disk.bl_angle.map(math.cos) * RADIUS
    fy = disk.w * disk.bl_angle.map(math.sin) * RADIUS
    return fx, fy


def proba_sampling(ordered_index, alpha=0.05, num_of_blades=140):
    """
    compute probability for bernoulli sampling.
    The probability function is a sigmoid
    """
    proba = 1 / (np.exp(alpha * (num_of_blades / 2 - ordered_index)) + 1)
    return proba


def dispose_blades(disk, alpha=0.05):
    """
    dispose blades on disk to minimize unbalance
    """
    disk = disk.sort_values(['w'], ascending=False)
    disk = disk.reset_index()
    disk['index'] = disk.index
    group1, group2 = group_creation(disk, alpha)
    group1 = build_lobes(group1)
    group2 = build_lobes(group2)
    return group1, group2


def group_creation(disk, alpha):
    """
    split blades into 2 groups using random sampling
    based on weight
    """
    disk.loc[:, 'proba'] = disk['index'].apply(proba_sampling,
                                               args=(alpha, disk.shape[0]))
    picked = disk.proba.apply(picking_blades)
    while picked.sum() != disk.shape[0]/2:
        picked = disk.proba.apply(picking_blades)
        print(picked.sum(), ": sum of picks")
    disk['picked'] = picked
    group1 = disk[disk.picked == 0].reset_index(drop=True)
    group2 = disk[disk.picked == 1].reset_index(drop=True)
    group1['index'] = group1.index
    group2['index'] = group2.index
    return group1, group2


class Simulated_annealing:
    """
    encapsulated simulated annealing class
    """
    def __init__(self, init_state, chg_state_func, ener_func, temp_func,
                 max_steps, accept_proba, search_param):
        self.init_state = init_state
        self.chg_state_func = chg_state_func
        self.ener_func = ener_func
        self.temp_func = temp_func
        self.max_steps = max_steps
        self.accept_proba = accept_proba
        self.search_param = search_param

    def optimize(self):
        state = self.init_state
        print('def final state')
        final_state = self.init_state
        for k in range(self.max_steps):
            # print(k, ":k")
            # print(self.max_steps, ": self.max_steps")
            temp = self.temp_func(k, self.max_steps)
            # state_new = self.chg_state_func(state, search_param)
            state_new = self.chg_state_func(state)
            print(temp, ": temperature")
            # print(max(0.01, min(1, 1 - float(k)/float(self.max_steps))), ":computed temp")
            state_ener = self.ener_func(state)
            state_nw_ener = self.ener_func(state_new)
            print(state_nw_ener, ": state_nw_ener")
            proba = accept_proba(state_ener,
                                 state_nw_ener,
                                 temp)
            print(self.ener_func(final_state), ": self.ener_func(final state)")
            if proba > np.random.sample():
                state = state_new
            if self.ener_func(state) < self.ener_func(final_state):
                print('<<<<<<<<<OOOOKKKK>>>>>>>>>')
                final_state = state
        return final_state


# temporary functions defined for initial test:

INTERVAL = (-100, 100)

def f(x):
    """ Function to minimize."""
    return x ** 2


def clip(x):
    """ Force x to be in the interval."""
    a, b = INTERVAL
    return max(min(x, b), a)


def random_start():
    """ Random point in the interval."""
    a, b = INTERVAL
    return a + (b - a) * rn.random_sample()


def cost_function(x):
    """ Cost of x = f(x)."""
    return f(x)


def random_neighbour(x, fraction=1):
    """Move a little bit x, from the left or the right."""
    amplitude = (max(INTERVAL) - min(INTERVAL)) * fraction / 10
    delta = (-amplitude/2.) + amplitude * rn.random_sample()
    return clip(x + delta)


def acceptance_probability(cost, new_cost, temperature):
    if new_cost < cost:
        # print("    - Acceptance probabilty = 1 as new_cost = {} < cost = {}...".format(new_cost, cost))
        return 1
    else:
        p = np.exp(- (new_cost - cost) / temperature)
        # print("    - Acceptance probabilty = {:.3g}...".format(p))
        return p


def temperature(k, kmax):
    """ Example of temperature dicreasing as the process goes on."""
    return max(0.01, min(1, 1 - float(k)/float(kmax)))


def plot_blades(disk):
    """
    """
    wmin = disk.w.min()
    xplot = disk.bl_angle.map(math.cos) * (disk.w - wmin) * RADIUS
    yplot = disk.bl_angle.map(math.sin) * (disk.w - wmin) * RADIUS
    for x in range(len(xplot)):
        plt.plot([0, xplot[x]],[0,yplot[x]],'ro-',label='python')
    limit=np.max(np.ceil(np.absolute(xplot).astype(int))) # set limits for axis
    plt.xlim((-limit,limit))
    plt.ylim((-limit,limit))
    plt.ylabel('Imaginary')
    plt.xlabel('Real')
    plt.show()


def picking_blades(p):
    return np.random.binomial(1, p)


def build_lobes(group):
    """
    creates 2 lobes from an input group of blades
    :input:
        group :: pd.DataFrame corresponding to group of blades
    Note: this only works if the number of rows of group is a
    multiple of 2
    """
    df = pd.DataFrame(columns=group.columns, index=group.index)
    ind_first_bl = int((group.shape[0]/2 + 1)/2) - 1
    ind_seco_bl = int((group.shape[0]/2 + 1)/2 + group.shape[0]/2) - 1
    df.iloc[ind_first_bl, :] = group.iloc[0, :]
    df.iloc[ind_seco_bl, :] = group.iloc[1, :]

    para = 0
    for blades in range(2, group.shape[0], 4):
        depo_blade_1 = int((group.shape[0]/2 + para + 3) / 2 +
                           group.shape[0]/2) - 1
        depo_blade_2 = int((group.shape[0]/2 - para - 1) / 2) - 1
        depo_blade_3 = int((group.shape[0]/2 + para + 3) / 2) - 1
        depo_blade_4 = int((group.shape[0]/2 - para - 1) / 2 +
                           group.shape[0]/2) - 1
        para = para + 2
        df.iloc[depo_blade_1, :] = group.iloc[blades, :]
        df.iloc[depo_blade_2, :] = group.iloc[blades + 1, :]
        df.iloc[depo_blade_3, :] = group.iloc[blades + 2, :]
        df.iloc[depo_blade_4, :] = group.iloc[blades + 3, :]
    return df


def cost_function_unbalance(disk):
    """ Cost of disk"""
    fx, fy = cpt_blades_imbalance(disk)
    unbalance_blades = math.sqrt(fx.sum() * fx.sum() + fy.sum() * fy.sum())
    return unbalance_blades


def split_in_groups(disk):
    """
    """
    group1 = disk.iloc[:int(disk.shape[0]/2), :].reset_index(drop=True)
    group2 = disk.iloc[int(disk.shape[0]/2):, :].reset_index(drop=True)
    return group1, group2


def search_rand_blades_1(group1, group2):
    """
    """
    rand1 = random.choice(group1.index)
    group2.loc[:, "options"] = group1.loc[rand1].w > group2.w
    rand2 = random.choice((group2.options == True).index)
    return rand1, rand2


def search_rand_blades_2(group1, group2, search_param=10):
    """
    """
    rand1 = random.randrange(0, group1.shape[0])
    if (rand1 + search_param > group1.index.max()):
        w_min = group1.loc[(rand1 - search_param)].w
        w_max = group1.loc[group1.index.max()].w
    elif (rand1 - search_param < group1.index.min()):
        w_min = group1.loc[group1.index.min()].w
        w_max = group1.loc[rand1 + search_param].w
    else:
        w_min = group1.loc[(rand1 - search_param)].w
        w_max = group1.loc[rand1 + search_param].w

    min_index = group2[(group2.w - w_min).apply(abs) ==
                       abs(group2.w - w_min).min()].index[0]
    max_index = group2[(group2.w - w_max).apply(abs) ==
                       abs(group2.w - w_max).min()].index[0]
    choice = group2[min_index : max_index + 1]

    rand2 = random.randrange(choice.index.min(),
                             choice.index.max())
    return rand1, rand2


def random_neighbour_disk_naive(disk):
    """swap a blade from group 1 with a blade from group 2"""
    group1, group2 = split_in_groups(disk)
    rand1 = random.randrange(0, group1.shape[0])
    rand2 = random.randrange(0, group2.shape[0])
    temp2 = group2.iloc[rand2, :].copy(deep=True)
    temp1 = group1.iloc[rand1, :].copy(deep=True)
    group2.iloc[rand2, :] = temp1
    group1.iloc[rand1, :] = temp2
    group1 = build_lobes(group1)
    group2 = build_lobes(group2)
    return concat_groups(group1, group2)


def random_neighbour_disk(disk): #, search_param):
    """swap a blade from group 1 with a blade from group 2"""
    group1, group2 = split_in_groups(disk)
    group1 = group1.sort_values(['w']).reset_index(drop=True)
    group2 = group2.sort_values(['w']).reset_index(drop=True)
    rand1, rand2 = search_rand_blades_1(group1, group2)
    temp2 = group2.iloc[rand2, :].copy(deep=True)
    temp1 = group1.iloc[rand1, :].copy(deep=True)
    group2.iloc[rand2, :] = temp1
    group1.iloc[rand1, :] = temp2
    # print(temp1, ": blade swap 1")
    # print(temp2, ": blade swap 2")
    # print((group1.loc[rand1].w > group2.w).value_counts())
    group1 = build_lobes(group1)
    group2 = build_lobes(group2)
    return concat_groups(group1, group2)


def concat_groups(group1, group2):
    """
    """
    disk = pd.concat([group1, group2]).reset_index(drop=True)
    return disk

if __name__ == '__main__':

    # plotting unbalance:
    data_file = os.path.join(DATA_SOURCE, "input-data.csv")
    disk = read_data(data_file)

    print("start preplacing blades")
    # alpha=0.05
    alpha = 0.01
    # group1, group2 = group_creation(disk, alpha)
    group1, group2 = dispose_blades(disk, alpha)
    disposed = concat_groups(group1, group2)
    print("done preplacing blades")

    disk.loc[:, ["w", "blade", "ht"]] = disposed[["w", "blade", "ht"]]
    plot_blades(disk)

    # # fx, fy = cpt_blades_imbalance(disk)
    # # unbalance_blades = math.sqrt(fx.sum() * fx.sum() + fy.sum() * fy.sum())

    # init_state = disk
    # chg_state_func = random_neighbour_disk
    # ener_func = cost_function_unbalance
    # temp_func = temperature
    # max_steps = 500
    # search_param = 50
    # accept_proba = acceptance_probability
    # Simu = Simulated_annealing(init_state, chg_state_func, ener_func,
    #                            temp_func, max_steps, accept_proba,
    #                            search_param)
    # group1, group2 = split_in_groups(disk)
    # res = Simu.optimize()
    # plot_blades(res)
