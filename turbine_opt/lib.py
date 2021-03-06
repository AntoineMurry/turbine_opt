# -*- coding: UTF-8 -*-
# Copyright (C) 2018 Antoine Murry
""" Main lib for turbine-opt Project
"""
import math
import random

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

pd.set_option('display.width', 200)

RADIUS = 10


def random_neighbour_choice(disk, energy):
    """
    swap a blade from group 1 with a blade from group 2
    if energy > 40: use boosted blade swap
    if energy < 40: use naive blade swap
    """
    if energy > 40:
        return boosted_blade_swap(disk.copy())
    else:
        return naive_blade_swap(disk.copy())


def cost_function_unbalance(disk):
    """
    Cost of disk
    :input:
        disk :: pd.DataFrame read from read_data
    """
    fx, fy = cpt_blades_imbalance(disk)
    unbalance_blades = math.sqrt(fx.sum() * fx.sum() + fy.sum() * fy.sum())
    return unbalance_blades


def temperature(k, kmax):
    """
    decreasing temperature function
    :input:
        k :: current iteration
        kmax :: max number of iterations
    """
    return max(0.01, min(1, 1 - float(k)/float(kmax)))


def acceptance_probability(energy, new_energy, temperature):
    """
    acceptance probability function, which decreases with
    temperature
    :input:
        energy :: energy value from current state
        new_energy :: energy value from new state
        temperature :: value of temperature from temperature function
    """
    if new_energy < energy:
        return 1
    else:
        p = np.exp(- (new_energy - energy) / temperature)
        return p


def read_data(data_file):
    """
    reads input vector, computes all angles
    :input:
        data_file :: path to file
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
        disk :: pd.DataFrame read from read_data
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
    :input:
        disk :: pd.DataFrame read from read_data
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
    :input:
        disk :: pd.DataFrame read from read_data
        alpha :: distortion of Sigmoid
    """
    disk.loc[:, 'proba'] = disk['index'].apply(proba_sampling,
                                               args=(alpha, disk.shape[0]))
    picked = disk.proba.apply(picking_blades)
    while picked.sum() != disk.shape[0]/2:
        picked = disk.proba.apply(picking_blades)
    disk['picked'] = picked
    group1 = disk[disk.picked == 0].reset_index(drop=True)
    group2 = disk[disk.picked == 1].reset_index(drop=True)
    group1['index'] = group1.index
    group2['index'] = group2.index
    return group1, group2


def plot_blades(disk):
    """
    function to display blade weights on the turbine disk
    :input:
        disk :: pd.DataFrame read from read_data
    """
    wmin = disk.w.min()
    xplot = disk.bl_angle.map(math.cos) * (disk.w - wmin) * RADIUS
    yplot = disk.bl_angle.map(math.sin) * (disk.w - wmin) * RADIUS
    for x in range(len(xplot)):
        plt.plot([0, xplot[x]], [0, yplot[x]], 'ro-', label='python')
    limit = np.max(np.ceil(np.absolute(xplot).astype(int)))
    plt.xlim((-limit, limit))
    plt.ylim((-limit, limit))
    plt.show()


def picking_blades(p):
    return np.random.binomial(1, p)


def build_lobes(group):
    """
    creates 2 lobes from an input group of blades
    :input:
        group :: pd.DataFrame corresponding to group of blades
    Note: i) this only works if the number of rows of group is a
    multiple of 2
          ii) input blades must be sorted by weight (increasingly)
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


def split_in_groups(disk):
    """
    splits disk into 2 groups of blades:
        - group1 = group of heavier blades
        - group2 = group of lighter blades
    :input:
        disk :: pd.DataFrame read from read_data
    """
    group1 = disk.iloc[:int(disk.shape[0]/2), :].reset_index(drop=True)
    group2 = disk.iloc[int(disk.shape[0]/2):, :].reset_index(drop=True)
    return group1, group2


def search_rand_blades_1(group1, group2):
    """
    randomly searches for 2 blades to swap between the
    2 input groups
    :input:
        group1 :: pd.DataFrame corresponding to heavier group of blades
    :input:
        group2 :: pd.DataFrame corresponding to lighter group of blades
    """
    rand1 = random.choice(group1.index)
    group2.loc[:, "options"] = group1.loc[rand1].w > group2.w
    rand2 = random.choice((group2.options).index)
    return rand1, rand2


def search_rand_blades_2(group1, group2, search_param=10):
    """
    randomly searches for 2 blades to swap between the
    2 input groups.
    limits the difference of weight between the swapped blades by using
    search_param
    :input:
        group1 :: pd.DataFrame corresponding to heavier group of blades
        group2 :: pd.DataFrame corresponding to lighter group of blades
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
    choice = group2[min_index: max_index + 1]

    rand2 = random.randrange(choice.index.min(),
                             choice.index.max())
    return rand1, rand2


def random_neighbour_disk(disk):
    """
    swap a blade from group 1 with a blade from group 2
    :input:
        disk :: pd.DataFrame read from read_data
    """
    group1, group2 = split_in_groups(disk)
    group1 = group1.sort_values(['w']).reset_index(drop=True)
    group2 = group2.sort_values(['w']).reset_index(drop=True)
    rand1, rand2 = search_rand_blades_1(group1.copy(), group2.copy())
    temp2 = group2.iloc[rand2, :].copy(deep=True)
    temp1 = group1.iloc[rand1, :].copy(deep=True)
    group2.iloc[rand2, :] = temp1
    group1.iloc[rand1, :] = temp2
    group1 = build_lobes(group1.copy())
    group2 = build_lobes(group2.copy())
    return concat_groups(group1, group2)


def naive_blade_swap(disk):
    """
    swap a blade from group 1 with a blade from group 2
    :input:
        disk :: pd.DataFrame read from read_data
    """
    group1, group2 = split_in_groups(disk)
    # rand1 = random.randrange(0, group1.shape[0])
    rand1 = random.choice(group1.index)
    # rand2 = random.randrange(0, group2.shape[0])
    rand2 = random.choice(group2.index)
    temp2 = group2.iloc[rand2, :].copy(deep=True)
    temp1 = group1.iloc[rand1, :].copy(deep=True)
    group2.iloc[rand2, :] = temp1
    group1.iloc[rand1, :] = temp2
    group1 = build_lobes(group1.sort_values(['w']))
    group2 = build_lobes(group2.sort_values(['w']))
    return concat_groups(group1, group2)


def boosted_blade_swap(disk):
    """
    swap a blade from group 1 with a blade from group 2
    :input:
        disk :: pd.DataFrame read from read_data
    """
    group1, group2 = split_in_groups(disk)
    group1 = group1.sort_values(['w']).reset_index(drop=True)
    group2 = group2.sort_values(['w']).reset_index(drop=True)
    group2.loc[:, 'options'] = group2.w < group1.w
    option_blades = group2[group2.options]
    if len(option_blades) > 0:
        rand = random.choice(option_blades.index)
    else:
        return disk
    temp2 = group2.iloc[rand, :].copy(deep=True)
    temp1 = group1.iloc[rand, :].copy(deep=True)
    group2.iloc[rand, :] = temp1
    group1.iloc[rand, :] = temp2
    group1 = build_lobes(group1.sort_values(['w']))
    group2 = build_lobes(group2.sort_values(['w']))
    return concat_groups(group1, group2)


def concat_groups(group1, group2):
    """
    concatenate groups of blades to get assembled disk
    :input:
        group1 :: pd.DataFrame corresponding to heavier group of blades
        group2 :: pd.DataFrame corresponding to lighter group of blades
    """
    disk = pd.concat([group1, group2]).reset_index(drop=True)
    return disk


if __name__ == '__main__':
    pass
