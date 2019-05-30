# -*- coding: UTF-8 -*-
# Copyright (C) 2018 Antoine Murry
""" Main lib for papillon Project
"""
import os
import math

import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np

import numpy.random as rn

from papillon.data import DATA_SOURCE

pd.set_option('display.width', 200)

RADIUS = 100


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
    fx = disk.w * math.cos(disk.bl_angle) * RADIUS
    fy = disk.w * math.sin(disk.bl_angle) * RADIUS
    return fx, fy


class Simulated_annealing:
    """
    encapsulated simulated annealing class
    """
    def __init__(self, init_state, chg_state_func, ener_func, temp_func,
                 max_steps, accept_proba):
        self.init_state = init_state
        self.chg_state_func = chg_state_func
        self.ener_func = ener_func
        self.temp_func = temp_func
        self.max_steps = max_steps
        self.accept_proba = accept_proba

    def optimize(self):
        state = self.init_state
        for k in range(self.max_steps):
            temp = self.temp_func(k, self.max_steps)
            state_new = self.chg_state_func(state)
            proba = accept_proba(self.ener_func(state),
                                 self.ener_func(state_new),
                                 temp)
            if proba > np.random.sample():
                state = state_new
        return state


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



if __name__ == '__main__':

    init_state = random_start()
    chg_state_func = random_neighbour
    ener_func = cost_function
    temp_func = temperature
    max_steps = 100000
    accept_proba = acceptance_probability


    #TODO: finish testing this fist funtion. inspiration = https://perso.crans.org/besson/notebooks/Simulated_annealing_in_Python.html

    Simu = Simulated_annealing(init_state, chg_state_func, ener_func, temp_func,
            max_steps, accept_proba)

    # plotting unbalance:
    # data_file = os.path.join(DATA_SOURCE, "input-data.csv")
    # disk = read_data(data_file)
    # X = [disk.bl_angle.map(math.cos) * disk.w]
    # Y = [disk.bl_angle.map(math.sin) * disk.w]
    # plt.scatter(X,Y, color='red')
    # plt.show()
