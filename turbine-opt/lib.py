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


if __name__ == '__main__':
    # For introspection purposes to quickly get this functions on ipython
    data_file = os.path.join(DATA_SOURCE, "input-data.csv")
    disk = read_data(data_file)
    X = [disk.bl_angle.map(math.cos) * disk.w]
    Y = [disk.bl_angle.map(math.sin) * disk.w]
    plt.scatter(X,Y, color='red')
    plt.show()
