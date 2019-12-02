# -*- coding: UTF-8 -*-

# Import from standard library
import os
import math
import random
import unittest

# Import from classical libraries
import pandas as pd

from turbine_opt.data import DATA_SOURCE
from turbine_opt.lib import group_creation
from turbine_opt.lib import dispose_blades
from turbine_opt.lib import acceptance_probability
from turbine_opt.lib import random_neighbour_choice
from turbine_opt.lib import cost_function_unbalance



class TestUtils(unittest.TestCase):

    # @unittest.skip('')
    def test_dispose_blades(self):
        random.seed(0)
        disk = pd.read_csv(os.path.join(DATA_SOURCE, "test_data" ,
                                        "input-data.csv"))
        group1, group2 = dispose_blades(disk, alpha=0.05)
        res = list(group1.w > group1.w.shift(1))
        exp = [False,  True, True, True, False, False, False, False, True,
               True, True, False, False, False]
        self.assertEqual(res, exp)

    def test_accept_proba(self):
        random.seed(0)
        energy = 1000
        new_energy_1 = 1500
        new_energy_2 = 500
        proba_1 = acceptance_probability(energy, new_energy_1, 5000)
        proba_1 = round(proba_1, 3)
        proba_2 = acceptance_probability(energy, new_energy_2, 5000)
        self.assertEqual(proba_1, 0.905)
        self.assertEqual(proba_2, 1)

    def test_cost_function_unbalance(self):
        disk = pd.read_csv(os.path.join(DATA_SOURCE, "test_data" ,
                                        "input-data.csv"))
        disk.loc[:, 'bl_angle'] = disk.blade * 360/disk.shape[0]
        disk.loc[:, 'bl_angle'] = disk.bl_angle.map(math.radians)
        unbalance_blades = round(cost_function_unbalance(disk), 3)
        self.assertEqual(unbalance_blades, 56.379)

    def test_random_neighbour_choice(self):
        disk = pd.read_csv(os.path.join(DATA_SOURCE, "test_data" ,
                                        "input-data.csv"))
        energy = 50
        swap = random_neighbour_choice(disk, energy)



if __name__ == '__main__':
    unittest.main()
