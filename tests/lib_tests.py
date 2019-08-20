# -*- coding: UTF-8 -*-

# Import from standard library
import os
import random
import unittest

# Import from classical libraries
import pandas as pd

from turbine_opt.data import DATA_SOURCE
from turbine_opt.lib import group_creation
from turbine_opt.lib import dispose_blades


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


if __name__ == '__main__':
    unittest.main()
