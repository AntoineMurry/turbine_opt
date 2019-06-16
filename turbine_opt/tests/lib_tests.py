# -*- coding: UTF-8 -*-

# Import from standard library
import os
import random
import unittest

# Import from classical libraries
import pandas as pd

from turbine_opt.tests.test_data import DATA_SOURCE_TEST
from turbine_opt.lib import group_creation
from turbine_opt.lib import dispose_blades


class TestUtils(unittest.TestCase):

    # @unittest.skip('')
    def test_dispose_blades(self):
        random.seed(0)
        disk = pd.read_csv(os.path.join(DATA_SOURCE_TEST, "input-data.csv"))
        group1, group2 = dispose_blades(disk, alpha=0.05)
        list(group1.w > group1.w.shift(1))
        self.assertEqual(df.shape, (29134, 3))


if __name__ == '__main__':
    unittest.main()
