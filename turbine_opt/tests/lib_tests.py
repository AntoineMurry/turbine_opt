# -*- coding: UTF-8 -*-

# Import from standard library
import unittest

# Import from classical libraries
import pandas as pd

from turbine_opt.tests.data import DATA_SOURCE_TEST
from turbine_opt.lib import group_creation


class TestUtils(unittest.TestCase):

    # @unittest.skip('')
    def test_dispose_blades(self):
        random.seed(0)
        disk = pd.read_csv(os.join(DATA_SOURCE_TEST, "input-data.csv"))
        group1, group2 = (disk, alpha=0.05)
        list(disk.w > disk.w.shift(1))
        self.assertEqual(df.shape, (29134, 3))


if __name__ == '__main__':
    unittest.main()
