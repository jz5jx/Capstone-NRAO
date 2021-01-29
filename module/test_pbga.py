import unittest
import numpy as np
from scipy import stats
from pbga import PBGA


class TestPBGA(unittest.TestCase):

    @staticmethod
    def generate_data(x_bar, y_bar, x_var, y_var, cov, sigma):
        x = np.arange(0, 1000, 1)
        y = np.arange(0, 1000, 1)
        x, y = np.meshgrid(x, y)
        pos = np.dstack((x, y))

        cov_mat = [[x_var, cov], [cov, y_var]]
        rv = stats.multivariate_normal([x_bar, y_bar], cov_mat)
        z = rv.pdf(pos)

        std = np.std(z)
        z[z < std * sigma] = 0

        return z

    def setUp(self):
        self.pbga = PBGA(buffer_size=10, group_size=40)

    # Test one central group with no covariance
    def test_one_group_no_covariance(self):
        z = self.generate_data(500, 500, 50, 50, 0, 5)

        self.pbga.run(image=z)

        self.assertEqual(len(self.pbga.group_ranges), 1)
        self.assertEqual(len(self.pbga.group_stats), 1)

        group_stats = self.pbga.group_stats[0]
        self.assertEqual(group_stats['X_BAR'], 500)
        self.assertEqual(group_stats['Y_BAR'], 500)

    # Test two row adjacent groups with high covariance
    def test_two_adjacent_groups_high_covariance(self):
        z = self.generate_data(333, 500, 150, 150, 125, 5)
        z += self.generate_data(667, 500, 150, 150, -125, 5)

        self.pbga.run(image=z)

        self.assertEqual(len(self.pbga.group_ranges), 2)
        self.assertEqual(len(self.pbga.group_stats), 2)

        group1_stats = self.pbga.group_stats[0]
        group2_stats = self.pbga.group_stats[1]

        self.assertAlmostEqual(group1_stats['X_BAR'], 333)
        self.assertAlmostEqual(group1_stats['Y_BAR'], 500)

        self.assertAlmostEqual(group2_stats['X_BAR'], 667)
        self.assertAlmostEqual(group2_stats['Y_BAR'], 500)

    # Test four row, col adjacent groups with no covariance
    def test_four_adjacent_groups_no_covariance(self):
        z = self.generate_data(333, 333, 50, 50, 0, 5)
        z += self.generate_data(667, 333, 50, 50, 0, 5)
        z += self.generate_data(333, 667, 50, 50, 0, 5)
        z += self.generate_data(667, 667, 50, 50, 0, 5)

        self.pbga.run(image=z)

        self.assertEqual(len(self.pbga.group_ranges), 4)
        self.assertEqual(len(self.pbga.group_stats), 4)
        self.assertEqual(len(self.pbga.group_data), 4)

        group1_stats = self.pbga.group_stats[0]
        group2_stats = self.pbga.group_stats[1]
        group3_stats = self.pbga.group_stats[2]
        group4_stats = self.pbga.group_stats[3]

        self.assertEqual(group1_stats['X_BAR'], 333)
        self.assertEqual(group1_stats['Y_BAR'], 333)

        self.assertEqual(group2_stats['X_BAR'], 667)
        self.assertEqual(group2_stats['Y_BAR'], 333)

        self.assertEqual(group3_stats['X_BAR'], 333)
        self.assertEqual(group3_stats['Y_BAR'], 667)

        self.assertEqual(group4_stats['X_BAR'], 667)
        self.assertEqual(group4_stats['Y_BAR'], 667)

    # Test four row, col adjacent groups with high covariance
    def test_four_adjacent_groups_high_covariance(self):
        z = self.generate_data(333, 333, 150, 150, 125, 5)
        z += self.generate_data(667, 333, 150, 150, -125, 5)
        z += self.generate_data(333, 667, 150, 150, -125, 5)
        z += self.generate_data(667, 667, 150, 150, 125, 5)

        self.pbga.run(image=z)

        self.assertEqual(len(self.pbga.group_ranges), 4)
        self.assertEqual(len(self.pbga.group_stats), 4)
        self.assertEqual(len(self.pbga.group_data), 4)

        group1_stats = self.pbga.group_stats[0]
        group2_stats = self.pbga.group_stats[1]
        group3_stats = self.pbga.group_stats[2]
        group4_stats = self.pbga.group_stats[3]

        self.assertAlmostEqual(group1_stats['X_BAR'], 333)
        self.assertAlmostEqual(group1_stats['Y_BAR'], 333)

        self.assertAlmostEqual(group2_stats['X_BAR'], 667)
        self.assertAlmostEqual(group2_stats['Y_BAR'], 333)

        self.assertAlmostEqual(group3_stats['X_BAR'], 333)
        self.assertAlmostEqual(group3_stats['Y_BAR'], 667)

        self.assertAlmostEqual(group4_stats['X_BAR'], 667)
        self.assertAlmostEqual(group4_stats['Y_BAR'], 667)


if __name__ == "__main__":
    unittest.main()
