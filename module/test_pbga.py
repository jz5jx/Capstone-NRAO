import unittest
import numpy as np
from scipy import stats
from pbga import PBGA


class TestPBGA(unittest.TestCase):

    @staticmethod
    def generate_data(x_bar, y_bar, x_var, y_var, cov, sigma, n_rows, n_cols):
        x = np.arange(0, n_rows, 1)
        y = np.arange(0, n_cols, 1)
        x, y = np.meshgrid(x, y)
        pos = np.dstack((x, y))

        cov_mat = [[x_var, cov], [cov, y_var]]
        rv = stats.multivariate_normal([x_bar, y_bar], cov_mat)
        z = rv.pdf(pos)

        std = np.std(z)
        z[z < std * sigma] = 0

        return z

    def generate_image(self, params, n_rows, n_cols):
        # generates an image containing groups following the parameters of
        # generate_data(); groups must be arranged from bottom to top with
        # precedence over left from right in order for the tests within the
        # evaluate methods to function properly
        image_data = np.zeros((n_rows, n_cols))
        for param in params:
            x_bar, y_bar, x_var, y_var, cov, sigma = param
            image_data += self.generate_data(x_bar, y_bar, x_var, y_var,
                                             cov, sigma, n_rows, n_cols)
        return image_data

    def evaluate_number_of_groups(self, pbga, group_count):
        # check number of groups detected by pbga matches group count
        self.assertEqual(len(pbga.group_ranges), group_count,
                         msg=f"Number of group ranges "
                             f"({len(pbga.group_ranges)}) does match specified"
                             f" number of groups ({group_count}).")
        self.assertEqual(len(pbga.group_stats), group_count,
                         msg=f"Number of group statistics "
                             f"({len(pbga.group_stats)}) does match specified"
                             f" number of groups ({group_count}).")
        self.assertEqual(len(pbga.group_data), group_count,
                         msg=f"Number of group data "
                             f"({len(pbga.group_data)}) does match specified"
                             f" number of groups ({group_count}).")

    def evaluate_group_location(self, pbga, params, margin):
        for i, (stats_, param) in enumerate(zip(pbga.group_stats, params)):
            x_bar, y_bar, x_var, y_var, cov, sigma = param
            x_min, x_max = x_bar * (1 - margin), x_bar * (1 + margin)
            y_min, y_max = y_bar * (1 - margin), y_bar * (1 + margin)
            # check if x_bar, y_bar fall within the original parameters plus
            # the specified margin
            self.assertTrue((x_min <= stats_['X_BAR'] <= x_max),
                            msg=f"Group {i}'s \"X_BAR\" ({stats_['X_BAR']})\n"
                                f"does not fall within [{x_min}, {x_max}].")
            self.assertTrue((y_min <= stats_['Y_BAR'] <= y_max),
                            msg=f"Group {i}'s \"Y_BAR\" ({stats_['Y_BAR']})\n"
                                f"does not fall within [{y_min}, {y_max}].")

    def evaluate_groups_from_params(self, params, n_rows, n_cols, buffer_size,
                                    group_size, margin):
        image_data = self.generate_image(params, n_rows=n_rows, n_cols=n_cols)

        pbga = PBGA(buffer_size=buffer_size, group_size=group_size)
        pbga.run(image=image_data)

        self.evaluate_number_of_groups(pbga, len(params))
        self.evaluate_group_location(pbga, params, margin=margin)

    # Test one central group with no covariance
    def test_one_group_no_covariance(self):
        # parameters for generate_data() excluding n_rows, n_cols
        # (i.e. x_bar, y_bar, x_var, y_var, cov, sigma)
        params = [[500, 500, 50, 50, 0, 5]]
        self.evaluate_groups_from_params(params, n_rows=1000, n_cols=1000,
                                         buffer_size=10, group_size=50,
                                         margin=0.025)

    # Test two row adjacent groups with high covariance
    def test_two_adjacent_groups_high_covariance(self):
        # parameters for generate_data() excluding n_rows, n_cols
        # (i.e. x_bar, y_bar, x_var, y_var, cov, sigma)
        params = [[333, 500, 150, 150, 125, 5],
                  [667, 500, 150, 150, -125, 5]]
        self.evaluate_groups_from_params(params, n_rows=1000, n_cols=1000,
                                         buffer_size=10, group_size=50,
                                         margin=0.025)

    # Test four row, col adjacent groups with no covariance
    def test_four_adjacent_groups_no_covariance(self):
        # parameters for generate_data() excluding n_rows, n_cols
        # (i.e. x_bar, y_bar, x_var, y_var, cov, sigma)
        params = [[333, 333, 50, 50, 0, 5],
                  [667, 333, 50, 50, 0, 5],
                  [333, 667, 50, 50, 0, 5],
                  [667, 667, 50, 50, 0, 5]]
        self.evaluate_groups_from_params(params, n_rows=1000, n_cols=1000,
                                         buffer_size=10, group_size=50,
                                         margin=0.025)

    # Test four row, col adjacent groups with high covariance
    def test_four_adjacent_groups_high_covariance(self):
        # parameters for generate_data() excluding n_rows, n_cols
        # (i.e. x_bar, y_bar, x_var, y_var, cov, sigma)
        params = [[333, 333, 150, 150, 125, 5],
                  [667, 333, 150, 150, -125, 5],
                  [333, 667, 150, 150, -125, 5],
                  [667, 667, 150, 150, 125, 5]]
        self.evaluate_groups_from_params(params, n_rows=1000, n_cols=1000,
                                         buffer_size=10, group_size=50,
                                         margin=0.025)

    # Test four row, col adjacent groups at the corners
    def test_four_adjacent_groups_at_corners(self):
        # parameters for generate_data() excluding n_rows, n_cols
        # (i.e. x_bar, y_bar, x_var, y_var, cov, sigma)
        params = [[25, 25, 150, 150, 125, 5],
                  [975, 25, 150, 150, -125, 5],
                  [25, 975, 150, 150, -125, 5],
                  [975, 975, 150, 150, 125, 5]]
        self.evaluate_groups_from_params(params, n_rows=1000, n_cols=1000,
                                         buffer_size=10, group_size=50,
                                         margin=0.025)

    def test_nine_adjacent_groups_at_center(self):
        # parameters for generate_data() excluding n_rows, n_cols
        # (i.e. x_bar, y_bar, x_var, y_var, cov, sigma)
        params = [[500, 450, 25, 150, 0, 5],
                  [450, 450, 150, 150, 125, 5],
                  [550, 450, 150, 150, -125, 5],
                  [500, 500, 50, 50, 0, 5],
                  [450, 500, 150, 25, 0, 5],
                  [550, 500, 150, 25, 0, 5],
                  [500, 550, 25, 150, 0, 5],
                  [450, 550, 150, 150, -125, 5],
                  [550, 550, 150, 150, 125, 5]]
        self.evaluate_groups_from_params(params, n_rows=1000, n_cols=1000,
                                         buffer_size=5, group_size=50,
                                         margin=0.025)


if __name__ == "__main__":
    unittest.main()
