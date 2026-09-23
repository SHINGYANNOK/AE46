import unittest

import numpy as np

from gnss_ls import SAT_WORKED, solve_gnss_ls


def synthetic_problem():
    """Return well-spread satellites and exact pseudoranges from a known state."""
    sat_xyz = np.array([
        [15_600_000.0,  7_540_000.0,  20_140_000.0],
        [-18_760_000.0, 2_750_000.0,  18_610_000.0],
        [17_610_000.0, -14_630_000.0, 13_480_000.0],
        [-19_170_000.0, -610_000.0, 18_390_000.0],
        [17_800_000.0,  6_400_000.0, -19_700_000.0],
        [-21_100_000.0, 9_000_000.0,  7_000_000.0],
    ])
    true_q = np.array([1_115_195.0, -4_842_952.0, 3_985_350.0, 75.0])
    pseudoranges = np.linalg.norm(sat_xyz - true_q[:3], axis=1) + true_q[3]
    SAT = np.column_stack([sat_xyz, pseudoranges])
    q0 = true_q + np.array([1_000.0, -1_000.0, 500.0, -50.0])
    return SAT, true_q, q0


class TestGaussianWeightedGNSS(unittest.TestCase):
    def test_unit_or_equal_standard_deviations_preserve_ordinary_ls_solution(self):
        ordinary = solve_gnss_ls(
            SAT_WORKED, q0=[0, 0, 6_370_000, 0]
        )
        scalar = solve_gnss_ls(
            SAT_WORKED, q0=[0, 0, 6_370_000, 0], pseudorange_std_m=3.5
        )
        vector = solve_gnss_ls(
            SAT_WORKED,
            q0=[0, 0, 6_370_000, 0],
            pseudorange_std_m=np.full(len(SAT_WORKED), 8.0),
        )

        self.assertTrue(ordinary.converged)
        self.assertTrue(scalar.converged)
        self.assertTrue(vector.converged)
        np.testing.assert_allclose(scalar.position, ordinary.position, atol=1e-7)
        np.testing.assert_allclose(vector.position, ordinary.position, atol=1e-7)
        self.assertAlmostEqual(scalar.clock_bias_m, ordinary.clock_bias_m, places=7)
        self.assertAlmostEqual(vector.clock_bias_m, ordinary.clock_bias_m, places=7)

    def test_exact_synthetic_measurements_recover_known_state(self):
        SAT, true_q, q0 = synthetic_problem()
        result = solve_gnss_ls(SAT, q0=q0)

        self.assertTrue(result.converged, result.termination_reason)
        np.testing.assert_allclose(result.position, true_q[:3], atol=1e-5)
        self.assertAlmostEqual(result.clock_bias_m, true_q[3], places=5)

    def test_uncertain_outlier_is_downweighted(self):
        SAT, true_q, q0 = synthetic_problem()
        SAT_with_outlier = SAT.copy()
        SAT_with_outlier[-1, 3] += 300.0

        ordinary = solve_gnss_ls(SAT_with_outlier, q0=q0)
        std = np.ones(len(SAT_with_outlier))
        std[-1] = 1_000.0
        weighted = solve_gnss_ls(
            SAT_with_outlier, q0=q0, pseudorange_std_m=std
        )

        ordinary_error = np.linalg.norm(ordinary.position - true_q[:3])
        weighted_error = np.linalg.norm(weighted.position - true_q[:3])
        self.assertTrue(weighted.converged, weighted.termination_reason)
        self.assertLess(weighted_error, ordinary_error * 0.05)

    def test_reported_weighted_statistics_match_definition(self):
        SAT, _, q0 = synthetic_problem()
        SAT[-1, 3] += 10.0
        std = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 8.0])
        result = solve_gnss_ls(SAT, q0=q0, pseudorange_std_m=std)

        expected_normalized = result.residuals / std
        expected_weighted = float(expected_normalized @ expected_normalized)
        np.testing.assert_allclose(result.pseudorange_std_m, std)
        np.testing.assert_allclose(result.normalized_residuals, expected_normalized)
        self.assertAlmostEqual(
            result.weighted_sum_squared_residuals, expected_weighted, places=10
        )
        self.assertAlmostEqual(
            result.normalized_rmse,
            np.sqrt(expected_weighted / len(std)),
            places=10,
        )
        self.assertEqual(result.state_covariance.shape, (4, 4))
        self.assertTrue(np.all(np.isfinite(result.state_covariance)))
        np.testing.assert_allclose(
            result.state_covariance, result.state_covariance.T, atol=1e-12
        )
        self.assertTrue(
            all(
                "weighted_objective_before" in item
                and "weighted_objective_after" in item
                for item in result.history
            )
        )

    def test_invalid_standard_deviations_are_rejected(self):
        invalid_values = [
            True,
            0.0,
            -1.0,
            np.nan,
            [1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, np.inf],
        ]
        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    solve_gnss_ls(
                        SAT_WORKED,
                        q0=[0, 0, 6_370_000, 0],
                        pseudorange_std_m=value,
                    )


if __name__ == "__main__":
    unittest.main()
