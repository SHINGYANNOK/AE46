"""Independent checks of the combined stationary ordinary LS estimator."""
import unittest
import numpy as np
from run_least_squares import stationary_least_squares, minimum_elevation_mask
from test_gnss_ls import synthetic_problem


def epochs_with_clocks(noise=False):
    SAT,true,_=synthetic_problem();epochs=[];rng=np.random.default_rng(7)
    clocks=np.array([80.0,1000.0,-500000.0,2000000.0])
    for k,bias in enumerate(clocks):
        xyz=SAT[:,:3]+np.array([2000*k,-500*k,300*k])
        pseudo=np.linalg.norm(xyz-true[:3],axis=1)+bias
        if noise:pseudo+=rng.normal(0,3,len(pseudo))
        epochs.append((xyz,pseudo))
    return epochs,true[:3],clocks


class TestStationaryLeastSquares(unittest.TestCase):
    def test_exact_known_position_and_separate_epoch_clocks(self):
        epochs,true,clocks=epochs_with_clocks()
        result=stationary_least_squares(epochs)
        self.assertTrue(result['converged'])
        np.testing.assert_allclose(result['position'],true,rtol=0,atol=1e-5)
        np.testing.assert_allclose(result['clock_biases_m'],clocks,rtol=0,atol=1e-5)
        self.assertLess(result['rms_m'],1e-6)

    def test_clock_shifts_do_not_change_receiver_position(self):
        epochs,_,_=epochs_with_clocks(noise=True)
        shifted=[(xyz,p+offset) for (xyz,p),offset in zip(epochs,[1e6,-2e6,50000,-100000])]
        before=stationary_least_squares(epochs);after=stationary_least_squares(shifted)
        np.testing.assert_allclose(after['position'],before['position'],rtol=0,atol=1e-5)
        self.assertAlmostEqual(after['objective_m2'],before['objective_m2'],places=5)

    def test_profiled_solution_matches_independent_augmented_state_fit(self):
        epochs,true,_=epochs_with_clocks(noise=True)
        q=np.r_[true+[100,-100,50],np.zeros(len(epochs))]
        for _ in range(20):
            Hs,vs=[],[]
            for k,(xyz,pseudo) in enumerate(epochs):
                rho=np.linalg.norm(xyz-q[:3],axis=1)
                H=np.zeros((len(pseudo),len(q)))
                H[:,:3]=(q[:3]-xyz)/rho[:,None];H[:,3+k]=1
                Hs.append(H);vs.append(pseudo-rho-q[3+k])
            delta,_,rank,_=np.linalg.lstsq(np.vstack(Hs),np.concatenate(vs),rcond=None)
            self.assertEqual(rank,len(q));q+=delta
            if np.linalg.norm(delta)<1e-7:break
        profiled=stationary_least_squares(epochs)
        np.testing.assert_allclose(profiled['position'],q[:3],rtol=0,atol=1e-5)
        np.testing.assert_allclose(profiled['clock_biases_m'],q[3:],rtol=0,atol=1e-5)

    def test_unidentifiable_geometry_is_rejected(self):
        xyz=np.tile([20e6,0,0],(5,1))
        with self.assertRaises(ValueError):stationary_least_squares([(xyz,np.full(5,20e6))])


class TestElevationFilter(unittest.TestCase):
    def test_uses_geometry_and_keeps_boundary_observation(self):
        receiver=np.array([6378137.0,0,0])
        expected=np.array([-5.0,5.0,14.0,15.0,25.0,80.0])
        angles=np.radians(expected)
        # At latitude=longitude=0, ECEF X is local up and Z is local north.
        xyz=receiver+20e6*np.column_stack([np.sin(angles),np.zeros(len(angles)),np.cos(angles)])
        elevation,keep=minimum_elevation_mask(xyz,receiver,15)
        np.testing.assert_allclose(elevation,expected,atol=1e-10)
        np.testing.assert_array_equal(keep,[False,False,False,True,True,True])

    def test_rejects_invalid_cutoffs_and_shapes(self):
        receiver=np.array([6378137.0,0,0]);xyz=np.array([[20e6,10e6,10e6]])
        for minimum in [-1,90,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):minimum_elevation_mask(xyz,receiver,minimum)
        with self.assertRaises(ValueError):minimum_elevation_mask(xyz[:,0],receiver,15)


if __name__=='__main__':unittest.main(verbosity=2)
