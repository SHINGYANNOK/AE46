import unittest
import numpy as np
from staged_positioning import (
    estimate, extract_xyz_pseudorange, geometric_elevation, saastamoinen_delay,
)
from analyze_spp import enu_rotation


def problem():
    # Known geodetic reference, independently converted to ECEF.
    lat, lon = np.radians([22.31, 114.19])
    e2 = 6.6943799901413165e-3
    N = 6378137.0 / np.sqrt(1-e2*np.sin(lat)**2)
    receiver = np.array([(N+30)*np.cos(lat)*np.cos(lon),
                         (N+30)*np.cos(lat)*np.sin(lon),
                         (N*(1-e2)+30)*np.sin(lat)])
    az = np.radians([0,35,70,110,150,190,225,260,300,340])
    el = np.radians([8,15,25,35,45,55,65,75,40,20])
    local = np.column_stack([np.cos(el)*np.sin(az),np.cos(el)*np.cos(az),np.sin(el)])
    satellite = receiver + 21e6*local@enu_rotation(receiver)
    pseudo = np.linalg.norm(satellite-receiver,axis=1) + 80.0
    return satellite,pseudo,receiver,el


class TestStagedPositioning(unittest.TestCase):
    def test_baseline_reads_only_xyz_and_pseudorange(self):
        sat,p,true,_=problem()
        row={'sat_x':sat[:,0:1].T,'sat_y':sat[:,1:2].T,
             'sat_z':sat[:,2:3].T,'cor_pseudo':p.reshape(1,-1)}
        xyz,pr=extract_xyz_pseudorange(row,'cor_pseudo')
        r=estimate(xyz,pr)
        self.assertTrue(r['converged'])
        self.assertFalse(r['ionosphere_applied'])
        np.testing.assert_allclose(r['xyz'],true,atol=1e-5,rtol=0)
        self.assertAlmostEqual(r['clock_bias_m'],80,places=5)

    def test_external_ionosphere_delay_is_subtracted_in_metres(self):
        sat,p,true,_=problem();ion=np.linspace(3,30,len(p))
        r=estimate(sat,p+ion,ionosphere_delay_m=ion)
        self.assertTrue(r['converged'])
        self.assertTrue(r['ionosphere_applied'])
        np.testing.assert_allclose(r['xyz'],true,atol=1e-4,rtol=0)
        with self.assertRaises(ValueError):estimate(sat,p,ionosphere_delay_m=[1,2])

    def test_troposphere_geometry_iteration_recovers_injected_known_state(self):
        sat,p,true,el=problem()
        model=saastamoinen_delay(true,el)
        r=estimate(sat,p+model,troposphere=True)
        self.assertTrue(r['converged'])
        self.assertGreater(r['outer_iterations'],1)
        np.testing.assert_allclose(r['xyz'],true,atol=2e-4,rtol=0)
        self.assertLess(r['residual_rms_m'],1e-4)

    def test_low_elevation_outlier_is_downweighted(self):
        sat,p,true,_=problem();p[0]+=100
        ordinary=estimate(sat,p)
        weighted=estimate(sat,p,satellite_weight=True)
        self.assertTrue(weighted['converged'])
        self.assertLess(np.linalg.norm(weighted['xyz']-true),np.linalg.norm(ordinary['xyz']-true)*0.5)

    def test_elevation_and_atmosphere_domains(self):
        sat,_,true,el=problem()
        np.testing.assert_allclose(geometric_elevation(sat,true),el,atol=1e-12)
        delay=saastamoinen_delay(true,np.radians([90,30,5]))
        self.assertGreater(delay[0],2.1);self.assertLess(delay[0],2.8)
        np.testing.assert_allclose(delay[1],2*delay[0],atol=1e-10)
        self.assertGreater(delay[2],10*delay[0])
        with self.assertRaises(ValueError):saastamoinen_delay(true,[0])
        with self.assertRaises(ValueError):saastamoinen_delay(true,el,relative_humidity=2)


if __name__=='__main__':unittest.main(verbosity=2)
