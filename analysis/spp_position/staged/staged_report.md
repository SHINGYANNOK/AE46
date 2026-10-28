# Staged positioning: XYZ and pseudorange only

All stages estimate the same four unknowns [x, y, z, receiver clock bias]. The estimator reads only sat_x, sat_y, sat_z and the selected pseudorange. Supplied satellite IDs, constellation codes, elevation, azimuth and signal strength are not used. Each epoch is solved independently from an Earth-centre initial guess.

## Full recording comparison

| Stage | Latitude (deg) | Longitude (deg) | Ellipsoid height (m) | Median residual RMS (m) | 95% horizontal scatter radius (m) |
|---|---:|---:|---:|---:|---:|
| Raw pseudorange baseline | 22.48183302 | 114.23061729 | 85628.73 | 115202.68 | 35343.95 |
| cor_pseudo baseline | 22.31311557 | 114.19179658 | 31.48 | 9.41 | 8.80 |
| + assumed troposphere | 22.31311533 | 114.19180664 | 19.62 | 9.14 | 8.46 |
| + troposphere + satellite weights | 22.31308546 | 114.19180466 | 19.37 | 9.45 | 4.73 |
| Weights only (control) | 22.31308795 | 114.19180538 | 25.79 | 9.89 | 4.46 |

Every row uses all 6,693 epochs. The same satellite observations are retained between the cor_pseudo stages. No constellation bias terms, Earth-rotation change, satellite exclusion or residual rejection were added. The weights-only control separates the effect of weighting from the assumed atmosphere correction.

## What the comparison shows

The 95% horizontal scatter radius changes from 8.80 m (baseline) to 8.46 m (assumed troposphere) to 4.73 m (assumed troposphere plus weights). The weights-only control gives 4.46 m. Therefore most of the tighter clustering comes from weighting, and this recording does not demonstrate that the added troposphere model improves horizontal accuracy.

The median ellipsoidal height changes from 31.48 m to 19.62 m after the assumed troposphere model, then to 19.37 m after weighting. This substantial height change needs a surveyed reference and confirmation of the existing cor_pseudo corrections to judge.

The raw-only model numerically converges but gives a median altitude about 85.6 km and a median pseudorange fit RMS about 115.2 km. It is not a credible ground-receiver solution. Atmospheric corrections of metres to tens of metres cannot account for those residuals; satellite clock and other upstream corrections need checking.

The final unweighted fit RMS increases from 9.14 m to 9.45 m when weights are applied, while the cluster tightens. These metrics optimize or measure different things. A more stable cluster alone does not establish a more accurate receiver location.


## Ionosphere stage: unavailable

No ionospheric correction was computed or implied. The file has no broadcast model coefficients, signal-frequency metadata, TEC map or paired dual-frequency pseudoranges. Time and geometry alone do not determine ionospheric delay. A proper Klobuchar correction needs the broadcast alpha/beta parameters and measurement frequencies. The estimate() function accepts independently supplied per-satellite ionospheric delays in metres for a future measured-data run.

## Troposphere experiment

The standard-atmosphere Saastamoinen dry and wet model was added to the cor_pseudo observation equation: P = geometric range + receiver clock + I + T. Equivalently, subtract I and T from P before the least-squares update. Here I is not available and is not applied. The temperature is 15 C at sea level with a 6.5 K/km lapse rate, sea-level pressure is 1013.25 hPa, and relative humidity is assumed to be 70%. Ellipsoid height is used as an approximation for atmospheric height; there is no measured weather or geoid information in the file. Delay and geometry are recomputed until the outer position/clock iteration converges.

The correction history of cor_pseudo is unknown. This extra troposphere stage is a sensitivity experiment, not a verified removal of a known uncorrected error. It can apply a correction twice if the source already included it. Low-elevation paths are retained to keep the comparison controlled; the simple mapping is less reliable near the horizon.

## Satellite weights

Weights use elevation recomputed from the satellite XYZ and current receiver estimate: w_i = max(sin(E_i), sin(5 degrees))^2. The existing weighted solver receives relative standard deviations sigma_i = 1/sqrt(w_i). These are relative weights, not calibrated uncertainties or a verified statistical noise model. No sat_el, sat_az or sat_snr values are read.

A weighted estimate may have a larger unweighted residual RMS: it optimizes a different objective. Position scatter about its own median measures repeatability, not accuracy against ground truth. A shift or a lower scatter alone does not prove the correction is right.

## Reproduction

From the repository root with NumPy installed:

```
python analysis/spp_position/staged_positioning.py "C:\Users\User\Downloads\sppStruct_user.mat"
```

Outputs are stages.json and a per-epoch CSV for each compared stage in this folder. Original solver and previous analyses remain separate.

## Sources

- [ESA: Klobuchar ionospheric model and required inputs](https://gssc.esa.int/navipedia/index.php/Klobuchar_Ionospheric_Model)
- [RTKLIB: Saastamoinen standard-atmosphere equations, tropmodel](https://github.com/tomojitakasu/RTKLIB/blob/master/src/rtkcmn.c)
- [GNSS-SDR: pseudorange equation, troposphere and weighting models](https://gnss-sdr.org/docs/sp-blocks/pvt/)
