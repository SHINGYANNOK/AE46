# Receiver position recovered from sppStruct_user.mat

The corrected pseudoranges place the receiver near **22.313118 N, 114.191777 E**, in Hong Kong.
[Open estimated position in Google Maps](https://www.google.com/maps?q=22.3131184,114.1917765)

## Result

- Median ECEF: X = -2419182.726 m, Y = 5384992.666 m, Z = 2406537.133 m.
- WGS-84 latitude: 22.313118375 degrees; longitude: 114.191776523 degrees.
- Ellipsoidal height: 32.26 m. This is not height above sea level.
- All 6,693 epochs converged. The recording spans 7,220 seconds (2 h 20 s), with gaps.
- The coordinates appear consistent with a stationary receiver. This is a conclusion from the clustering, not an external confirmation.
- 95% of epoch estimates are within 9.56 m horizontally of the median position.
- Median pseudorange residual RMS: 4.92 m. This measures fit, not position accuracy.
- Numerical digits are retained for reproducibility and do not imply centimetre accuracy.

## Method

For each epoch, solve P_i = ||satellite_i - receiver|| + b_constellation(i) using iterative nonlinear least squares. The unknowns are three ECEF coordinates and four independent constellation clock/bias terms, corresponding to the file codes 1, 4, 8, and 32. The primary calculation uses every corrected pseudorange and the supplied satellite coordinates without applying extra corrections whose prior application is unknown. Convert ECEF to WGS-84 geodetic coordinates and take the component-wise median ECEF over the recording.

The first epoch agrees with the existing project solver for the common-clock model to better than 0.1 mm. Exact synthetic measurements recover the known receiver position and separate constellation biases. Independent single-constellation calculations on 112 epochs per constellation place their median horizontal positions within about 10 m of the main estimate.

## Model sensitivity

| Model | Latitude (degrees) | Longitude (degrees) | Ellipsoidal height (m) | 95% horizontal scatter radius (m) | Median fit RMS (m) |
|---|---:|---:|---:|---:|---:|
| common_clock | 22.31311557 | 114.19179658 | 31.48 | 8.80 | 9.41 |
| multi_clock | 22.31311837 | 114.19177652 | 32.26 | 9.56 | 4.92 |
| multi_clock_elevation_weighted | 22.31308727 | 114.19177291 | 23.89 | 5.72 | 5.11 |
| multi_clock_elevation_weighted_rotation | 22.31309156 | 114.19145315 | 24.24 | 5.42 | 4.76 |

The weighted models discard satellites below 10 degrees of recomputed geometric elevation and use sin(elevation)^2 observation weights. They do not use the supplied angle fields.

## Limits of the recovered position

- The file does not state whether satellite coordinates or corrected pseudoranges already include Earth rotation, ionosphere, troposphere, satellite clock or code-bias corrections. No additional atmosphere correction was assumed.
- Applying the standard Earth-rotation transformation shifts the weighted median about 33 m west. Both alternatives remain in the same immediate area, but this ambiguity prevents claiming a precisely verified map point.
- The supplied satellite elevation and azimuth values disagree with the geometry of the recovered position and supplied ECEF coordinates: sampled RMS differences are about 32.3 degrees in elevation and 72.6 degrees in azimuth. Their definition or computation needs checking.
- The raw pseudoranges give a median sampled fit RMS of about 115 km, compared with about 4.9 m using corrected values and constellation biases. Raw values require further correction before positioning.
- Scatter about a dataset median is repeatability, not an accuracy bound. There is no surveyed ground truth in the file.

## Outputs and reproduction

- `positions.csv`: per-epoch primary position, clock-model fit RMS and ENU offsets from the median.
- `results.json`: full numerical summaries, model comparisons, checks and input SHA-256.
- `analyze_spp.py`: NumPy-only reader for this numeric MATLAB v5 struct and positioning functions.
- `run_analysis.py`: full analysis driver, which also compares the basic result to the existing project solver.

Run from the repository root with NumPy installed:
```
python analysis/spp_position/run_analysis.py "C:\Users\User\Downloads\sppStruct_user.mat"
```

## Method references

- [ESA Navipedia: Code Based Positioning](https://gssc.esa.int/navipedia/index.php/Code_Based_Positioning_%28SPS%29)
- [ESA Navipedia: Satellite Coordinates Computation and Earth rotation](https://gssc.esa.int/navipedia/index.php/Satellite_Coordinates_Computation)
