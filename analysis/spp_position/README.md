# GNSS dataset analysis

These scripts inspect the supplied `sppStruct_user.mat` and reuse the repository's
NumPy Gauss-Newton solver. The input MAT file is not bundled. The lightweight
reader supports this file's little-endian MATLAB v5 real numeric struct format;
it is not a general replacement for SciPy's MAT reader.

Run commands from the repository root with the dependencies in `requirements.txt`.

## Ordinary least squares

```bash
python analysis/spp_position/run_least_squares.py /path/to/sppStruct_user.mat
```

The estimator reads only `sat_x`, `sat_y`, `sat_z` and `cor_pseudo`. It solves each
epoch independently with four unknowns `[receiver_x, receiver_y, receiver_z, b]`,
using equal weights. `week` and `tow` appear only as exported labels. The combined
fit assumes a stationary receiver and solves one shared XYZ with a separate
clock offset at each epoch, eliminating clock offsets analytically.

Results are written to `analysis/spp_position/least_squares/`. The tracked report
and JSON summary are compact snapshots; per-epoch positions, all observation
residuals, clock offsets and iteration traces are regenerated as ignored CSVs.
Use `--field raw_pseudo --output /path/to/raw_results` to inspect the raw column
separately. Raw pseudoranges do not give a credible ground-receiver solution with
this simplified model and require upstream corrections.

## Remove low satellites and recalculate

```bash
python analysis/spp_position/run_least_squares.py /path/to/sppStruct_user.mat --min-elevation-deg 15
```

This removes individual readings when the satellite is less than 15 degrees above
the horizon. Angles are recomputed from satellite coordinates and the receiver
position recovered by the unfiltered combined fit; the questionable supplied
angle fields are not used. The selected readings are then fitted with ordinary
least squares and equal weights. A satellite can be kept at one time and removed
at another. This rule identifies low satellites, not proven building obstruction.

Results go to a separate `least_squares_filtered_15deg/` folder, preserving the
unfiltered results. `satellite_filter.csv` records the original observation index,
recomputed angle and keep/drop decision for all readings. Residual observation
indices retain their original input order. Use a different numeric cutoff to run
another comparison, or `--output` to choose the destination.

## Staged atmosphere and weighting experiment

```bash
python analysis/spp_position/staged_positioning.py /path/to/sppStruct_user.mat
```

This compares equal weighting, an assumed standard-atmosphere Saastamoinen
correction, elevation-based weights, and a weights-only control. Elevation is
recomputed from geometry; supplied angle and signal-strength fields are unused.
The source of the existing `cor_pseudo` corrections is unknown, so an added
atmosphere model is a sensitivity experiment that may duplicate prior processing.
No ionosphere correction is invented when coefficients and frequencies are absent.

## Constellation and Earth-rotation sensitivity

```bash
python analysis/spp_position/run_analysis.py /path/to/sppStruct_user.mat
```

This earlier analysis compares common-clock and separate-constellation-clock
models, weighting and satellite-coordinate rotation. Its report is separate from
the strict XYZ-plus-pseudorange ordinary least-squares result.

## Google My Maps export

After the staged run:

```bash
python analysis/spp_position/export_google_maps.py
```

The KML contains median markers, 500 sampled epochs per stage and circles covering
95% of each stage's full-data horizontal solutions. These circles describe
repeatability and are not verified accuracy bounds. Import the generated KML
using Google My Maps. CSV, KML and screenshot exports are ignored by Git.

## Validation

```bash
python -m unittest -v
python -m unittest discover -s analysis/spp_position -p 'test_*.py' -v
```

The analysis tests cover recovery of known synthetic positions and epoch clocks,
clock-offset invariance, equivalence to an explicit augmented-state LS solve,
rank-deficient geometry, atmosphere correction signs and geometry-derived
weighting. There is no surveyed ground truth in the supplied file.
