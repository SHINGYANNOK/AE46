# Position calculation after removing low satellites

We removed a reading whenever its satellite was less than 15 degrees above the horizon. We worked out the direction from the satellite coordinates and the receiver position found by the original calculation. The questionable direction numbers stored in the file were not used to decide what to remove.

A low satellite is not automatically blocked. This is a cautious rule to test whether removing low readings makes the position results more consistent. A satellite can be kept at one time and removed at another.

## What changed

| | Before | After |
|---|---:|---:|
| Satellite readings used | 196,405 | 179,200 |
| Successful position calculations | 6,693 | 6,693 |
| Satellites available at each time | 27-32 | 25-29 |
| Distance containing 95 out of 100 results, measured from the middle of each group | 8.80 m | 6.79 m |

Removed **17,205 readings (8.76%)**. Every remaining reading has equal importance in the least-squares calculation.

The middle of the recalculated position group is **22.31309821 N, 114.19179401 E**. Its height above the WGS-84 reference shape is 30.44 m; this is not height above sea level.

The separate calculation that assumes one fixed position for the whole recording gives [22.31310672092433, 114.1917999087488, 29.64907663874328]. The two position summaries answer slightly different questions; the table above describes the separate calculations at each recorded time.

The filtered results are more closely grouped. We still need a known receiver location to tell whether they are closer to the true position.

## Saved files

- positions.csv: recalculated position at every recorded time.
- satellite_filter.csv: every original reading, its recalculated angle, and whether it was kept.
- residuals.csv: remaining disagreements between measurements and the position calculations.
- summary.json: full numerical summary.
- least_squares_report.md: the calculation in more detail.

The filtering and saved row counts were checked for every input reading. Original row numbers are preserved so every keep/drop decision can be traced back to the MAT file.

To repeat from the repository root:

    python analysis/spp_position/run_least_squares.py /path/to/sppStruct_user.mat --min-elevation-deg 15
