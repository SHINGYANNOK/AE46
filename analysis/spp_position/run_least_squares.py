"""Ordinary least squares for sppStruct: per-epoch and stationary combined fits.

Estimator inputs: satellite ECEF X/Y/Z and one selected pseudorange column.
The existing project solver supplies the four-unknown per-epoch calculation.
The combined fit assumes one stationary XYZ and a separate clock per epoch;
clock nuisance parameters are eliminated analytically by within-epoch centring.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from gnss_ls import C, solve_gnss_ls
from analyze_spp import load, llh, enu_rotation


def stationary_least_squares(epochs, q0=None, tolerance_m=1e-4, max_iterations=30):
    """Minimize sum_ki (P_ki - ||S_ki-x|| - b_k)^2 with equal weights.

    For fixed x, b_k=mean_i(P_ki-rho_ki). Subtract each epoch's mean
    residual and mean XYZ Jacobian to profile out all b_k exactly. Solve the
    resulting N-by-3 linear least-squares problem with NumPy's SVD solver.
    """
    q = np.zeros(3) if q0 is None else np.asarray(q0,dtype=float).copy()
    history = []
    converged = False
    for iteration in range(1,max_iterations+1):
        matrices, vectors = [], []
        for xyz, pseudo in epochs:
            rho = np.linalg.norm(xyz-q,axis=1)
            if np.any(rho <= 0): raise ValueError('Zero geometric range')
            H = (q-xyz)/rho[:,None]
            v = pseudo-rho
            matrices.append(H-H.mean(axis=0))
            vectors.append(v-v.mean())
        H = np.vstack(matrices); residual = np.concatenate(vectors)
        delta,_,rank,_ = np.linalg.lstsq(H,residual,rcond=None)
        if rank < 3: raise ValueError('Combined XYZ geometry is rank deficient')
        q += delta
        history.append({'iteration':iteration,'x_m':float(q[0]),'y_m':float(q[1]),'z_m':float(q[2]),
                        'step_norm_m':float(np.linalg.norm(delta)),
                        'objective_before_m2':float(residual@residual)})
        if not np.all(np.isfinite(q)): raise ValueError('Nonfinite combined estimate')
        if np.linalg.norm(delta) < tolerance_m:
            converged = True
            break
    biases,residuals = [],[]
    for xyz,pseudo in epochs:
        v = pseudo-np.linalg.norm(xyz-q,axis=1)
        b = float(v.mean()); biases.append(b);residuals.append(v-b)
    all_residuals = np.concatenate(residuals)
    return {'position':q,'clock_biases_m':np.array(biases),'residuals':residuals,
            'converged':converged,'iterations':len(history),'history':history,
            'rms_m':float(np.sqrt(np.mean(all_residuals**2))),
            'objective_m2':float(all_residuals@all_residuals)}


def write_csv(path, header, rows):
    with path.open('w',newline='',encoding='utf8') as f:
        writer=csv.writer(f);writer.writerow(header);writer.writerows(rows)


def run(input_path, field, output_dir):
    rows=load(input_path);output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    # Time fields below are output labels only; they do not enter either estimator.
    epochs=[(np.column_stack([row[k].ravel() for k in ('sat_x','sat_y','sat_z')]),
             row[field].ravel().copy()) for row in rows]
    results=[];normal_eq_checks=[];first_trace=[]
    for i,(xyz,pseudo) in enumerate(epochs):
        result=solve_gnss_ls(np.column_stack([xyz,pseudo]))
        if not result.converged: raise ValueError(f'Epoch {i+1}: {result.termination_reason}')
        results.append(result)
        rho=np.linalg.norm(xyz-result.position,axis=1)
        H=np.column_stack([(result.position-xyz)/rho[:,None],np.ones(len(xyz))])
        final_delta,_,rank,_=np.linalg.lstsq(H,result.residuals,rcond=None)
        if rank != 4: raise ValueError(f'Epoch {i+1}: final geometry rank {rank}')
        normal_eq_checks.append([float(np.max(np.abs(H.T@result.residuals))),float(np.linalg.norm(final_delta))])
        if i==0:
            for j,step in enumerate(result.history):
                first_trace.append([j+1,*step['q_old'],*step['dq'],*step['q_new'],step['objective_before'],step['objective_after']])
        if (i+1)%2000==0:print(f'Least-squares epochs solved: {i+1}/{len(rows)}',flush=True)
    positions=np.array([r.position for r in results]);median=np.median(positions,axis=0)
    local=(positions-median)@enu_rotation(median).T;horizontal=np.linalg.norm(local[:,:2],axis=1)
    stationary=stationary_least_squares(epochs)
    if not stationary['converged']:raise ValueError('Combined stationary fit did not converge')
    report={
        'input_sha256':hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
        'pseudorange_field':field,'estimator_inputs':['sat_x','sat_y','sat_z',field],
        'time_metadata':'week/tow label exported rows only; not used to estimate position',
        'weights':'equal for every observation',
        'additional_corrections_applied':[],
        'input_correction_history':'cor_pseudo is supplied already processed; exact source corrections unknown',
        'epochs':len(rows),'satellite_observations':sum(len(xyz) for xyz,_ in epochs),
        'satellites_per_epoch_min_max':[min(len(xyz) for xyz,_ in epochs),max(len(xyz) for xyz,_ in epochs)],
        'per_epoch':{
            'converged_epochs':sum(r.converged for r in results),
            'median_ecef_m':median.tolist(), 'mean_ecef_m':positions.mean(axis=0).tolist(),
            'median_position_lat_lon_ellipsoidal_height':llh(median).tolist(),
            'horizontal_scatter_percentiles_50_95_99_m':np.percentile(horizontal,[50,95,99]).tolist(),
            'enu_scatter_standard_deviation_m':np.std(local,axis=0).tolist(),
            'residual_rms_percentiles_50_95_m':np.percentile([r.pseudorange_rmse_m for r in results],[50,95]).tolist(),
            'iterations_min_median_max':[min(r.iterations for r in results),float(np.median([r.iterations for r in results])),max(r.iterations for r in results)],
            'max_final_normal_equation_absolute_component_m':float(np.max(np.array(normal_eq_checks)[:,0])),
            'max_remaining_linearized_state_step_norm_m':float(np.max(np.array(normal_eq_checks)[:,1])),
        },
        'stationary_combined':{
            'assumption':'One constant receiver XYZ over the recording; independent clock bias at each epoch',
            'position_ecef_m':stationary['position'].tolist(),
            'position_lat_lon_ellipsoidal_height':llh(stationary['position']).tolist(),
            'converged':stationary['converged'],'iterations':stationary['iterations'],
            'residual_rms_m':stationary['rms_m'],'objective_sum_squared_residuals_m2':stationary['objective_m2'],
            'offset_from_per_epoch_median_enu_m':((stationary['position']-median)@enu_rotation(median).T).tolist(),
        },
        'limits':['Residual RMS is model fit, not position accuracy.',
                  'Scatter is relative to the per-epoch median; no surveyed ground truth is available.',
                  'The combined fit is only appropriate if the receiver remained stationary.',
                  'A single common receiver clock per epoch does not separately model inter-constellation offsets.']}
    write_csv(output_dir/'positions.csv',
              ['epoch_1based','gps_week','tow_s','x_ecef_m','y_ecef_m','z_ecef_m','latitude_deg','longitude_deg','ellipsoidal_height_m','receiver_clock_bias_m','receiver_clock_bias_s','pseudorange_residual_rms_m','satellite_count','iterations','east_from_median_m','north_from_median_m','up_from_median_m'],
              ([i+1,float(row['week'].flat[0]),float(row['tow'].flat[0]),*r.position,*llh(r.position),r.clock_bias_m,r.clock_bias_s,r.pseudorange_rmse_m,len(epochs[i][0]),r.iterations,*local[i]] for i,(row,r) in enumerate(zip(rows,results))))
    write_csv(output_dir/'residuals.csv',
              ['epoch_1based','observation_1based','per_epoch_ls_residual_m','stationary_ls_residual_m'],
              ([i+1,j+1,float(v),float(stationary['residuals'][i][j])] for i,r in enumerate(results) for j,v in enumerate(r.residuals)))
    write_csv(output_dir/'stationary_epoch_clocks.csv',
              ['epoch_1based','gps_week','tow_s','clock_bias_m','clock_bias_s'],
              ([i+1,float(row['week'].flat[0]),float(row['tow'].flat[0]),b,b/C] for i,(row,b) in enumerate(zip(rows,stationary['clock_biases_m']))))
    write_csv(output_dir/'first_epoch_iterations.csv',
              ['iteration','old_x','old_y','old_z','old_clock_m','delta_x','delta_y','delta_z','delta_clock_m','new_x','new_y','new_z','new_clock_m','sum_squared_residuals_before_m2','sum_squared_residuals_after_m2'],first_trace)
    write_csv(output_dir/'stationary_iterations.csv',list(stationary['history'][0]),(list(s.values()) for s in stationary['history']))
    (output_dir/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    per=report['per_epoch'];joint=report['stationary_combined'];lat,lon,h=joint['position_lat_lon_ellipsoidal_height']
    text=f'''# Ordinary least-squares results

Input: sppStruct_user.mat, measurement column **{field}**. Every satellite has equal weight. The numerical estimators read only satellite X, Y, Z and pseudorange. Time fields are exported as labels.

For each epoch, minimize

    sum_i (P_i - sqrt((X_i-x)^2+(Y_i-y)^2+(Z_i-z)^2) - b)^2.

The four unknowns are receiver ECEF x, y, z and common receiver clock bias b in metres. The existing project function solve_gnss_ls iterates Gauss-Newton: build the range Jacobian H, solve min ||v-H*dq||^2 with numpy.linalg.lstsq, update q, and repeat. Each epoch starts from [0,0,0,0].

## Combined fit assuming a stationary receiver

- Latitude: **{lat:.9f} degrees N**.
- Longitude: **{lon:.9f} degrees E**.
- WGS-84 ellipsoidal height: **{h:.3f} m** (not height above sea level).
- ECEF XYZ (m): {joint['position_ecef_m']}.
- All {report['satellite_observations']:,} observations are fitted together with one shared XYZ and {len(rows):,} independent epoch clock offsets.
- Converged in {joint['iterations']} iterations; pooled pseudorange residual RMS: {joint['residual_rms_m']:.3f} m.

The combined objective is sum over epochs k and satellites i of (P_ki-rho_ki(x)-b_k)^2. At any candidate position, the exact least-squares clock offset is b_k=mean_i(P_ki-rho_ki). Subtracting each epoch mean from both residuals and XYZ Jacobian eliminates the clock offsets, leaving an SVD least-squares solve for the three shared position corrections. This is an ordinary least-squares fit of all measurements; it is not an average or median of already solved positions.

## Independent per-epoch fits

- {per['converged_epochs']:,}/{len(rows):,} epochs converged, using {report['satellites_per_epoch_min_max'][0]}-{report['satellites_per_epoch_min_max'][1]} satellites each.
- Median of epoch ECEF solutions, converted to latitude/longitude/height: {per['median_position_lat_lon_ellipsoidal_height']}.
- Median pseudorange residual RMS: {per['residual_rms_percentiles_50_95_m'][0]:.3f} m.
- 95% horizontal scatter radius about the per-epoch median: {per['horizontal_scatter_percentiles_50_95_99_m'][1]:.3f} m.
- Largest remaining final linearized correction norm: {per['max_remaining_linearized_state_step_norm_m']:.3g} m.

The median above is a descriptive summary of independent epoch solutions, distinct from the combined stationary least-squares estimate.

## Interpretation limits

The supplied cor_pseudo was already processed, but its exact correction history is unknown. The run adds no further atmosphere or Earth-rotation correction. It uses one receiver clock per epoch and no separate constellation offsets. The combined fit assumes a stationary receiver. Fit residuals and repeatability do not establish accuracy without a surveyed reference.

## Files

- positions.csv: independent epoch positions, clock offsets, fit RMS and local ENU scatter.
- residuals.csv: both per-epoch and stationary-fit residuals for every input observation, with input-order indices.
- stationary_epoch_clocks.csv: clock offsets from the combined fit.
- first_epoch_iterations.csv: every four-parameter update for the first epoch.
- stationary_iterations.csv: shared-position solver iterations.
- summary.json: complete machine-readable summary and input hash.

## Reproduce

From the repository root, with NumPy installed:

    python analysis/spp_position/run_least_squares.py "C:\\Users\\User\\Downloads\\sppStruct_user.mat"

Use --field raw_pseudo and a separate --output directory to test the unprocessed pseudorange column. Its earlier fit was not a credible ground-receiver solution.
'''
    (output_dir/'least_squares_report.md').write_text(text,encoding='utf8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('--field',choices=['cor_pseudo','raw_pseudo'],default='cor_pseudo');p.add_argument('--output',default=str(HERE/'least_squares'))
    a=p.parse_args();run(a.input,a.field,a.output)
