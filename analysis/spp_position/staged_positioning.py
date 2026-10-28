"""Staged XYZ+pseudorange experiment using the existing four-unknown solver.

The estimator never uses sat_const, sat_id, sat_el, sat_az or sat_snr. The
Saastamoinen stage is an explicitly assumed standard-atmosphere sensitivity
test on cor_pseudo. A real ionosphere correction needs external information.
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
from gnss_ls import solve_gnss_ls
from analyze_spp import load, llh, enu_rotation


def extract_xyz_pseudorange(row, field):
    """Read exactly the three satellite coordinates and selected pseudorange."""
    xyz = np.column_stack([row[k].ravel() for k in ('sat_x', 'sat_y', 'sat_z')])
    return xyz, row[field].ravel().copy()


def geometric_elevation(xyz, receiver):
    local = (xyz - receiver) @ enu_rotation(receiver).T
    return np.arctan2(local[:, 2], np.linalg.norm(local[:, :2], axis=1))


def saastamoinen_delay(receiver, elevation, relative_humidity=0.7):
    """Standard-atmosphere dry+wet Saastamoinen slant delay, metres.

    Sea-level pressure 1013.25 hPa, temperature 15 C, lapse 6.5 K/km.
    Ellipsoid height approximates meteorological height; negative heights use
    sea-level meteorology. RH=70% is an assumption, not a file measurement.
    Equations: RTKLIB rtkcmn.c tropmodel; no old/default ionosphere coefficients.
    """
    latitude, _, height = llh(receiver)
    if not -100 <= height <= 10000:
        raise ValueError('Receiver height outside standard-atmosphere model domain')
    elevation = np.asarray(elevation, dtype=float)
    if np.any(elevation <= 0) or np.any(elevation > np.pi/2 + 1e-12):
        raise ValueError('Troposphere model requires satellites above the horizon')
    if not 0 <= relative_humidity <= 1:
        raise ValueError('Relative humidity must be between zero and one')
    h = max(0.0, float(height))
    pressure = 1013.25 * (1 - 2.2557e-5*h)**5.2568
    temperature = 288.16 - 0.0065*h
    vapor = 6.108 * relative_humidity * np.exp((17.15*temperature-4684)/(temperature-38.45))
    dry = 0.0022768*pressure / (1 - 0.00266*np.cos(2*np.radians(latitude)) - 0.00028*h/1000)
    wet = 0.002277*(1255/temperature + 0.05)*vapor
    return (dry + wet) / np.sin(elevation)


def estimate(xyz, pseudorange, *, troposphere=False, satellite_weight=False,
             ionosphere_delay_m=None, relative_humidity=0.7, initial_state=None):
    """Estimate [receiver x,y,z, common clock bias] from XYZ and pseudorange.

    Optional ionosphere_delay_m must be independently supplied in metres at
    the measurement frequencies. None means unavailable/not applied. It does
    not silently select a fallback ionosphere model. Correction sign:
    P_corrected = P - I - T. Elevation and relative weights are derived from
    the current estimated receiver position, not supplied MAT angle fields.
    """
    xyz = np.asarray(xyz, dtype=float)
    pseudorange = np.asarray(pseudorange, dtype=float)
    SAT = np.column_stack([xyz, pseudorange])
    ion = np.zeros(len(pseudorange)) if ionosphere_delay_m is None else np.asarray(ionosphere_delay_m, dtype=float)
    if ion.shape != pseudorange.shape or not np.all(np.isfinite(ion)) or np.any(ion < 0):
        raise ValueError('Ionosphere delays must be one finite nonnegative metre value per observation')
    first = solve_gnss_ls(SAT, q0=initial_state)
    if not first.converged:
        raise ValueError(first.termination_reason)
    q = np.r_[first.position, first.clock_bias_m]
    corrected = troposphere or satellite_weight or ionosphere_delay_m is not None
    outer_converged = not corrected
    result = first
    outer_iterations = 0
    if corrected:
        for outer_iterations in range(1, 11):
            elev = geometric_elevation(xyz, q[:3])
            trop = saastamoinen_delay(q[:3], elev, relative_humidity) if troposphere else np.zeros(len(pseudorange))
            # Relative variance is csc^2(elevation); cap below 5 degrees.
            std = 1/np.maximum(np.sin(elev), np.sin(np.radians(5))) if satellite_weight else None
            result = solve_gnss_ls(np.column_stack([xyz, pseudorange-ion-trop]), q0=q, pseudorange_std_m=std)
            if not result.converged:
                raise ValueError(result.termination_reason)
            next_q = np.r_[result.position, result.clock_bias_m]
            if np.linalg.norm(next_q-q) < 1e-4:
                q = next_q
                outer_converged = True
                break
            q = next_q
    elev = geometric_elevation(xyz, q[:3])
    trop = saastamoinen_delay(q[:3], elev, relative_humidity) if troposphere else np.zeros(len(pseudorange))
    residual = pseudorange - (np.linalg.norm(xyz-q[:3], axis=1)+q[3]+ion+trop)
    relative_weight = np.maximum(np.sin(elev), np.sin(np.radians(5)))**2 if satellite_weight else np.ones(len(pseudorange))
    return dict(xyz=q[:3], llh=llh(q[:3]), clock_bias_m=float(q[3]),
                converged=result.converged and outer_converged, satellite_count=len(pseudorange),
                residual_rms_m=float(np.sqrt(np.mean(residual**2))),
                relative_weighted_rms_m=float(np.sqrt(np.sum(relative_weight*residual**2)/np.sum(relative_weight))),
                elevation_min_deg=float(np.degrees(elev.min())),
                trop_min_m=float(trop.min()), trop_max_m=float(trop.max()),
                ionosphere_applied=ionosphere_delay_m is not None, outer_iterations=outer_iterations)


def summarize(records):
    good=[r for r in records if r['converged']]
    if not good: return {'converged_epochs':0}
    xyz=np.array([r['xyz'] for r in good]);median=np.median(xyz,axis=0)
    enu=(xyz-median)@enu_rotation(median).T
    horizontal=np.linalg.norm(enu[:,:2],axis=1)
    return {'converged_epochs':len(good),'median_ecef_m':median.tolist(),'median_lat_lon_ellipsoid_height':llh(median).tolist(),
            'horizontal_scatter_radius_50_95_99_m':np.percentile(horizontal,[50,95,99]).tolist(),
            'enu_standard_deviation_m':np.std(enu,axis=0).tolist(),
            'ellipsoid_height_percentiles_5_50_95_m':np.percentile([r['llh'][2] for r in good],[5,50,95]).tolist(),
            'residual_rms_percentiles_50_95_m':np.percentile([r['residual_rms_m'] for r in good],[50,95]).tolist(),
            'median_relative_weighted_rms_m':float(np.median([r['relative_weighted_rms_m'] for r in good])),
            'satellite_count_min_max':[min(r['satellite_count'] for r in good),max(r['satellite_count'] for r in good)],
            'minimum_geometric_elevation_deg':min(r['elevation_min_deg'] for r in good),
            'troposphere_delay_range_m':[min(r['trop_min_m'] for r in good),max(r['trop_max_m'] for r in good)]}


def run(input_path, output_dir):
    rows=load(input_path);output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    configs={
        'raw_xyz_pseudorange_only':('raw_pseudo',{}),
        'corrected_xyz_pseudorange_only':('cor_pseudo',{}),
        'corrected_plus_assumed_troposphere':('cor_pseudo',{'troposphere':True}),
        'corrected_plus_assumed_troposphere_and_weights':('cor_pseudo',{'troposphere':True,'satellite_weight':True}),
        'corrected_weights_only_control':('cor_pseudo',{'satellite_weight':True}),
    }
    results={name:[] for name in configs}
    for i,row in enumerate(rows):
        for name,(field,options) in configs.items():
            xyz,pseudo=extract_xyz_pseudorange(row,field)
            results[name].append(estimate(xyz,pseudo,**options))
        if (i+1)%1500==0:print(f'Processed {i+1}/{len(rows)} epochs for all five comparisons.',flush=True)
    report={'input_sha256':hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
            'epochs':len(rows),'estimator_inputs':['sat_x','sat_y','sat_z','selected pseudorange'],
            'estimated_state':['receiver_x','receiver_y','receiver_z','one common receiver_clock_bias_m'],
            'ionosphere_stage':{'status':'unavailable_not_applied','reason':'No model coefficients, signal frequencies, TEC products, or dual-frequency measurements are provided.'},
            'assumptions':{'cor_pseudo_processing_history':'unknown: further atmosphere correction may duplicate existing corrections',
                           'troposphere_model':'Saastamoinen, standard atmosphere, relative humidity 70%; ellipsoid height approximates atmospheric height',
                           'weight_model':'relative w_i=max(sin(E_i),sin(5 deg))^2, geometry-derived elevation',
                           'satellite_selection':'same observations retained in every corrected-pseudorange stage; no added masks or rejection',
                           'earth_rotation':'no additional correction in any stage',
                           'constellation_biases':'not used, to keep the baseline strictly XYZ+pseudorange',
                           'accuracy_reference':'none; all scatter is relative to each stage median, not true position error'},
            'stages':{name:summarize(records) for name,records in results.items()}}
    baseline=np.array(report['stages']['corrected_xyz_pseudorange_only']['median_ecef_m'])
    R=enu_rotation(baseline)
    for name,records in results.items():
        median=np.array(report['stages'][name]['median_ecef_m'])
        report['stages'][name]['median_shift_from_corrected_baseline_enu_m']=((median-baseline)@R.T).tolist()
        with (output_dir/(name+'.csv')).open('w',newline='',encoding='utf8') as f:
            writer=csv.writer(f)
            writer.writerow(['epoch_index_1based','x_m','y_m','z_m','latitude_deg','longitude_deg','ellipsoid_height_m','clock_bias_m','residual_rms_m','satellite_count','converged'])
            for i,r in enumerate(records):writer.writerow([i+1,*r['xyz'],*r['llh'],r['clock_bias_m'],r['residual_rms_m'],r['satellite_count'],r['converged']])
    (output_dir/'stages.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    write_report(report,output_dir)
    for name,stats in report['stages'].items():
        print(name,json.dumps({k:stats[k] for k in ['converged_epochs','median_lat_lon_ellipsoid_height','horizontal_scatter_radius_50_95_99_m','residual_rms_percentiles_50_95_m']}),flush=True)
    return report


def write_report(report,output_dir):
    labels={
        'raw_xyz_pseudorange_only':'Raw pseudorange baseline',
        'corrected_xyz_pseudorange_only':'cor_pseudo baseline',
        'corrected_plus_assumed_troposphere':'+ assumed troposphere',
        'corrected_plus_assumed_troposphere_and_weights':'+ troposphere + satellite weights',
        'corrected_weights_only_control':'Weights only (control)',
    }
    text=['# Staged positioning: XYZ and pseudorange only','',
          'All stages estimate the same four unknowns [x, y, z, receiver clock bias]. The estimator reads only sat_x, sat_y, sat_z and the selected pseudorange. Supplied satellite IDs, constellation codes, elevation, azimuth and signal strength are not used. Each epoch is solved independently from an Earth-centre initial guess.','',
          '## Full recording comparison','',
          '| Stage | Latitude (deg) | Longitude (deg) | Ellipsoid height (m) | Median residual RMS (m) | 95% horizontal scatter radius (m) |',
          '|---|---:|---:|---:|---:|---:|']
    for name,stats in report['stages'].items():
        lat,lon,h=stats['median_lat_lon_ellipsoid_height'];rms=stats['residual_rms_percentiles_50_95_m'][0];scatter=stats['horizontal_scatter_radius_50_95_99_m'][1]
        text.append(f'| {labels[name]} | {lat:.8f} | {lon:.8f} | {h:.2f} | {rms:.2f} | {scatter:.2f} |')
    text+=['','Every row uses all 6,693 epochs. The same satellite observations are retained between the cor_pseudo stages. No constellation bias terms, Earth-rotation change, satellite exclusion or residual rejection were added. The weights-only control separates the effect of weighting from the assumed atmosphere correction.','',
           '## Ionosphere stage: unavailable','',
           'No ionospheric correction was computed or implied. The file has no broadcast model coefficients, signal-frequency metadata, TEC map or paired dual-frequency pseudoranges. Time and geometry alone do not determine ionospheric delay. A proper Klobuchar correction needs the broadcast alpha/beta parameters and measurement frequencies. The estimate() function accepts independently supplied per-satellite ionospheric delays in metres for a future measured-data run.','',
           '## Troposphere experiment','',
           'The standard-atmosphere Saastamoinen dry and wet model was added to the cor_pseudo observation equation: P = geometric range + receiver clock + I + T. Equivalently, subtract I and T from P before the least-squares update. Here I is not available and is not applied. The temperature is 15 C at sea level with a 6.5 K/km lapse rate, sea-level pressure is 1013.25 hPa, and relative humidity is assumed to be 70%. Ellipsoid height is used as an approximation for atmospheric height; there is no measured weather or geoid information in the file. Delay and geometry are recomputed until the outer position/clock iteration converges.','',
           'The correction history of cor_pseudo is unknown. This extra troposphere stage is a sensitivity experiment, not a verified removal of a known uncorrected error. It can apply a correction twice if the source already included it. Low-elevation paths are retained to keep the comparison controlled; the simple mapping is less reliable near the horizon.','',
           '## Satellite weights','',
           'Weights use elevation recomputed from the satellite XYZ and current receiver estimate: w_i = max(sin(E_i), sin(5 degrees))^2. The existing weighted solver receives relative standard deviations sigma_i = 1/sqrt(w_i). These are relative weights, not calibrated uncertainties or a verified statistical noise model. No sat_el, sat_az or sat_snr values are read.','',
           'A weighted estimate may have a larger unweighted residual RMS: it optimizes a different objective. Position scatter about its own median measures repeatability, not accuracy against ground truth. A shift or a lower scatter alone does not prove the correction is right.','',
           '## Reproduction','',
           'From the repository root with NumPy installed:','',
           '```','python analysis/spp_position/staged_positioning.py "C:\\Users\\User\\Downloads\\sppStruct_user.mat"','```','',
           'Outputs are stages.json and a per-epoch CSV for each compared stage in this folder. Original solver and previous analyses remain separate.','',
           '## Sources','',
           '- [ESA: Klobuchar ionospheric model and required inputs](https://gssc.esa.int/navipedia/index.php/Klobuchar_Ionospheric_Model)',
           '- [RTKLIB: Saastamoinen standard-atmosphere equations, tropmodel](https://github.com/tomojitakasu/RTKLIB/blob/master/src/rtkcmn.c)',
           '- [GNSS-SDR: pseudorange equation, troposphere and weighting models](https://gnss-sdr.org/docs/sp-blocks/pvt/)',
           '']
    marker=text.index('## Ionosphere stage: unavailable')
    text[marker:marker]=['## What the comparison shows\n\nThe 95% horizontal scatter radius changes from 8.80 m (baseline) to 8.46 m (assumed troposphere) to 4.73 m (assumed troposphere plus weights). The weights-only control gives 4.46 m. Therefore most of the tighter clustering comes from weighting, and this recording does not demonstrate that the added troposphere model improves horizontal accuracy.\n\nThe median ellipsoidal height changes from 31.48 m to 19.62 m after the assumed troposphere model, then to 19.37 m after weighting. This substantial height change needs a surveyed reference and confirmation of the existing cor_pseudo corrections to judge.\n\nThe raw-only model numerically converges but gives a median altitude about 85.6 km and a median pseudorange fit RMS about 115.2 km. It is not a credible ground-receiver solution. Atmospheric corrections of metres to tens of metres cannot account for those residuals; satellite clock and other upstream corrections need checking.\n\nThe final unweighted fit RMS increases from 9.14 m to 9.45 m when weights are applied, while the cluster tightens. These metrics optimize or measure different things. A more stable cluster alone does not establish a more accurate receiver location.\n\n']
    (output_dir/'staged_report.md').write_text('\n'.join(text),encoding='utf8')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('input');parser.add_argument('--output',default=str(HERE/'staged'))
    args=parser.parse_args();run(args.input,args.output)
