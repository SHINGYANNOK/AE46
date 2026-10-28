"""Run full-data positioning and sensitivity checks; write reproducible outputs."""
import argparse,csv,json,hashlib,sys
from pathlib import Path
import numpy as np
import analyze_spp as a

def main():
    p=argparse.ArgumentParser();p.add_argument('input');args=p.parse_args()
    outdir=Path(__file__).resolve().parent
    rows=a.load(args.input)
    first=a.solve(rows[0],multiclock=True)['xyz']
    # Independently compare the basic four-unknown calculation with the project solver.
    sys.path.insert(0,str(outdir.parent.parent))
    from gnss_ls import solve_gnss_ls
    sat,P,_,_=a.arrays(rows[0]);check=solve_gnss_ls(np.column_stack([sat,P]))
    assert check.converged
    np.testing.assert_allclose(check.position,a.solve(rows[0])['xyz'],atol=1e-4,rtol=0)
    np.testing.assert_allclose(a.llh([6378137,0,0]),[0,0,0],atol=1e-8)
    np.testing.assert_allclose(a.llh([0,6378137,0]),[0,90,0],atol=1e-8)
    # Recover a known position with independent constellation biases and exact ranges.
    mock={k:v.copy() for k,v in rows[0].items()}
    codes=mock['sat_const'].ravel().astype(int)
    mock['cor_pseudo']=(np.linalg.norm(sat-first,axis=1)+np.array([{1:10,4:30,8:-20,32:40}[c] for c in codes])).reshape(1,-1)
    np.testing.assert_allclose(a.solve(mock,multiclock=True)['xyz'],first,atol=1e-5,rtol=0)
    print('Independent solver comparison, coordinate conversion and synthetic recovery passed.',flush=True)
    configs={
        'common_clock':{},
        'multi_clock':{'multiclock':True},
        'multi_clock_elevation_weighted':{'multiclock':True,'geometric_elevation':True,'mask_deg':10,'weighted':True},
        'multi_clock_elevation_weighted_rotation':{'multiclock':True,'geometric_elevation':True,'mask_deg':10,'weighted':True,'rotation':1},
    }
    solutions={k:[] for k in configs};times=[]
    for i,row in enumerate(rows):
        times.append(float(row['tow'].flat[0]))
        for name,config in configs.items():
            solutions[name].append(a.solve(row,initial=first,**config))
        if (i+1)%2000==0:print(f'Solved {i+1}/{len(rows)} epochs for all four models.',flush=True)
    report={'input':str(Path(args.input).resolve()),'input_sha256':hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
        'epochs':len(rows),'gps_week':float(rows[0]['week'].flat[0]),'tow_first':times[0],'tow_last':times[-1],
        'elapsed_seconds':times[-1]-times[0], 'primary_model':'multi_clock',
        'independent_solver_position_difference_m':float(np.linalg.norm(check.position-a.solve(rows[0])['xyz'])),
        'models':{},'notes':[
        'All model comparisons use corrected pseudoranges. No atmospheric model added: correction history unknown.',
        'Primary result uses supplied ECEF coordinates directly; no additional Earth rotation correction because prior processing is unknown.',
        'Rotation sensitivity applies R3(omega*geometric_range/c) to each satellite coordinate.',
        'Elevation weighting uses recomputed geometric elevation, not the inconsistent supplied sat_el/sat_az.',
        'Epoch scatter and residual RMS are not independently verified position accuracy.',
        'No surveyed receiver coordinates or processing metadata are present in this MAT file.']}
    ref=np.median(np.array([r['xyz'] for r in solutions['multi_clock']]),axis=0)
    R=a.enu_rotation(ref);t=np.array(times)
    for name,results in solutions.items():
        xyz=np.array([r['xyz'] for r in results]);med=np.median(xyz,axis=0);enu=(xyz-med)@a.enu_rotation(med).T;h=np.hypot(enu[:,0],enu[:,1])
        rm=np.array([r['rms'] for r in results]);n=np.array([r['n'] for r in results]);heights=np.array([r['llh'][2] for r in results])
        head=np.median(xyz[t<t[0]+300],axis=0);tail=np.median(xyz[t>t[-1]-300],axis=0)
        block=np.array([np.median(xyz[(t>=lo)&(t<lo+600)],axis=0) for lo in np.arange(t[0],t[-1],600)])
        block_enu=(block-med)@R.T
        report['models'][name]={
            'converged_epochs':sum(r['success'] for r in results),'median_ecef_m':med.tolist(),'median_position_lat_lon_ellipsoid_height':a.llh(med).tolist(),
            'offset_from_primary_enu_m':((med-ref)@R.T).tolist(),
            'horizontal_scatter_radius_m_percentiles_50_95_99':np.percentile(h,[50,95,99]).tolist(),
            'enu_scatter_std_m':np.std(enu,axis=0).tolist(),
            'ellipsoid_height_m_percentiles_5_50_95':np.percentile(heights,[5,50,95]).tolist(),
            'pseudorange_residual_rms_m_percentiles_50_95':np.percentile(rm,[50,95]).tolist(),
            'satellites_used_min_median_max':[int(n.min()),float(np.median(n)),int(n.max())],
            'first_last_5min_median_change_enu_m':((tail-head)@R.T).tolist(),
            'ten_minute_block_medians_enu_m':block_enu.tolist(),
            'bias_m_median':np.median(np.array([r['bias'] for r in results]),axis=0).tolist()}
        if name=='multi_clock':
            with (outdir/'positions.csv').open('w',newline='',encoding='utf8') as f:
                w=csv.writer(f);w.writerow(['gps_week','tow_s','x_ecef_m','y_ecef_m','z_ecef_m','latitude_deg','longitude_deg','ellipsoid_height_m','east_from_median_m','north_from_median_m','up_from_median_m','pseudorange_residual_rms_m','satellite_count','converged'])
                for row,result,local in zip(rows,results,enu):
                    w.writerow([float(row['week'].flat[0]),float(row['tow'].flat[0]),*result['xyz'],*result['llh'],*local,result['rms'],result['n'],result['success']])
    # Same measurements solved separately by constellation, sampled throughout recording.
    report['single_constellation_crosschecks']={}
    for code in [1,4,8,32]:
        result=[a.solve(rows[i],constellation=code,initial=first) for i in range(0,len(rows),60)]
        xyz=np.array([r['xyz'] for r in result if r['success']]);med=np.median(xyz,axis=0)
        report['single_constellation_crosschecks'][str(code)]={'epochs_sampled':len(result),'converged':len(xyz),'median_llh':a.llh(med).tolist(),'offset_from_primary_enu_m':((med-ref)@R.T).tolist(),'median_residual_rms_m':float(np.median([r['rms'] for r in result if r['success']]))}
    raw=[a.solve(rows[i],field='raw_pseudo',initial=first) for i in range(0,len(rows),60)]
    report['raw_pseudorange_median_residual_rms_m']=float(np.median([r['rms'] for r in raw]))
    angles=[];azimuth_errors=[]
    for i in range(0,len(rows),60):
        row=rows[i];sat,_,_,el=a.arrays(row);local=(sat-ref)@R.T
        elev=np.degrees(np.arctan2(local[:,2],np.linalg.norm(local[:,:2],axis=1)))
        az=np.degrees(np.arctan2(local[:,0],local[:,1]))%360
        angles.extend((el-elev).tolist());azimuth_errors.extend(((row['sat_az'].ravel()-az+180)%360-180).tolist())
    report['supplied_angle_discrepancy_deg']={'elevation_rms':float(np.sqrt(np.mean(np.square(angles)))),'azimuth_rms':float(np.sqrt(np.mean(np.square(azimuth_errors))))}
    (outdir/'results.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    for name,result in report['models'].items():
        print(name,json.dumps({k:result[k] for k in ['converged_epochs','median_position_lat_lon_ellipsoid_height','horizontal_scatter_radius_m_percentiles_50_95_99','pseudorange_residual_rms_m_percentiles_50_95','offset_from_primary_enu_m','first_last_5min_median_change_enu_m']}),flush=True)
    print('Full results:',outdir/'results.json',flush=True)

if __name__=='__main__':main()
