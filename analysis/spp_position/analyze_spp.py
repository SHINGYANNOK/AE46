"""Inspect this MATLAB v5 sppStruct and estimate receiver ECEF coordinates.
Only NumPy required. MAT reader supports this file's real numeric structs;
it is deliberately not a general-purpose replacement for scipy.io.loadmat.
"""
import argparse
import csv
import json
import struct
import zlib
from pathlib import Path
import numpy as np

C = 299792458.0
OMEGA = 7.2921151467e-5
DT = {1:'i1',2:'u1',3:'<i2',4:'<u2',5:'<i4',6:'<u4',7:'<f4',9:'<f8',12:'<i8',13:'<u8'}

def element(b, offset):
    tag, size = struct.unpack_from('<II', b, offset)
    if tag >> 16:
        return tag & 65535, b[offset+4:offset+4+(tag>>16)], offset+8
    end = offset+8+size
    return tag, b[offset+8:end], end if tag == 15 else (end+7)//8*8

def matrix(b):
    _, flags, offset = element(b, 0)
    cls = struct.unpack_from('<I', flags)[0] & 255
    _, data, offset = element(b, offset)
    dims = tuple(np.frombuffer(data, '<i4'))
    _, name, offset = element(b, offset)
    if cls == 2:
        _, data, offset = element(b, offset)
        length = struct.unpack_from('<i', data)[0]
        _, data, offset = element(b, offset)
        fields = [data[i:i+length].split(b'\0')[0].decode() for i in range(0,len(data),length)]
        rows = []
        for _ in range(int(np.prod(dims))):
            row = {}
            for field in fields:
                tag, data, offset = element(b, offset)
                if tag != 14: raise ValueError('Expected matrix field')
                row[field] = matrix(data)
            rows.append(row)
        return rows
    if cls not in range(6,16): raise ValueError(f'Unsupported MATLAB class {cls}')
    tag, data, offset = element(b, offset)
    return np.frombuffer(data, DT[tag]).reshape(dims, order='F').astype(float)

def load(path):
    b = Path(path).read_bytes()
    if b[126:128] != b'IM': raise ValueError('Expected little-endian MATLAB v5 file')
    tag, data, offset = element(b,128)
    if tag == 15:
        tag, data, _ = element(zlib.decompress(data),0)
    if tag != 14: raise ValueError('Expected MATLAB matrix')
    return matrix(data)

def llh(xyz):
    x,y,z = np.asarray(xyz)
    a=6378137.0; e2=6.6943799901413165e-3
    p=np.hypot(x,y); lon=np.arctan2(y,x); lat=np.arctan2(z,p*(1-e2))
    for _ in range(12):
        N=a/np.sqrt(1-e2*np.sin(lat)**2)
        lat=np.arctan2(z+e2*N*np.sin(lat),p)
    N=a/np.sqrt(1-e2*np.sin(lat)**2)
    h=p*np.cos(lat)+z*np.sin(lat)-a*np.sqrt(1-e2*np.sin(lat)**2)
    return np.array([np.degrees(lat),np.degrees(lon),h])

def enu_rotation(xyz):
    lat,lon,_=llh(xyz); lat,lon=np.radians([lat,lon])
    sl,cl=np.sin(lon),np.cos(lon); sp,cp=np.sin(lat),np.cos(lat)
    return np.array([[-sl,cl,0],[-sp*cl,-sp*sl,cp],[cp*cl,cp*sl,sp]])

def arrays(row, field='cor_pseudo', mask_deg=None, constellation=None):
    sat=np.column_stack([row[k].ravel() for k in ['sat_x','sat_y','sat_z']])
    P=row[field].ravel(); codes=row['sat_const'].ravel().astype(int)
    mask=np.ones(len(P),dtype=bool)
    if mask_deg is not None: mask &= row['sat_el'].ravel()>=mask_deg
    if constellation is not None: mask &= codes==constellation
    return sat[mask],P[mask],codes[mask],row['sat_el'].ravel()[mask]

def solve(row,field='cor_pseudo',mask_deg=None,constellation=None,multiclock=False,rotation=0,weighted=False,initial=None,geometric_elevation=False):
    sat,P,codes,elev=arrays(row,field,None if geometric_elevation else mask_deg,constellation)
    if geometric_elevation:
        if initial is None: raise ValueError('Geometric elevation needs approximate receiver XYZ')
        local=(sat-initial)@enu_rotation(initial).T
        elev=np.degrees(np.arctan2(local[:,2],np.linalg.norm(local[:,:2],axis=1)))
        mask=elev>=(mask_deg if mask_deg is not None else -90)
        sat,P,codes,elev=sat[mask],P[mask],codes[mask],elev[mask]
    groups=np.unique(codes) if multiclock else np.array([0])
    B=(codes[:,None]==groups).astype(float) if multiclock else np.ones((len(P),1))
    q=np.zeros(3+len(groups));
    if initial is not None: q[:3]=initial
    w=np.maximum(np.sin(np.radians(elev)),0.1) if weighted else np.ones(len(P))
    success=False
    for it in range(30):
        S=sat.copy()
        if rotation:
            angle=rotation*OMEGA*np.linalg.norm(sat-q[:3],axis=1)/C
            S[:,0]=np.cos(angle)*sat[:,0]+np.sin(angle)*sat[:,1]
            S[:,1]=-np.sin(angle)*sat[:,0]+np.cos(angle)*sat[:,1]
        rho=np.linalg.norm(S-q[:3],axis=1)
        v=P-rho-B@q[3:]
        H=np.column_stack([(q[:3]-S)/rho[:,None],B])
        dq,_,rank,_=np.linalg.lstsq(H*w[:,None],v*w,rcond=None)
        if rank<len(q): break
        q+=dq
        if not np.all(np.isfinite(q)) or np.linalg.norm(q[:3])>1e9: break
        if np.linalg.norm(dq)<1e-4: success=True;break
    rho=np.linalg.norm(S-q[:3],axis=1);v=P-rho-B@q[3:]
    return {'xyz':q[:3], 'llh':llh(q[:3]),'bias':q[3:],'rms':float(np.sqrt(np.mean(v*v))), 'residual':v,'n':len(P),'success':success,'groups':groups.tolist(),'iterations':it+1}

def serial(result):
    return {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in result.items() if k!='residual'}

def probe(rows):
    for idx in [0,len(rows)//2,len(rows)-1]:
        print('EPOCH',idx,'TOW',float(rows[idx]['tow'].flat[0]))
        for field in ['raw_pseudo','cor_pseudo']:
            for multi,mask in [(False,None),(True,None),(False,10),(True,10)]:
                result=solve(rows[idx],field,mask,multiclock=multi)
                print(field,'multi',multi,'mask',mask,json.dumps(serial(result)))
        for code in [1,4,8,32]:
            result=solve(rows[idx],constellation=code)
            print('const',code,json.dumps(serial(result)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('--probe',action='store_true');args=p.parse_args()
    rows=load(args.input)
    probe(rows)
