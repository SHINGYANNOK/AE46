"""Export the three staged GNSS estimates as a Google My Maps KML."""
from pathlib import Path
import csv,json,math,xml.etree.ElementTree as ET
import numpy as np

BASE=Path(__file__).resolve().parent/'staged'
NS='http://www.opengis.net/kml/2.2';ET.register_namespace('',NS)
def sub(parent,tag,text=None,**attrs):
    child=ET.SubElement(parent,'{'+NS+'}'+tag,attrs)
    if text is not None:child.text=str(text)
    return child

def main():
    report=json.loads((BASE/'stages.json').read_text())
    root=ET.Element('{'+NS+'}kml');doc=sub(root,'Document')
    sub(doc,'name','GNSS position: baseline, troposphere, weights')
    sub(doc,'description','Computed GNSS positions, not a surveyed reference. Additional troposphere is an assumed-atmosphere experiment; ionosphere unavailable. Points show stationary solution scatter, not a travelled route. Each cloud contains 500 evenly sampled epochs out of 6693. Circles contain 95% of the full-data horizontal solutions about their own median; they are not accuracy bounds. Heights are stored as descriptive ellipsoid heights; the map uses ground-clamped positions.')
    specs=[('corrected_xyz_pseudorange_only','1 Basic XYZ + cor_pseudo','ffff7733','http://maps.google.com/mapfiles/kml/paddle/blu-circle.png'),
           ('corrected_plus_assumed_troposphere','2 Plus assumed troposphere','ff0099ff','http://maps.google.com/mapfiles/kml/paddle/orange-circle.png'),
           ('corrected_plus_assumed_troposphere_and_weights','3 Plus satellite weights','ff44aa22','http://maps.google.com/mapfiles/kml/paddle/grn-circle.png')]
    for idx,(key,label,color,icon_url) in enumerate(specs):
        for kind,scale in [('median',1.1),('epoch',0.35)]:
            st=sub(doc,'Style',id=f's{idx}_{kind}');ic=sub(st,'IconStyle');sub(ic,'scale',scale)
            icon=sub(ic,'Icon');sub(icon,'href',icon_url)
            ls=sub(st,'LabelStyle');sub(ls,'scale',0 if kind=='epoch' else .9)
        st=sub(doc,'Style',id=f's{idx}_circle');line=sub(st,'LineStyle');sub(line,'color',color);sub(line,'width',2)
        poly=sub(st,'PolyStyle');sub(poly,'color','20'+color[2:]);sub(poly,'fill',1);sub(poly,'outline',1)
        stats=report['stages'][key];lat,lon,h=stats['median_lat_lon_ellipsoid_height'];radius=stats['horizontal_scatter_radius_50_95_99_m'][1]
        folder=sub(doc,'Folder');sub(folder,'name',label)
        pm=sub(folder,'Placemark');sub(pm,'name',label+' median');sub(pm,'styleUrl',f'#s{idx}_median')
        sub(pm,'description',f'Latitude {lat:.9f}, longitude {lon:.9f}. Ellipsoidal height {h:.2f} m. Full-data 95% horizontal scatter radius {radius:.2f} m, not verified position accuracy. Median pseudorange residual RMS {stats["residual_rms_percentiles_50_95_m"][0]:.2f} m.')
        point=sub(pm,'Point');sub(point,'altitudeMode','clampToGround');sub(point,'coordinates',f'{lon:.10f},{lat:.10f},0')
        pm=sub(folder,'Placemark');sub(pm,'name',label+' 95% scatter circle');sub(pm,'styleUrl',f'#s{idx}_circle')
        sub(pm,'description',f'{radius:.2f} m repeatability radius from all 6693 solutions, not an accuracy or confidence boundary.')
        polygon=sub(pm,'Polygon');sub(polygon,'altitudeMode','clampToGround');outer=sub(polygon,'outerBoundaryIs');ring=sub(outer,'LinearRing')
        # Small circles are adequate at these metre-scale radii.
        a=6378137.;e2=6.6943799901413165e-3;phi=math.radians(lat);v=math.sqrt(1-e2*math.sin(phi)**2)
        N=a/v;M=a*(1-e2)/v**3
        coords=[]
        for angle in np.linspace(0,2*np.pi,73):
            la=lat+math.degrees(radius*math.cos(angle)/M)
            lo=lon+math.degrees(radius*math.sin(angle)/(N*math.cos(phi)))
            coords.append(f'{lo:.10f},{la:.10f},0')
        sub(ring,'coordinates',' '.join(coords))
        with (BASE/(key+'.csv')).open(newline='') as f:rows=list(csv.DictReader(f))
        indices=np.unique(np.linspace(0,len(rows)-1,500,dtype=int))
        for i in indices:
            row=rows[i];pm=sub(folder,'Placemark');sub(pm,'name',label+f' epoch {i+1}');sub(pm,'styleUrl',f'#s{idx}_epoch')
            sub(pm,'description',f'Epoch {i+1} of {len(rows)}. Ellipsoidal height {float(row["ellipsoid_height_m"]):.2f} m; fit RMS {float(row["residual_rms_m"]):.2f} m.')
            point=sub(pm,'Point');sub(point,'altitudeMode','clampToGround');sub(point,'coordinates',f'{row["longitude_deg"]},{row["latitude_deg"]},0')
    lat,lon,_=report['stages'][specs[-1][0]]['median_lat_lon_ellipsoid_height']
    look=sub(doc,'LookAt');sub(look,'longitude',lon);sub(look,'latitude',lat);sub(look,'range',150);sub(look,'tilt',0);sub(look,'heading',0)
    out=BASE/'gnss_google_maps.kml';ET.ElementTree(root).write(out,encoding='utf-8',xml_declaration=True)
    # Separate compact CSV gives an alternative three-pin import.
    with (BASE/'gnss_map_centres.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.writer(f);w.writerow(['Stage','Latitude','Longitude','Ellipsoidal height m','95 percent horizontal scatter m'])
        for key,label,_,_ in specs:
            stats=report['stages'][key];w.writerow([label,*stats['median_lat_lon_ellipsoid_height'],stats['horizontal_scatter_radius_50_95_99_m'][1]])
    check=ET.parse(out);ns={'k':NS};points=check.findall('.//k:Point',ns);placemarks=check.findall('.//k:Placemark',ns)
    assert len(points)==1503 and len(placemarks)==1506
    assert out.stat().st_size<5*1024*1024
    for item in check.findall('.//k:coordinates',ns):
        for coord in item.text.split():
            lo,la,alt=map(float,coord.split(','));assert -180<=lo<=180 and -90<=la<=90 and alt==0
    print(out);print('Validated:',len(placemarks),'features;',out.stat().st_size,'bytes')

if __name__=='__main__':main()
