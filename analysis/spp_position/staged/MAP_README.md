# View the GNSS estimates in Google Maps

Open the final estimated position:
https://www.google.com/maps?q=22.3130854583,114.1918046608

For the three-stage comparison:
1. Sign in at https://www.google.com/maps/d/ .
2. Create a new map, then choose Import in its layer.
3. Select gnss_google_maps.kml from this folder.

Blue: basic XYZ + cor_pseudo estimate.
Orange: plus the assumed tropospheric correction.
Green: plus satellite weighting.

Each stage includes its median marker, 500 evenly sampled epoch estimates,
and the 95% horizontal scatter circle calculated from all 6,693 estimates.
The circles show repeatability, not verified absolute accuracy. Positions
are clamped to the map surface; ellipsoidal heights are in the descriptions.
No ionospheric correction was available. The tropospheric model is an
assumed-atmosphere experiment because cor_pseudo's correction history is unknown.

The compact gnss_map_centres.csv contains just the three median markers as
an alternative import. Select Latitude and Longitude as the location columns,
and Stage as the marker title.

Google's KML import instructions:
https://support.google.com/mymaps/answer/3024836?hl=en-GB
