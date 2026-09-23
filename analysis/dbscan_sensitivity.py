# -*- coding: utf-8 -*-
"""
dbscan_sensitivity.py — Sensitivity of the incident clustering (Section 4.7 / Table note) to the
two DBSCAN parameters, on the geolocated points of the replayed 12 August 2024 stream.

Input : data/fire_events_geoparsed.csv  (output of part5 on the simulation stream; 600 geolocated mentions)
Output: data/dbscan_sensitivity.csv      (one row per (eps_km, minPts) combination)

Columns of the output:
  eps_km, minPts        DBSCAN radius (haversine, km) and minimum cluster size
  clusters              number of clusters nationwide
  isolated              number of noise points (isolated reports)
  largest               size of the largest cluster
  attica_clusters       number of clusters having at least one point within 30 km of the NE-Attica fire
  attica_largest        number of points within 30 km of the fire that belong to the largest such cluster
  attica_noise          noise points within 30 km of the fire

The operating point of the paper is eps_km = 15, minPts = 2 (part5_geoparsing_and_mapping.py).
Usage:  python analysis/dbscan_sensitivity.py
"""
import ast, sys, numpy as np, pandas as pd
from sklearn.cluster import DBSCAN

SRC = "data/fire_events_geoparsed.csv"
OUT = "data/dbscan_sensitivity.csv"
EPS_KM = [5, 10, 15, 20, 30]
MINPTS = [2, 3, 5]
FIRE_LAT, FIRE_LON = 38.14, 23.90          # approximate centroid of the burned area (Varnavas–Marathon–Penteli), NE Attica
FIRE_RADIUS_KM = 30
R = 6371.0088

def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dlmb = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))

df = pd.read_csv(SRC, dtype=str)
pts = []
for c in df["Coordinates"].dropna():
    try:
        for p in ast.literal_eval(c):
            pts.append((float(p["lat"]), float(p["lon"])))
    except (ValueError, SyntaxError, KeyError):
        continue
P = np.array(pts)
print(f"{len(P)} geolocated points")
near_fire = haversine_km(P[:, 0], P[:, 1], FIRE_LAT, FIRE_LON) <= FIRE_RADIUS_KM

rows = []
for eps in EPS_KM:
    for m in MINPTS:
        lab = DBSCAN(eps=eps / R, min_samples=m, algorithm="ball_tree", metric="haversine").fit_predict(np.radians(P))
        ids = [k for k in set(lab) if k != -1]
        sizes = {k: int((lab == k).sum()) for k in ids}
        att = [k for k in ids if ((lab == k) & near_fire).any()]
        att_in = {k: int(((lab == k) & near_fire).sum()) for k in att}
        rows.append(dict(eps_km=eps, minPts=m, clusters=len(ids), isolated=int((lab == -1).sum()),
                         largest=max(sizes.values()) if sizes else 0,
                         attica_clusters=len(att), attica_largest=max(att_in.values()) if att_in else 0,
                         attica_noise=int(((lab == -1) & near_fire).sum())))
res = pd.DataFrame(rows)
res.to_csv(OUT, index=False)
print(res.to_string(index=False))
