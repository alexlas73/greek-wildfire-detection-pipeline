#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export per-toponym geocoding error distances (for the Section 4.5 error-distribution figure).

Runs the geoparser on the 147-item gold standard in BOTH configurations and saves,
for every gold toponym, the distance (km) between the predicted and true coordinates:
  - BASE      : accept the top-ranked Nominatim candidate (no disambiguation)
  - OPTIMIZED : full disambiguation framework

Output: geoparse_errors.csv  with columns
  Tweet_ID, toponym, actual_toponym, base_dist_km, optimized_dist_km

This lets us plot the distribution of errors and show how disambiguation collapses
the long tail of grossly-misplaced toponyms (the effect that lowers the MEAN while
leaving the MEDIAN unchanged).

NOTE: geocoding hits the live Nominatim service. If nominatim_cache.pkl from the
sensitivity run is present, it is reused so the numbers stay consistent with that run.

RUN (Colab, inside the repo folder)
    !pip install gr-nlp-toolkit geopy folium scikit-learn emoji spacy pandas openpyxl -q
    !python -m spacy download el_core_news_lg -q
    # ensure master_cleaned_dataset.xlsx and ground_truth_geoparsing.xlsx are in data/
    !python export_geoparse_errors.py
"""
import os, sys, json, pickle, copy
import numpy as np
import pandas as pd
from geopy.distance import geodesic

REPO = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(REPO, 'data')
MASTER = os.path.join(DATA, 'master_cleaned_dataset.xlsx')
GOLD = os.path.join(DATA, 'ground_truth_geoparsing.xlsx')
CACHE = os.path.join(REPO, 'nominatim_cache.pkl')
NER_CACHE = os.path.join(REPO, 'ner_cache.pkl')

sys.path.insert(0, REPO)
import part5_geoparsing_and_mapping as P5

# ---- persistent Nominatim cache (reuse sensitivity run if present) ----
disk = {}
if os.path.exists(CACHE):
    disk = pickle.load(open(CACHE, 'rb'))
    print(f'[cache] reusing {len(disk)} cached toponyms')
_orig = P5.get_candidates_cached
def cached(loc):
    name = loc['name'] if isinstance(loc, dict) else loc
    k = str(name).strip().lower()
    if k in disk:
        return copy.deepcopy(disk[k])
    r = _orig(name); disk[k] = copy.deepcopy(r); return r
P5.get_candidates_cached = cached

# ---- data ----
df = pd.read_excel(MASTER, sheet_name='master_cleaned_dataset')
gold = pd.read_excel(GOLD, sheet_name='golden_dataset_fixed')
fire = df[df.is_fire == 1].copy()
fire['Date'] = pd.to_datetime(fire['Date'], format='%a %b %d %H:%M:%S %z %Y', utc=True, errors='coerce')
fire = fire.sort_values('Date').reset_index(drop=True)

# ---- NER once (cached) ----
if os.path.exists(NER_CACHE):
    fire['Extracted_Locations'] = pickle.load(open(NER_CACHE, 'rb'))
    print('[ner] reused cached NER')
else:
    print('[ner] running NER once...')
    fire['Extracted_Locations'] = fire['Light Cleaned Text'].apply(P5.extract_locations_smart)
    pickle.dump(fire['Extracted_Locations'], open(NER_CACHE, 'wb'))


def run(optimized):
    """Return a dict Tweet_ID(str) -> list of (toponym, pred_lat, pred_lon)."""
    P5.spatiotemporal_memory = []
    work = fire.copy()
    if optimized:
        work['Coordinates'] = work.apply(P5.resolve_coordinates, axis=1)
    else:
        # BASE: take the top Nominatim candidate for each extracted toponym, no disambiguation
        def base_resolve(row):
            out = []
            for loc in (row.get('Extracted_Locations') or []):
                name = loc['name'] if isinstance(loc, dict) else loc
                cands = P5.get_candidates_cached(name)
                if cands:
                    c = cands[0]
                    out.append({'original_name': name, 'lat': c['lat'], 'lon': c['lon']})
            return out
        work['Coordinates'] = work.apply(base_resolve, axis=1)
    res = {}
    for _, r in work.iterrows():
        res[str(r['Tweet_ID'])] = r['Coordinates']
    return res

def clean_id(x):
    s = str(x).strip()
    if 'e' in s.lower():
        try: s = '%.0f' % float(s)
        except: pass
    return s[:-2] if s.endswith('.0') else s

print('[run] base...'); base = run(False)
pickle.dump(disk, open(CACHE, 'wb'))
print('[run] optimized...'); opt = run(True)
pickle.dump(disk, open(CACHE, 'wb'))

def best_dist(coords_list, topo_norm, alat, alon):
    """Distance of the predicted coord whose toponym matches, to the true point."""
    best = None
    for c in (coords_list or []):
        nm = P5.normalize_text(c.get('original_name', ''))
        if nm and (nm == topo_norm or nm in topo_norm or topo_norm in nm):
            try:
                d = geodesic((float(c['lat']), float(c['lon'])), (float(alat), float(alon))).km
                best = d if best is None else min(best, d)
            except Exception:
                pass
    return best

rows = []
for _, g in gold.iterrows():
    if pd.isna(g.get('Actual_Lat')):
        continue
    tid = clean_id(g['Tweet_ID'])
    topo = P5.normalize_text(g.get('Actual_Toponym', ''))
    bd = best_dist(base.get(tid, []), topo, g['Actual_Lat'], g['Actual_Lon'])
    od = best_dist(opt.get(tid, []),  topo, g['Actual_Lat'], g['Actual_Lon'])
    rows.append({'Tweet_ID': tid, 'toponym': g.get('Actual_Toponym'),
                 'base_dist_km': round(bd, 3) if bd is not None else np.nan,
                 'optimized_dist_km': round(od, 3) if od is not None else np.nan})

out = pd.DataFrame(rows)
out.to_csv(os.path.join(REPO, 'geoparse_errors.csv'), index=False)

def summ(col):
    v = out[col].dropna()
    within10 = (v <= 10).mean() * 100
    return f'n={len(v)}  mean={v.mean():.2f}km  median={v.median():.2f}km  <=10km={within10:.1f}%'

print('\n=== per-toponym error export ===')
print('BASE     ', summ('base_dist_km'))
print('OPTIMIZED', summ('optimized_dist_km'))
print('\nSaved geoparse_errors.csv')
print('(Live Nominatim: values reflect the current gazetteer, consistent with the sensitivity re-run.)')
