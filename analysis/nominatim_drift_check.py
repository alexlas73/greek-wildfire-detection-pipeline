#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nominatim drift diagnostic (supports Section 3.5.3 / 4.x).

WHY
---
Re-running the geoparsing pipeline reproduces the published NER figures exactly
(strict 53.74%, operational 89.8%) but yields 73.47% operational geocoding
accuracy against the 80.58% originally reported. Since NER is identical, the
divergence must arise downstream, in the geocoding stage. Nominatim is a live
service backed by OpenStreetMap, whose content and candidate ranking change over
time. This script tests that hypothesis directly.

HOW
---
ground_truth_geoparsing.xlsx preserves the ORIGINAL run's output
(Geocoded_As, Predicted_Lat, Predicted_Lon). We re-query Nominatim today for the
same NER-extracted toponyms and compare, per toponym:

  * did the original run geocode it, and does it geocode now?
  * is the returned place the same (name and coordinates)?
  * did the top-ranked candidate change position?
  * does the accept/reject decision against the 10 km criterion change?

OUTPUT
------
nominatim_drift.csv  - per-toponym comparison
nominatim_drift.json - summary counts

RUN (Colab, inside the repo folder, after the sensitivity script so the cache exists)
    !python nominatim_drift_check.py
"""
import os, sys, json, pickle, time
import pandas as pd
import numpy as np

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
GOLD     = os.path.join(REPO_DIR, 'data', 'ground_truth_geoparsing.xlsx')
CACHE    = os.path.join(REPO_DIR, 'nominatim_cache.pkl')
OUT_CSV  = os.path.join(REPO_DIR, 'nominatim_drift.csv')
OUT_JSON = os.path.join(REPO_DIR, 'nominatim_drift.json')

sys.path.insert(0, REPO_DIR)
import part5_geoparsing_and_mapping as P5
from geopy.distance import geodesic

TOL_KM = 10.0          # same operational criterion as the paper
SAME_PLACE_KM = 1.0    # below this, treat as the same place returned

gold = pd.read_excel(GOLD)
print(f'[data] gold standard entries: {len(gold)}')

# reuse the cache built by the sensitivity run, if present
disk = {}
if os.path.exists(CACHE):
    with open(CACHE, 'rb') as f:
        disk = pickle.load(f)
    print(f'[cache] {len(disk)} toponyms already cached from the sweep')

_orig = P5.get_candidates_cached
def cached(loc):
    k = str(loc).strip().lower()
    if k in disk:
        return disk[k]
    r = _orig(loc)
    disk[k] = r
    return r

rows = []
for i, g in gold.iterrows():
    topo = str(g['NER_Extracted']).strip()
    if not topo or topo.lower() in ('nan', 'missing', 'none'):
        continue

    cands = cached(topo) or []
    now_top = cands[0] if cands else None

    orig_name = str(g.get('Geocoded_As', '') or '')
    orig_ok   = pd.notna(g.get('Predicted_Lat'))
    orig_lat, orig_lon = g.get('Predicted_Lat'), g.get('Predicted_Lon')
    act_lat,  act_lon  = g.get('Actual_Lat'),  g.get('Actual_Lon')

    rec = {
        'Tweet_ID': g['Tweet_ID'],
        'toponym': topo,
        'gold_toponym': g.get('Actual_Toponym'),
        'orig_geocoded_as': orig_name,
        'orig_success': bool(orig_ok),
        'now_n_candidates': len(cands),
        'now_top_name': (now_top or {}).get('name', ''),
        'now_success': now_top is not None,
    }

    # distance between the two runs' chosen points
    if orig_ok and now_top:
        try:
            rec['shift_km'] = round(geodesic((float(orig_lat), float(orig_lon)),
                                             (float(now_top['lat']), float(now_top['lon']))).km, 3)
        except Exception:
            rec['shift_km'] = None
    else:
        rec['shift_km'] = None

    # accuracy against ground truth, then vs now
    def within(lat, lon):
        if pd.isna(act_lat) or lat is None or pd.isna(lat):
            return None
        try:
            return geodesic((float(lat), float(lon)), (float(act_lat), float(act_lon))).km <= TOL_KM
        except Exception:
            return None

    rec['orig_within_10km'] = within(orig_lat, orig_lon) if orig_ok else False
    rec['now_within_10km']  = within(now_top['lat'], now_top['lon']) if now_top else False

    # where did the originally chosen place rank in today's candidate list?
    rank = None
    if orig_ok and cands:
        for j, c in enumerate(cands):
            try:
                if geodesic((float(c['lat']), float(c['lon'])),
                            (float(orig_lat), float(orig_lon))).km <= SAME_PLACE_KM:
                    rank = j
                    break
            except Exception:
                pass
    rec['orig_choice_rank_today'] = rank
    rows.append(rec)

    if len(rows) % 25 == 0:
        print(f'  ...{len(rows)} toponyms checked')

with open(CACHE, 'wb') as f:
    pickle.dump(disk, f)

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)

# ----------------------------------------------------------------- SUMMARY
n = len(df)
same_place   = int((df['shift_km'].fillna(9e9) <= SAME_PLACE_KM).sum())
moved        = int(((df['shift_km'] > SAME_PLACE_KM) & df['shift_km'].notna()).sum())
newly_fail   = int((df['orig_success'] & ~df['now_success']).sum())
newly_work   = int((~df['orig_success'] & df['now_success']).sum())
was_ok_now_no = int((df['orig_within_10km'] == True).mul(df['now_within_10km'] == False).sum())
was_no_now_ok = int((df['orig_within_10km'] == False).mul(df['now_within_10km'] == True).sum())
rank_changed = int((df['orig_choice_rank_today'].fillna(-1) > 0).sum())
rank_lost    = int(df['orig_choice_rank_today'].isna().mul(df['orig_success']).sum())

print('\n' + '=' * 70)
print('NOMINATIM DRIFT SUMMARY')
print(f'  toponyms compared                         : {n}')
print(f'  identical place returned (<= {SAME_PLACE_KM} km)      : {same_place}')
print(f'  different place returned                  : {moved}')
print(f'  geocoded originally, fails today          : {newly_fail}')
print(f'  failed originally, geocodes today         : {newly_work}')
print(f'  originally within 10 km, now outside      : {was_ok_now_no}')
print(f'  originally outside 10 km, now within      : {was_no_now_ok}')
print(f'  original choice no longer ranked first    : {rank_changed}')
print(f'  original choice absent from results today : {rank_lost}')

if df['shift_km'].notna().any():
    s = df.loc[df['shift_km'].notna(), 'shift_km']
    print(f'\n  shift distance: median {s.median():.3f} km, mean {s.mean():.3f} km, max {s.max():.1f} km')

print('\nLargest disagreements between the two runs:')
top = df.dropna(subset=['shift_km']).nlargest(10, 'shift_km')
for _, r in top.iterrows():
    print(f"  {r['toponym'][:28]:30s} orig='{str(r['orig_geocoded_as'])[:22]:24s}' "
          f"now='{str(r['now_top_name'])[:22]:24s}' {r['shift_km']:.1f} km")

summary = {'n_compared': n, 'same_place': same_place, 'different_place': moved,
           'newly_failing': newly_fail, 'newly_working': newly_work,
           'lost_accuracy': was_ok_now_no, 'gained_accuracy': was_no_now_ok,
           'rank_changed': rank_changed, 'choice_absent_today': rank_lost,
           'tolerance_km': TOL_KM, 'same_place_km': SAME_PLACE_KM}
with open(OUT_JSON, 'w') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f'\nSaved {OUT_CSV} and {OUT_JSON}')
print('\nINTERPRETATION GUIDE')
print('  If "different place returned" and "originally within 10 km, now outside" are')
print('  substantial, the gap between 80.58% and 73.47% is attributable to changes in')
print('  the live gazetteer rather than to any change in the pipeline, and should be')
print('  disclosed as such. If those counts are near zero, the discrepancy has another')
print('  cause and must be investigated further before anything is reported.')
