#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sensitivity analysis of the geoparsing disambiguation thresholds (Section 3.5.3).

WHY
---
The disambiguation thresholds (40 km / 20 km / 5 h / 15 km / 2 samples) were fixed
a priori from operational reasoning, not tuned on the benchmark. This script varies
each ONE AT A TIME around its chosen value and reports operational geocoding
accuracy, to demonstrate that the reported performance sits on a stable plateau
rather than at a sharp optimum. The purpose is NOT to find a better setting: if a
higher value appears, it must NOT be adopted post hoc, because selecting a threshold
on the 147-item evaluation set would constitute fitting to the benchmark and would
invalidate the very robustness claim being made.

HOW TO RUN (Google Colab, same environment as part5)
----------------------------------------------------
  !pip install gr-nlp-toolkit geopy folium scikit-learn emoji spacy -q
  !python -m spacy download el_core_news_lg -q
  !git clone https://github.com/alexlas73/greek-wildfire-detection-pipeline.git
  %cd greek-wildfire-detection-pipeline
  # place this file in the repo root, then:
  !python sensitivity_analysis.py

The script geocodes ONCE, caches every Nominatim response to disk, and then replays
the resolution stage for each configuration. Nominatim is therefore queried only on
the first run (roughly one request per second, as per its usage policy); subsequent
runs read from the cache. Expect the first pass to take a while, later passes seconds.
"""
import os, sys, json, pickle, copy, itertools
import pandas as pd
import numpy as np
from datetime import datetime

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
REPO_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(REPO_DIR, 'data')
MASTER     = os.path.join(DATA_DIR, 'master_cleaned_dataset.csv')     # or .xlsx
GOLD       = os.path.join(DATA_DIR, 'ground_truth_geoparsing.xlsx')
CACHE      = os.path.join(REPO_DIR, 'nominatim_cache.pkl')
OUT_CSV    = os.path.join(REPO_DIR, 'sensitivity_results.csv')
OUT_JSON   = os.path.join(REPO_DIR, 'sensitivity_results.json')

sys.path.insert(0, REPO_DIR)
import part5_geoparsing_and_mapping as P5

BASELINE = {
    'DISAMBIGUATION_THRESHOLD_KM': 40.0,
    'STRICT_THRESHOLD_KM':         20.0,
    'TIME_WINDOW_HOURS':            5.0,
    'FUZZY_MATCH_THRESHOLD':        0.80,
    'CLUSTERING_EPSILON_KM':       15.0,
    'CLUSTERING_MIN_SAMPLES':       2,
}

# One-at-a-time sweep grids (baseline value must appear in each grid)
GRID = {
    'DISAMBIGUATION_THRESHOLD_KM': [20.0, 30.0, 40.0, 50.0, 60.0, 80.0],
    'STRICT_THRESHOLD_KM':         [5.0, 10.0, 15.0, 20.0, 30.0, 40.0],
    'TIME_WINDOW_HOURS':           [1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 24.0],
    'FUZZY_MATCH_THRESHOLD':       [0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
}

# --------------------------------------------------------------------------
# PERSISTENT NOMINATIM CACHE
# --------------------------------------------------------------------------
_disk_cache = {}
if os.path.exists(CACHE):
    with open(CACHE, 'rb') as f:
        _disk_cache = pickle.load(f)
    print(f'[cache] loaded {len(_disk_cache)} cached toponyms')

# ---------------------------------------------------------------------------
# BUGFIX: part5 defines  is_fuzzy_match(s1, s2, threshold=FUZZY_MATCH_THRESHOLD)
# The default argument is bound once, when the function is defined, so reassigning
# P5.FUZZY_MATCH_THRESHOLD afterwards has NO effect. We rebind the function so the
# threshold is looked up on every call and the sweep is actually exercised.
import difflib as _difflib

def _live_fuzzy_match(s1, s2, threshold=None):
    if threshold is None:
        threshold = P5.FUZZY_MATCH_THRESHOLD
    return _difflib.SequenceMatcher(None, s1.lower(), s2.lower()).ratio() >= threshold

P5.is_fuzzy_match = _live_fuzzy_match

# ---------------------------------------------------------------------------
# BRANCH INSTRUMENTATION: count how often each disambiguation path is taken, so
# we can tell whether a flat sweep means "robust" or "this code never runs".
# ---------------------------------------------------------------------------
BRANCH = {}

def _count(name):
    BRANCH[name] = BRANCH.get(name, 0) + 1

_orig_get_min_dist = P5.get_min_dist
_orig_get_fallback = P5.get_fallback

def _counting_get_fallback(cands):
    _count('fallback_used')
    return _orig_get_fallback(cands)

P5.get_fallback = _counting_get_fallback

_original_get_candidates = P5.get_candidates_cached

def cached_get_candidates(loc_name):
    key = str(loc_name).strip().lower()
    if key in _disk_cache:
        return copy.deepcopy(_disk_cache[key])
    res = _original_get_candidates(loc_name)
    _disk_cache[key] = copy.deepcopy(res)
    return res

P5.get_candidates_cached = cached_get_candidates

def save_cache():
    with open(CACHE, 'wb') as f:
        pickle.dump(_disk_cache, f)
    print(f'[cache] saved {len(_disk_cache)} toponyms')

# --------------------------------------------------------------------------
# LOAD DATA
# --------------------------------------------------------------------------
def load_master():
    if os.path.exists(MASTER):
        return pd.read_csv(MASTER)
    alt = MASTER.replace('.csv', '.xlsx')
    return pd.read_excel(alt)

df = load_master()
gold = pd.read_excel(GOLD)
print(f'[data] master = {len(df)} rows | gold standard = {len(gold)} entries')

fire = df[pd.to_numeric(df['is_fire'], errors='coerce') == 1].copy()
fire['Date'] = pd.to_datetime(fire['Date'], format='%a %b %d %H:%M:%S %z %Y',
                              utc=True, errors='coerce')
fire = fire.sort_values('Date').reset_index(drop=True)
print(f'[data] active-fire posts to process sequentially: {len(fire)}')

# --------------------------------------------------------------------------
# STAGE 1 — NER ONCE (independent of every swept threshold)
# --------------------------------------------------------------------------
NER_CACHE = os.path.join(REPO_DIR, 'ner_cache.pkl')
if os.path.exists(NER_CACHE):
    with open(NER_CACHE, 'rb') as f:
        fire['Extracted_Locations'] = pickle.load(f)
    print('[ner] loaded cached NER output')
else:
    print('[ner] running GR-NLP-TOOLKIT over all active-fire posts (one pass)...')
    fire['Extracted_Locations'] = fire['Light Cleaned Text'].apply(P5.extract_locations_smart)
    with open(NER_CACHE, 'wb') as f:
        pickle.dump(fire['Extracted_Locations'], f)
    print('[ner] cached')

# --------------------------------------------------------------------------
# ID NORMALISATION
# --------------------------------------------------------------------------
def clean_id(x):
    """Tweet IDs are 19 digits and survive neither float64 nor Excel intact.
    Normalise anything (int, float, '1.23e+18', '123.0') to a plain digit string."""
    s = str(x).strip()
    if s in ('', 'nan', 'None'):
        return ''
    if 'e' in s.lower():
        try:
            s = '%.0f' % float(s)
        except ValueError:
            pass
    if s.endswith('.0'):
        s = s[:-2]
    return s

gold['Tweet_ID'] = gold['Tweet_ID'].apply(clean_id)
fire['Tweet_ID'] = fire['Tweet_ID'].apply(clean_id)

_overlap = len(set(gold['Tweet_ID']) & set(fire['Tweet_ID']))
print(f'[check] gold IDs matched inside active-fire set: {_overlap} / {gold["Tweet_ID"].nunique()} unique')
if _overlap == 0:
    print('[FATAL] No gold tweet appears in the processed set. Check that the master')
    print('        dataset really is the labelled corpus and that is_fire is populated.')
    sys.exit(1)

# --------------------------------------------------------------------------
# STAGE 2 — REPLAY RESOLUTION FOR A GIVEN CONFIGURATION
# --------------------------------------------------------------------------
_orig_resolve = P5.resolve_coordinates

def _instrumented_resolve(row):
    locs = row.get('Extracted_Locations') or []
    n = len(locs)
    if n == 0:
        _count('no_toponyms')
    elif n == 1:
        _count('single_toponym')
    else:
        _count('multi_toponym')
    _count('memory_size_%s' % ('empty' if not P5.spatiotemporal_memory else 'nonempty'))
    return _orig_resolve(row)

P5.resolve_coordinates = _instrumented_resolve


def run_config(overrides):
    """Apply threshold overrides, replay resolution sequentially, evaluate on gold."""
    BRANCH.clear()
    for k, v in BASELINE.items():
        setattr(P5, k, v)
    for k, v in overrides.items():
        setattr(P5, k, v)

    P5.spatiotemporal_memory = []          # reset stateful memory
    work = fire.copy()
    # align_geoparsed_results expects ONE ROW PER TWEET carrying two list columns:
    #   'Extracted_Locations' (NER output) and 'Coordinates' (resolved candidates).
    work['Coordinates'] = work.apply(P5.resolve_coordinates, axis=1)
    geo_df = work[['Tweet_ID', 'Extracted_Locations', 'Coordinates']].copy()
    if geo_df.empty:
        return None

    g = gold.copy()
    g['Tweet_ID'] = g['Tweet_ID'].apply(clean_id)
    geo_df['Tweet_ID'] = geo_df['Tweet_ID'].apply(clean_id)

    if not getattr(run_config, '_checked', False):
        ov = len(set(geo_df['Tweet_ID']) & set(g['Tweet_ID']))
        print(f'[check] predicted tweets overlapping gold: {ov} '
              f'(predicted={geo_df["Tweet_ID"].nunique()}, gold={g["Tweet_ID"].nunique()})')
        if ov == 0:
            print('[FATAL] Predictions and gold share no Tweet_ID. Aborting rather than')
            print('        reporting a spurious 0.00% result.')
            print('  sample predicted:', list(geo_df['Tweet_ID'])[:3])
            print('  sample gold     :', list(g['Tweet_ID'])[:3])
            sys.exit(1)
        run_config._checked = True

    aligned = P5.align_geoparsed_results(geo_df, g)
    evaluated = P5.evaluate_geoparsing(aligned)
    res = summarise(evaluated)
    if not getattr(run_config, '_sane', False):
        if res['ner_operational_acc'] == 0.0:
            print('[FATAL] Baseline operational NER accuracy is 0.00%, which cannot be')
            print('        correct. Aborting rather than producing a meaningless sweep.')
            sys.exit(1)
        run_config._sane = True
    return res


def summarise(ev):
    """Recompute headline metrics from the evaluated frame (robust to print-only eval)."""
    from geopy.distance import geodesic
    tot = len(ev)
    strict = partial = 0
    dists = []
    for _, r in ev.iterrows():
        e = P5.normalize_text(r.get('NER_Extracted', ''))
        a = P5.normalize_text(r.get('Actual_Toponym', ''))
        if e == 'none' and a == '':
            strict += 1; partial += 1
        elif e and a:
            if e == a:
                strict += 1; partial += 1
            elif e in a or a in e:
                partial += 1
        if pd.notna(r.get('Predicted_Lat')) and pd.notna(r.get('Actual_Lat')):
            try:
                dists.append(geodesic((float(r['Predicted_Lat']), float(r['Predicted_Lon'])),
                                      (float(r['Actual_Lat']), float(r['Actual_Lon']))).km)
            except Exception:
                pass
    d = np.array(dists) if dists else np.array([np.nan])
    ok = int((d <= P5.EVALUATION_DISTANCE_KM).sum()) if dists else 0
    return {
        'n_gold': tot,
        'ner_strict_acc':      round(strict / tot, 4) if tot else None,
        'ner_operational_acc': round(partial / tot, 4) if tot else None,
        'n_geocoded':          int(len(dists)),
        'geocoding_op_acc':    round(ok / tot, 4) if tot else None,
        'mean_error_km':       round(float(np.nanmean(d)), 3) if dists else None,
        'median_error_km':     round(float(np.nanmedian(d)), 3) if dists else None,
    }

# --------------------------------------------------------------------------
# STAGE 3 — SWEEP
# --------------------------------------------------------------------------
results = []

print('\n' + '=' * 74)
print('BASELINE CONFIGURATION')
base = run_config({})
save_cache()
print(json.dumps(base, indent=2))

print('\n--- BRANCH USAGE AT BASELINE ---')
for k in sorted(BRANCH):
    print(f'  {k:28s} {BRANCH[k]}')
tot_top = BRANCH.get('single_toponym', 0) + BRANCH.get('multi_toponym', 0)
if tot_top:
    print(f'  posts with >1 toponym       : '
          f'{100*BRANCH.get("multi_toponym",0)/tot_top:.1f}% of posts carrying toponyms')
print('  NOTE: if almost every post has a single toponym and the memory is usually')
print('        empty, the pairwise-distance and memory branches are rarely exercised,')
print('        and a flat sweep reflects low branch coverage rather than robustness.')

THESIS_GEOCODING_ACC = 0.8058
print('\n--- COMPARISON WITH THE PUBLISHED BASELINE ---')
print(f'  reported in thesis : {THESIS_GEOCODING_ACC:.4f}')
print(f'  reproduced here    : {base["geocoding_op_acc"]:.4f}')
gap = base['geocoding_op_acc'] - THESIS_GEOCODING_ACC
print(f'  difference         : {gap:+.4f}')
if abs(gap) > 0.02:
    print('  [WARNING] The run does not reproduce the published figure. Nominatim is a')
    print('            live service and OpenStreetMap content changes over time, so the')
    print('            candidate rankings returned today may differ from those obtained')
    print('            when the original experiments were run. This must be investigated')
    print('            and disclosed before any of these numbers are reported.')

results.append({'parameter': 'BASELINE', 'value': None, 'is_baseline': True,
                'branch_stats': json.dumps(BRANCH), **base})

for param, values in GRID.items():
    print('\n' + '=' * 74)
    print(f'SWEEPING {param}   (baseline = {BASELINE[param]})')
    for v in values:
        r = run_config({param: v})
        if r is None:
            continue
        flag = ' <- baseline' if v == BASELINE[param] else ''
        print(f'  {param} = {v:<6} geocoding_acc = {r["geocoding_op_acc"]:.4f} '
              f'median_err = {r["median_error_km"]}{flag}')
        results.append({'parameter': param, 'value': v,
                        'is_baseline': v == BASELINE[param],
                        'branch_stats': json.dumps(BRANCH), **r})
    save_cache()

out = pd.DataFrame(results)
out.to_csv(OUT_CSV, index=False)
with open(OUT_JSON, 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print('\n' + '=' * 74)
print('SUMMARY — operational geocoding accuracy by parameter value')
for param in GRID:
    sub = out[out.parameter == param]
    if sub.empty:
        continue
    span = sub.geocoding_op_acc.max() - sub.geocoding_op_acc.min()
    print(f'\n{param}: range = {span:.4f}')
    for _, r in sub.iterrows():
        mark = '*' if r.is_baseline else ' '
        print(f'  {mark} {r.value:<8} {r.geocoding_op_acc:.4f}')

print(f'\nSaved {OUT_CSV} and {OUT_JSON}')
print('\nREMINDER: do not adopt a better-scoring threshold found here. The purpose is')
print('to demonstrate stability, not to re-tune on the evaluation set.')
