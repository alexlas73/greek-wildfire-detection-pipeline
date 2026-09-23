#!/usr/bin/env python3
"""
Section 3.2.3 — Step 2: compute inter-annotator agreement between
annotator 1 (M.S., original thesis labels) and annotator 2 (A.L.).

Outputs: Cohen's kappa (binary), per-label kappa + Krippendorff's alpha
(multilabel), and toponym/distance agreement (geoparsing).
"""
import pandas as pd, numpy as np, json, unicodedata, re
from sklearn.metrics import cohen_kappa_score, confusion_matrix

SEED = 20260810
SRC = 'data/master_cleaned_dataset.csv'
GEO = 'data/ground_truth_geoparsing.xlsx'
WS  = 'data/IAA_worksheet_ANNOTATOR2.xlsx'
OUT = 'data/'

df = pd.read_csv(SRC, dtype={'Tweet_ID': str})
geo = pd.read_excel(GEO, sheet_name='golden_dataset_fixed')
results = {}

# NOTE: Excel stores the 19-digit Tweet_ID as a float64, which silently truncates
# the final digits. Merging on Tweet_ID is therefore unreliable. The worksheets were
# generated deterministically (seed below), so annotator-1 labels are recovered by
# regenerating the identical sample and aligning on ROW ORDER, which is verified
# against the Raw Text column before use.

def regen_binary():
    prop = df['is_fire'].value_counts(normalize=True)
    n0 = int(round(400 * prop[0])); n_per = {0: n0, 1: 400 - n0}
    parts = [df[df.is_fire == k].sample(n=n_per[k], random_state=SEED) for k in (0, 1)]
    return pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)

def regen_multi():
    def combo(r):
        w, u = int(r.is_wildland), int(r.is_urban)
        return {(1,0):'wildland',(0,1):'urban',(1,1):'mixed',(0,0):'unidentified'}[(w,u)]
    fire = df[df.is_fire == 1].copy()
    fire['combo'] = fire.apply(combo, axis=1)
    picks = []
    for c in ['mixed', 'wildland', 'urban', 'unidentified']:
        pool = fire[fire.combo == c]
        picks.append(pool.sample(n=min(75, len(pool)), random_state=SEED))
    return pd.concat(picks).sample(frac=1, random_state=SEED).reset_index(drop=True)

def regen_geo():
    return geo.sample(frac=1, random_state=SEED).reset_index(drop=True)

def check(a, b, col_a, col_b, name):
    m = (a[col_a].astype(str).values == b[col_b].astype(str).values).mean()
    print(f'[align] {name}: row-order text match = {m:.4f}')
    assert m > 0.99, f'alignment failed for {name}'

# ---------------------------------------------------------------- helpers
def kappa_ci(y1, y2, n_boot=5000, seed=1):
    """Bootstrap 95% CI for Cohen's kappa."""
    rng = np.random.RandomState(seed)
    y1, y2 = np.asarray(y1), np.asarray(y2)
    n = len(y1)
    stats = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y1[idx])) < 2 or len(np.unique(y2[idx])) < 2:
            continue
        stats.append(cohen_kappa_score(y1[idx], y2[idx]))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))

def landis_koch(k):
    if k < 0:    return 'poor'
    if k <= .20: return 'slight'
    if k <= .40: return 'fair'
    if k <= .60: return 'moderate'
    if k <= .80: return 'substantial'
    return 'almost perfect'

def krippendorff_alpha_nominal(data):
    """
    data: array (n_coders x n_items), np.nan for missing.
    Nominal-level Krippendorff's alpha via coincidence matrix.
    """
    data = np.asarray(data, dtype=float)
    vals = np.unique(data[~np.isnan(data)])
    v2i = {v: i for i, v in enumerate(vals)}
    V = len(vals)
    coinc = np.zeros((V, V))
    for j in range(data.shape[1]):
        col = data[:, j]
        col = col[~np.isnan(col)]
        m = len(col)
        if m < 2:
            continue
        for a in col:
            for b in col:
                if a is not b:
                    pass
        # pairwise within unit, normalised by (m-1)
        for ia in range(m):
            for ib in range(m):
                if ia != ib:
                    coinc[v2i[col[ia]], v2i[col[ib]]] += 1.0 / (m - 1)
    n_total = coinc.sum()
    if n_total == 0:
        return float('nan')
    nc = coinc.sum(axis=1)
    Do = sum(coinc[i, j] for i in range(V) for j in range(V) if i != j)
    De = sum(nc[i] * nc[j] for i in range(V) for j in range(V) if i != j) / (n_total - 1)
    return float(1 - (Do / De)) if De else float('nan')

# ================================================================ BINARY
ws_b = pd.read_excel(WS, sheet_name='1_binary')
src_b = regen_binary()
check(src_b, ws_b, 'Raw Text', 'Raw Text', 'binary')
mb = ws_b.copy()
mb['is_fire'] = src_b['is_fire'].values
mb['Tweet_ID'] = src_b['Tweet_ID'].values
a1 = mb['is_fire'].astype(int).values
a2 = mb['is_fire_A2'].astype(int).values

k = cohen_kappa_score(a1, a2)
lo, hi = kappa_ci(a1, a2)
agree = (a1 == a2).mean()
cm = confusion_matrix(a1, a2, labels=[0, 1])
alpha_b = krippendorff_alpha_nominal(np.vstack([a1.astype(float), a2.astype(float)]))

print('=' * 70)
print(f'BINARY TASK  (n = {len(mb)})')
print(f"  Cohen's kappa      : {k:.4f}   95% CI [{lo:.4f}, {hi:.4f}]  ({landis_koch(k)})")
print(f"  Krippendorff alpha : {alpha_b:.4f}")
print(f'  Raw agreement      : {agree:.4f}  ({int((a1==a2).sum())}/{len(mb)})')
print(f'  Disagreements      : {int((a1!=a2).sum())}')
print('  Confusion matrix (rows = A1, cols = A2), labels [0,1]:')
print('   ', cm.tolist())
results['binary'] = {'n': int(len(mb)), 'kappa': round(float(k), 4),
                     'kappa_ci95': [round(lo, 4), round(hi, 4)],
                     'krippendorff_alpha': round(float(alpha_b), 4),
                     'raw_agreement': round(float(agree), 4),
                     'n_disagreements': int((a1 != a2).sum()),
                     'confusion_matrix_A1_rows_A2_cols': cm.tolist(),
                     'interpretation': landis_koch(k)}

mb['disagree'] = (a1 != a2)
mb[mb.disagree][['row', 'Tweet_ID', 'Raw Text', 'is_fire', 'is_fire_A2', 'note']] \
    .rename(columns={'is_fire': 'A1_is_fire', 'is_fire_A2': 'A2_is_fire'}) \
    .to_excel(OUT + 'IAA_disagreements_binary.xlsx', index=False)

# ============================================================ MULTILABEL
ws_m = pd.read_excel(WS, sheet_name='2_multilabel')
src_m = regen_multi()
check(src_m, ws_m, 'Raw Text', 'Raw Text', 'multilabel')
mm = ws_m.copy()
mm['is_wildland'] = src_m['is_wildland'].values
mm['is_urban'] = src_m['is_urban'].values
mm['Tweet_ID'] = src_m['Tweet_ID'].values
print('\n' + '=' * 70)
print(f'MULTILABEL TASK  (n = {len(mm)})')

results['multilabel'] = {'n': int(len(mm)), 'per_label': {}}
per_label_alpha = []
for lab, c1, c2 in [('is_wildland', 'is_wildland', 'is_wildland_A2'),
                    ('is_urban', 'is_urban', 'is_urban_A2')]:
    x1 = mm[c1].astype(int).values
    x2 = mm[c2].astype(int).values
    kk = cohen_kappa_score(x1, x2)
    l, h = kappa_ci(x1, x2)
    al = krippendorff_alpha_nominal(np.vstack([x1.astype(float), x2.astype(float)]))
    per_label_alpha.append(al)
    print(f'  {lab:12s} kappa = {kk:.4f}  95% CI [{l:.4f}, {h:.4f}]  '
          f'alpha = {al:.4f}  raw = {(x1==x2).mean():.4f}  ({landis_koch(kk)})')
    results['multilabel']['per_label'][lab] = {
        'kappa': round(float(kk), 4), 'kappa_ci95': [round(l, 4), round(h, 4)],
        'krippendorff_alpha': round(float(al), 4),
        'raw_agreement': round(float((x1 == x2).mean()), 4),
        'n_disagreements': int((x1 != x2).sum()),
        'interpretation': landis_koch(kk)}

# 4-state combined
def combo4(w, u):
    return np.array([{(0,0):0,(1,0):1,(0,1):2,(1,1):3}[(int(a), int(b))] for a, b in zip(w, u)])

c1 = combo4(mm['is_wildland'], mm['is_urban'])
c2 = combo4(mm['is_wildland_A2'], mm['is_urban_A2'])
k4 = cohen_kappa_score(c1, c2)
l4, h4 = kappa_ci(c1, c2)
a4 = krippendorff_alpha_nominal(np.vstack([c1.astype(float), c2.astype(float)]))
exact = (c1 == c2).mean()
names = ['unidentified', 'wildland', 'urban', 'mixed']
print(f'\n  4-state combined: kappa = {k4:.4f}  95% CI [{l4:.4f}, {h4:.4f}]  '
      f'alpha = {a4:.4f}  ({landis_koch(k4)})')
print(f'  Exact-match ratio (both labels correct): {exact:.4f}')
cm4 = confusion_matrix(c1, c2, labels=[0, 1, 2, 3])
print('  4-state confusion (rows A1, cols A2):', names)
for i, r in enumerate(cm4):
    print(f'    {names[i]:14s} {r.tolist()}')

# per-class agreement
print('\n  Per-class agreement (A1 class -> proportion matched by A2):')
per_class = {}
for i, nm in enumerate(names):
    mask = c1 == i
    if mask.sum():
        pc = float((c2[mask] == i).mean())
        per_class[nm] = {'n': int(mask.sum()), 'matched': round(pc, 4)}
        print(f'    {nm:14s} n={int(mask.sum()):3d}  matched={pc:.4f}')

results['multilabel']['four_state'] = {
    'kappa': round(float(k4), 4), 'kappa_ci95': [round(l4, 4), round(h4, 4)],
    'krippendorff_alpha': round(float(a4), 4),
    'exact_match_ratio': round(float(exact), 4),
    'confusion_matrix': cm4.tolist(), 'class_order': names,
    'per_class': per_class, 'interpretation': landis_koch(k4)}
results['multilabel']['mean_per_label_alpha'] = round(float(np.mean(per_label_alpha)), 4)

mm['disagree'] = (c1 != c2)
mm[mm.disagree][['row', 'Tweet_ID', 'Raw Text', 'is_wildland', 'is_urban',
                 'is_wildland_A2', 'is_urban_A2', 'note']] \
    .to_excel(OUT + 'IAA_disagreements_multilabel.xlsx', index=False)

# ============================================================ GEOPARSING
ws_g = pd.read_excel(WS, sheet_name='3_geoparsing')
src_g = regen_geo()
check(src_g, ws_g, 'Raw_Text', 'Raw_Text', 'geoparsing')
mg = ws_g.copy()
for c in ['Actual_Toponym', 'Actual_Lat', 'Actual_Lon']:
    mg[c] = src_g[c].values
mg['Tweet_ID'] = src_g['Tweet_ID'].values

def norm(s):
    if pd.isna(s):
        return ''
    s = str(s).strip().lower()
    s = ''.join(ch for ch in unicodedata.normalize('NFD', s)
                if unicodedata.category(ch) != 'Mn')
    s = re.sub(r'[^\w\s]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

def hav(a, b, c, d):
    R = 6371.0088
    p1, p2 = np.radians(a), np.radians(c)
    dp, dl = np.radians(c - a), np.radians(d - b)
    x = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(x))

t1 = mg['Actual_Toponym'].apply(norm)
t2 = mg['toponym_A2'].apply(norm)
none1 = t1.isin(['', 'none'])
none2 = t2.isin(['', 'none'])

exact_t = (t1 == t2)
# containment counts as agreement (e.g. "Μηλίτσα" vs "Μηλίτσα Μεσσηνίας")
contain = [(x in y or y in x) if (x and y) else False for x, y in zip(t1, t2)]
contain = np.array(contain) | exact_t.values

print('\n' + '=' * 70)
print(f'GEOPARSING TASK  (n = {len(mg)})')
print(f'  A1 said NONE: {int(none1.sum())} | A2 said NONE: {int(none2.sum())}')
print(f'  NONE-decision agreement: {float((none1==none2).mean()):.4f}')
k_none = cohen_kappa_score(none1.astype(int), none2.astype(int))
print(f"  Cohen's kappa on locatable/not decision: {k_none:.4f} ({landis_koch(k_none)})")
print(f'  Exact normalised toponym match     : {float(exact_t.mean()):.4f}')
print(f'  Match allowing containment         : {float(contain.mean()):.4f}')

both = (~none1) & (~none2) & mg['Actual_Lat'].notna() & mg['lat_A2'].notna()
d = hav(mg.loc[both, 'Actual_Lat'].astype(float).values,
        mg.loc[both, 'Actual_Lon'].astype(float).values,
        mg.loc[both, 'lat_A2'].astype(float).values,
        mg.loc[both, 'lon_A2'].astype(float).values)
print(f'\n  Coordinate comparison on n = {int(both.sum())} items where both gave coordinates:')
for tol in (1, 5, 10):
    print(f'    within {tol:2d} km : {float((d<=tol).mean()):.4f}  ({int((d<=tol).sum())}/{len(d)})')
print(f'    median distance : {float(np.median(d)):.3f} km')
print(f'    mean distance   : {float(np.mean(d)):.3f} km')

results['geoparsing'] = {
    'n': int(len(mg)),
    'n_none_A1': int(none1.sum()), 'n_none_A2': int(none2.sum()),
    'none_decision_agreement': round(float((none1 == none2).mean()), 4),
    'kappa_locatable_decision': round(float(k_none), 4),
    'exact_normalised_toponym_match': round(float(exact_t.mean()), 4),
    'toponym_match_with_containment': round(float(contain.mean()), 4),
    'n_both_coordinates': int(both.sum()),
    'within_1km': round(float((d <= 1).mean()), 4),
    'within_5km': round(float((d <= 5).mean()), 4),
    'within_10km': round(float((d <= 10).mean()), 4),
    'median_distance_km': round(float(np.median(d)), 3),
    'mean_distance_km': round(float(np.mean(d)), 3)}

gg = mg.copy()
gg['dist_km'] = np.nan
gg.loc[both, 'dist_km'] = d
gg['toponym_mismatch'] = ~contain
gg[gg.toponym_mismatch | (gg.dist_km > 10)][
    ['row', 'Tweet_ID', 'Raw_Text', 'Actual_Toponym', 'toponym_A2',
     'Actual_Lat', 'Actual_Lon', 'lat_A2', 'lon_A2', 'dist_km', 'note']] \
    .to_excel(OUT + 'IAA_disagreements_geoparsing.xlsx', index=False)

with open(OUT + 'IAA_results.json', 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print('\nSaved IAA_results.json and three disagreement worksheets.')
