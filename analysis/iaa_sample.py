#!/usr/bin/env python3
"""
Section 3.2.3 — Step 1: draw the inter-annotator agreement (IAA) subsets and
produce BLIND worksheets for the second annotator.

Design rationale
----------------
BINARY: 400 posts, stratified by is_fire in the corpus proportion (2177/1100).
  400 gives a 95% CI half-width of roughly +/-0.05 on kappa in the typical range,
  which is adequate for a reliability statement and feasible to label by hand.

MULTILABEL: 300 active-fire posts, stratified over the 4-state combo but with the
  rare 'mixed' class deliberately OVER-SAMPLED (target ~25% of the subset versus
  7.8% in the corpus). Rare-category agreement is what reviewers question, and
  proportional sampling would yield too few mixed items to estimate per-label
  kappa stably. Weighting is recorded so corpus-level estimates can be recovered.

GEOPARSING: all 147 gold items double-annotated (small enough for full coverage).

Blindness
---------
Worksheets contain NO original labels and are shuffled, so the second annotator
cannot anchor on the first annotator's decisions. The key file mapping row -> Tweet_ID
-> original label is written separately and must NOT be opened before labelling.
"""
import pandas as pd, numpy as np, json

SRC = 'data/master_cleaned_dataset.csv'
GEO = 'data/ground_truth_geoparsing.xlsx'
OUT = 'data/'
SEED = 20260810                      # fixed for reproducibility of the sample

rng = np.random.RandomState(SEED)
df = pd.read_csv(SRC, dtype={'Tweet_ID': str})

def combo(r):
    if r.is_fire == 0:
        return 'nonfire'
    w, u = int(r.is_wildland), int(r.is_urban)
    return {(1,0):'wildland', (0,1):'urban', (1,1):'mixed', (0,0):'unidentified'}[(w,u)]

df['combo'] = df.apply(combo, axis=1)

# ---------------------------------------------------------------- BINARY
N_BIN = 400
prop = df['is_fire'].value_counts(normalize=True)
n_per = {0: int(round(N_BIN * prop[0])), 1: N_BIN - int(round(N_BIN * prop[0]))}
parts = [df[df.is_fire == k].sample(n=n_per[k], random_state=SEED) for k in (0, 1)]
bin_sample = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)

bin_key = bin_sample[['Tweet_ID', 'is_fire']].copy()
bin_key.insert(0, 'row', range(1, len(bin_key) + 1))
bin_key.rename(columns={'is_fire': 'annotator1_is_fire'}, inplace=True)

bin_sheet = pd.DataFrame({
    'row': range(1, len(bin_sample) + 1),
    'Tweet_ID': bin_sample['Tweet_ID'],
    'Raw Text': bin_sample['Raw Text'],
    'is_fire_A2': '',                       # <- annotator 2 fills 0 or 1
    'uncertain (x)': '',
    'note': '',
})

# ------------------------------------------------------------ MULTILABEL
N_MULTI = 300
fire = df[df.is_fire == 1]
TARGET = {'mixed': 0.25, 'wildland': 0.25, 'urban': 0.25, 'unidentified': 0.25}
picks = []
for c, frac in TARGET.items():
    pool = fire[fire.combo == c]
    k = min(int(round(N_MULTI * frac)), len(pool))
    picks.append(pool.sample(n=k, random_state=SEED))
multi_sample = pd.concat(picks).sample(frac=1, random_state=SEED).reset_index(drop=True)

multi_key = multi_sample[['Tweet_ID', 'is_wildland', 'is_urban', 'combo']].copy()
multi_key.insert(0, 'row', range(1, len(multi_key) + 1))
multi_key.rename(columns={'is_wildland': 'annotator1_is_wildland',
                          'is_urban': 'annotator1_is_urban',
                          'combo': 'annotator1_combo'}, inplace=True)

multi_sheet = pd.DataFrame({
    'row': range(1, len(multi_sample) + 1),
    'Tweet_ID': multi_sample['Tweet_ID'],
    'Raw Text': multi_sample['Raw Text'],
    'is_wildland_A2': '',                   # <- annotator 2 fills 0 or 1
    'is_urban_A2': '',                      # <- annotator 2 fills 0 or 1
    'uncertain (x)': '',
    'note': '',
})

# ------------------------------------------------------------- GEOPARSING
geo = pd.read_excel(GEO, sheet_name='golden_dataset_fixed')
geo_shuf = geo.sample(frac=1, random_state=SEED).reset_index(drop=True)

geo_key = geo_shuf[['Tweet_ID', 'Actual_Toponym', 'Actual_Lat', 'Actual_Lon']].copy()
geo_key.insert(0, 'row', range(1, len(geo_key) + 1))
geo_key.rename(columns={'Actual_Toponym': 'annotator1_toponym',
                        'Actual_Lat': 'annotator1_lat',
                        'Actual_Lon': 'annotator1_lon'}, inplace=True)

geo_sheet = pd.DataFrame({
    'row': range(1, len(geo_shuf) + 1),
    'Tweet_ID': geo_shuf['Tweet_ID'],
    'Raw_Text': geo_shuf['Raw_Text'],
    'toponym_A2': '',                       # <- true toponym, or NONE
    'lat_A2': '',
    'lon_A2': '',
    'note': '',
})

# ------------------------------------------------------------------ WRITE
with pd.ExcelWriter(OUT + 'IAA_worksheet_ANNOTATOR2.xlsx', engine='openpyxl') as w:
    bin_sheet.to_excel(w, sheet_name='1_binary', index=False)
    multi_sheet.to_excel(w, sheet_name='2_multilabel', index=False)
    geo_sheet.to_excel(w, sheet_name='3_geoparsing', index=False)

with pd.ExcelWriter(OUT + 'IAA_key_DO_NOT_OPEN_BEFORE.xlsx', engine='openpyxl') as w:
    bin_key.to_excel(w, sheet_name='1_binary_key', index=False)
    multi_key.to_excel(w, sheet_name='2_multilabel_key', index=False)
    geo_key.to_excel(w, sheet_name='3_geoparsing_key', index=False)

meta = {
    'seed': SEED,
    'binary': {'n': len(bin_sheet),
               'strata': {str(k): int(v) for k, v in n_per.items()},
               'sampling': 'proportional to corpus class balance'},
    'multilabel': {'n': len(multi_sheet),
                   'per_class': multi_sample['combo'].value_counts().to_dict(),
                   'corpus_per_class': fire['combo'].value_counts().to_dict(),
                   'sampling': 'equal allocation; mixed deliberately over-sampled'},
    'geoparsing': {'n': len(geo_sheet), 'sampling': 'full gold set'},
}
with open(OUT + 'IAA_sampling_metadata.json', 'w') as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

print('BINARY  :', len(bin_sheet), 'posts |', n_per)
print('MULTI   :', len(multi_sheet), 'posts |', multi_sample['combo'].value_counts().to_dict())
print('  corpus proportions :', fire['combo'].value_counts(normalize=True).round(3).to_dict())
print('GEO     :', len(geo_sheet), 'items (full gold set)')
print('\nWritten: IAA_worksheet_ANNOTATOR2.xlsx, IAA_key_DO_NOT_OPEN_BEFORE.xlsx, IAA_sampling_metadata.json')
