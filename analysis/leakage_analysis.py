#!/usr/bin/env python3
"""
Near-duplicate / event-leakage analysis (Section 3.4.3).

Reproduces the EXACT split procedure of the original pipeline:
  BINARY (part3):     df.sample(frac=1, random_state=42) -> stratify on is_fire
                      test_size=0.30 then 0.50   (seed 42)
  MULTILABEL (part4): df_fire.sample(frac=1, random_state=789) -> stratify on
                      4-state combo (is_wildland*2 + is_urban)
                      test_size=0.30 then 0.50   (seed 789)

Note: part2_text_cleaning.py already de-duplicates on 'Raw Text'. Residual
duplicates therefore arise only AFTER cleaning, when distinct raw posts collapse
to identical cleaned text (e.g. differing only by URL or @mention).
"""
import pandas as pd, numpy as np, json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

SRC = 'data/master_cleaned_dataset.csv'
THRESHOLDS = [0.95, 0.90, 0.85, 0.80]

df0 = pd.read_csv(SRC, dtype={'Tweet_ID': str})


def assign_splits(frame, seed, strat):
    d = frame.sample(frac=1, random_state=seed).reset_index(drop=True)
    y = strat(d)
    idx = np.arange(len(d))
    tr, tmp = train_test_split(idx, test_size=0.30, random_state=seed, stratify=y)
    va, te = train_test_split(tmp, test_size=0.50, random_state=seed, stratify=y[tmp])
    s = pd.Series('train', index=d.index)
    s.iloc[va] = 'val'
    s.iloc[te] = 'test'
    d['split'] = s.values
    return d


def analyse(d, text_col, label, out_rows):
    texts = d[text_col].fillna('').astype(str).values
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2, sublinear_tf=True)
    X = vec.fit_transform(texts)
    splits = d['split'].values
    n = X.shape[0]

    res = {t: {'pairs': 0, 'cross': 0, 'cross_test': 0, 'docs': set()} for t in THRESHOLDS}
    lo = min(THRESHOLDS)
    for start in range(0, n, 500):
        end = min(start + 500, n)
        S = cosine_similarity(X[start:end], X)
        for li in range(end - start):
            gi = start + li
            row = S[li]; row[gi] = 0.0
            cand = np.where(row >= lo)[0]
            for gj in cand[cand > gi]:
                sim = row[gj]
                for t in THRESHOLDS:
                    if sim >= t:
                        r = res[t]
                        r['pairs'] += 1
                        if splits[gi] != splits[gj]:
                            r['cross'] += 1
                            r['docs'].update([gi, gj])
                            if 'test' in (splits[gi], splits[gj]):
                                r['cross_test'] += 1

    print(f'\n{"="*74}\n{label}   (n = {n})')
    print(d['split'].value_counts().reindex(['train', 'val', 'test']).to_string())
    print(f'\n{"thresh":>7} {"dup pairs":>10} {"cross-split":>12} {"incl. test":>11} '
          f'{"docs":>7} {"% corpus":>9}')
    for t in THRESHOLDS:
        r = res[t]
        pct = 100 * len(r['docs']) / n
        print(f'{t:>7.2f} {r["pairs"]:>10} {r["cross"]:>12} {r["cross_test"]:>11} '
              f'{len(r["docs"]):>7} {pct:>8.2f}%')
        out_rows.append({'task': label, 'threshold': t, 'n_docs': n,
                         'duplicate_pairs': r['pairs'], 'cross_split_pairs': r['cross'],
                         'pairs_involving_test': r['cross_test'],
                         'docs_affected': len(r['docs']), 'pct_of_corpus': round(pct, 2)})

    ex = d[d.duplicated(subset=[text_col], keep=False)]
    spanning = 0
    if len(ex):
        spanning = int((ex.groupby(text_col)['split'].nunique() > 1).sum())
    print(f'\nExact duplicates after cleaning: {len(ex)} rows / '
          f'{ex[text_col].nunique() if len(ex) else 0} groups; spanning >1 split: {spanning}')
    return {'n': n, 'exact_rows': int(len(ex)),
            'exact_groups': int(ex[text_col].nunique()) if len(ex) else 0,
            'exact_groups_spanning_splits': spanning}


rows, meta = [], {}

binary = assign_splits(df0, 42, lambda d: d['is_fire'].values)
meta['binary'] = analyse(binary, 'Advanced Cleaned Text',
                         'BINARY TASK (seed 42, stratify=is_fire)', rows)

fire = df0[df0.is_fire == 1].copy()
multi = assign_splits(fire, 789,
                      lambda d: (d['is_wildland'].astype(int) * 2 + d['is_urban'].astype(int)).values)
meta['multilabel'] = analyse(multi, 'Light Cleaned Text',
                             'MULTILABEL TASK (seed 789, stratify=4-state combo)', rows)

pd.DataFrame(rows).to_csv('data/leakage_analysis_summary.csv', index=False)
with open('data/leakage_analysis_results.json', 'w') as f:
    json.dump({'meta': meta, 'rows': rows}, f, indent=2, ensure_ascii=False)
print('\nSaved leakage_analysis_summary.csv / .json')
