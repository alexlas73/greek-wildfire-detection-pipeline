#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Headless Varnavas simulation runner (for the Section 4.7 map figure).

Reuses the project's own functions to process the raw Varnavas tweets end to end:
  raw tweets -> text cleaning -> GreekBERT binary -> XLM-RoBERTa multilabel
             -> geoparsing (part5.run_mapping) -> DBSCAN clustering + map

Produces:
  fire_events_geoparsed.csv   (geoparsed fire events with coordinates)
  fire_events_map.html        (interactive Folium map)
  simulation_clustered.csv    (events + DBSCAN cluster_id, added by this script)

RUN (Colab), from inside the repo folder:
  !pip install gr-nlp-toolkit geopy folium scikit-learn emoji spacy transformers torch datasets pandas -q
  !python -m spacy download el_core_news_lg -q
  !python run_simulation.py
"""
import os, sys
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

DATA = os.path.join(REPO, 'data', 'varnavas_fire.csv')
BIN_MODEL   = 'mariossmrs/greek-bert-fire-detection-binary-classification'
MULTI_MODEL = 'mariossmrs/greek-xlm-roberta-fire-type-multilabel-classification'

from part2_text_cleaning import run_text_cleaning
import part5_geoparsing_and_mapping as P5

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from datasets import Dataset

# ----------------------------------------------------------------- load + clean
raw = pd.read_csv(DATA)
print(f'[data] {len(raw)} raw tweets')
cleaned = run_text_cleaning(raw.copy())

# ----------------------------------------------------------------- binary inference
def tok_fn(tokenizer):
    def f(ex):
        return tokenizer(ex['text'], padding='max_length', truncation=True, max_length=128)
    return f

def binary(df, thr=0.30):
    tk = AutoTokenizer.from_pretrained(BIN_MODEL)
    md = AutoModelForSequenceClassification.from_pretrained(BIN_MODEL)
    ds = Dataset.from_dict({'text': df['Advanced Cleaned Text'].astype(str).tolist()}).map(tok_fn(tk), batched=True)
    logits, _, _ = Trainer(model=md).predict(ds)
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
    df['Fire Probability'] = np.round(probs, 3)
    df['Predicted Label'] = (probs >= thr).astype(int)
    print(f'[binary] fire-related: {int(df["Predicted Label"].sum())}/{len(df)}')
    return df

def multilabel(df):
    fire = df[df['Predicted Label'] == 1]
    df['is_wildland_pred'] = 0; df['is_urban_pred'] = 0
    if fire.empty:
        return df
    tk = AutoTokenizer.from_pretrained(MULTI_MODEL)
    md = AutoModelForSequenceClassification.from_pretrained(MULTI_MODEL)
    ds = Dataset.from_dict({'text': fire['Light Cleaned Text'].astype(str).tolist()}).map(tok_fn(tk), batched=True)
    logits, _, _ = Trainer(model=md).predict(ds)
    p = torch.sigmoid(torch.tensor(logits)).numpy()
    df.loc[df['Predicted Label'] == 1, 'is_wildland_pred'] = (p[:, 0] > 0.5).astype(int)
    df.loc[df['Predicted Label'] == 1, 'is_urban_pred'] = (p[:, 1] > 0.5).astype(int)
    return df

# ---- checkpoint: classification (binary + multilabel) ----
CKPT_CLS = 'simulation_predicted_tweets.csv'
if os.path.exists(CKPT_CLS):
    print(f'[resume] loading cached classification from {CKPT_CLS}')
    cleaned = pd.read_csv(CKPT_CLS)
    if 'Predicted Label' not in cleaned.columns:
        print('[resume] cached file missing predictions; recomputing')
        cleaned = multilabel(binary(cleaned))
        cleaned.to_csv(CKPT_CLS, index=False)
else:
    cleaned = binary(cleaned)
    cleaned = multilabel(cleaned)
    cleaned.to_csv(CKPT_CLS, index=False)
    print(f'[checkpoint] classification saved to {CKPT_CLS}')

# ---- checkpoint: geoparsing + map ----
# run_mapping saves fire_events_geoparsed.csv and fire_events_map.html; if the
# geoparsed CSV already exists (and a nominatim_cache.pkl is present) the geocoding
# stage inside run_mapping reuses the cache, so re-running is fast.
if os.path.exists('fire_events_geoparsed.csv'):
    print('[resume] fire_events_geoparsed.csv already exists; skipping geoparse/map stage')
else:
    P5.run_mapping(cleaned, target_label_col='Predicted Label')

# ----------------------------------------------------------------- re-derive clusters into the CSV
# run_mapping computes DBSCAN internally for the map but does not always write cluster ids to
# the CSV; we recompute them here with the SAME parameters so the figure can colour clusters.
from sklearn.cluster import DBSCAN
if os.path.exists('fire_events_geoparsed.csv'):
    ev = pd.read_csv('fire_events_geoparsed.csv')
    latc = next((c for c in ev.columns if 'lat' in c.lower()), None)
    lonc = next((c for c in ev.columns if 'lon' in c.lower()), None)
    if latc and lonc:
        pts = ev[[latc, lonc]].dropna()
        if len(pts) >= 2:
            db = DBSCAN(eps=15.0/6371.0088, min_samples=2, algorithm='ball_tree', metric='haversine')
            labels = db.fit_predict(np.radians(pts.values))
            ev.loc[pts.index, 'cluster_id'] = labels
            ev.to_csv('simulation_clustered.csv', index=False)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            print(f'[cluster] {n_clusters} clusters over {len(pts)} located events '
                  f'({(labels==-1).sum()} isolated); saved simulation_clustered.csv')
        else:
            print('[cluster] too few located points to cluster')
    else:
        print('[cluster] no lat/lon columns found in fire_events_geoparsed.csv:', list(ev.columns))

print('\nDONE. Upload simulation_clustered.csv (or fire_events_geoparsed.csv) for the figure.')
