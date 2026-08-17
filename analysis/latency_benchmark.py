#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Latency benchmark for the fine-tuned encoders (for the Section 4.4 trade-off figure).

Measures per-post inference latency for GreekBERT (binary) and XLM-RoBERTa
(multilabel) on the SAME held-out test sets, loaded exactly as the operational
pipeline loads them (Hugging Face, max_length=128). Reports single-item latency
(batch size 1, the realistic streaming-inference case) and, for reference,
batched throughput.

The hardware is recorded in the output so the figure caption can state it. Run
this on the SAME Colab GPU used for the rest of the work so the comparison with
the LLM APIs is on a documented footing.

RUN (Colab)
    !pip install transformers torch pandas openpyxl scikit-learn -q
    # upload master_cleaned_dataset.xlsx, then:
    !python latency_benchmark.py
"""
import time, json, platform
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split

BIN_MODEL   = 'mariossmrs/greek-bert-fire-detection-binary-classification'
MULTI_MODEL = 'mariossmrs/greek-xlm-roberta-fire-type-multilabel-classification'
DATA = 'master_cleaned_dataset.xlsx'
MAXLEN = 128

device = 'cuda' if torch.cuda.is_available() else 'cpu'
gpu = torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU only'
print(f'[hw] device={device} | {gpu}')

df = pd.read_excel(DATA, sheet_name='master_cleaned_dataset')

# same splits/text columns as the pipeline
def binary_test():
    d = df.sample(frac=1, random_state=42).reset_index(drop=True)
    X = d['Advanced Cleaned Text'].astype(str).values      # GreekBERT input
    y = d['is_fire'].astype(int).values
    _, Xtmp, _, ytmp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    _, Xte, _, _ = train_test_split(Xtmp, ytmp, test_size=0.50, random_state=42, stratify=ytmp)
    return Xte

def multi_test():
    d = df[df.is_fire == 1].copy().sample(frac=1, random_state=789).reset_index(drop=True)
    X = d['Light Cleaned Text'].astype(str).values         # XLM-RoBERTa input
    y = d[['is_wildland', 'is_urban']].values.astype(int)
    s = y[:, 0] * 2 + y[:, 1]
    _, Xtmp, _, _, _, stmp = train_test_split(X, y, s, test_size=0.30, random_state=789, stratify=s)
    _, Xte, _, _, _, _ = train_test_split(Xtmp, np.zeros((len(Xtmp), 2)), stmp,
                                          test_size=0.50, random_state=789, stratify=stmp)
    return Xte

def bench(model_id, texts, label):
    print(f'\n[{label}] loading {model_id}')
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_id).to(device).eval()

    # warm-up (excluded from timing)
    with torch.no_grad():
        enc = tok(list(texts[:8]), padding='max_length', truncation=True,
                  max_length=MAXLEN, return_tensors='pt').to(device)
        _ = mdl(**enc)
    if device == 'cuda':
        torch.cuda.synchronize()

    # single-item latency (batch size 1 — realistic streaming case)
    per_item = []
    with torch.no_grad():
        for t in texts:
            t0 = time.perf_counter()
            enc = tok([t], padding='max_length', truncation=True,
                      max_length=MAXLEN, return_tensors='pt').to(device)
            _ = mdl(**enc)
            if device == 'cuda':
                torch.cuda.synchronize()
            per_item.append(time.perf_counter() - t0)
    per_item = np.array(per_item)

    # batched throughput (reference)
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            batch = list(texts[i:i+32])
            enc = tok(batch, padding='max_length', truncation=True,
                      max_length=MAXLEN, return_tensors='pt').to(device)
            _ = mdl(**enc)
            if device == 'cuda':
                torch.cuda.synchronize()
    batch_total = time.perf_counter() - t0

    res = {
        'model': model_id, 'n': int(len(texts)),
        'single_item_latency_s_mean': round(float(per_item.mean()), 4),
        'single_item_latency_s_median': round(float(np.median(per_item)), 4),
        'batched_latency_s_per_item': round(batch_total / len(texts), 4),
    }
    print(f'  single-item: mean {res["single_item_latency_s_mean"]}s, '
          f'median {res["single_item_latency_s_median"]}s')
    print(f'  batched(32): {res["batched_latency_s_per_item"]}s/item')
    return res

out = {
    'hardware': {'device': device, 'gpu': gpu, 'torch': torch.__version__,
                 'platform': platform.platform()},
    'binary_greekbert': bench(BIN_MODEL, binary_test(), 'binary/GreekBERT'),
    'multilabel_xlmr': bench(MULTI_MODEL, multi_test(), 'multilabel/XLM-RoBERTa'),
}
json.dump(out, open('latency_results.json', 'w'), indent=2, ensure_ascii=False)
print('\nSaved latency_results.json')
print('\nReport the SINGLE-ITEM latency in the paper as the realtime/streaming figure,')
print('and note the hardware. The LLM API latency (~1.3 s) is remote and includes network.')
