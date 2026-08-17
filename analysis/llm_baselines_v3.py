#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM baselines for binary + multilabel classification (Section 3.4.2 / 4.4).

Compares two large language models against the fine-tuned encoders on the SAME
held-out test sets, under zero-shot and few-shot prompting, and records accuracy,
the same metrics used elsewhere in the paper (F2 for binary; 4-state macro-F1 and
exact-match for multilabel), plus latency and per-item cost.

MODELS
  proprietary : Claude Sonnet     (Anthropic API)   -> upper-bound reference
  open        : Llama 3.3 70B     (DeepInfra API)   -> self-hostable reference

The test sets are regenerated with the EXACT splits used by part3/part4:
  binary     : df.sample(frac=1, seed 42)  -> stratify is_fire, 70/15/15  -> 492 test posts
  multilabel : fire.sample(frac=1, seed 789) -> stratify 4-state, 70/15/15 -> 165 test posts
Few-shot examples are drawn from the TRAINING split only, so no test post is ever
shown to the model as an example.

------------------------------------------------------------------------------
SETUP (do this once, in Colab)
------------------------------------------------------------------------------
  !pip install anthropic openai scikit-learn pandas openpyxl -q

  import os
  os.environ['ANTHROPIC_API_KEY'] = 'sk-ant-...'     # from console.anthropic.com
  os.environ['DEEPINFRA_API_KEY']  = '...'            # from deepinfra.com

  # upload master_cleaned_dataset.xlsx into the working folder, then:
  !python llm_baselines.py
------------------------------------------------------------------------------
"""
import os, sys, time, json, re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_recall_fscore_support, fbeta_score,
                             accuracy_score, f1_score, classification_report)

# ----------------------------------------------------------------- CONFIG
DATA = 'master_cleaned_dataset.xlsx'
TEXT_COL = 'Light Cleaned Text'          # cased text: best for a general LLM
N_FEWSHOT = 8                            # examples per prompt (from TRAIN only)
SEED_FEWSHOT = 2026

CLAUDE_MODEL = 'claude-sonnet-4-5'       # update to the current Sonnet if needed
LLAMA_MODEL  = 'meta-llama/Llama-3.3-70B-Instruct'   # via DeepInfra (OpenAI-compatible)

# Approx pricing (USD per 1M tokens) for the per-item cost estimate. UPDATE to the
# current published rates before reporting; these are only for an order-of-magnitude
# operational comparison, not an exact bill.
PRICING = {
    CLAUDE_MODEL: {'in': 3.00,  'out': 15.00},
    LLAMA_MODEL:  {'in': 0.35,  'out': 0.40},
}

RUN = {'claude_zero': True, 'claude_few': True, 'llama_zero': True, 'llama_few': True}

# ---------------------------------------------------------------------------
# ROBUSTNESS: automatic retry with exponential back-off (handles DeepInfra/Anthropic
# rate limits and transient network errors) + per-item checkpointing so an
# interrupted session resumes instead of starting over.
# ---------------------------------------------------------------------------
import pickle

CKPT = 'llm_ckpt.pkl'
_ckpt = {}
if os.path.exists(CKPT):
    try:
        _ckpt = pickle.load(open(CKPT, 'rb'))
        print(f'[resume] checkpoint found with {sum(len(v) for v in _ckpt.values())} cached predictions')
    except Exception:
        _ckpt = {}

def _save_ckpt():
    pickle.dump(_ckpt, open(CKPT, 'wb'))

def with_retry(fn, *a, max_tries=8, **kw):
    delay = 2.0
    for attempt in range(max_tries):
        try:
            return fn(*a, **kw)
        except Exception as e:
            msg = str(e).lower()
            transient = any(k in msg for k in
                            ['rate', '429', 'overload', 'timeout', 'timed out',
                             'connection', 'temporarily', '529', '503', '500'])
            if attempt == max_tries - 1 or not transient:
                raise
            wait = delay * (2 ** attempt) + np.random.uniform(0, 1)
            print(f'      [retry {attempt+1}/{max_tries}] {type(e).__name__}: waiting {wait:.0f}s')
            time.sleep(min(wait, 90))
    raise RuntimeError('unreachable')

# ----------------------------------------------------------------- DATA / SPLITS
df = pd.read_excel(DATA, sheet_name='master_cleaned_dataset')

def binary_splits():
    d = df.sample(frac=1, random_state=42).reset_index(drop=True)
    X = d[TEXT_COL].astype(str).values
    y = d['is_fire'].astype(int).values
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    Xv, Xte, yv, yte = train_test_split(Xtmp, ytmp, test_size=0.50, random_state=42, stratify=ytmp)
    return Xtr, ytr, Xte, yte

def multi_splits():
    d = df[df.is_fire == 1].copy().sample(frac=1, random_state=789).reset_index(drop=True)
    X = d[TEXT_COL].astype(str).values
    y = d[['is_wildland', 'is_urban']].values.astype(int)
    s = y[:, 0] * 2 + y[:, 1]
    Xtr, Xtmp, ytr, ytmp, str_, stmp = train_test_split(X, y, s, test_size=0.30,
                                                        random_state=789, stratify=s)
    Xv, Xte, yv, yte, sv, ste = train_test_split(Xtmp, ytmp, stmp, test_size=0.50,
                                                 random_state=789, stratify=stmp)
    return Xtr, ytr, Xte, yte

Xb_tr, yb_tr, Xb_te, yb_te = binary_splits()
Xm_tr, ym_tr, Xm_te, ym_te = multi_splits()
print(f'[data] binary test = {len(Xb_te)} (fire {int(yb_te.sum())}, non-fire {int((yb_te==0).sum())})')
print(f'[data] multilabel test = {len(Xm_te)}')

# ----------------------------------------------------------------- PROMPTS (Greek)
BIN_SYS = (
    "Είσαι ταξινομητής κειμένου για ελληνικές αναρτήσεις στο X σχετικά με πυρκαγιές. "
    "Κρίνε ΜΟΝΟ αν η ανάρτηση αναφέρει ΕΝΕΡΓΗ πυρκαγιά σε εξέλιξη στην Ελλάδα αυτή τη στιγμή "
    "(θέση, εξάπλωση, εκκένωση, επιχείρηση κατάσβεσης). "
    "Μεταφορικές χρήσεις, παλιές πυρκαγιές, πολιτικά σχόλια, προληπτικές προειδοποιήσεις χωρίς "
    "ενεργή φωτιά, και πυρκαγιές εκτός Ελλάδας θεωρούνται ΟΧΙ ενεργές. "
    "Απάντησε ΜΟΝΟ με JSON: {\"active_fire\": 0 ή 1}."
)
MULTI_SYS = (
    "Είσαι ταξινομητής τύπου πυρκαγιάς για ελληνικές αναρτήσεις που αφορούν ΕΝΕΡΓΗ πυρκαγιά. "
    "Δώσε δύο δυαδικές ετικέτες με βάση ΜΟΝΟ το ρητό περιεχόμενο του κειμένου: "
    "is_wildland (δασική/αγροτική/χορτολιβαδική) και is_urban (κτίρια/οχήματα/αστικές δομές). "
    "mixed = 1,1 όταν απειλείται ή έχει φτάσει ο οικισμός. Αν δεν υπάρχει ρητή ένδειξη τύπου, "
    "δώσε 0,0. ΜΗΝ χρησιμοποιείς γεωγραφική γνώση για τοπωνύμια. "
    "Απάντησε ΜΟΝΟ με JSON: {\"is_wildland\": 0 ή 1, \"is_urban\": 0 ή 1}."
)

def fewshot_binary():
    rng = np.random.RandomState(SEED_FEWSHOT)
    pos = np.where(yb_tr == 1)[0]; neg = np.where(yb_tr == 0)[0]
    idx = list(rng.choice(pos, N_FEWSHOT // 2, replace=False)) + \
          list(rng.choice(neg, N_FEWSHOT // 2, replace=False))
    rng.shuffle(idx)
    return [{'text': Xb_tr[i], 'label': int(yb_tr[i])} for i in idx]

def fewshot_multi():
    rng = np.random.RandomState(SEED_FEWSHOT)
    s = ym_tr[:, 0] * 2 + ym_tr[:, 1]
    idx = []
    for cls in [0, 1, 2, 3]:
        pool = np.where(s == cls)[0]
        idx += list(rng.choice(pool, min(2, len(pool)), replace=False))
    rng.shuffle(idx)
    return [{'text': Xm_tr[i], 'w': int(ym_tr[i, 0]), 'u': int(ym_tr[i, 1])} for i in idx]

# ----------------------------------------------------------------- CLIENTS
def get_claude():
    from anthropic import Anthropic
    return Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])

def get_deepinfra():
    # DeepInfra exposes an OpenAI-compatible endpoint; reuse the OpenAI SDK.
    from openai import OpenAI
    return OpenAI(api_key=os.environ['DEEPINFRA_API_KEY'],
                  base_url='https://api.deepinfra.com/v1/openai')

def parse_json(txt):
    m = re.search(r'\{[^{}]*\}', txt, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def call_claude(client, system, messages, max_tokens=20):
    r = client.messages.create(model=CLAUDE_MODEL, max_tokens=max_tokens,
                               system=system, messages=messages)
    return r.content[0].text, r.usage.input_tokens, r.usage.output_tokens

def call_llama(client, system, user, max_tokens=20):
    r = client.chat.completions.create(
        model=LLAMA_MODEL, max_tokens=max_tokens, temperature=0,
        messages=[{'role': 'system', 'content': system},
                  {'role': 'user', 'content': user}])
    u = r.usage
    return r.choices[0].message.content, u.prompt_tokens, u.completion_tokens

# ----------------------------------------------------------------- RUN ONE TASK
def run_binary(provider, few, client):
    key = f'binary_{provider}_{"few" if few else "zero"}'
    cache = _ckpt.get(key, {})
    shots = fewshot_binary() if few else []
    preds, in_tok, out_tok, t0 = [], 0, 0, time.time()
    for i, text in enumerate(Xb_te):
        if i in cache:
            preds.append(cache[i]['p']); in_tok += cache[i]['it']; out_tok += cache[i]['ot']
            continue
        if provider == 'claude':
            msgs = []
            for s in shots:
                msgs.append({'role': 'user', 'content': f'Ανάρτηση: {s["text"]}'})
                msgs.append({'role': 'assistant', 'content': json.dumps({'active_fire': s['label']})})
            msgs.append({'role': 'user', 'content': f'Ανάρτηση: {text}'})
            out, it, ot = with_retry(call_claude, client, BIN_SYS, msgs)
        else:
            shot_txt = ''.join(f'Ανάρτηση: {s["text"]}\n{{"active_fire": {s["label"]}}}\n\n' for s in shots)
            out, it, ot = with_retry(call_llama, client, BIN_SYS, shot_txt + f'Ανάρτηση: {text}')
        j = parse_json(out)
        pv = int(j['active_fire']) if j and 'active_fire' in j else 0
        preds.append(pv); in_tok += it; out_tok += ot
        cache[i] = {'p': pv, 'it': it, 'ot': ot}; _ckpt[key] = cache
        if (i + 1) % 25 == 0:
            _save_ckpt(); print(f'    {provider} binary {"few" if few else "zero"}: {i+1}/{len(Xb_te)}')
    _save_ckpt()
    return np.array(preds), in_tok, out_tok, time.time() - t0

def run_multi(provider, few, client):
    key = f'multi_{provider}_{"few" if few else "zero"}'
    cache = _ckpt.get(key, {})
    shots = fewshot_multi() if few else []
    preds, in_tok, out_tok, t0 = [], 0, 0, time.time()
    for i, text in enumerate(Xm_te):
        if i in cache:
            preds.append(cache[i]['p']); in_tok += cache[i]['it']; out_tok += cache[i]['ot']
            continue
        if provider == 'claude':
            msgs = []
            for s in shots:
                msgs.append({'role': 'user', 'content': f'Ανάρτηση: {s["text"]}'})
                msgs.append({'role': 'assistant', 'content': json.dumps({'is_wildland': s['w'], 'is_urban': s['u']})})
            msgs.append({'role': 'user', 'content': f'Ανάρτηση: {text}'})
            out, it, ot = with_retry(call_claude, client, MULTI_SYS, msgs, max_tokens=30)
        else:
            shot_txt = ''.join(f'Ανάρτηση: {s["text"]}\n{{"is_wildland": {s["w"]}, "is_urban": {s["u"]}}}\n\n' for s in shots)
            out, it, ot = with_retry(call_llama, client, MULTI_SYS, shot_txt + f'Ανάρτηση: {text}', max_tokens=30)
        j = parse_json(out)
        w = int(j['is_wildland']) if j and 'is_wildland' in j else 0
        u = int(j['is_urban']) if j and 'is_urban' in j else 0
        preds.append([w, u]); in_tok += it; out_tok += ot
        cache[i] = {'p': [w, u], 'it': it, 'ot': ot}; _ckpt[key] = cache
        if (i + 1) % 25 == 0:
            _save_ckpt(); print(f'    {provider} multi {"few" if few else "zero"}: {i+1}/{len(Xm_te)}')
    _save_ckpt()
    return np.array(preds), in_tok, out_tok, time.time() - t0

# ----------------------------------------------------------------- METRICS
def binary_metrics(y, p):
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average='binary', zero_division=0)
    return {'accuracy': round(accuracy_score(y, p), 4), 'precision': round(pr, 4),
            'recall': round(rc, 4), 'f1': round(f1, 4),
            'f2': round(fbeta_score(y, p, beta=2.0, average='binary', zero_division=0), 4)}

def multi_metrics(y, p):
    y4 = y[:, 0] * 2 + y[:, 1]; p4 = p[:, 0] * 2 + p[:, 1]
    pr, rc, f1, _ = precision_recall_fscore_support(y4, p4, average='macro', zero_division=0)
    return {'exact_match': round(accuracy_score(y, p), 4),
            'macro_precision': round(pr, 4), 'macro_recall': round(rc, 4),
            'macro_f1': round(f1, 4)}

def cost(model, it, ot, n):
    pr = PRICING[model]
    total = it / 1e6 * pr['in'] + ot / 1e6 * pr['out']
    return {'input_tokens': int(it), 'output_tokens': int(ot),
            'total_usd': round(total, 4), 'usd_per_1000_items': round(total / n * 1000, 4)}

# ----------------------------------------------------------------- MAIN
results = {}
claude = get_claude() if (RUN['claude_zero'] or RUN['claude_few']) else None
llm_open = get_deepinfra() if (RUN['llama_zero'] or RUN['llama_few']) else None

plan = [
    ('claude_zero', 'claude', False, claude), ('claude_few', 'claude', True, claude),
    ('llama_zero',  'llama',  False, llm_open), ('llama_few',  'llama',  True, llm_open),
]
model_of = {'claude': CLAUDE_MODEL, 'llama': LLAMA_MODEL}

for tag, provider, few, client in plan:
    if not RUN[tag]:
        continue
    print(f'\n=== {tag.upper()} ===')
    pb, ib, ob, tb = run_binary(provider, few, client)
    pm, im, om, tm = run_multi(provider, few, client)
    results[tag] = {
        'model': model_of[provider], 'shot': 'few' if few else 'zero',
        'binary': {**binary_metrics(yb_te, pb),
                   'latency_s_per_item': round(tb / len(Xb_te), 3),
                   'cost': cost(model_of[provider], ib, ob, len(Xb_te))},
        'multilabel': {**multi_metrics(ym_te, pm),
                       'latency_s_per_item': round(tm / len(Xm_te), 3),
                       'cost': cost(model_of[provider], im, om, len(Xm_te))},
    }
    b = results[tag]['binary']; m = results[tag]['multilabel']
    print(f"  binary   : F2={b['f2']} recall={b['recall']} acc={b['accuracy']} "
          f"lat={b['latency_s_per_item']}s/item")
    print(f"  multilabel: macroF1={m['macro_f1']} exact={m['exact_match']} "
          f"lat={m['latency_s_per_item']}s/item")

with open('llm_baseline_results.json', 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

rows = []
for tag, r in results.items():
    rows.append({'config': tag, 'model': r['model'], 'shot': r['shot'],
                 'bin_F2': r['binary']['f2'], 'bin_recall': r['binary']['recall'],
                 'bin_acc': r['binary']['accuracy'], 'bin_lat_s': r['binary']['latency_s_per_item'],
                 'bin_usd_per_1k': r['binary']['cost']['usd_per_1000_items'],
                 'multi_macroF1': r['multilabel']['macro_f1'],
                 'multi_exact': r['multilabel']['exact_match'],
                 'multi_lat_s': r['multilabel']['latency_s_per_item']})
pd.DataFrame(rows).to_csv('llm_baseline_results.csv', index=False)
print('\nSaved llm_baseline_results.json and .csv')
print('\nNOTE: pricing and model IDs are set at the top of this file — update them to the')
print('current published values before the numbers go in the paper.')
