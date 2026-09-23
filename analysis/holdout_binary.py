# -*- coding: utf-8 -*-
"""
holdout_binary.py  —  Binary active-fire classification: RANDOM vs TEMPORAL (chronological) hold-out
Run in Google Colab with a GPU runtime. Upload master_cleaned_dataset.csv when prompted.
Outputs go to ./holdout_binary/  (zip it and download at the end).

Identical to part3_binary_classification.py in models, hyperparameters and metrics.
Only the data split changes, and extra artefacts are written for the revision:
  - per-post test probabilities (for bootstrap CIs, paired tests, threshold analysis)
  - misclassified posts (for the qualitative error-analysis table)
  - metrics at the operational threshold 0.30 as well as 0.50
"""
# ---------------------------------------------------------------- SETTINGS
SPLIT_MODES = ["random", "temporal"]      # run both
SEEDS       = [42, 1, 2, 3, 4]            # 5 seeds as in the paper; use [42] for a quick test
CUTOFF      = "2025-08-01"                # temporal split: test = posts on/after this date (UTC)
VAL_FRAC    = 0.15                        # temporal split: latest 15 % of the training period -> validation
RUN_SVM     = True                        # advanced SVM (fast) for completeness
OUT         = "./holdout_binary"
# --------------------------------------------------------------------------
import os, io, json, gc, shutil
import numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_fscore_support, fbeta_score, confusion_matrix, make_scorer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, set_seed
from datasets import Dataset
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- DATA
try:
    from google.colab import files
    print("Upload master_cleaned_dataset.csv"); up = files.upload(); fn = list(up.keys())[0]
    df = pd.read_csv(io.BytesIO(up[fn]), dtype={"Tweet_ID": str}) if fn.endswith(".csv") else pd.read_excel(io.BytesIO(up[fn]), dtype={"Tweet_ID": str})
except ImportError:
    df = pd.read_csv("data/master_cleaned_dataset.csv", dtype={"Tweet_ID": str})
df["dt"] = pd.to_datetime(df["Date"], format="%a %b %d %H:%M:%S %z %Y", utc=True, errors="coerce")
assert df["dt"].notna().all(), "Date parsing failed"
df = df.dropna(subset=["Advanced Cleaned Text", "Light Cleaned Text", "Hard Cleaned Text"]).reset_index(drop=True)
df["is_fire"] = df["is_fire"].astype(int)
print(f"Rows: {len(df)}  fire={df.is_fire.sum()}  range {df.dt.min()} -> {df.dt.max()}")

def make_split(mode, seed):
    """Return index arrays (train, val, test)."""
    idx = np.arange(len(df)); y = df["is_fire"].values
    if mode == "random":
        tr, tmp = train_test_split(idx, test_size=0.30, random_state=42, stratify=y)
        va, te  = train_test_split(tmp, test_size=0.50, random_state=42, stratify=y[tmp])
    else:
        cut = pd.Timestamp(CUTOFF, tz="UTC")
        te  = idx[df["dt"].values >= np.datetime64(cut)]
        pre = idx[df["dt"].values <  np.datetime64(cut)]
        pre = pre[np.argsort(df.loc[pre, "dt"].values)]          # chronological
        n_val = int(round(VAL_FRAC * len(pre)))
        tr, va = pre[:-n_val], pre[-n_val:]                        # validation = latest slice of the pre-cutoff period
    return tr, va, te

# ---------------------------------------------------------------- METRICS
def binmetrics(y, p):
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return dict(precision=pr, recall=rc, f1=f1, f2=fbeta_score(y, p, beta=2.0, zero_division=0),
                fpr=fp / max(fp + tn, 1), tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn), n=int(len(y)))

def bootstrap_ci(y, prob, thr=0.5, B=2000, seed=0):
    rng = np.random.default_rng(seed); n = len(y); out = {"f2": [], "recall": [], "precision": []}
    for _ in range(B):
        s = rng.integers(0, n, n); m = binmetrics(y[s], (prob[s] >= thr).astype(int))
        for k in out: out[k].append(m[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in out.items()}

def compute_metrics(eval_pred):
    logits, labels = eval_pred; preds = np.argmax(logits, axis=-1)
    m = binmetrics(labels, preds); return {k: m[k] for k in ["precision", "recall", "f1", "f2"]}

def softmax(x): e = np.exp(x - x.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)

# ---------------------------------------------------------------- TRANSFORMER RUN
MODELS = {"GreekBERT": ("nlpaueb/bert-base-greek-uncased-v1", "Advanced Cleaned Text"),
          "XLM-RoBERTa": ("xlm-roberta-base", "Light Cleaned Text")}

def run_transformer(name, mode, seed, tr, va, te):
    set_seed(seed); mid, col = MODELS[name]
    tok = AutoTokenizer.from_pretrained(mid)
    X = df[col].astype(str).tolist(); y = df["is_fire"].tolist()
    def ds(ix): return Dataset.from_dict({"text": [X[i] for i in ix], "label": [y[i] for i in ix]}).map(
        lambda e: tok(e["text"], padding="max_length", truncation=True, max_length=128), batched=True)
    dtr, dva, dte = ds(tr), ds(va), ds(te)
    model = AutoModelForSequenceClassification.from_pretrained(mid, num_labels=2)
    odir = f"{OUT}/tmp_{name}_{mode}_{seed}"
    args = TrainingArguments(output_dir=odir, num_train_epochs=4, per_device_train_batch_size=16,
        per_device_eval_batch_size=32, learning_rate=5e-6, weight_decay=0.1, warmup_ratio=0.1,
        eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="f2", greater_is_better=True, fp16=torch.cuda.is_available(),
        report_to="none", logging_strategy="epoch", seed=seed, save_total_limit=1)
    trainer = Trainer(model=model, args=args, train_dataset=dtr, eval_dataset=dva, compute_metrics=compute_metrics)
    trainer.train()
    logits, _, _ = trainer.predict(dte); prob = softmax(logits)[:, 1]; yte = np.array([y[i] for i in te])
    res = {"model": name, "split": mode, "seed": seed, "n_train": len(tr), "n_val": len(va), "n_test": len(te),
           "thr_0.50": binmetrics(yte, (prob >= 0.5).astype(int)),
           "thr_0.30": binmetrics(yte, (prob >= 0.3).astype(int)),
           "ci95_thr_0.50": bootstrap_ci(yte, prob, 0.5), "ci95_thr_0.30": bootstrap_ci(yte, prob, 0.3)}
    pd.DataFrame({"Tweet_ID": df.loc[te, "Tweet_ID"].values, "Date": df.loc[te, "Date"].values,
                  "text": df.loc[te, "Light Cleaned Text"].values, "y_true": yte, "prob_fire": prob,
                  "pred_0.50": (prob >= 0.5).astype(int), "pred_0.30": (prob >= 0.3).astype(int)}
                 ).to_csv(f"{OUT}/probs_{name}_{mode}_seed{seed}.csv", index=False)
    shutil.rmtree(odir, ignore_errors=True); del trainer, model; gc.collect(); torch.cuda.empty_cache()
    return res

def run_svm(mode, tr, va, te):
    X = df["Hard Cleaned Text"].astype(str).values; y = df["is_fire"].values
    pipe = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2))), ("svm", SVC(kernel="linear", class_weight="balanced", random_state=42))])
    gs = GridSearchCV(pipe, {"tfidf__max_features": [5000, 10000], "svm__C": [0.1, 1, 5, 10, 50]},
                      cv=5, scoring=make_scorer(fbeta_score, beta=2.0), n_jobs=-1)
    gs.fit(X[tr], y[tr]); p = gs.predict(X[te])
    return {"model": "AdvancedSVM", "split": mode, "seed": 42, "best_params": gs.best_params_, "thr_0.50": binmetrics(y[te], p)}

# ---------------------------------------------------------------- MAIN
all_res = []
for mode in SPLIT_MODES:
    tr, va, te = make_split(mode, 42)
    print(f"\n=== {mode.upper()} split: train {len(tr)} / val {len(va)} / test {len(te)} "
          f"(test fire={df.loc[te,'is_fire'].sum()})")
    if mode == "temporal":
        print("   test period:", df.loc[te, "dt"].min(), "->", df.loc[te, "dt"].max())
        print("   val period :", df.loc[va, "dt"].min(), "->", df.loc[va, "dt"].max())
    if RUN_SVM: all_res.append(run_svm(mode, tr, va, te)); print("   SVM done")
    for seed in SEEDS:
        for name in MODELS:
            r = run_transformer(name, mode, seed, tr, va, te); all_res.append(r)
            print(f"   {name} seed {seed}: F2@0.5={r['thr_0.50']['f2']:.4f}  F2@0.3={r['thr_0.30']['f2']:.4f}  "
                  f"CI={r['ci95_thr_0.50']['f2']}")
            json.dump(all_res, open(f"{OUT}/results_binary.json", "w"), indent=2, default=float)

# ---------------------------------------------------------------- SUMMARY + PAIRED BOOTSTRAP
rows = []
for r in all_res:
    for thr in ["thr_0.50", "thr_0.30"]:
        if thr in r: rows.append(dict(model=r["model"], split=r["split"], seed=r["seed"], thr=thr, **r[thr]))
summ = pd.DataFrame(rows); summ.to_csv(f"{OUT}/summary_binary_per_seed.csv", index=False)
agg = summ.groupby(["split", "model", "thr"])[["precision", "recall", "f1", "f2", "fpr"]].agg(["mean", "std"]).round(4)
agg.to_csv(f"{OUT}/summary_binary_mean_sd.csv"); print("\n", agg)

# paired bootstrap: GreekBERT vs XLM-R on identical test posts (seed 42, threshold 0.5), per split
for mode in SPLIT_MODES:
    try:
        a = pd.read_csv(f"{OUT}/probs_GreekBERT_{mode}_seed42.csv"); b = pd.read_csv(f"{OUT}/probs_XLM-RoBERTa_{mode}_seed42.csv")
        y = a.y_true.values; pa, pb = a["pred_0.50"].values, b["pred_0.50"].values
        rng = np.random.default_rng(0); d = []
        for _ in range(2000):
            s = rng.integers(0, len(y), len(y))
            d.append(fbeta_score(y[s], pa[s], beta=2) - fbeta_score(y[s], pb[s], beta=2))
        print(f"{mode}: ΔF2 (GreekBERT − XLM-R) = {np.mean(d):.4f}  95% CI [{np.percentile(d,2.5):.4f}, {np.percentile(d,97.5):.4f}]")
        json.dump({"delta_f2_mean": float(np.mean(d)), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]},
                  open(f"{OUT}/paired_bootstrap_{mode}.json", "w"))
    except FileNotFoundError: pass

# misclassified posts (seed 42, both thresholds) for the error-analysis table
for mode in SPLIT_MODES:
    for name in MODELS:
        try:
            p = pd.read_csv(f"{OUT}/probs_{name}_{mode}_seed42.csv")
            e = p[p.y_true != p["pred_0.50"]].copy()
            e["error_type"] = np.where(e.y_true == 1, "False Negative (missed fire)", "False Positive (false alarm)")
            e.sort_values(["error_type", "prob_fire"]).to_csv(f"{OUT}/errors_{name}_{mode}_seed42.csv", index=False)
        except FileNotFoundError: pass
print("\nDONE. Zip and download the folder:", OUT)
