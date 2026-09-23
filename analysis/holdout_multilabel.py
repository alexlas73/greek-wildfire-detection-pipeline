# -*- coding: utf-8 -*-
"""
holdout_multilabel.py  —  Multilabel fire-type classification: RANDOM vs TEMPORAL (chronological) hold-out
Run in Google Colab with a GPU runtime. Upload master_cleaned_dataset.csv when prompted.
Outputs go to ./holdout_multilabel/

Identical to part4_multilabel_classification.py in models, hyperparameters, weighted loss and
threshold optimisation. Only the split changes; extra artefacts are written for the revision
(per-post probabilities, misclassified posts, bootstrap CIs, per-class metrics).
"""
# ---------------------------------------------------------------- SETTINGS
SPLIT_MODES = ["random", "temporal"]
SEEDS       = [789, 1, 2, 3, 4]           # 5 seeds; use [789] for a quick test
CUTOFF      = "2025-08-01"
VAL_FRAC    = 0.15
RUN_SVM     = True
OUT         = "./holdout_multilabel"
# --------------------------------------------------------------------------
import os, io, json, gc, shutil
import numpy as np, pandas as pd, torch
from torch import nn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score, make_scorer, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, set_seed
from datasets import Dataset
os.makedirs(OUT, exist_ok=True)
CLS = ["Unidentified (0,0)", "Urban (0,1)", "Wildland (1,0)", "Mixed (1,1)"]

# ---------------------------------------------------------------- DATA
try:
    from google.colab import files
    print("Upload master_cleaned_dataset.csv"); up = files.upload(); fn = list(up.keys())[0]
    df0 = pd.read_excel(io.BytesIO(up[fn]))
except ImportError:
    df0 = pd.read_excel("master_cleaned_dataset.xlsx")
df0["dt"] = pd.to_datetime(df0["Date"], format="%a %b %d %H:%M:%S %z %Y", utc=True, errors="coerce")
df = df0[df0["is_fire"] == 1].copy()
df["is_wildland"] = df["is_wildland"].fillna(0).astype(float); df["is_urban"] = df["is_urban"].fillna(0).astype(float)
df = df.dropna(subset=["Hard Cleaned Text", "Advanced Cleaned Text"]).reset_index(drop=True)
df["cls4"] = (df.is_wildland * 2 + df.is_urban).astype(int)
Y = df[["is_wildland", "is_urban"]].values
print(f"Fire posts: {len(df)}  class counts: {np.bincount(df.cls4, minlength=4)}")

def make_split(mode):
    idx = np.arange(len(df)); strat = df["cls4"].values
    if mode == "random":
        tr, tmp = train_test_split(idx, test_size=0.30, random_state=789, stratify=strat)
        va, te  = train_test_split(tmp, test_size=0.50, random_state=789, stratify=strat[tmp])
    else:
        cut = np.datetime64(pd.Timestamp(CUTOFF, tz="UTC"))
        te  = idx[df["dt"].values >= cut]; pre = idx[df["dt"].values < cut]
        pre = pre[np.argsort(df.loc[pre, "dt"].values)]; n_val = int(round(VAL_FRAC * len(pre)))
        tr, va = pre[:-n_val], pre[-n_val:]
    return tr, va, te

# ---------------------------------------------------------------- METRICS
def to4(y): y = np.asarray(y).astype(int); return y[:, 0] * 2 + y[:, 1]
def mlmetrics(yt, yp):
    yt, yp = np.asarray(yt).astype(int), np.asarray(yp).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(to4(yt), to4(yp), average="macro", zero_division=0)
    pc_p, pc_r, pc_f, pc_n = precision_recall_fscore_support(to4(yt), to4(yp), labels=[0, 1, 2, 3], zero_division=0)
    return dict(exact_match=accuracy_score(yt, yp), macro_precision=pr, macro_recall=rc, macro_f1=f1,
                per_class={CLS[i]: dict(precision=pc_p[i], recall=pc_r[i], f1=pc_f[i], support=int(pc_n[i])) for i in range(4)})
def bootstrap_ci(yt, yp, B=2000, seed=0):
    rng = np.random.default_rng(seed); n = len(yt); mf, em, mixed = [], [], []
    for _ in range(B):
        s = rng.integers(0, n, n); m = mlmetrics(yt[s], yp[s]); mf.append(m["macro_f1"]); em.append(m["exact_match"]); mixed.append(m["per_class"][CLS[3]]["f1"])
    pct = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {"macro_f1": pct(mf), "exact_match": pct(em), "mixed_f1": pct(mixed)}
def compute_metrics(eval_pred):
    logits, labels = eval_pred; probs = torch.sigmoid(torch.tensor(logits)).numpy(); preds = (probs > 0.5).astype(int)
    m = mlmetrics(labels, preds); return {k: m[k] for k in ["exact_match", "macro_precision", "macro_recall", "macro_f1"]}
def optimize_thresholds(yv, pv):
    best = (0.5, 0.5, -1); y4 = to4(yv)
    for tw in np.arange(0.10, 0.95, 0.05):
        for tu in np.arange(0.10, 0.95, 0.05):
            p = np.stack([(pv[:, 0] > tw), (pv[:, 1] > tu)], 1).astype(int)
            f = f1_score(y4, to4(p), average="macro", zero_division=0)
            if f > best[2]: best = (tw, tu, f)
    return best

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs.pop("labels"); out = model(**inputs)
        loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.45, 1.32]).to(model.device))(out.logits, labels)
        return (loss, out) if return_outputs else loss

# ---------------------------------------------------------------- MODELS
MODELS = {"GreekBERT":  dict(mid="nlpaueb/bert-base-greek-uncased-v1", col="Advanced Cleaned Text", epochs=4, lr=2e-5),
          "XLM-RoBERTa": dict(mid="xlm-roberta-base", col="Light Cleaned Text", epochs=5, lr=3e-5)}

def run_transformer(name, mode, seed, tr, va, te):
    set_seed(seed); cfg = MODELS[name]; tok = AutoTokenizer.from_pretrained(cfg["mid"])
    X = df[cfg["col"]].astype(str).tolist(); yl = Y.tolist()
    def ds(ix): return Dataset.from_dict({"text": [X[i] for i in ix], "label": [yl[i] for i in ix]}).map(
        lambda e: tok(e["text"], padding="max_length", truncation=True, max_length=128), batched=True)
    dtr, dva, dte = ds(tr), ds(va), ds(te)
    model = AutoModelForSequenceClassification.from_pretrained(cfg["mid"], num_labels=2, problem_type="multi_label_classification")
    odir = f"{OUT}/tmp_{name}_{mode}_{seed}"
    args = TrainingArguments(output_dir=odir, num_train_epochs=cfg["epochs"], per_device_train_batch_size=8, learning_rate=cfg["lr"],
        warmup_ratio=0.15, eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True, metric_for_best_model="macro_f1",
        greater_is_better=True, fp16=torch.cuda.is_available(), report_to="none", logging_strategy="epoch", weight_decay=0.1,
        lr_scheduler_type="cosine", seed=seed, save_total_limit=1)
    trainer = WeightedTrainer(model=model, args=args, train_dataset=dtr, eval_dataset=dva, compute_metrics=compute_metrics)
    trainer.train()
    lv, _, _ = trainer.predict(dva); pv = torch.sigmoid(torch.tensor(lv)).numpy(); tw, tu, fval = optimize_thresholds(Y[va], pv)
    lt, _, _ = trainer.predict(dte); pt = torch.sigmoid(torch.tensor(lt)).numpy()
    yp = np.stack([(pt[:, 0] > tw), (pt[:, 1] > tu)], 1).astype(int); yt = Y[te].astype(int)
    res = {"model": name, "split": mode, "seed": seed, "n_train": len(tr), "n_val": len(va), "n_test": len(te),
           "thr_wild": float(tw), "thr_urban": float(tu), "val_macro_f1": float(fval), **mlmetrics(yt, yp), "ci95": bootstrap_ci(yt, yp)}
    pd.DataFrame({"Tweet_ID": df.loc[te, "Tweet_ID"].values, "Date": df.loc[te, "Date"].values, "text": df.loc[te, "Light Cleaned Text"].values,
                  "true_wild": yt[:, 0], "true_urban": yt[:, 1], "prob_wild": pt[:, 0], "prob_urban": pt[:, 1],
                  "pred_wild": yp[:, 0], "pred_urban": yp[:, 1], "true_class": [CLS[i] for i in to4(yt)], "pred_class": [CLS[i] for i in to4(yp)]}
                 ).to_csv(f"{OUT}/probs_{name}_{mode}_seed{seed}.csv", index=False)
    shutil.rmtree(odir, ignore_errors=True); del trainer, model; gc.collect(); torch.cuda.empty_cache()
    return res

def run_svm(mode, tr, va, te):
    X = df["Hard Cleaned Text"].astype(str).values
    pipe = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
                     ("svm", MultiOutputClassifier(SVC(kernel="linear", class_weight="balanced", probability=True, random_state=789)))])
    scorer = make_scorer(lambda a, b: f1_score(to4(a), to4(b), average="macro", zero_division=0))
    gs = GridSearchCV(pipe, {"tfidf__max_features": [3000, 5000], "tfidf__ngram_range": [(1, 1), (1, 2)], "svm__estimator__C": [0.1, 1, 5, 10]},
                      cv=5, scoring=scorer, n_jobs=-1)
    gs.fit(X[tr], Y[tr]); yp = gs.predict(X[te]).astype(int); yt = Y[te].astype(int)
    return {"model": "AdvancedSVM", "split": mode, "seed": 789, "best_params": str(gs.best_params_), **mlmetrics(yt, yp), "ci95": bootstrap_ci(yt, yp)}

# ---------------------------------------------------------------- MAIN
all_res = []
for mode in SPLIT_MODES:
    tr, va, te = make_split(mode)
    print(f"\n=== {mode.upper()} split: train {len(tr)} / val {len(va)} / test {len(te)}  test classes={np.bincount(df.loc[te,'cls4'], minlength=4)}")
    if RUN_SVM: all_res.append(run_svm(mode, tr, va, te)); print("   SVM done")
    for seed in SEEDS:
        for name in MODELS:
            r = run_transformer(name, mode, seed, tr, va, te); all_res.append(r)
            print(f"   {name} seed {seed}: macroF1={r['macro_f1']:.4f} exact={r['exact_match']:.4f} mixedF1={r['per_class'][CLS[3]]['f1']:.4f} CI={r['ci95']['macro_f1']}")
            json.dump(all_res, open(f"{OUT}/results_multilabel.json", "w"), indent=2, default=float)

rows = [dict(model=r["model"], split=r["split"], seed=r["seed"], exact_match=r["exact_match"], macro_precision=r["macro_precision"],
             macro_recall=r["macro_recall"], macro_f1=r["macro_f1"], **{f"f1_{c.split(' ')[0]}": r["per_class"][c]["f1"] for c in CLS}) for r in all_res]
summ = pd.DataFrame(rows); summ.to_csv(f"{OUT}/summary_multilabel_per_seed.csv", index=False)
agg = summ.groupby(["split", "model"]).agg(["mean", "std"]).round(4); agg.to_csv(f"{OUT}/summary_multilabel_mean_sd.csv"); print("\n", agg)

for mode in SPLIT_MODES:
    for name in MODELS:
        try:
            p = pd.read_csv(f"{OUT}/probs_{name}_{mode}_seed{SEEDS[0]}.csv"); e = p[p.true_class != p.pred_class]
            e.to_csv(f"{OUT}/errors_{name}_{mode}_seed{SEEDS[0]}.csv", index=False)
        except FileNotFoundError: pass
print("\nDONE. Zip and download the folder:", OUT)
