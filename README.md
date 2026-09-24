# People as Sensors: Greek-Language Wildfire Detection Pipeline

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21982026.svg)](https://doi.org/10.5281/zenodo.21982026)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

End-to-end, Greek-language NLP pipeline that turns keyword-retrieved, unlabeled X posts into a near-real-time, geolocated wildfire map: active-fire detection → fire-type classification (wildland / urban / mixed) → toponym extraction and spatiotemporal disambiguation → DBSCAN incident clustering → interactive Folium map.

> **Associated paper:** Lazanas, A.; Samaras, M. *People as Sensors: An End-to-End Deep Learning Framework for Near-Real-Time Wildfire Detection, Classification, and Geospatial Intelligence from Greek Social Media.* **AI** (MDPI), 2026, under review.
>
> **Interactive demonstration map (12 August 2024, NE Attica):** https://alexlas73.github.io/FireMap/

**Release v2.2 (September 2026)** adds the material of the revised manuscript: chronological hold-out evaluation (§4.8), DBSCAN sensitivity analysis and comparison with the official 112 warnings (§4.7), near-duplicate leakage analysis (§3.4.3), the full inter-annotator agreement (IAA) package (§3.2.3, §4.1), the annotation guidelines, and a corrected data release (see *Data* below).

---

## Pipeline overview

```
Keyword-retrieved X posts (Twikit)
    │
    ▼
Part 2 · Text cleaning            light → advanced → hard
    │
    ▼
Part 3 · Active-fire detection    GreekBERT (binary)        F2 = 0.911 (random split) · 0.850 (chronological hold-out)
    │
    ▼
Part 4 · Fire-type classification XLM-RoBERTa (multilabel)  macro-F1 = 0.874 (random) · 0.769 ± 0.085 (hold-out)
    │
    ▼
Part 5 · Geoparsing & mapping     GR-NLP-TOOLKIT NER → Nominatim → spatiotemporal memory (5 h, 40/20 km)
    │                              → DBSCAN (ε = 15 km, minPts = 2) → Folium map
    ▼
Part 6 · End-to-end replay        unlabeled stream of 12 August 2024 (2,000 posts)
```

`figures/fig_sequence_diagram.png` shows the inference sequence for one post (paper Figure 2).

---

## Repository structure

```
greek-wildfire-detection-pipeline/
├── scripts/
│   ├── part1_x_scraping.py                 X post collection via Twikit
│   ├── part2_text_cleaning.py              Greek text normalisation (three levels)
│   ├── part3_binary_classification.py      Active-fire classifier (SVM / GreekBERT / XLM-RoBERTa)
│   ├── part4_multilabel_classification.py  Fire-type classifier (SVM / GreekBERT / XLM-RoBERTa)
│   ├── part5_geoparsing_and_mapping.py     NER → geocoding → disambiguation → DBSCAN → Folium
│   └── part6_simulation.py                 End-to-end replay on an unlabeled stream
├── analysis/                               Scripts reproducing the paper's evaluation (table below)
├── data/                                   Corpus, gold standards, IAA package, results (see Data)
├── docs/ANNOTATION_GUIDELINES.md           Annotation protocol used for the corpus and the IAA audit
├── figures/                                Paper figures
├── Click_to_see_Fire_Map.html              Redirect to the interactive map
├── CITATION.cff
└── LICENSE
```

---

## Data

All files are in `data/`. **No usernames are distributed; in-text `@mentions` are replaced by `@user` in every file.**

### Annotated corpus (paper §3.2)

| File | Content |
|---|---|
| `master_cleaned_dataset.csv` | 3,277 posts, 3 July 2024 – 21 September 2025: `Date`, `Tweet_ID`, `Tweet_ID_precision`, `snowflake_time_utc`, `Raw Text`, `Likes`, `Retweets`, labels `is_fire`, `is_wildland`, `is_urban`, and the three cleaned-text columns. 1,100 posts are active-fire posts (`is_fire = 1`); fire type is encoded as (`is_wildland`, `is_urban`) = wildland (1,0), urban (0,1), mixed (1,1), unidentified (0,0). |
| `ground_truth_classification.xlsx` | The same posts and labels in spreadsheet form (kept for continuity with v2.0/v2.1). |
| `ground_truth_geoparsing.xlsx` | Geoparsing gold standard: 147 toponyms in 100 posts with gold toponym and coordinates (§3.5.3, §4.5). |

> **Post identifiers of the annotated corpus are approximate.** The corpus passed through a spreadsheet at annotation time, and spreadsheet software stores 19-digit integers as double-precision floats, which rounds an X post ID to the nearest multiple of 256. The original scrape files no longer exist, so every `Tweet_ID` in `master_cleaned_dataset.csv`, `ground_truth_*.xlsx`, the IAA files and the hold-out results is within ±128 of the true identifier (`Tweet_ID_precision = approx_pm128`). The timestamp encoded in the upper bits of the identifier is unaffected and is given to the millisecond in `snowflake_time_utc`. Identifiers are consistent *within* this release, so all joins between files work; they cannot be used directly as `x.com/i/status/<id>` links. The identifiers of the 2,000-post demonstration stream are exact (`Tweet_ID_precision = exact`).

### Demonstration stream (paper §4.7)

| File | Content |
|---|---|
| `raw_tweets_for_simulation.csv` | 2,000 unlabeled posts of 12 August 2024, 19:04–23:58 UTC (exact identifiers). Input of `scripts/part6_simulation.py`. |
| `simulation_predicted_tweets.csv` | Output of Parts 3–4 on the stream (fire probability, predicted labels). |
| `fire_events_geoparsed.csv` | Output of Part 5: the 434 posts with at least one resolved toponym; 600 geolocated mentions with coordinates and OSM metadata. |
| `112_alerts_matching.csv` | The ten public-warning (112) messages issued for the fire, 11–13 August 2024, matched to the earliest corpus/stream post naming each locality (area-level recall 8/9). |
| `dbscan_sensitivity.csv` | Clustering outcome for ε ∈ {5, 10, 15, 20, 30} km × minPts ∈ {2, 3, 5}. |

### Inter-annotator agreement package (paper §3.2.3, §4.1)

| File | Content |
|---|---|
| `IAA_sampling_metadata.json` | Sampling design (seed 20260810; 400 binary, 300 multilabel with the mixed class over-sampled, all 147 geoparsing items). |
| `IAA_worksheet_ANNOTATOR2.xlsx` | The blind worksheets as filled in by the second annotator (three sheets). |
| `IAA_key_annotator1.xlsx` | First-annotator labels for the same rows (the sealed key of the audit). |
| `IAA_results.json`, `IAA_geoparsing_setbased.json` | Cohen's κ, Krippendorff's α, bootstrap CIs; set-based (Jaccard) and distance agreement for geoparsing. |
| `IAA_disagreements_*.xlsx` | The 4 binary, 6 multilabel and the geoparsing disagreements. |
| `IAA_adjudication.json` | Adjudication outcome (1 of 700 audited labels revised) and the two clarified coding rules. |
| `docs/ANNOTATION_GUIDELINES.md` | The written protocol given to the second annotator. |

### Evaluation outputs

| File / folder | Paper section |
|---|---|
| `leakage_analysis_summary.csv`, `leakage_analysis_results.json` | §3.4.3 near-duplicate posts across splits |
| `llm_baseline_results.csv/.json`, `latency_results.json` | §4.4 LLM baselines, latency and cost |
| `geoparse_errors.csv` | §4.5 per-toponym error of base vs. disambiguated geocoding |
| `sensitivity_results.csv`, `nominatim_drift.csv` | §4.6 threshold sensitivity; gazetteer drift re-run |
| `holdout_binary/`, `holdout_multilabel/` | §4.8 random vs. chronological hold-out: per-seed metrics (`summary_*`), per-post test probabilities (`probs_*`, five seeds × two models × two splits), misclassified posts (`errors_*`), bootstrap CIs (`results_*.json`), paired bootstrap of the GreekBERT–XLM-RoBERTa difference |

---

## Pre-trained models

| Task | Model | Hugging Face ID |
|---|---|---|
| Active-fire detection | GreekBERT | [`mariossmrs/greek-bert-fire-detection-binary-classification`](https://huggingface.co/mariossmrs/greek-bert-fire-detection-binary-classification) |
| Fire-type classification | XLM-RoBERTa | [`mariossmrs/greek-xlm-roberta-fire-type-multilabel-classification`](https://huggingface.co/mariossmrs/greek-xlm-roberta-fire-type-multilabel-classification) |

---

## Requirements

Python 3.10+. The classification scripts were run on Google Colab (GPU) with pinned versions:

```bash
pip install "transformers==4.46.3" "datasets==3.1.0" "accelerate==1.1.1" torch scikit-learn \
            gr-nlp-toolkit spacy geopy folium emoji pandas openpyxl matplotlib twikit
python -m spacy download el_core_news_lg
```

> Parts 3, 4 and the hold-out scripts need a GPU for reasonable run times (≈ 4 min per seed and model on a T4). Part 5 queries the public Nominatim instance at 1 request/s; results depend on the state of the gazetteer on the day of the run (paper §4.6).

---

## Reproducing the paper

Run everything from the repository root.

| Step | Script | Paper |
|---|---|---|
| Classification, random split | `scripts/part3_binary_classification.py`, `scripts/part4_multilabel_classification.py` | §4.2, §4.3 |
| Chronological hold-out (both splits, five seeds, CIs, paired bootstrap) | `analysis/holdout_binary.py`, `analysis/holdout_multilabel.py` | §4.8, Tables 7–8 |
| LLM baselines and latency | `analysis/llm_baselines_v3.py`, `analysis/latency_benchmark.py` (needs `ANTHROPIC_API_KEY`, `DEEPINFRA_API_KEY`) | §4.4 |
| Inter-annotator agreement | `analysis/iaa_sample.py` (draws the blind worksheets), `analysis/iaa_compute.py` | §4.1 |
| Near-duplicate leakage | `analysis/leakage_analysis.py` | §3.4.3 |
| Geoparsing and disambiguation | `scripts/part5_geoparsing_and_mapping.py`, `analysis/export_geoparse_errors.py` | §4.5 |
| Threshold sensitivity; gazetteer drift | `analysis/sensitivity_analysis_v4.py`, `analysis/nominatim_drift_check.py` | §4.6 |
| End-to-end replay and map | `analysis/run_simulation.py` (wraps `scripts/part6_simulation.py`) | §4.7 |
| Clustering sensitivity | `analysis/dbscan_sensitivity.py` | §4.7 |

`analysis/iaa_compute.py` and `analysis/leakage_analysis.py` reproduce `data/IAA_results.json` and `data/leakage_analysis_*` exactly from `data/master_cleaned_dataset.csv`; `analysis/dbscan_sensitivity.py` reproduces `data/dbscan_sensitivity.csv` from `data/fire_events_geoparsed.csv`.

---

## Data provenance, ethics and terms of use

- Posts were collected by keyword (fire-related Greek terms and Greeklish hashtags), not by account, from public X posts, using Twikit.
- The release contains post text so that the corpus is usable as a benchmark; usernames are not distributed and in-text mentions are masked. Post text remains the property of its authors and is provided for **non-commercial research use only**. Authors of posts who wish their post removed can contact the corresponding author; removed posts will be dropped in the next release.
- Geolocation is derived from toponyms in the text, never from device positions.
- The demonstration map shows source posts for research purposes; an operational deployment would not (paper §5.5).

---

## Citation

```bibtex
@article{lazanas2026people,
  title     = {People as Sensors: An End-to-End Deep Learning Framework for Near-Real-Time
               Wildfire Detection, Classification, and Geospatial Intelligence from Greek Social Media},
  author    = {Lazanas, Alexios and Samaras, Marios},
  journal   = {AI},
  publisher = {MDPI},
  year      = {2026},
  note      = {under review}
}

@software{lazanas2026pipeline,
  title     = {greek-wildfire-detection-pipeline (v2.2)},
  author    = {Lazanas, Alexios and Samaras, Marios},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21982026}
}
```

---

## Authors

- **Alexios Lazanas** — Department of Mechanical Engineering and Aeronautics, University of Patras — alexlas@upatras.gr (corresponding author)
- **Marios Samaras** — Department of Mechanical Engineering and Aeronautics, University of Patras

## License

Code, annotations, guidelines and evaluation outputs: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Post text: see *Data provenance, ethics and terms of use*.
