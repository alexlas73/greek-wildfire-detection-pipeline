Citizens as Sensors: Fine-Tuned Transformers for Greek-Language Wildfire Detection, Toponym Disambiguation, and Real-Time Mapping

Wildfires are increasing around the world, and usually the first sign of a fire is a citi-zen's social media posting, not a satellite. This paper conceptualizes "Citizens as sen-sors" as a complete, operational system: an end-to-end, Greek-language NLP pipeline that transforms raw, unlabeled X posts into a live, geolocated wildfire map, with no human intervention at inference. The pipeline identifies active fires, categorizes them as wildland, urban, or mixed, resolves the referenced toponym, and groups fire reports into incident zones on an interactive, time-sliced satellite map. On a corpus of 3,277 manu-ally annotated Greek posts, validated by a formal inter-annotator audit (Cohen's κ = 0.978), fine-tuned transformers achieved F2 = 0.911 for fire detection and macro-F1 = 0.874 for fire-type classification, decisively outperforming two LLMs and classical base-lines on the same test sets, while being more accurate, over twice as fast, and free of per-call cost. Spatiotemporal toponym disambiguation raised geocoding accuracy from 54.68% to 80.58% (p < 0.001), and a later repeatability test revealed measurable gazetteer drift. Applied to raw posts from 12 August 2024, the pipeline automatically identified and mapped a major Attica wildfire in near real time, resolving hundreds of citizen re-ports into a single incident cluster.

# Citizens as Sensors: Greek-Language Wildfire Detection Pipeline

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21982027.svg)](https://doi.org/10.5281/zenodo.21982027)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

End-to-end NLP pipeline that transforms raw Greek social-media posts into a live, geolocated wildfire map — no human intervention at inference time.

> **Associated paper:** Lazanas, A.; Samaras, M. Citizens as Sensors: Fine-Tuned Transformers for Greek-Language Wildfire Detection, Toponym Disambiguation, and Real-Time Mapping. *AI* 2026 (MDPI, JCR Q1). https://doi.org/10.5281/zenodo.21982027

---

## Pipeline Overview

```
Raw X posts
    │
    ▼
Part 1 · Scraping          (Twikit)
    │
    ▼
Part 2 · Text Cleaning     (4 levels: light → advanced → hard)
    │
    ▼
Part 3 · Binary Classification    GreekBERT  →  F2 = 0.911
    │         (active fire?)
    ▼
Part 4 · Multilabel Classification    XLM-RoBERTa  →  Macro-F1 = 0.874
    │         (wildland / urban / mixed)
    ▼
Part 5 · Geoparsing & Mapping   NER → Nominatim → spatiotemporal disambiguation
    │         →  DBSCAN clustering (ε = 15 km)  →  Folium interactive map
    ▼
Part 6 · Simulation        (end-to-end demo on Varnavas/Attica 2024 fire)
```

---

## Repository Structure

```
greek-wildfire-detection-pipeline/
├── part1_x_scraping.py               # X (Twitter) post collection via Twikit
├── part2_text_cleaning.py            # Greek text normalisation (4 levels)
├── part3_binary_classification.py    # Active-fire binary classifier (GreekBERT / SVM)
├── part4_multilabel_classification.py# Fire-type multilabel classifier (XLM-RoBERTa / SVM)
├── part5_geoparsing_and_mapping.py   # NER + disambiguation + DBSCAN + Folium map
├── part6_simulation.py               # End-to-end simulation on unlabelled posts
├── data/
│   ├── tweet_ids.txt                 # Post IDs for corpus re-hydration (raw text not distributed)
│   ├── ground_truth_geoparsing.xlsx  # 147-item geoparsing gold standard
│   ├── ground_truth_classification.xlsx  # Post-level binary + multilabel labels
│   ├── IAA_results.json              # Inter-annotator agreement scores
│   ├── IAA_adjudication.json         # Adjudication outcomes
│   └── ANNOTATION_GUIDELINES.md     # Full annotation protocol
└── README.md
```

> **Note on raw tweet text:** In accordance with the X Developer Agreement and Policy, full post text is not redistributed. The file `tweet_ids.txt` contains the post identifiers; the original text can be re-collected using the [Academic Research Product Track](https://developer.twitter.com/en/products/twitter-api/academic-research) or a compatible scraping tool.

---

## Pre-trained Models

Both fine-tuned models are publicly available on Hugging Face:

| Task | Model | Hugging Face ID |
|---|---|---|
| Binary (active fire?) | GreekBERT | [`mariossmrs/greek-bert-fire-detection-binary-classification`](https://huggingface.co/mariossmrs/greek-bert-fire-detection-binary-classification) |
| Multilabel (fire type) | XLM-RoBERTa | [`mariossmrs/greek-xlm-roberta-fire-type-multilabel-classification`](https://huggingface.co/mariossmrs/greek-xlm-roberta-fire-type-multilabel-classification) |

---

## Requirements

Python 3.10+ recommended. Install all dependencies:

```bash
pip install twikit transformers datasets torch scikit-learn \
            gr-nlp-toolkit spacy geopy folium emoji \
            pandas openpyxl matplotlib
python -m spacy download el_core_news_lg
```

> **GPU:** Parts 3, 4, and 6 run substantially faster on a CUDA-enabled GPU. The pipeline is functional on CPU (Parts 3 and 4 take ~0.5 s/post on CPU; ~5–10× faster on GPU).

---

## Quickstart: Reproduce the Paper's Results

### 1. Clone the repository

```bash
git clone https://github.com/alexlas73/greek-wildfire-detection-pipeline.git
cd greek-wildfire-detection-pipeline
```

### 2. Re-hydrate the corpus (optional)

If you have X API access, re-collect the post text using the IDs in `data/tweet_ids.txt`. Place the resulting dataset at `data/master_cleaned_dataset.xlsx` (see column schema in the annotation guidelines).

### 3. Run binary classification

```bash
python part3_binary_classification.py
```

Trains and evaluates GreekBERT and baseline SVMs on the 70/15/15 stratified split (seed 42). Outputs per-seed metrics across 5 runs.

### 4. Run multilabel classification

```bash
python part4_multilabel_classification.py
```

Trains and evaluates XLM-RoBERTa on active-fire posts only, with per-label threshold tuning (seed 789).

### 5. Run geoparsing and mapping

```bash
python part5_geoparsing_and_mapping.py
```

Evaluates geoparsing on the 147-item gold standard; generates the interactive Folium map (`fire_events_map.html`).

### 6. Run the end-to-end simulation

```bash
python part6_simulation.py
```

Processes the August 2024 Attica fire dataset end-to-end and produces `fire_events_geoparsed.csv` and `fire_events_map.html`.

---

## Reproducing Additional Experiments

The following analysis scripts reproduce the paper's supplementary experiments. Run them from the repo root after completing Steps 3–5 above:

| Script | Experiment | Section |
|---|---|---|
| `analysis/iaa_compute.py` | Inter-annotator agreement (Cohen's κ, Krippendorff's α) | §4.1 |
| `analysis/leakage_analysis.py` | Near-duplicate leakage check across splits | §3.4.3 |
| `analysis/sensitivity_analysis.py` | One-at-a-time threshold sensitivity sweep | §4.6 |
| `analysis/nominatim_drift_check.py` | Gazetteer drift quantification | §4.6 |
| `analysis/llm_baselines.py` | LLM baseline comparison (Claude + Llama) | §4.4 |
| `analysis/latency_benchmark.py` | Per-post inference latency measurement | §4.4 |

> **LLM baselines:** require an Anthropic API key (`ANTHROPIC_API_KEY`) and a DeepInfra API key (`DEEPINFRA_API_KEY`) set as environment variables. Expected cost: < $1 for the full test-set evaluation.

---

## Data Availability

The gold-standard annotation datasets, inter-annotator agreement materials, and all analysis scripts are archived at:

> Lazanas, A.; Samaras, M. alexlas73/greek-wildfire-detection-pipeline (v2.0). *Zenodo* **2026**. https://doi.org/10.5281/zenodo.21982027

---

## Citation

If you use this code or data in your research, please cite:

```bibtex
@article{lazanas2026citizens,
  title     = {Citizens as Sensors: Fine-Tuned Transformers for Greek-Language
               Wildfire Detection, Toponym Disambiguation, and Real-Time Mapping},
  author    = {Lazanas, Alexis and Samaras, Marios},
  journal   = {AI},
  publisher = {MDPI},
  year      = {2026},
  doi       = {10.5281/zenodo.21982027}
}
```

---

## Authors

- **Alexis Lazanas** — Department of Mechanical Engineering and Aeronautics, University of Patras, Rion-Patras 26500, Greece ([alexlas73@upatras.gr](mailto:alexlas73@upatras.gr)) — *corresponding author*
- **Marios Samaras** — Department of Mechanical Engineering and Aeronautics, University of Patras

---

## License

This repository is licensed under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) licence. You are free to share and adapt the material for any purpose, provided appropriate credit is given.

