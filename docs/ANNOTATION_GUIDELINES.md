# Annotation Guidelines — Greek Wildfire Corpus
### For the second annotator (inter-annotator agreement study)

Version 1.0 · Reconstructed from the coding rules used for the original annotation.

---

## Before you start — three rules that protect the validity of the exercise

1. **Do not open `IAA_key_DO_NOT_OPEN_BEFORE.xlsx`** until you have finished labelling
   every sheet. It contains the first annotator's labels. Seeing them makes the
   agreement coefficient meaningless.
2. **Judge each post in isolation.** Do not scroll back to compare with similar posts
   you labelled earlier, and do not use knowledge of what fire was happening at that
   date. Each row is an independent judgment.
3. **Do not look anything up.** No searching the toponym, no checking news archives for
   what actually happened. You label only what the text itself supports. (The one
   exception is the geoparsing sheet, where coordinates must be looked up — see Task 3.)

If you are unsure, put an `x` in the `uncertain (x)` column and still give your best
label. Never leave a label blank. The uncertain flag is analysed separately and is
useful evidence about which rules are ambiguous.

---

## Task 1 — Binary: is this an active fire report?
Sheet `1_binary`, 400 posts. Fill `is_fire_A2` with **1** or **0**.

### Label 1 (active-fire-related)
The post conveys **real-time situational awareness of an ongoing fire in Greece**.
Typical content: where the fire is, how it is spreading, evacuation warnings, road
closures, requests for help, reports of firefighting resources deployed.

The fire must be:
- **actual** (not metaphorical, not hypothetical),
- **ongoing or just-started** (not a retrospective account),
- **in Greece**.

### Label 0 (not active-fire-related)
Everything else, including:
- **metaphorical uses** — "η αγορά πήρε φωτιά", "φωτιά στα σεντόνια", "καίγομαι στο διάβασμα";
- **past fires** — anniversaries, court cases, reconstruction reports, "πέρσι κάηκε";
- **political or emotional commentary** using fire vocabulary — blame, government
  criticism, general complaints about fire policy, even during a fire season;
- **prevention, preparedness or risk warnings** with no fire currently burning
  (e.g. a high-risk-index map, "προσοχή, κίνδυνος αναζωπύρωσης");
- **fires outside Greece**;
- fire-adjacent noise: smoke of unknown origin with no fire claim, controlled burns.

### Boundary cases — decide as follows
| Situation | Label |
|---|---|
| Official 112 evacuation message for an ongoing fire | **1** |
| "Ξέσπασε φωτιά στη Χ" with no further detail | **1** |
| Fire reported as fully extinguished ("τέθηκε υπό έλεγχο") | **0** |
| Reignition (αναζωπύρωση) reported as happening now | **1** |
| Thanking firefighters while the fire still burns | **1** |
| Thanking firefighters after it is over | **0** |
| Reposting a news headline about a fire burning now | **1** |
| Asking "τι γίνεται με τη φωτιά;" during an event | **1** |
| Photo caption with no text indicating an active fire | **0** |
| Fire in Greece but clearly a small controlled incident (e.g. rubbish bin) | **1** if the post reports it as an active fire incident |

---

## Task 2 — Multilabel: what type of fire?
Sheet `2_multilabel`, 300 posts, all of which the first annotator judged to be active
fires. Fill **both** `is_wildland_A2` and `is_urban_A2` with **1** or **0**.

This yields four possible states:

| is_wildland | is_urban | Meaning |
|---|---|---|
| 1 | 0 | **Wildland** — forest, brush, grassland, agricultural land |
| 0 | 1 | **Urban** — building, house, vehicle, factory, rubbish, urban structure |
| 1 | 1 | **Mixed** — fire affecting both, e.g. wildland fire reaching houses |
| 0 | 0 | **Unidentified** — an active fire, but the text gives no explicit type cue |

### The three rules that govern this task

**Rule 1 — Explicit textual content only.**
Label what the post says. If it does not say or clearly imply the type, the answer is
`0,0` (unidentified). "Unidentified" is a legitimate and common label — do not guess to
avoid it.

**Rule 2 — Implicit inference is allowed only when the context is unambiguous.**
Permitted inferences:
- mention of **aerial support** (canadair, ελικόπτερα, πυροσβεστικά αεροσκάφη) → wildland;
- **specialised forest crews** (πεζοπόρα τμήματα, δασοκομάντος) → wildland;
- **evacuation of a settlement** because of an approaching fire front → mixed, only if
  the text indicates the fire has reached or is threatening the built area;
- fire brigade attending a **flat/shop/vehicle** → urban.

Not permitted: inferring from season, from the scale of response alone, or from what you
personally remember about the incident.

**Rule 3 — Geographic knowledge is deliberately excluded.**
A toponym you know to be mountainous, forested, or a city centre does **not** by itself
determine the label. If the post says "φωτιά στον Υμηττό" and nothing else, that is
`0,0`, not wildland — even though you know Ymittos is a mountain. This rule exists to
stop the model memorising place→type associations, so it matters that you apply it
strictly even when it feels wrong.

### Boundary cases
| Situation | Label |
|---|---|
| "Φωτιά σε δασική έκταση στη Χ" | 1,0 |
| "Φωτιά σε διαμέρισμα" | 0,1 |
| "Η φωτιά πλησιάζει τα πρώτα σπίτια" | 1,1 |
| "Εκκενώνεται ο οικισμός Χ λόγω της πυρκαγιάς" | 1,1 |
| "Φωτιά στη Χ, επιχειρούν 20 πυροσβέστες" | 0,0 |
| "Φωτιά σε αποθήκη σε βιομηχανική περιοχή" | 0,1 |
| "Φωτιά σε ξερά χόρτα δίπλα στον δρόμο" | 1,0 |
| "Φωτιά σε αυτοκίνητο στην εθνική οδό" | 0,1 |
| Canadair mentioned, no other cue | 1,0 |
| "Καίγεται το βουνό" | 1,0 (explicit in text, not geographic knowledge) |

---

## Task 3 — Geoparsing: where is the fire?
Sheet `3_geoparsing`, all 147 gold items. Fill `toponym_A2`, `lat_A2`, `lon_A2`.

For each post, identify **the toponym that indicates where the fire actually is**, then
record its coordinates.

- Write the toponym in its **nominative form** in Greek (e.g. `Γλυκά Νερά`, not `Γλυκών Νερών`).
- If the post contains several place names, choose the one that locates **the fire**, not
  a secondary reference (a road the reporter is on, where a witness lives, a comparison
  to a past fire elsewhere).
- If the post contains **no usable location**, write `NONE` in `toponym_A2` and leave the
  coordinates blank.
- If the location is a facility rather than a settlement (e.g. `Νεκροταφείο Γλυκών Νερών`,
  `Λόφος Φιλοπάππου`), record the facility as the toponym.
- Coordinates: look up the toponym and give decimal degrees to **four decimal places**.
  Use the centre of the named place. Any standard gazetteer or map service is acceptable —
  note which one you used in the `note` column if it was not obvious.

This is the only task where looking things up is expected, because coordinates cannot be
recalled from text alone. Do not, however, look up *which fire* occurred on that date.

---

## After labelling

Save the completed worksheet and return it. Agreement will be computed as:
- **Cohen's κ** for Task 1;
- **per-label Cohen's κ** plus **Krippendorff's α** for Task 2;
- **exact toponym match, normalised match, and distance-based agreement** for Task 3.

Disagreements will then be reviewed jointly, resolved by discussion, and the adjudicated
labels become the final gold standard. Interpretation follows the Landis–Koch scale
(0.01–0.20 slight; 0.21–0.40 fair; 0.41–0.60 moderate; 0.61–0.80 substantial;
0.81–1.00 almost perfect).

**A note on what a low κ would mean.** If agreement on some label turns out to be poor,
that is a finding, not a failure — it says the coding rule for that category is
underspecified. The honest response is to refine the guideline, document the change, and
report both the original and revised agreement. Do not adjust your labels to match the
first annotator's in order to raise the number.
