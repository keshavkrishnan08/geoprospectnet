# GeoProspectNet

**Geothermal prospecting at continental scale. A multi-modal deep network ranks 235,470 cells of
the western United States, and it is judged on wells permitted after its training inventory, not
on test cells it has effectively already seen.**

![Box plots of continental percentile rank by validation cohort: random deep negatives sit near the bottom of the map, Colorado Plateau and random western cells fall in the middle, known geothermal cells mostly rank above the 90th percentile, and all 22 post-2008 hold-out wells rank above the 90th percentile](outputs/figures/fig3_cohort_distributions.png)

The hard part of geothermal exploration is not finding hot regions on a map. It is picking the
next drilling target among thousands of plausible cells. Most machine-learning prospectivity maps
are graded on held-out cells near known fields, where almost any model scores perfectly.
GeoProspectNet aligns gravity, magnetics, heat flow, spring geochemistry and fault geology in one
learned embedding, ranks every cell on a continental scale, and is tested on 22 Nevada wells that
the model never saw, including Zanskar's Big Blind discovery. A manuscript on this work has been
submitted to *Geothermics*.

```
USGS gravity, magnetics, SMU heat flow (6-channel 32x32 patches)
  + GEOTHERM and thermal-spring geochemistry (9 features)
  + Quaternary faults, Holocene volcanoes, elevation (11 features)
    ->  per-modality encoders -> 128-d embeddings, aligned by a contrastive (InfoNCE) loss
    ->  attention fusion -> 256-d -> blend with the 8 neighbouring cells -> classifier
    ->  score all 235,470 cells (4 km grid) -> continental percentile rank
    ->  DBSCAN clusters of top cells, kept only where two model variants agree -> 33 sites
    ->  Monte Carlo volumetric capacity per site -> SHA-256-hashed candidate manifest
```

Score the map on wells it never saw.

## The core idea

1. **Different surveys see the same subsurface.** Each modality has its own encoder, and a
   contrastive loss pulls the embeddings of the same cell together across modalities before they
   are fused.
2. **In-distribution accuracy is not evidence.** Every learned method, including random forest
   and XGBoost, reaches 1.00 AUROC on the random test split. The post-2008 hold-out wells are
   what separate the methods.
3. **Rank, then check specificity.** The model's output is used as a continental percentile, and
   negative-control cohorts chosen on geological grounds (deep random cells, cratonic low heat
   flow, Colorado Plateau) test whether high ranks are earned.
4. **Freeze candidates before outcomes.** The 33 sites that both model variants flag, with their
   capacity estimates, are written to a hashed manifest with a stated rule for when a site counts
   as verified or falsified.

## What it does (measured on the post-2008 hold-out and negative cohorts)

| Check | Metric | Result |
|---|---|---|
| Post-2008 hold-out, 22 Nevada wells never used in training | share in the top 10% of the map / mean percentile | **22 of 22** / **98.2** |
| Zanskar Big Blind well (permit 1678) | continental percentile | 97.4 |
| Random forest and XGBoost on the same hold-out | top-10% capture (test-split AUROC 1.00) | **0%** |
| Logistic regression / heat-flow rule | top-10% capture | 91% / 82% |
| Random deep negatives (NC3, n = 100) | mean percentile | **5.3** |
| Cratonic, low-heat-flow cells (NC4, n = 100) | mean percentile | 10.1 |
| Colorado Plateau interior (NC5, n = 100) | mean percentile | 43.0 |
| Great Basin vs seven published Mordensky et al. maps | NC3 mean percentile, ours vs theirs | **1.2** vs 31.1 to 40.8 |
| Seed stability (cpu_margin, 4 seeds) | hold-out top-10% capture | 77% to 100% (mean 92%) |
| Candidate sites that both variants flag | count / summed P50 capacity | 33 / 6,607 MWe (P10 2,492, P90 17,130) |

The temporal hold-out is the result that matters. Tree ensembles that score perfectly on the
random split put none of the post-2008 wells in the top 10%, while GeoProspectNet puts all of
them there and keeps random deep cells near the bottom of the ranking. Hold-out scores are in
`outputs/results/temporal_holdout.csv`, the baseline comparison in `baselines.csv`, and the
cohorts in `negative_controls_cpu_max.csv` and `hard_negative_cohorts.csv`.

## How it works

**Data.** A 4 km grid over 31 to 49 N and 125 to 103 W, 235,470 cells. Geophysics: USGS Bouguer
and isostatic gravity, magnetic anomaly, interpolated SMU heat flow (leave-one-out check over
3,289 stations: RMSE 8.7 mW/m², R-squared 0.61) and derived temperature channels, as 6-channel
32x32 patches. Geochemistry: GEOTHERM and thermal-spring records within 50 km (9 features).
Geology: Quaternary faults, Holocene volcanoes and SRTM elevation (11 features). Sources,
versions and licences are listed in `DATA.md`.

**Labels.** Positive cells lie within 5 km of one of 280 known fields (1,370 cells). Negative
cells are more than 50 km from any positive, more than 25 km from any thermal spring, and below
60 mW/m² heat flow where heat flow is known. The configs ask for up to 80 negatives per positive;
the eligible pool caps the reported model at 26,213 (about 19 per positive).

**Model.** A CNN encodes the geophysics patches and MLPs encode the tabular modalities, each to a
128-d embedding. Pairwise InfoNCE aligns modalities (temperature initialized at 0.07), attention
fusion projects to 256-d, and each cell's embedding is mixed with the mean of its 8 neighbours
(alpha 0.3) before the classifier. The reported variants train with BCE, focal and margin losses
(positives pushed above 0.7, negatives below 0.3) plus 0.5 times the contrastive loss, using
AdamW (learning rate 1e-3, cosine schedule) for 80 epochs (cpu_max) or 60 (cpu_margin).

**Discovery.** Monte Carlo dropout over the full grid, cells above a variant-specific probability
threshold, DBSCAN clustering (15 km radius, at least 3 cells), a 25 km exclusion around known
fields and a geological plausibility filter. A cpu_margin site is kept only if a cpu_max site lies
within 25 km.

**Capacity.** A volumetric Monte Carlo per site (2,000 draws): conversion efficiency of 0.45 times
the Carnot efficiency (capped at 0.18), a recovery factor that depends on plant type, a lognormal
reservoir thickness around 250 m, and reservoir temperature with a 20 °C standard deviation,
taken from geothermometers at 27 sites and from a heat-flow proxy at 6.

**Manifest.** `outputs/results/preregistration_manifest.json` stores the 33 sites, their
capacities, a SHA-256 hash and a frozen timestamp (2026-05-13). A site is verified if a producing
well or successful exploration result is reported within 25 km afterwards, and falsified if a dry
hole or negative survey is reported within 5 km. `tests/test_manifest_stability.py` checks that the
hash is stable.

## Run it

Trained checkpoints and every results table are committed, so evaluation can rerun without
training. The 5.4 GB geophysics patch array and a few large raw files are not in git and are
rebuilt by the processing step.

```bash
pip install -r requirements.txt
bash scripts/01_download_data.sh     # automated public downloads; prints the manual ones
bash scripts/02_process_data.sh      # grid, modalities, labels, splits, neighbours
python -m src.training.train --config configs/cpu_max.yaml --random --seed 42
python -m src.evaluation.temporal_holdout
python -m src.evaluation.baselines --config configs/cpu_max.yaml
python -m src.evaluation.negative_controls --config configs/cpu_max.yaml \
    --ckpt outputs/checkpoints/random_cpu_max_seed42.pt
pytest tests/
```

`notebooks/kaggle_full_pipeline.ipynb` runs every stage end to end on a GPU (both variants,
multi-seed, baselines, cohorts, discovery, capacity, manifest, figures) and skips any step whose
outputs already exist.

## Layout

```
src/
  data/             downloads, grid, per-modality processing, labels, splits, neighbours
  models/           encoders, contrastive loss, attention fusion, spatial smoothing, classifier, baselines
  training/         training loop, multi-seed runs, separation tuning, single-modality ablation, eastern fine-tune
  evaluation/       temporal hold-out, baselines, negative and hard-negative cohorts, Mordensky comparison,
                    discovery, capacity Monte Carlo, field checks, manifest, eastern US test, calibration
  visualization/    maps, tables and figures
configs/            training configs (cpu_smoke through cpu_max, plus ablation variants)
data/               processed grid arrays and splits, field metadata, public raw downloads
outputs/
  checkpoints/      trained models for each variant, seed, fold and ablation
  results/          every table behind the numbers above (CSV and JSON) and continental scores
  maps/             prospectivity and uncertainty GeoTIFFs
  figures/          PNG and PDF figures
notebooks/          kaggle_full_pipeline.ipynb (end to end) and earlier pipeline notebooks
scripts/            numbered shell stages and a smoke test
tests/              data integrity, manifest stability, model loading, reproducibility
paper/              early LaTeX draft and bibliography
RESULTS_SUMMARY.md  index of results files
DATA.md             dataset sources, versions and licences
```

## Honesty

* In-distribution numbers are saturated and carry no information: AUROC 1.00 for every learned
  method, and calibration error 0.0007 on labelled cells. The negatives are defined to be far
  from fields and springs, so they are easy. Only the hold-out and the negative cohorts test the
  model.
* Raw probabilities are not meaningful across the map. In the measured variants 77% to 87% of
  cells score above 0.5, and Colorado Plateau cells average 0.92 while ranking at the 43rd
  percentile. Use the percentile rank.
* The hold-out tests near-field drilling more than blind discovery. 20 of the 22 wells lie within
  25 km of a field in the training inventory (median 13.9 km), and 8 of them are one cluster of
  Nevada Bureau of Mines and Geology wells, so they represent far fewer than 22 independent
  systems.
* Candidate #45, the largest site (1,298 MWe P50, about 20% of the total), sits 13.5 km from the
  Beowawe field in the training inventory, and three of the five permits near it are hold-out
  wells. Its nearby permits are not independent confirmation.
* Capacity figures are screening estimates and depend heavily on reservoir temperature: shifting
  every site by -20 °C cuts the summed P50 from 6,578 to 4,267 MWe. The P10 and P90 totals are
  sums of per-site percentiles, not percentiles of the total.
* Only the contrastive loss measurably matters. Without it, hold-out capture drops from 100% to
  77%. Removing spatial smoothing or attention fusion gives identical scores to the full model,
  and the satellite thermal modality is implemented but switched off (`use_thermal: false`).
  `paper/main.tex` predates these results, claims 2% to 8% AUROC gains from those components and
  still has placeholders.
* The model does not transfer east: seven eastern US springs average the 49th percentile and
  only 1 of 7 lands in the top 10%. A fine-tuned eastern model finds no electric-grade resource.
* Two known pipeline bugs: `negative_controls.py` copies the hold-out row from a cached file
  instead of rescoring, so older summaries report 95.7 for the hold-out where the current cpu_max
  scores give 98.2; and the negative-ratio sweep trained four identical models because splits
  were cached. `scripts/run_all.sh` uses `configs/default.yaml`, not the cpu_max and cpu_margin
  configs behind the reported numbers.

MIT licensed. Datasets keep their original licences; see `DATA.md`.
