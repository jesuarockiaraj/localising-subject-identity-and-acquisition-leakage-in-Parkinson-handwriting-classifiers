# Full reproduction commands, table by table

All commands assume you are in the repository root with `code/` on your path
(`cd code` or add it to `PYTHONPATH`), the datasets downloaded and laid out
per the main `README.md`, and the derived `Datasets_2a/` / `Datasets_marker/`
/ `Datasets_marker_2a/` mirrors already generated (commands for that are in
the README's "Datasets" section — run those first if you haven't).

Every `--seeds` value below is copied directly from what was actually run
for the numbers in the manuscript, not a guess.

---

## Table 1 — Baseline replication

**HandPD spiral, P1 vs. P2 (all rows, verified: seeds 999, 1001–1003):**
```bash
python pd_loso_pipeline_cross_dataset.py --dataset handpd --root Datasets/HandPD \
    --task spiral --models efficientnet_b0 --seeds 999 1001 1002 1003
```

**Cross-cohort HandPD→Kaggle, EfficientNet-B0 (verified):**
```bash
python pd_loso_pipeline_cross_dataset.py --dataset handpd --root Datasets/HandPD \
    --task spiral --test-root Datasets/kaggle --test-task spiral --test-name kaggle \
    --models efficientnet_b0 --seeds 999 1001 1002 1003
```

**Cross-cohort Kaggle→HandPD, EfficientNet-B0 (verified):**
```bash
python pd_loso_pipeline_cross_dataset.py --dataset kaggle --root Datasets/kaggle \
    --task spiral --test-root Datasets/HandPD --test-task spiral --test-name handpd \
    --models efficientnet_b0 --seeds 999 1001 1002 1003
```

**VGG-16 / ResNet-50 cross-cohort cells (both directions) — NOT independently
re-verified in this study** (see Limitations). The exact original invocation
that produced the numbers printed in the manuscript is unknown/under-specified
and is exactly the provenance issue the manuscript discloses — do not treat a
fresh run of the command below as reproducing the *same* numbers, only as
regenerating the *comparison* honestly from scratch:
```bash
python pd_loso_pipeline_cross_dataset.py --dataset handpd --root Datasets/HandPD \
    --task spiral --test-root Datasets/kaggle --test-task spiral --test-name kaggle \
    --models vgg16 resnet50 --seeds 1 2 3 4 5
python pd_loso_pipeline_cross_dataset.py --dataset kaggle --root Datasets/kaggle \
    --task spiral --test-root Datasets/HandPD --test-task spiral --test-name handpd \
    --models vgg16 resnet50 --seeds 1 2 3 4 5
```

**Figure 3** (cross-cohort AUC by architecture, visualising Table 1) — edit
the recorded `H2K`/`K2H` values at the top of `make_baseline_fig.py` to
match the table above, then:
```bash
python make_baseline_fig.py
```

---

## Table 2 — Identity/cohort probe, raw images

```bash
python leak_decomposition_probe.py \
    --handpd-glob "Datasets/HandPD/**/*.jpg" \
    --kaggle-glob "Datasets/kaggle/drawings/**/*.png"
```
Prints subject re-ID@1 and cohort balanced accuracy per mode (full /
background / trace) directly to stdout — this table is transcribed from
console output, not saved to a file by the script as shipped.

---

## Table 3 — Stratified re-identification + label-shuffle control

`leak_decomposition_probe.py` caches its extracted features on request
(`--save-features`), which is what `stratified_reid.py` needs:

```bash
python leak_decomposition_probe.py \
    --handpd-glob "Datasets/HandPD/**/*.jpg" \
    --kaggle-glob "Datasets/kaggle/drawings/**/*.png" \
    --modes full --save-features
python stratified_reid.py
```
The first command writes `scratch_feats_full.npy` / `scratch_subjects_full.npy`
in the current directory; `stratified_reid.py` reads them directly (no
flags of its own). Verified end to end: this reproduces every number in
Table 3 exactly, including the pooled row matching Table 2's "Full" row to
three decimal places (0.327) — Table 3 was regenerated with this exact
pipeline run after an earlier version was found to be out of sync with
Table 2 (see the manuscript's Table 3 caption for the disclosure).

---

## Table 4 / Table 6 — Rendering-invariance ladder (raw / suppressed)

(Note: this used to be mislabeled "Table 4 / Table 5" here — Table 5 is
actually the separate probe-supp table below. Fixed.)

**Raw images (Table 4):**
```bash
python rendering_invariance_probe.py \
    --handpd-glob "Datasets/HandPD/**/*.jpg" \
    --kaggle-glob "Datasets/kaggle/drawings/**/*.png" \
    --out rendering_invariance_results.json
```

**Suppressed images (Table 6 / "rendering-supp"):**
```bash
python rendering_invariance_probe.py \
    --handpd-glob "Datasets_2a/HandPD/**/*.jpg" \
    --kaggle-glob "Datasets_2a/kaggle/drawings/**/*.png" \
    --out rendering_invariance_results_suppressed.json
```

**Figure 5** (raw vs. suppressed ladder, side by side) — edit the recorded
`RAW_REID`/`RAW_AUC`/`SUPP_REID`/`SUPP_AUC` values at the top of
`make_rendering_ladder_fig.py` to match Tables 4 and 6, then:
```bash
python make_rendering_ladder_fig.py
```

---

## Table 5 (probe-supp) — Identity/cohort probe, suppressed images

```bash
python leak_decomposition_probe.py \
    --handpd-glob "Datasets_2a/HandPD/**/*.jpg" \
    --kaggle-glob "Datasets_2a/kaggle/drawings/**/*.png"
```

---

## Aspect-ratio / resolution-artefact control (Section 4.6)

```bash
python aspect_ratio_control.py
```
No flags — paths are hardcoded to `Datasets/HandPD` and `Datasets/kaggle`
inside the script.

---

## HandPD clean pen-only split (Section 4.7)

First verify the colour direction (dark ink = printed guide, blue ink = pen
stroke) before trusting the split below:
```bash
python verify_guide_color.py
```
Then reproduce the clean pen-only re-ID number:
```bash
python handpd_clean_split.py
```
No flags — same hardcoded paths as above.

---

## Table 8 — Guide-excluded, colour-separated pen-only rendering-invariance ladder

**Raw images:**
```bash
python rendering_invariance_penonly.py \
    --handpd-glob "Datasets/HandPD/**/*.jpg" \
    --kaggle-glob "Datasets/kaggle/drawings/**/*.png" \
    --out rendering_invariance_penonly_results.json
```

**Suppressed images** (reconstructs the guide-excluded pen-only view by
isolating the pen mask on the raw image first, then rendering it through
`mech2_suppression.render()` — equivalent to what a corrected suppression
pipeline would output, without regenerating `Datasets_2a/`):
```bash
python rendering_invariance_penonly_suppressed.py \
    --handpd-glob "Datasets/HandPD/**/*.jpg" \
    --kaggle-glob "Datasets_2a/kaggle/drawings/**/*.png" \
    --out rendering_invariance_penonly_suppressed_results.json
```

---

## Table 9 — Background-only disease decodability (raw vs. suppressed, both cohorts)

```bash
python background_disease_probe.py
```
No flags — paths for both cohorts (raw and suppressed) are hardcoded inside
the script. Writes `background_disease_probe_results.json`.

**Suppressed P1 vs. P2 identity-leakage gap** (aggregates existing
predictions from `results_2a_verify_seeds_999_1001_1002_1003/` — no new
training needed):
```bash
python compute_suppressed_p1.py
```

---

## Table 10 — Calibration (Brier / ECE)

```bash
python make_calibration_table.py
```
Reads `results/results_baseline/predictions_handpd_spiral.csv` (P1/P2) and
`predictions_cross_handpd_spiral_to_kaggle_spiral.csv` /
`predictions_cross_kaggle_spiral_to_handpd_spiral.csv` (both cross-cohort
directions) — the same *original*, pre-correction predictions the
manuscript's own text says this table reuses, not the newly verified ones
— aggregates to one mean probability per (seed, subject), computes Brier
score and 10-equal-width-bin ECE per seed, and reports mean &plusmn; SD.
Writes `calibration_table.csv`. **Verified exact**: reproduces all eight
numbers in Table 8 to three decimal places.

**Figure 7** (Brier/ECE bar chart) — edit the recorded `BRIER`/`ECE` values
at the top of `make_calibration_fig.py` to match Table 8, then:
```bash
python make_calibration_fig.py
```

---

## Table 11 — Generalisation to meander / wave

**HandPD meander, raw P1/P2 (original, five seeds):**
```bash
python pd_loso_pipeline_cross_dataset.py --dataset handpd --root Datasets/HandPD \
    --task meander --models efficientnet_b0 --seeds 1 2 3 4 5
```
**HandPD meander, suppressed:**
```bash
python pd_loso_pipeline_cross_dataset.py --dataset handpd --root Datasets_2a/HandPD \
    --task meander --models efficientnet_b0 --seeds 1 2 3 4 5
```
**Kaggle wave, raw:**
```bash
python pd_loso_pipeline_cross_dataset.py --dataset kaggle --root Datasets/kaggle \
    --task wave --models efficientnet_b0 --seeds 1 2 3 4 5
```
**Kaggle wave, suppressed:**
```bash
python pd_loso_pipeline_cross_dataset.py --dataset kaggle --root Datasets_2a/kaggle \
    --task wave --models efficientnet_b0 --seeds 1 2 3 4 5
```
**Paired comparison (repeat for wave with its own prediction files):**
```bash
python compare_2a_vs_baseline.py --protocol P2_subject --model efficientnet_b0 \
    --baseline results/predictions_handpd_meander_raw.csv \
    --treat    results/predictions_handpd_meander_2a.csv
```
(rename/move each run's `results/predictions_handpd_meander.csv` before the
next run, per the "GOTCHA" in the harness noted below.)

---

## Table 7 — Causal comparison, all three rows

**1. Within-cohort HandPD spiral, suppressed (verified):**
```bash
python pd_loso_pipeline_cross_dataset.py --dataset handpd --root Datasets_2a/HandPD \
    --task spiral --models efficientnet_b0 --seeds 999 1001 1002 1003
python compare_2a_vs_baseline.py --protocol P2_subject --model efficientnet_b0 \
    --baseline results/results_verify_combined_4seed/predictions_handpd_spiral.csv \
    --treat    results/results_2a_verify_seeds_999_1001_1002_1003/predictions_handpd_spiral.csv
```

**2. Cross-cohort HandPD→Kaggle, suppressed (verified):**
```bash
python pd_loso_pipeline_cross_dataset.py --dataset handpd --root Datasets_2a/HandPD \
    --task spiral --test-root Datasets_2a/kaggle --test-task spiral --test-name kaggle \
    --models efficientnet_b0 --seeds 999 1001 1002 1003
python compare_2a_vs_baseline.py --model efficientnet_b0 \
    --baseline results/results_verify_cross_h2k_4seed/predictions_cross_handpd_spiral_to_kaggle_spiral.csv \
    --treat    results/results_2a_verify_cross_h2k_4seed/predictions_cross_handpd_spiral_to_kaggle_spiral.csv
```

**3. Cross-cohort Kaggle→HandPD, suppressed (verified):**
```bash
python pd_loso_pipeline_cross_dataset.py --dataset kaggle --root Datasets_2a/kaggle \
    --task spiral --test-root Datasets_2a/HandPD --test-task spiral --test-name handpd \
    --models efficientnet_b0 --seeds 999 1001 1002 1003
python compare_2a_vs_baseline.py --model efficientnet_b0 \
    --baseline results/results_verify_cross_k2h_4seed/predictions_cross_kaggle_spiral_to_handpd_spiral.csv \
    --treat    results/results_2a_verify_cross_k2h_4seed/predictions_cross_kaggle_spiral_to_handpd_spiral.csv
```

**Figure 6 (forest plot)** — edit the recorded `ROWS` values at the top of
`make_forest_plot.py` to match the `compare_2a_vs_baseline.py` console output
above (by design, the figure never re-derives numbers — see the script's own
docstring), then:
```bash
python make_forest_plot.py
```

---

## Positive control (Section 3.3 / Figure 4)

```bash
python inject_marker.py --in-root Datasets/HandPD --out-root Datasets_marker/HandPD --task spiral
python mech2_suppression.py --in-root Datasets_marker/HandPD --out-root Datasets_marker_2a/HandPD --target-height 512

# step (a): probe on raw-marked
python marker_probe.py --root Datasets_marker/HandPD --label raw-marked
# step (b): probe on suppressed-marked
python marker_probe.py --root Datasets_marker_2a/HandPD --label suppressed-marked

# step (c): trained classifier, raw-marked vs. suppressed-marked, five seeds
python pd_loso_pipeline_cross_dataset.py --dataset handpd --root Datasets_marker/HandPD \
    --task spiral --models efficientnet_b0 --seeds 1 2 3 4 5
python pd_loso_pipeline_cross_dataset.py --dataset handpd --root Datasets_marker_2a/HandPD \
    --task spiral --models efficientnet_b0 --seeds 1 2 3 4 5

python make_positive_control_fig.py
```

---

## Deep CORAL baseline (Section "Implications for future mechanisms", Figure 9)

```bash
python coral_baseline.py --train-root Datasets/HandPD --train-task spiral \
    --test-root Datasets/kaggle --test-task spiral --direction handpd_to_kaggle \
    --model efficientnet_b0 --seeds 999 1001 1002 1003 --out coral_results_h2k.json
python make_intervention_comparison_fig.py
```

---

## Legacy: architecture diagram thumbnails (unused)

`make_pipeline_thumbs.py` cropped real-data thumbnails for an earlier,
TikZ-rendered version of Figure 1. Figure 1 is now a standalone supplied
diagram (`figures/handwriting_architecture.pdf`), so this script's output
is no longer referenced in the manuscript; kept only for provenance.
```bash
python make_pipeline_thumbs.py
```

## Figure 2 (masking panel)

```bash
python make_masking_panel.py
```

## Figure 8 (worked example)

```bash
python train_demo_checkpoint.py --root Datasets/HandPD --task spiral \
    --model efficientnet_b0 --seed 1234 --out demo_ckpt/efficientnet_b0_handpd_spiral_full.pt
python make_worked_example.py
```

---

## Important operational note (from the original harness)

The harness hard-codes `out_dir="results"` — every
`pd_loso_pipeline_cross_dataset.py` run **overwrites** the previous run's
CSVs in `results/`. Move or rename the `results/` folder after each run
before starting the next one (as the folder names under `results/` in this
repository already reflect), e.g.:
```bash
mv results results_my_new_run_name
```
