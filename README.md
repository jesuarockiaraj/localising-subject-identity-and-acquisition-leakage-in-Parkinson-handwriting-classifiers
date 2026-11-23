# Localising subject-identity and acquisition leakage in Parkinson's handwriting classifiers

Code, result data, and figures backing every table and figure in the manuscript
*"Localising subject-identity and acquisition leakage in Parkinson's handwriting
classifiers: background suppression is verified but insufficient."*
The full manuscript (with methods and all equations/algorithms) is in
`manuscript/`. For the exact command that reproduces every single table and
figure, one by one, see **[REPRODUCE.md](REPRODUCE.md)**.

## Repository structure

```
code/         all analysis/training/figure-generation scripts (29 files)
results/      predictions, summary CSVs, and JSON results behind every
              table and figure -- lets you regenerate tables/figures
              without re-running any GPU training
figures/      every PNG/PDF figure used in the manuscript, including the
              Figure 1 architecture diagram
manuscript/   the manuscript itself (.tex source + compiled PDF + highlights)
requirements.txt
```

## Datasets (not included in this repository)

This study performs a secondary analysis of two pre-existing, publicly
available datasets. They are **not redistributed here** -- download them from
their original sources and place them locally before running any script:

- **HandPD**: Botucatu/UNESP dataset,
  <https://wwwp.fc.unesp.br/~papa/pub/datasets/Handpd/>
- **Kaggle "Parkinson's Drawings"**:
  <https://www.kaggle.com/datasets/kmader/parkinsons-drawings>

Expected local layout (matching the paths hard-coded in the scripts' default
arguments -- override with `--root` / `--test-root` flags if you place them
elsewhere):

```
Datasets/HandPD/{SpiralControl,SpiralPatients,MeanderControl,MeanderPatients}/
Datasets/kaggle/{spiral,wave}/{training,testing}/{healthy,parkinson}/
```

The **suppressed** ("2a") and **marker-injected** mirrors used in the causal
lane and positive-control experiment are not included either -- they are
*derived* images generated from the datasets above by this repository's own
code, not separate data:

```bash
python code/mech2_suppression.py --in-root Datasets/HandPD --out-root Datasets_2a/HandPD --target-height 512
python code/mech2_suppression.py --in-root Datasets/kaggle --out-root Datasets_2a/kaggle --target-height 512
python code/inject_marker.py --in-root Datasets/HandPD --out-root Datasets_marker/HandPD
python code/mech2_suppression.py --in-root Datasets_marker/HandPD --out-root Datasets_marker_2a/HandPD --target-height 512
```

## Environment

```bash
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Trained on a single NVIDIA RTX 3050 (4 GB), PyTorch 2.5.1 (CUDA 12.1). All
training scripts default to `epochs=40, n_folds=10` and will fall back to CPU
automatically if no GPU is available (much slower).

## Script &rarr; result/figure map

Figure and table numbers below match the current manuscript
(`manuscript/Phase2_manuscript.tex`). Figure 1 (the architecture diagram,
`handwriting_architecture.pdf`) is a standalone diagram asset, not rendered
by a script.

| Script | What it does | Produces |
|---|---|---|
| `pub_style.py` | Shared chart styling (palette, gradient-filled rounded bars, typography) imported by every `make_*_fig.py` script below -- not run standalone | -- |
| `pd_loso_pipeline_cross_dataset.py` | Core training harness: P1 (image-level) / P2 (subject-grouped) k-fold, and cross-cohort external validation | `results_*/predictions_*.csv`, `summary_*.csv` -- Table 1 |
| `make_baseline_fig.py` | Renders the Table 1 cross-cohort AUC-by-architecture bar chart | `baseline_fig.png/.pdf` (Figure 3) |
| `mech2_suppression.py` | Background/template suppression ("2a") pipeline | `Datasets_2a/` mirrors |
| `compare_2a_vs_baseline.py` | Paired subject-level bootstrap &Delta;AUC, raw vs. suppressed | Table 7, `auc_forest.png/.pdf` data |
| `make_forest_plot.py` | Renders the forest plot from recorded `compare_2a_vs_baseline.py` output | `auc_forest.png/.pdf` (Figure 6) |
| `leak_decomposition_probe.py` | Frozen-feature identity re-identification + cohort decodability probe (full/background/trace); `--save-features` caches features for `stratified_reid.py` | Tables 2, 5 |
| `stratified_reid.py` | Re-ID stratified by cohort/task, using `leak_decomposition_probe.py`'s cached features | Table 3 |
| `make_calibration_table.py` | Brier score + 10-bin ECE from the original Table 1 predictions | Table 10, `calibration_table.csv` |
| `make_calibration_fig.py` | Renders the Table 10 Brier/ECE bar chart | `calibration_fig.png/.pdf` (Figure 7) |
| `rendering_invariance_probe.py` | Rendering-invariance ladder (color/gray/uniform) | Tables 4, 6 (rendering), `rendering_invariance_results*.json` |
| `make_rendering_ladder_fig.py` | Renders the raw-vs-suppressed rendering-invariance ladder line chart | `rendering_ladder_fig.png/.pdf` (Figure 5) |
| `aspect_ratio_control.py` | Pad-to-square vs. stretch control | Resolution/aspect-ratio limitation check |
| `handpd_clean_split.py` | Colour-separates HandPD's printed guide (dark) from the pen stroke (blue); corrected direction verified by `verify_guide_color.py` | Clean pen-only re-ID number (re-ID@1 = 0.236) |
| `verify_guide_color.py` | Cross-subject mask IoU check establishing that the dark ink is the printed guide and the blue ink is the pen stroke | Colour-direction verification backing `handpd_clean_split.py` |
| `rendering_invariance_penonly.py` | Guide-excluded rendering-invariance ladder on raw images (colour-separated pen-only trace) | Table 8 (raw rows), `rendering_invariance_penonly_results.json` |
| `rendering_invariance_penonly_suppressed.py` | Same guide-excluded ladder reconstructed on suppressed images | Table 8 (suppressed rows), `rendering_invariance_penonly_suppressed*_results.json` |
| `background_disease_probe.py` | Within-cohort background-only disease-decodability probe, raw vs. suppressed, both cohorts | Table 9, `background_disease_probe_results.json` |
| `compute_suppressed_p1.py` | Aggregates suppressed-image P1 (image-level) vs. P2 (subject-grouped) AUC from existing predictions -- no new training | Suppressed P1/P2 identity-leakage gap numbers (Section on background-only disease decodability) |
| `inject_marker.py` | Injects the synthetic, background-confined positive-control marker | `Datasets_marker/` |
| `marker_probe.py` | Probes disease-decodability on marked images (steps a/b of the positive control) | `positive_control_fig.png/.pdf` data |
| `make_positive_control_fig.py` | Renders the positive-control summary figure | `positive_control_fig.png/.pdf` (Figure 4) |
| `coral_baseline.py` | Deep CORAL domain-generalisation baseline (single pilot run) | `coral_results_h2k.json` |
| `make_intervention_comparison_fig.py` | Renders baseline vs. suppression vs. CORAL comparison | `intervention_comparison_fig.png/.pdf` (Figure 9) |
| `make_masking_panel.py` | Renders the real full/background/trace example panel | `masking_panel.png/.pdf` (Figure 2) |
| `make_pipeline_thumbs.py` | Legacy: crops real-data thumbnails that were previously embedded in a TikZ-rendered architecture diagram. Figure 1 is now a standalone supplied diagram (`handwriting_architecture.pdf`), so this script's output is no longer used in the manuscript; kept for provenance | `thumb_full.png`, `thumb_background.png`, `thumb_trace.png` (unused) |
| `train_demo_checkpoint.py` / `demo_predict.py` | Trains and runs the single-checkpoint worked-example demo | Checkpoint + predictions for the worked example |
| `make_worked_example.py` | Renders the worked-example figure | `worked_example.png/.pdf` (Figure 8) |

## Results folder guide

Every subfolder under `results/` is a `pd_loso_pipeline_cross_dataset.py` (or
`coral_baseline.py` / `marker_probe.py`) output directory, kept as generated
rather than merged, so provenance stays traceable:

| Folder | Contents |
|---|---|
| `results_baseline/` | **Original**, under-specified (3-fold) HandPD spiral baseline run -- the provenance bug discussed in the manuscript's Limitations |
| `results_verify_seed999/`, `results_verify_seeds_1001_1003/`, `results_verify_combined_4seed/` | **Independently verified** (k=10, 40 epochs) HandPD spiral baseline, seeds 999/1001-1003 |
| `results_2a/`, `results_2a_cross_h2k/`, `results_2a_cross_k2h/` | **Original**, five-seed (1-5) suppressed-image runs |
| `results_2a_verify_seeds_999_1001_1002_1003/`, `results_2a_verify_cross_h2k_4seed/`, `results_2a_verify_cross_k2h_4seed/` | **Independently verified** four-seed suppressed-image runs |
| `results_verify_cross_h2k_4seed/`, `results_verify_cross_k2h_4seed/` | **Independently verified** four-seed raw cross-cohort runs |
| `results_marker_raw/`, `results_marker_2a/` | Positive-control marker experiment, raw-marked and suppressed-marked training runs |
| `coral_results_h2k.json`, `coral_smoketest.json` | Deep CORAL baseline results |
| `rendering_invariance_results*.json` | Rendering-invariance ladder raw output (full-trace, guide-contaminated) |
| `rendering_invariance_penonly_results.json`, `rendering_invariance_penonly_suppressed*_results.json` | Guide-excluded, colour-separated pen-only rendering-invariance ladder output (Table 8) |
| `background_disease_probe_results.json` | Within-cohort background-only disease-decodability results (Table 9) |
| `calibration_table.csv` | Brier score / ECE values behind Table 10 |

## Reproducing a table or figure end to end

See **[REPRODUCE.md](REPRODUCE.md)** for the exact command(s) behind every
single table and figure in the manuscript, one section per table. Every
table -- including Table 3 (stratified re-ID, via
`leak_decomposition_probe.py --save-features`), Table 8 (the guide-excluded,
colour-separated pen-only rendering-invariance ladder), Table 9 (background-only
disease decodability), and Table 10 (calibration, via `make_calibration_table.py`)
-- is reproducible from the scripts in `code/` alone; both were checked end to
end against this repository's own result data before release.

## Citation

If you use this code or build on this study, please cite the manuscript (see
`manuscript/Phase2_manuscript.tex` for full bibliographic details once
published).

## License note

The analysis code in `code/` is released for reproducibility. The HandPD and
Kaggle "Parkinson's Drawings" datasets are third-party data under their own
original authors' terms -- see the links above.
