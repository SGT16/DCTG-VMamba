# Recorded results

These files summarize retained outputs from the original experiments and the major-revision experiments on CystoDS, EBTC and Kaggle WLI, including source-dataset-held-out transfer evaluations. Numerical results are preserved. Models were not retrained for this export.

## Files

| File | Contents |
| --- | --- |
| `metrics_pre_major.csv` | Original and earlier development runs, including performance and recorded training times. |
| `metrics_post_major.csv` | Major-revision training and evaluation summaries. |
| `class_counts_pre_major.csv` | Exact correct and incorrect counts for each recorded class before the major revision. |
| `class_counts_post_major.csv` | Exact class counts for major-revision evaluations. |
| `confusion_matrix_cells_pre_major.csv` | Integer confusion-matrix cells, with true classes as rows and predicted classes as columns. |
| `confusion_matrix_cells_post_major.csv` | Corresponding confusion-matrix cells after the major revision. |
| `count_coverage.csv` | Availability of exact count evidence for each training record, including any missing evidence. |
| `source_fingerprints.csv` | Opaque source identifiers and original-file SHA-256 fingerprints. |

CSV files use UTF-8, commas and decimal points. Empty fields are unavailable or inapplicable. A recorded `nan` remains undefined, and a numeric zero remains zero.

## Reading class counts

For a class with reference support `support`:

- `correct` is the diagonal count in its confusion matrix.
- `incorrect = support - correct` counts examples of that reference class assigned elsewhere. It equals the one-versus-rest false-negative count.
- `fp` counts examples from other reference classes assigned to this class.
- `predicted_count = correct + fp`.

The `tp`, `fn`, `fp` and `tn` fields use a one-versus-rest definition. True negatives must not be added across classes as a count of correctly classified images. A matrix cell is identified by `record_id`, `true_index` and `predicted_index`; its evaluation metadata is available in the matching class-count rows.

Rows marked `original_source_class_binary_correctness` describe the recorded binary decision within an original source-class subgroup. Their correct/incorrect counts are valid for that binary endpoint. They do not define a multiclass classifier over the subgroup names, so unsupported one-versus-rest fields are empty.

## Evaluation units

Use the protocol, model, dataset, split, output route and aggregation fields to select comparable evaluations. Earlier developmental protocols are retained separately. The original manuscript's 310 training runs are identified by `original_manuscript_run` in the metric summary and `official_original_310` in the class-count files. A joint training run can produce several dataset rows, with the same training duration repeated. Do not sum these repeated durations.

Retained test, validation, taxonomy, calibration, individual-member and ensemble views remain separate. `same_counts_as` links an identical count view for the same recorded evaluation context; it does not assert identical probability outputs. Repeated views must not be added together. Five source-fold models evaluated on one external cohort are not five independent target cohorts. Their descriptive metric SD, a probability-ensemble score, and a bootstrap confidence interval measure different quantities.

Dataset class names and reference labels are retained as recorded.

Counts are taken from integer matrices or reconstructed from retained true/predicted label pairs. Rounded precision, recall and F1 values are not used to infer counts. Source identifiers and fingerprints support traceability without exposing workstation locations or image identifiers.

## Additional paired analyses

The retained prediction scores were reanalyzed on 27 October 2026. These retrospective analyses compare the DCTG taxonomy readout with the two source-head readouts and compare DCTG with Joint-VMamba specificity at thresholds selected to reach at least 90% sensitivity on each validation fold.

- [Taxonomy comparisons](taxonomy_readout_p_values.csv): six macro-F1 contrasts with raw and Holm-adjusted p values.
- [Operating-point comparisons](operating_point_p_values.csv): two specificity contrasts, with error counts and raw and Holm-adjusted p values.
- [Validation thresholds](operating_point_thresholds.csv): model-specific thresholds for all five folds of each task.
- [Methods and interpretation](p_value_analysis_methods.md): test definitions, separate Holm families, exact and Monte Carlo calculations, and uncertainty limitations.

The complete calculation records are provided in `taxonomy_readout_results.json` and `operating_point_results.json`. Input fingerprints and the analysis metadata accompany these records. These analyses do not add training runs. The original metric summaries retain their existing definitions.
