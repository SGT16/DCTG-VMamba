# Retrospective paired p-value analysis

These results were recalculated on 27 October 2026 from 46 retained prediction CSV files. No training, model inference, prediction replacement, data split modification, or test-set threshold optimization was performed. The protocol was frozen before calculating these new p values. This is a retrospective analysis conducted after inspection of earlier results; it is not a prospectively specified confirmatory analysis.

## Comparisons and estimands

The taxonomy family comprises six comparisons: DCTG taxonomy readout minus Joint source fusion, and DCTG taxonomy readout minus DCTG source fusion, for each of the CystoDS, EBTC, and Kaggle WLI held-out targets. The statistic is pooled binary macro-F1 difference using a fixed two-label set and a probability threshold of 0.5 (positive when score is greater than or equal to the threshold). The saved scores are ensembles from five source-training folds evaluated on the same target images, not five disjoint target test folds. The taxonomy score is the saved global-branch neoplastic probability; it is distinct from the complete DCTG source-fusion score. Point estimates, class counts and differences are preserved in the result JSON.

The operating-point family comprises two comparisons: pooled outer-test specificity of DCTG minus Joint for CystoDS malignant-versus-rest and EBTC tumor-versus-nontumor. These use the corresponding matched binary-label-only runs and their validation and outer-test prediction CSV files. Positive class index is 0. For each model and fold, the threshold is the highest observed positive-class validation score attaining at least 90% empirical validation sensitivity. The required count is ceil(9 * number_of_validation_positives / 10); positive predictions use score >= threshold. Thresholds are determined from validation data alone, then applied unchanged to the corresponding outer test fold. Saved decimal scores are read as float64. The 20 thresholds and their validation statistics are included separately.

## Paired randomization

Under the conditional null, the two models' predictions are exchangeable within each retained group, conditional on observed labels and fitted models. A swap exchanges the models for all images in a group together; images are never transferred between groups or folds. The two-sided p value is the probability of an absolute statistic at least as large as observed, with equality tolerance 1e-12. Groups whose swap makes no change to the relevant sufficient statistic are integrated out exactly.

The method was selected from group structure before p-value computation. With at most 20 active nonsingleton groups, every active-group swap assignment is enumerated. When every group contains one image, the exact distribution is evaluated using independent binomial counts of discordant predictions within the two true-label classes. Otherwise, 1,000,000 independent group-swap assignments are sampled in batches of 4,096 using NumPy PCG64. Exact p values are their tail probabilities; Monte Carlo p values use (extreme_or_equal + 1) / (1,000,000 + 1). Large counts and random seeds in the JSON are decimal strings to preserve their exact integers.

For each sampled comparison, seed = int.from_bytes(SHA256(UTF8(str(20261027) + "|" + comparison_id))[:8], "little"). The comparison-specific seeds, assignment counts, evaluated-state counts, and method names are supplied in the CSV and JSON. CystoDS taxonomy comparisons and its specificity comparison use Monte Carlo group randomization. EBTC comparisons use exact active-group enumeration. Kaggle WLI comparisons use the exact singleton-group distribution.

Holm correction is applied separately to the six taxonomy comparisons and the two specificity comparisons. The columns p_raw and p_holm are unadjusted and family-adjusted p values, respectively. A positive difference favors model_a for the stated statistic; a negative difference favors model_b. P values are reported without replacing small values with zero.

## Grouping and limits

CystoDS uses retained patient groups. EBTC uses filename-derived case groups, which are not independently verified patient identities. Kaggle WLI uses one image per group because verified patient identities are unavailable. Therefore Kaggle WLI inference does not establish patient-level independence. Images are weighted equally in the pooled estimands; retained groups define the randomization unit, not an equal-case-weighted performance average. Fitted models and selected validation thresholds are treated as fixed, so training and threshold-selection uncertainty are not included.

No new confidence intervals were calculated in this analysis. monte_carlo_se is numerical sampling error of the Monte Carlo tail-probability estimate, not a confidence interval, model-performance standard deviation, or patient-sampling uncertainty. It is zero for exact calculations; a blank rng_seed means no random-number generator was used. Exact singleton calculations sum binomial-weighted states; states_evaluated therefore need not equal total_assignments. Total assignments and tail counts can be very large.

## Files and provenance

- taxonomy_readout_results.json and operating_point_results.json are byte-identical copies of the freshly computed result files.
- taxonomy_readout_p_values.csv contains the six comparisons; operating_point_p_values.csv contains the two specificity comparisons and both models' confusion counts and rates.
- operating_point_thresholds.csv contains the 20 model/fold validation thresholds.
- p_value_analysis_protocol.json is the frozen protocol, whose SHA-256 is 1fd13cb7e0acf82a886d35a2c789ada5073828a8277fcb32ac5ff67a57cc23a9.
- p_value_source_fingerprints.csv records 46 opaque source IDs and input SHA-256 fingerprints. Paths, image identifiers and individual group identifiers are omitted.

Use the exact JSON numeric values when transferring results into other tables. CSV files preserve numerical values in decimal form, but spreadsheet applications may display scientific notation or round long integer seed/count strings on import; the JSON is the authoritative record for those integers.
