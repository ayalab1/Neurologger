# Multi-device synchronization continuity

## Goal and scope

Implement the agreed stable-clock interpretation: weak correlation alone does not invalidate a time interval when reliable observations on both sides support one continuous affine mapping. Keep the final 1 ms alignment allowance and independent missing/repeated-data handling. Align post-merge QC features with synchronization features. Preserve public call compatibility and existing initial-alignment safeguards; no unrelated recording-length or PC-clock changes.

Source recordings and existing results under R:/shared/data/FieldRat2026 are read-only. New results, caches, temporary files and logs must be local, under this worktree's .codex-sync-validation directory. Data and generated outputs must not be published to the network drive. Code publication was separately authorized after validation, as recorded below.

## Implementation steps

1. Add bounded, two-sided affine continuity evidence for runs of rejected or robust-model-outlier observations. Preserve measured anchors as measured; record interpolated intervals separately. Use supported observations in coverage/rejection-run validation without pretending that a weak observation is an accepted correlation measurement.
2. Keep raw-localized discontinuities. Remove verified aliases and unlocalized, sub-tolerance adaptive candidates when the surrounding observations support the continuous model. Apply the same selected candidate list to attribution and segment creation.
3. Make post-merge QC use the configured second-order causal high-pass, valid filter warm-up, and the requested supported window duration. Keep the 1 ms allowance and reliability thresholds. Count non-overlapping windows when requiring independent failure, recovery, or correction support; overlapping 10-second windows must not collapse transitively into a single observation.
4. Update algorithm/schema versions and workflow documentation. Inspect the scoped diff, then validate WT2 day7 indoor, WT2 day14 outdoor and WT4 day14 outdoor with local outputs and unchanged source datasets. Add no tests or test infrastructure; use the explicitly requested real-data checks.

## Numerical semantics

Interpolation concerns the mapping from master sample position to slave source position, not invented waveform values. A run must be bracketed by reliable model-inlier observations with sufficient time span on each side. Both side fits must agree with the common model across the bracket within the existing analysis tolerance, including their observed residuals. Known offset steps and independently verified missing/repeated intervals remain barriers. Localized boundaries are not discarded merely because their jump is small. The new evidence is conditional on stable clocks; it is distinct from a direct correlation measurement.

## Follow-up: recording-length denominator and terminal boundaries

The user requested a recording-overlap denominator, the Day7 terminal-boundary correction, and another full check of the same three datasets. Existing local outputs may be overwritten; network inputs remain read-only.

1. Permit the existing three qualified terminal observations to support a sub-tolerance unlocalized boundary even when the final span is shorter than a full window. Require at least one tracking step of span, an observed recording endpoint, unchanged affine agreement, and all existing physical-discontinuity barriers. Do not extend the separate Day14 terminal support cutoff.
2. Project raw recording endpoints through the first/last authoritative affine mappings to define their common interval within the unchanged output timeline. Compute retention numerators and denominator on that interval; keep internal missing, repeated, and unsupported intervals in the denominator. Preserve full-output statistics and expose the interval/basis explicitly. If physical overlap cannot be established, report that and keep the existing output-based statistics.
3. Update the inspection figure, documentation and versions. Review only these changes and replay the saved Day7 endpoint evidence before rerunning the same three full recordings. Preserve small prior-result snapshots, overwrite local derived outputs as authorized, and retain network-write protection.
4. Compare the new physical-overlap rates with the old full-output rates separately, check output QC and source metadata, and update this record and the local report. No agents or extra datasets.

Implemented the terminal-boundary exception and common-recording-overlap summaries, including full-output statistics, inspection labels, documentation and pipeline v24 / sync v16 / schema 20. Reviewed the scoped source changes. Saved-output replay confirmed that Day7's terminal candidate is explained by three terminal anchors spanning 7.4192 seconds, with maximum model disagreement 0.974490 samples. Denominator-only replay retained Day14's unsupported tail: 99.8947294% within common recording overlap.

Prior v23 metrics, report and successful manifests were saved locally in `.codex-sync-validation/before-recording-overlap`. Removed only the three explicitly selected local base-output `amplifier.dat` files to make room for authorized replacement. Network inputs were not modified.

Completed `python .codex-sync-validation/run_selected_datasets.py --overwrite` for the same three full recordings with network mutation protection and local caches/temp/output. Conversational waiting stopped at the user's request while the job continued. On 2026-09-23 the original execution handle was unavailable, but all three completed manifests and runner summaries were present. No rerun was needed. Total recorded runtime: 82.58 minutes (27.96 / 33.56 / 21.06 minutes).

Final collection ran `.codex-sync-validation/review_final_results.py` successfully. All six pair models are OK; written continuity support and saved-measurement QC classifications match the current source. Common-overlap retention: WT2 day7 100%, WT2 day14 99.8947294%, WT4 day14 100%. Full-output retention: 99.9078696%, 99.7046688%, 100%, respectively. Structural and alignment retention agree.

Day7 recovered 325,393 common-valid samples (16.26965 seconds) through the terminal-boundary correction; a further 3.64875 seconds are recording-length mismatch excluded only from the denominator. WT2 day14's rendered valid count is unchanged: 8.0776 seconds of length mismatch are excluded from the denominator, while the unsupported 4.4608-second P2 tail inside common overlap remains invalid. WT4 remains fully retained under the supported clock-continuity assumption.

QC: 5,019 windows, 4,828 passes, 191 low-confidence measurements, no reliable failures beyond the unchanged 1 ms allowance; maximum reliable lag 7 samples (0.35 ms). No warning intervals or correction rounds. Low-confidence Day14 QC WARN diagnostics remain visible. Sizes and modification times matched for all 89 source files. Updated local `final_results.json` and `validation_report.md`; all three successful outputs now use the base session folders without retry suffixes. No remaining requested work, suites, agents, commits, pushes or server publication.

## Publication handoff

The user accepted the current behavior, including WT2 day14's excluded 4.4608-second terminal tail, and explicitly authorized commit and push. Publication scope is the validated eight Python files, `docs/analysis/data-processing.md`, and this combined plan/log on `fix/multi-device-sync-improvement`. Local datasets, generated outputs, caches and validation scripts are excluded. The existing three-dataset results and scoped diff review are reused; no additional tests or dataset runs are needed for publication. No merge or pull-request creation was requested.

## Previous completion log

2026-09-22; uncommitted on `fix/multi-device-sync-improvement`, base `5f8beb4`.

- Applied the updated root AGENTS.md: one owner, no new tests or broad suites, one combined plan/log. Previously dispatched agents were stopped before source edits. Unrelated channel-order files in the main checkout were not edited.
- Implemented timing interpolation with one shared anchor-quality predicate, separate inferred-evidence metadata, continuity-supported boundary selection, robust inlier step statistics, matched high-pass/window QC, and non-overlapping repeated-QC support. Versions: pipeline v23 / sync v15 / schema 19. Updated `docs/analysis/data-processing.md`; inspected the scoped diff and subsequent corrections.
- WT2 day14 reproduced the colleague branch's validation concern: an endpoint outlier of 7,936.95 samples and a coarse outlier of 26.37 samples entered global step statistics despite a stable 806-inlier model. Adopted the inlier-only interpretation for those statistics; retained separate checks for unsupported sustained outlier runs.
- WT4 required grouping the entire low-confidence run, including coarse aliases and a coarse model-inlier recovery point. The original narrow boundary selection split the run or chose an ineligible endpoint. Final support is 141 observations between 925 and 1,635 seconds, with eight qualified observations on each side and maximum model disagreement 0.117666 samples. A coarse inlier is counted only once in combined measured/inferred coverage.

| Dataset | Common data retention, before -> after | Common alignment retention, before -> after |
|---|---:|---:|
| WT2 day7 indoor | 95.5833% -> 99.4971% | 58.9710% -> 99.4971% |
| WT2 day14 outdoor | 98.5270% -> 99.7047% | 43.7165% -> 99.7047% |
| WT4 day14 outdoor | 82.1908% -> 100.0000% | 82.1442% -> 100.0000% |

All six pair models passed. Output QC: 5,018 windows, 4,827 reliable passes, 191 low-confidence measurements, no reliable failures beyond 1 ms; maximum reliable lag 7 samples (0.35 ms). Low-confidence windows remain diagnostic; retention is conditional on the supported clock model. Known discontinuities and missing/repeated data remain separate. Residual endpoint/recording-length support limitations remain in WT2.

Exact full-recording commands (worktree-relative):

```powershell
python .codex-sync-validation/run_selected_datasets.py
python .codex-sync-validation/run_selected_datasets.py --session WT2__WT2_day14_outdoormedium_071326 --output-suffix _retry
python .codex-sync-validation/run_selected_datasets.py --session WT4__WT4_day14_outdoormedium_071326 --output-suffix _retry
python .codex-sync-validation/run_selected_datasets.py --session WT4__WT4_day14_outdoormedium_071326 --output-suffix _retry2
python .codex-sync-validation/review_final_results.py
```

The retries followed concrete failures on the same selected datasets. Saved-observation replay isolated WT4's endpoint issue before the final full run. The final review confirmed identical QC warning/correction decisions for all three outputs. Earlier successful WT2 renders predate the final grouping adjustment; final diagnostics were re-evaluated and saved separately, with unchanged model/anchor/mapping inputs and no loss of previously supported observations. Original run manifests and failed local attempts remain available.

Network write operations were blocked by the runner. Size and modification time matched for all 89 source files after processing. All outputs/caches/temporary files stayed local. No test suite, lint/build, extra datasets, dependency installation, commits, pushes or server publication.

Local results: [validation report](../.codex-sync-validation/validation_report.md), [final metrics and source hashes](../.codex-sync-validation/final_results.json). Successful output folders are listed in the report. The comparison is against saved prior results; it is not a fresh baseline benchmark or direct proof of 1 ms accuracy throughout interpolated intervals.
