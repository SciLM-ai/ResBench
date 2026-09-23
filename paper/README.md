# Paper analysis scripts (archaeology)

One-off study scripts behind the results reported in `../RESULTS.md`
(calibration studies, CFG sweep, W1 sensitivity, compartmentalization,
entropy map comparisons). They are **not** part of the benchmark machinery
— scoring a model needs only the `resbench` package and the CLI.

Paths inside these scripts refer to the original evaluation workspace
(`$WORK/resbench_eval/…`) and the pre-release repository layout
(`results/*_reference/`, now `references/`); they are kept verbatim for
provenance and are re-runnable only with that workspace (raw draw pools
available in the dataset release).
