# Round-3 pre-registration and lock records

| File | Written by | Content |
|---|---|---|
| `prereg_round3.md` | authors, before the lock | hypotheses H1–H6, arms, chains, sample size, analysis and decision rules of round 3 |
| `prereg_round3.sha256` | `experiments/benefit_model_bdef_20261011/lock_config3.py --lock` | sha256 of `prereg_round3.md` at the lock |
| `lock_record.json` | same, 2026-10-11 21:00:43 | sha256 of the registration, models, frozen round-2 configuration, chain plan, sequence catalog, B-label files, offline reports, self-test fixtures and code; software versions |
| `lock_addendum.json` | `experiments/closedloop3_bdef_20261012/run.sh addendum`, 2026-10-11 21:30:28 | sha256 of the round-3 worker, analysis, run script and scheduler before the first round-3 job; smoke-test chain ids |

The registration refers to the project's work stages as rounds 6a (relabelling under B_def), 6b (retraining, re-tuning of δ_B, registration and lock) and 6c (round-3 closed-loop run and analysis); "stage 2" and "stage 3" denote the retraining and the closed-loop run.

On the experiment server, `prereg_round3.md`, `prereg_round3.sha256` and `lock_record.json` were located in `benefit_model_bdef_20261011/` and `lock_addendum.json` in `closedloop3_bdef_20261012/`.

`prereg_round3.md` is released unchanged: its sha256 equals the value recorded in `prereg_round3.sha256`, `lock_record.json` and `lock_addendum.json`. In `lock_record.json` and `lock_addendum.json` the absolute server paths were replaced by `<HANDOFF_RUN_ROOT>` and `<RHO_ROOT>`; the recorded hash values are unchanged.

## Recorded and released hashes

For this release, server paths in code and records were replaced by configurable roots and Chinese comments and messages were translated (see the main README). The table compares every hash recorded at the lock with the corresponding released file. Before these edits, the copies exported from the server matched every recorded hash that could be checked (all entries below except the models, the sequence catalog, the label-file list, the relabelling map, the fixtures, `v4_rows_all.json`, `cache_provenance.json` and `object_locator.py`, which were not exported). Files marked "identical" can still be verified against the lock; the others differ only by the edits described above. The integrity checks in `worker_cl3.py` and `analyze_cl3.py` that compare file hashes against these records will therefore refuse to run on the released files.

| Lock entry | File (relative to the run root) | sha256 at lock | sha256 of released file | Released as |
|---|---|---|---|---|
| model_v4 | `benefit_model_bdef_20261011/models/benefit_models_v4_all.joblib` | `3e3d3df496bf3be5…` | — | not included |
| model_v3_all_benefit_A | `benefit_model_20261003/models/benefit_models_v3_all.joblib` | `dba0dcb0fd1780c4…` | — | not included |
| jobs_cl3 | `benefit_model_bdef_20261011/jobs_cl3.json` | `405753cba134793d…` | identical | experiments/benefit_model_bdef_20261011/jobs_cl3.json |
| plan_report | `benefit_model_bdef_20261011/analysis/plan_report.json` | `6c408a2a179352a6…` | identical | data/raw/benefit_model_bdef_20261011/plan_report.json |
| prereg_round3 | `benefit_model_bdef_20261011/prereg_round3.md` | `4c1eaba310383dfb…` | identical | prereg/prereg_round3.md |
| round2_config_original | `closedloop2_20261005/locked_config.json` | `0edf0453217feb76…` | `93fe05a9ce9ce5dc…` | data/raw/closedloop2_20261005/locked_config.json |
| round2_config_frozen_copy | `benefit_model_bdef_20261011/config_benefit_A_round2_frozen.json` | `0edf0453217feb76…` | `93fe05a9ce9ce5dc…` | experiments/benefit_model_bdef_20261011/config_benefit_A_round2_frozen.json |
| round1_config | `closedloop_20261004/locked_config.json` | `b7af69e0dc36c17b…` | `a57b7eae5d2180cf…` | data/raw/closedloop_20261004/locked_config.json |
| catalog | `(outside run root) sequence_catalog.json` | `8a3b00a2a9bccf6c…` | — | not included |
| label_check | `benefit_model_bdef_20261011/analysis/label_check.json` | `14d7710b8424c9d2…` | `fa9d832972fa3378…` | data/raw/benefit_model_bdef_20261011/label_check.json |
| tune_report | `benefit_model_bdef_20261011/analysis/tune_report.json` | `eeec3fa25f5ad1a9…` | identical | data/raw/benefit_model_bdef_20261011/tune_report.json |
| v4_report | `benefit_model_bdef_20261011/analysis/v4_report_all.json` | `4856f16147a5ec62…` | `f733066b167b4782…` | data/raw/benefit_model_bdef_20261011/v4_report_all.json |
| v4_rows | `benefit_model_bdef_20261011/analysis/v4_rows_all.json` | `e114c580274d8608…` | — | not included |
| v4_report_rpexclude | `benefit_model_bdef_20261011/analysis/v4_report_all_rpexclude.json` | `ac5f65d46d9efd63…` | `1bd6cf95cbb5eba0…` | data/raw/benefit_model_bdef_20261011/v4_report_all_rpexclude.json |
| cache_provenance | `benefit_model_bdef_20261011/analysis/cache_provenance.json` | `e109d829bf1af575…` | — | not included |
| fixture_inputs | `benefit_model_bdef_20261011/analysis/fixture_inputs.npz` | `d0e9f0d3f9f81252…` | — | not included |
| fixture_expected | `benefit_model_bdef_20261011/analysis/fixture_expected.json` | `ee7e4b9ef2242d52…` | — | not included |
| label_files_list | `benefit_model_bdef_20261011/analysis/label_files.sha256` | `7c6e5b8b527e9f21…` | — | not included |
| code_train_benefit_v4 | `benefit_model_bdef_20261011/train_benefit_v4.py` | `0cdc144061d6525e…` | `cf89d38e81a1a7fd…` | experiments/benefit_model_bdef_20261011/train_benefit_v4.py |
| code_lock_config3 | `benefit_model_bdef_20261011/lock_config3.py` | `0774027780387f80…` | `4fa7645594089f25…` | experiments/benefit_model_bdef_20261011/lock_config3.py |
| code_plan_cl3 | `benefit_model_bdef_20261011/plan_cl3.py` | `152d34fc0c1bb337…` | `8ab884c01ad2f6d0…` | experiments/benefit_model_bdef_20261011/plan_cl3.py |
| code_run_sh | `benefit_model_bdef_20261011/run.sh` | `ba9f5455fe2b3a48…` | `cd68f959d7f630f9…` | experiments/benefit_model_bdef_20261011/run.sh |
| lib_benefit_lib | `benefit_model_20261003/benefit_lib.py` | `7b27f0ae4da73391…` | `cea8044225c38bf8…` | experiments/benefit_model_20261003/benefit_lib.py |
| lib_benefit_lib_v2 | `benefit_model_20261003/benefit_lib_v2.py` | `7195c17f1dc7cfb7…` | `c27f682b99ec4e41…` | experiments/benefit_model_20261003/benefit_lib_v2.py |
| lib_benefit_lib_v3 | `benefit_model_20261003/benefit_lib_v3.py` | `cce9c59398b4974a…` | `e3cd4296d976165a…` | experiments/benefit_model_20261003/benefit_lib_v3.py |
| lib_features_patch | `fix_20260928/features_patch.py` | `351fd26f7bc0f89e…` | `3c907f75510f9599…` | experiments/fix_20260928/features_patch.py |
| lib_object_locator | `fix_20260928/object_locator.py` | `20ca40af1ed29fc3…` | — | not included |
| lib_evaluate_cv | `fix_20260928/evaluate_cv.py` | `3f454781f158486d…` | `05dd6a9399b80190…` | experiments/fix_20260928/evaluate_cv.py |
| worker_cl2 | `closedloop_20261004/worker_cl2.py` | `e2612437c13cd4ed…` | `33824664f20dfa4b…` | experiments/closedloop_20261004/worker_cl2.py |
| worker_cl_perturb | `closedloop_20261004/worker_cl.py` | `cc1006852d6a8ac7…` | `5be194281c54e056…` | experiments/closedloop_20261004/worker_cl.py |
| worker_scripted_repair | `multitask_vocab_20261002/worker.py` | `cae961209b797380…` | `5121236d2d1d703e…` | experiments/multitask_vocab_20261002/worker.py |
| fc_worker_modeB | `fullchain_replay_20261010/fc_worker.py` | `5e460ecd83682e54…` | `ed173f1d57e530e8…` | experiments/fullchain_replay_20261010/fc_worker.py |
| relabel_mapping_0 | `relabel_bdef_20261011/mapping.json` | `9fe57a6ecaf6069d…` | — | not included |
| locked_config | `benefit_model_bdef_20261011/locked_config.json` | `7a0c297dfdd7a286…` | `b07ad247cf563cfb…` | experiments/benefit_model_bdef_20261011/locked_config.json |
| addendum:worker_cl3.py | `closedloop3_bdef_20261012/worker_cl3.py` | `b230384167a6ec41…` | `b3b8d6deab599b1c…` | experiments/closedloop3_bdef_20261012/worker_cl3.py |
| addendum:analyze_cl3.py | `closedloop3_bdef_20261012/analyze_cl3.py` | `e03d08a570157356…` | `ae2f8d33c00b64bd…` | experiments/closedloop3_bdef_20261012/analyze_cl3.py |
| addendum:run.sh | `closedloop3_bdef_20261012/run.sh` | `b11a0d8122cb15e7…` | `b63abde9290b4627…` | experiments/closedloop3_bdef_20261012/run.sh |
| addendum:supervise_multi.py | `closedloop3_bdef_20261012/supervise_multi.py` | `1928e22a1f4c9a31…` | `b7a20f40899d3022…` | experiments/newtasks_20261004/supervise_multi.py |
