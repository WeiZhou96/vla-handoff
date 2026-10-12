# Round 3 closed-loop confirmation under the handoff-referenced criterion B_def — pre-registration

Written in round 6b, after stage 2 (relabelling, retraining, re-tuning) and before any stage-3 closed-loop data exist.
`lock_config3.py --lock` records the sha256 of this file in `prereg_round3.sha256` and `lock_record.json`.
The lock refuses to run if `results/` is non-empty.

## 1. Background and purpose

- The benefit model was trained on repair labels judged by CALVIN's start-relative criterion A.
  - Criterion A credits undo–redo on push and rotation targets.
- Re-executing the 8,310 round-1/2 chain runs under the handoff-referenced criterion B_def changed the round-1 result.
  - The gain of the gate over retry fell from +18.9 pp to +2.6 pp [−1.6, 6.7].
- Round 6a relabelled all repair branches under B_def.
  - 2,848/2,848 branches were verified.
  - Labels changed only on push/rotate targets and four lift_pink_block_drawer branches.
- Stage 2 then retrained the model on B_def labels (`benefit_models_v4_all.joblib`) and re-tuned δ on development data only.
- This round tests, in closed loop and under B_def, whether the retrained gate improves whole-chain success over retry.
  - That improvement is what the B_def replay of the old gate failed to show.
- It also tests whether relabelling itself helped.

## 2. Locked components (`locked_config.json`)

**Model.** `models/benefit_models_v4_all.joblib`, trained by `train_benefit_v4.py`.

- Same data, features, structure and hyper-parameters as v3.
- Repair labels taken from `relabel_bdef_20261011/labels_bdef`.
- The hand-chosen repairs of `repair_20261001` are kept with their stored labels. For push_into_drawer and lift_red_block_slider, A and B_def showed zero disagreements:
  - in 2,520 closed-loop handoffs;
  - in 382 moved relabelled branches.

**Gate.**

- u* = argmax over the 37 options of the M7 score.
- Repair iff q(u*) − q(none) − λ·c̄(task,u*)/360 > δ.
- λ = 0.5, fixed as in rounds 1–2.
- δ re-derived with the round-1 objective and grid on B_def labels and cross-fitted M7_B scores:
  - development set: natural-distribution handoffs (gate, hand, and the natural-distribution group of the new-task data if identifiable; otherwise gate + hand);
  - rule and value recorded in `locked_config.json`.
- c̄ is the round-2 value.
- τ_risk and τ_retry are not used.

**benefit_A.** The round-2 policy is frozen byte-for-byte in `config_benefit_A_round2_frozen.json`:

- model `benefit_models_v3_all.joblib`;
- scorer M7;
- round-1 λ and δ;
- round-2 c̄.

**Common settings.**

- Targets: the 11 tasks covered by the model.
- Position 0 is never a target.
- Budget 2,160 steps per chain, 360 per subtask attempt, 120 per language repair.

## 3. Success criterion

**Driver.** Success of every subtask attempt, the retry decision and chain termination are driven by B_def, exactly as in mode B of `fc_worker` (sha256 recorded at lock).

- A target attempt is judged relative to the state at the handoff, i.e. before any repair.
- A retry is also judged relative to the handoff, with mode conditions taken at the start of that attempt.

**Criterion A** is computed and logged for every target attempt but never drives execution.

**Chain success** means all five subtasks succeed within the budget.

## 4. Chains (`plan_cl3.py`, `jobs_cl3.json`)

**Exclusions.** New catalog chains exclude:

1. every catalog index used in round 1 or round 2 (the full-chain replay used exactly these chains);
2. every index linked to development handoffs by the validated rule in `plan_cl3.py`;
3. any chain with the same (initial configuration, task sequence) as an excluded chain.

The linkage counts are recorded in `analysis/plan_report.json`.

**Base configurations.** A base configuration may recur with a new perturbation seed `sha256("cl3-perturb|i")`; the perturbation function is unchanged from rounds 1–2. Chains whose base configuration never appeared before are reported as a subset.

**Stratification.**

- 11 strata, one per target task, with the target at positions 1–4.
- Round-robin over base configurations; scarcest strata first.
- If a stratum is short, the remaining slots are filled from the other strata in the same order (recorded in the plan report).

**Sample size (primary hypothesis H1, paired chain-level difference d ∈ {−1, 0, 1}).**

- Assumptions:
  - one-sided α = 0.025 (equivalently, the lower bound of a two-sided 95% CI > 0);
  - power 80%;
  - smallest effect of interest +5 pp;
  - SD 0.411 (r1 benefit−retry in the replay, the larger of the two rounds);
  - design effect 1.11 (the largest observed).
- Calculation: n = (z₀.₉₇₅ + z₀.₈₀)² · SD² · deff / 0.05² = 589. Inflated by 5% for exclusions and rounded up to a multiple of 11, this gives **627 chains, 57 per stratum**.
- Minimum detectable effect at 596 analysed chains:
  - ≈ 5.0 pp under the r1 SD;
  - ≈ 3.8 pp under the r2 SD (0.327, deff 1.00).
- A true gain of the size seen for the old gate in the replay (+2.6 pp) would probably not be detected. This is stated in the paper whatever the outcome.

## 5. Arms (7; each chain runs all arms from the identical perturbed initial state)

| Arm | Policy | Why it is needed |
|---|---|---|
| none | no repair, no retry | absolute reference; H3, H5 |
| retry | no repair; one retry after a B_def failure | strongest simple baseline; primary comparator (H1) |
| prior | always execute the B_def prior-table argmax repair (v4 table); one retry | state-blind task-level rule (H2) |
| benefit_B | v4 M7, gate (λ, δ_B, c̄); one retry | the method |
| benefit_noretry_B | same repair decisions, no retry | detector-free variant (H5) |
| noloc_B | v4 M7n (no simulator-supervised locator), same λ, δ_B, c̄; one retry | privileged-information question (H6) |
| benefit_A | round-2 frozen policy (v3_all M7, round-1 λ/δ, round-2 c̄); one retry | did relabelling help? (H4) |

**Arms dropped and why.**

- inverse: under B_def the reverse skill no longer earns undo–redo credit. The replay of rounds 1–2 already measures it under B_def, and prior covers the task-rule baseline.
- home_all, risk, full and benefit_alt: their questions were answered in rounds 1–2 or are not claims of the paper.

## 6. Hypotheses

All outcomes are whole-chain success under B_def. Each comparison is a paired chain-level difference.

**Primary.**

- **H1:** benefit_B − retry > 0.

**Secondary (one family).**

- **H2:** benefit_B − prior > 0.
- **H3:** benefit_B − none > 0.
- **H4:** benefit_B − benefit_A > 0 (relabelling helped).
- **H5:** benefit_noretry_B − none > 0 (gain without failure detection).
- **H6 (non-inferiority, margin 5 pp):** noloc_B − benefit_B > −0.05.

**Pre-specified descriptive analyses (no confirmatory claims).**

- All 21 pairwise differences.
- Chains with vs without a push/rotate target at positions 1–4.
- Per planning stratum.
- The unseen-base subset.
- First-target-handoff success per arm and task, on chains where all arms share the state up to that handoff.
- Steps, repairs and retries per chain, and the repair choices per task.
- A/B_def disagreement counts per arm and task.

## 7. Analysis

**Paired differences.** For a comparison X − Y, d_c = S_c(X) − S_c(Y) per chain. The estimate is the mean of d_c over analysed chains, each chain weighted equally.

**Cluster bootstrap.**

- Clusters are base configurations (`job["base"]`).
- 5,000 resamples of clusters with replacement; all chains of a resampled cluster are included.
- Seed 20261011, numpy `RandomState`, one shared set of resampled cluster indices for all comparisons.

**Confidence intervals.** Two-sided 95% percentile CIs.

**One-sided p-values.**

- Superiority: p = (1 + #{b : d̄*_b ≤ 0}) / (5,000 + 1).
- H6 non-inferiority: the same with threshold −0.05.

**Analysis code** (`analyze_cl3.py`) is written in round 6c and hashed in `lock_addendum.json` before any stage-3 result file is opened.

## 8. Decision rules and claims

**H1 supported** if the lower 95% CI bound > 0.

- Claim: "a gate trained on handoff-referenced labels improves whole-chain success over retry under B_def", with the estimate and CI.

**H1 not supported.**

- The paper states that the method did not show an improvement over retry under B_def, and reports the estimate and CI.
- Claims about the method are restricted to secondary hypotheses rejected under Holm.
- No subgroup is promoted to a claim.

**H4.** Without rejection, the paper does not claim that relabelling improved closed-loop performance; offline B-label results are reported as offline only.

**H6.** Without rejection, the paper does not claim parity of the locator-free model.

**Harm.** If any CI for benefit_B − (retry, none, prior) lies entirely below 0, the paper reports that the method is harmful in that comparison.

## 9. Multiplicity

- **Primary H1:** tested alone at one-sided α = 0.025.
- **Secondary H2–H6:** one family, Holm step-down at familywise one-sided α = 0.025.
  - Sort the p-values ascending.
  - Reject H_(k) while p_(k) ≤ 0.025 / (5 − k + 1).
  - Stop at the first non-rejection.
- The secondary family is tested regardless of the H1 outcome.
- Unadjusted 95% CIs are reported for all comparisons.

## 10. Conduct: monitoring, stopping, exclusions

**Execution.**

- All 627 chains are run with all 7 arms.
- There is no interim analysis and no early stopping for efficacy or futility.

**Monitoring.** During the run only the following are inspected:

- completion counts;
- pairing identity (`paired_initial_state_identical`);
- worker self-checks (fixture test; criterion consistency logs).

Between-arm outcomes are not inspected.

**Technical failures.**

- A crashed job is re-run with the same seeds, up to 3 attempts.
- A chain is excluded from all arms only if:
  - the pairing check fails after one re-run; or
  - it crashes deterministically 3 times.
- Excluded chains are listed with reasons and not replaced.
- For H1, a worst-case sensitivity analysis is reported: excluded chains counted as benefit_B failure and retry success.
- If more than 5% of chains are excluded, this is reported prominently.

**Implementation errors.** If a defect in the worker affecting arm decisions or the criterion is found, then:

1. all chains of the affected arms are re-run after the fix;
2. the fix and the reason are logged as a deviation (Section 12).

## 11. Reported regardless of outcome

- For every arm: chain success, completed subtasks, steps, repairs and retries per chain.
- H1–H6 with estimates, CIs, p-values and Holm decisions.
- All descriptive analyses of Section 6.
- The offline B-label tables from `train_benefit_v4.py`, including:
  - M7_A (trained on A, evaluated on B, cross-fitted);
  - M7_v3all (in-sample reference);
  - the `--rp exclude` sensitivity.
- The development-set utilities from the δ tuning.
- The number of excluded chains, and the stratum fill if any.

## 12. Deviations policy

- Any change after the lock is appended to this file as a dated addendum.
  - Each addendum states what changed, why, and whether any stage-3 data existed at the time.
  - The new sha256 is appended to `prereg_round3.sha256`.
- Analyses as registered here remain primary; any additional analysis is labelled exploratory.
- The following are not deviations, provided the worker passes the fixture self-test before the first job:
  - writing the stage-3 worker (`worker_cl3.py`) and analysis (`analyze_cl3.py`) in round 6c;
  - recording their sha256 in `lock_addendum.json` before the first stage-3 job.

## 13. Files whose sha256 is recorded at lock time (`locked_config.json["sha256"]`, `lock_record.json`)

**Model, configuration and plan.**

- `models/benefit_models_v4_all.joblib`
- `benefit_model_20261003/models/benefit_models_v3_all.joblib` (benefit_A)
- `jobs_cl3.json`, `analysis/plan_report.json`, the sequence catalog
- `prereg_round3.md`
- `closedloop2_20261005/locked_config.json` and its copy `config_benefit_A_round2_frozen.json`
- `closedloop_20261004/locked_config.json`

**Labels.**

- the relabel `mapping.json`
- every `labels_bdef/*/results/label_*.json`: list in `analysis/label_files.sha256`, digest in `label_tree_digest`

**Analysis outputs.**

- `analysis/label_check.json`, `analysis/tune_report.json`
- `analysis/v4_report_all.json`, `analysis/v4_report_all_rpexclude.json`, `analysis/v4_rows_all.json`
- `analysis/cache_provenance.json`
- `analysis/fixture_inputs.npz`, `analysis/fixture_expected.json`

**Code.**

- `train_benefit_v4.py`, `lock_config3.py`, `plan_cl3.py`, `run.sh`
- `benefit_lib.py`, `benefit_lib_v2.py`, `benefit_lib_v3.py`, `features_patch.py`, `object_locator.py`, `evaluate_cv.py`
- `worker_cl2.py`, `worker_cl.py` (perturbation), `multitask_vocab_20261002/worker.py` (scripted repairs)
- `fc_worker.py` (mode B criterion)
