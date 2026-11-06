# Intermediated Computing Power Trading Network: Full Experimental Code

All experiments in the paper are launched from `run_all.ipynb`: hyperparameter tuning, the main experiment and benchmark comparison in the main text, the efficiency upper bound, ablations and attribution, multiple seeds, network experiments (network structure, scale and number of intermediaries, network shocks and recovery), exploitability, MARL sensitivity, and computational overhead. The core logic resides in the Python files under `cpm/`; the notebook is responsible only for invoking it.

## 1. Directory Structure (Place the Entire Folder in a New Directory)

```text
computing_market_v3/
├── run_all.ipynb      Launcher: run from top to bottom; one independent cell per experiment
├── README.md
├── requirements.txt
└── cpm/
    ├── config.py      Master parameter registry, unified experimental standard, joint tuning search space, values for sensitivity and structural experiments, and complete experiment manifest
    ├── specs.py       Card-type table, VRAM tiers, two electricity-tariff curves, and the single operating-cost function c^run
    ├── data.py        Synthetic data (profiles, hourly demand, per-task VRAM tiers)
    ├── mechanism.py   Uniform-price midpoint clearing and McAfee trade-reduction clearing
    ├── env.py         Market environment (three stages, bid caps, intermediary rules, rule-based benchmarks, U^base, W, common random numbers)
    ├── learner.py     PPO: IPPO (default) / MAPPO (centralized critic)
    ├── runner.py      Training loop and the single evaluation function evaluate()
    ├── collector.py   Data collection and network metrics during evaluation
    ├── upper_bound.py Efficiency upper bound (LP)
    ├── experiments.py Job orchestration: parallelism, resumption, result reuse, tuning selection, exploitability, overhead, and partial reruns (refresh/reset)
    ├── network.py     Network experiments: network shocks (node exit/entry, card-type layer removal, scale migration) and post-shock recovery
    ├── calibration.py Electricity tariffs, cost assumptions, and VRAM self-checks
    ├── analysis.py    Summary tables, paired tests, convergence diagnostics, and run checks
    └── figures.py     All figures
```

The following are generated automatically after running:

```text
data/<data key>/                      Dataset cache (a dataset is generated only once for the same set of data parameters)
results/tuning/best_params.json       Selected algorithm hyperparameters (automatically read in Stages 2 and 3)
results/<experiment name>/seed_<s>/   Per-job outputs: train_log.csv, val_log.csv, final.pt,
                                     train_meta.json, eval_<method>/...
results/tables/                       All summary tables (CSV); run_check.csv is the run-check table
figures/<subdirectory>/<figure>.png   1000 dpi PNG (also saved as PDF); subdirectories:
                                     00_data, main, robust, tuning, ablation,
                                     efficiency, network, exploit, sens_<parameter>,
                                     struct_<parameter>, overhead
figures/<subdirectory>/data/          All values and source information plotted in each figure (CSV),
                                     read directly when regenerating figures
```

## 2. Unified Standard (Controlled Variables)

- **All experiments use the same standard**: N = 3000, 2800 training days, 200 validation days, and 100 test days; the same dataset; and the same evaluation protocol (a fresh environment, a 20-day warm-up using the last 20 validation days, and common random numbers). The validation assessment at the end of training and the exploitability checks are also preceded by warm-up, whereas validation during training directly inherits the prosumers' price expectations from the training environment. No reduced-scale experiments are used.
- **Main text**: A single seed (`MAIN_SEED = 0`). MARL is compared with CDA, SDR, MMR, and McAfee using the same seed, data, and number of training days; all figures in the main text come from this run.
- **Supplementary information**: Each experiment changes only one factor; all other settings are identical to those in the main text.

| Experiment | Factor Changed | Seeds |
|---|---|---|
| Hyperparameter tuning (Stage 1) | Joint combinations of five algorithm hyperparameters: γ, learning rate, entropy coefficient, clipping range, and PPO epochs | Search seed 0; top three configurations rechecked with seeds 1 and 2 |
| Ablation / learning-based baselines / 2 × 2 attribution | One design component | Three seeds (0–2), paired with the full method using the same seed |
| Multiple seeds | Seed only (0–4) | Five |
| Sensitivity (MARL only) | One economic, calibration, or behavioral parameter: α, β, ρ̄, b, ξ (including the flat tariff), η, H, δ, demand-signal noise, share of pure buyers, reservation-price strength κ, expectation-updating speed λ̄, and loss-aversion strength | Single seed |
| Structural experiments | N ∈ {300, 1000, 2000, 5000}; \|M\| ∈ {1, 2, 4, 5, 8} | Single seed |
| Network shocks, post-shock recovery, exploitability | Starting from the main-text policy and changing only the network (nodes, connections, or scale) or a single agent | Single seed |

- **Tuning approach (joint random search)**: The five parameters are sampled jointly rather than scanned one at a time, avoiding missed interactions between parameters. The search space is specified by `config.TUNE_SPACE`: γ ∈ [0, 0.99], learning rate ∈ [3e-5, 1e-3] (log-uniform), entropy coefficient ∈ [1e-3, 5e-2] (log-uniform), clipping range ∈ [0.1, 0.3], and PPO epochs ∈ {3, …, 10}. We draw 24 configurations by Latin hypercube sampling using a fixed sampling seed of 2026 for reproducibility, and add the original default configuration as configuration 0, giving 25 configurations in total; all are trained with seed 0. The top three configurations by validation welfare are then retrained with seeds 1 and 2, and the final configuration is selected by the mean validation welfare across the three seeds, preventing selection of a configuration that merely happened to be paired with a lucky seed. Selection is based on deterministic welfare per unit of computing power over all 200 validation days after training; the test set is used only for reporting and never for selection. The selected values are written to `results/tuning/best_params.json`, after which all experiments use them.
- **Scope of tuning**: Only algorithm hyperparameters are tuned. Economic and calibration parameters describe the market itself; they are used only for sensitivity checks and are not "optimized" based on the results.
- **Training schedule and convergence criterion**: The 2800 training days are divided into two phases. The first 70% (1960 days) is the annealing phase, during which exploration noise and the entropy coefficient decrease linearly and the learning rate decays according to a cosine schedule to 5%. The remaining 30% (840 days) is the stabilization phase, during which all three are fixed at their lower bounds and training continues (`explore_lock_frac = 0.30`). Part of the rise in the training curve during annealing is merely due to decreasing exploration noise and cannot be used to determine convergence. Convergence is determined only within the stabilization phase, primarily from the deterministic validation curve: convergence is recorded when the relative change in the second half of the stabilization phase versus the first half is less than 2%, and the linear slope is either insignificant (p > 0.05) or less than 1% per 100 episodes. See the `converged` column in `convergence_<experiment>.csv`.
- **Prosumer reservation prices (willingness to sell)**: When a bid is below the reservation price, the prosumer prefers to keep the cards rather than sell them. The reservation price is constructed in two steps:  
  r⁰_{i,t} = c_{i,t} + φ_i · max(0, P̂_{i,t} − c_{i,t}),  
  r_{i,t} = r⁰_{i,t} + θ_i · max(0, ê_{i,t} − r⁰_{i,t}):
  - c_{i,t}: the prosumer's own operating cost (electricity price, power consumption, and depreciation, varying by time period);
  - P̂_{i,t}: the adaptive expectation of the resale price for the same card type in that period. It is initialized at the ECC retail price and subsequently updated daily by exponential smoothing of observed secondary-market transaction prices at an individual speed λ_i ~ U(0.5, 1.5) · λ̄ (λ̄ = 0.3);
  - φ_i: the minimum acceptable surplus share. It is heterogeneous across individuals, with φ_i ~ κ · Beta(1.29, 4.24) (κ = 1), a median of 0.20, and a mean of 0.23. This distribution is anchored in two empirical findings from ultimatum games: allocations below approximately 20% are rejected about half of the time, and when the average allocation is 40.4%, the average rejection rate is 16.2% (meta-analysis by Oosterbeek, Sloof, and van de Kuilen, 2004; 37 papers).
  - ê_{i,t}: the prosumer's expectation of its own selling price, smoothed using observed primary-market transaction prices at the same updating speed λ_i; θ_i ~ U(0.25, 0.35), based on Genesove and Mayer (2001): sellers facing nominal losses raise their asking prices by 25–35% of the expected loss, and the probability of sale declines markedly ("strategic withholding").
  - Reservation prices constrain the selling decision only (the same applies to selling to an intermediary or to the ECC buy-back). They do not enter utility or welfare; both are monetary measures.
  - The four rule-based benchmarks are subject to the same constraint: the CDA seller floor and McAfee truthful quotes are reservation prices. For SDR/MMR, the uniform price is obtained by solving a fixed point using the supply willing to sell at that price.
  - The reward baseline U^base is the counterfactual of "no intermediated market": no secondary-market price is available for reference, P̂ remains at the ECC retail price, and there is no selling-price experience, so the reservation price is c + φ_i(P^ES − c). This baseline is independent of the pricing method, training process, and actions. A prosumer therefore incurs no penalty for not selling; if it neither sells nor has demand, the reward is exactly zero (verified hour by hour for α = 0.1 and α = 0.4). The welfare baseline is also the same across methods.
  - The efficiency upper bound is first-best: the planner is not constrained by reservation prices, and W* ≥ W holds (verified hour by hour). The efficiency gap therefore includes losses caused by strategic withholding.
  - The individual parameters φ_i, θ_i, and λ_i are determined only by the data seed and are independent of the training seed. Multi-seed experiments change only training randomness; the market participants are the same.
  - κ = 0 with zero loss-aversion strength corresponds to the old model; both configurations are included as sensitivity settings.
- **Uniform-price rule-based benchmarks (consistent with the literature)**: The import price corresponds to P^ES, and the export price corresponds to P^EB; values are computed separately for each card type.
  - SDR follows Liu et al. (IEEE TPWRS 2017) and has no tunable parameters. When SDR ≥ 1, both the buying and selling prices equal P^EB. When SDR < 1:  
    p_s = P^ES · P^EB / ((P^ES − P^EB) · SDR + P^EB),  
    p_b = p_s · SDR + P^ES · (1 − SDR).
  - MMR follows Long et al. (IEEE PES GM 2017); its only parameter is the midpoint location, fixed at 0.5:  
    p_mid = (P^ES + P^EB) / 2.  
    When S ≥ D, buyers pay p_mid and sellers receive (D · p_mid + (S − D) · P^EB) / S. When S < D, sellers receive p_mid and buyers pay (S · p_mid + (D − S) · P^ES) / D.
  - Intermediaries are price takers: they purchase at p_s and set their retail listing price to p_b plus their own integration cost o_m. A buyer accepts a listing price no higher than its bid cap and trades at that listing price.
- **The secondary-market rules are identical across methods**: Buyers rank all feasible card types by preference and try them round by round; if the first-preferred card type is out of stock, they proceed to the next round. MARL buyers rank card types by their learned preferences; rule-based buyers sort by "(listing price / quote) × whole-order quantity" from lowest to highest; McAfee ranks by ECC cost. In the previous version, rule-based buyers tried only the tier with the lowest total expenditure, effectively giving the rule-based methods a worse secondary market and making the comparison unfair. This has been corrected. In both small-scale trials and full-scale evaluations with N = 3000, the welfare per unit for CDA therefore increased from about 120 to about 136, and that for MMR reached about 122. MARL (about 158 in the previous round) remains the best, but its lead is now about 16%, rather than more than 30% as in the old version. The new results should be used when writing.
- **Demand in SDR/MMR is counted only once**: Each buyer's whole-order demand is apportioned across its feasible card types according to their supply shares. In the single-commodity case, this reduces to the original S/D specification. The old version counted the buyer's full demand in every feasible layer (about 90% of tasks were feasible in all seven layers), overestimating D severalfold and systematically depressing S/D. After the correction, SDR performs substantially worse at α = 0.1 (about 68 at full scale): when S/D is not close to zero, the SDR formula rapidly pushes the seller price toward the export price P^EB = 0.1 · P^ES, which is below the reservation prices of most prosumers. This is a property of the SDR formula under the calibration used here and can be explained in the discussion.
- **Welfare includes operator surplus**: The budget surplus from McAfee trade reduction (the difference between the price paid by the buyer b_k and the amount received by the seller s_k) accrues to the market operator and is included in total surplus W, consistent with the LP upper bound (W ≤ W* was verified hour by hour). The table reports "Operator surplus" separately, making it possible to see how much of McAfee's welfare is captured by the operator. This term is zero under the other methods.
- **Adaptive newsvendor acquisition quantities for rule-based benchmarks**: The critical ratios for SDR/MMR directly use the current rule-based price for the period, whereas the other rules use transaction prices smoothed and retained across days. When an intermediary's smoothed sell-through rate for a card type falls below 0.3, its acquisition quantity is reduced proportionally, with a floor of 0.02: if the intermediary cannot sell the card type over the long run, it nearly stops purchasing it; under normal operations, no additional reduction is applied because inventory risk is already handled by the newsvendor quantile. In the old version, the price was reset to 0.5 at midnight every day and the intermediary did not observe its own sell-through, so it kept purchasing computing power that could not be sold.
- **Definition of network shocks**: After an intermediary exits, its acquisition capacity disappears, and the shares of the remaining intermediaries are not renormalized. A newly entering intermediary has θ = 1, uses the policy of a medium-scale intermediary, and adds new acquisition capacity. A retail participant exits by ceasing to generate demand and ceasing to provide computing capacity (using a fixed random ordering, with 10% ⊂ 20% ⊂ 30% ⊂ 50%).
- **Trading-network metrics** (reported in every evaluation under `eval_<method>/network_daily.csv` and `network_degrees.csv`): daily primary- and secondary-market edge counts (retail participant–intermediary pairs), Jaccard persistence between the edge sets of adjacent days, active seller and buyer shares, seller and buyer degree (number of intermediaries traded with per day), intermediary in-degree and out-degree, Gini coefficients and top-10% shares for seller and buyer transaction volumes, and the coefficient of variation of daily transaction volume.
- **Intermediary acquisition-quantity action**: Acquisition intensity is v = (a + 1) / 2 ∈ [0, 1], consistent with the mapping used for price actions, and has an initial value of 0.5. The old design used v = max(0, a), which made the acquisition quantity identically zero whenever a < 0 and caused vanishing gradients; the thinnest-margin 3090 tier could then be permanently abandoned (no inventory → no transaction → no learning signal). The old design is retained as the ablation `abl_relu_qty`.
- **Relearning protocol (`config.ADAPT`)**: Best-response retraining for exploitability tests and post-shock recovery use the same settings: starting from the trained policy, moderate exploration is re-enabled (exploration multiplier 0.2), the learning rate is set to half the tuned value, the entropy coefficient is set to its lower bound, and both the training and validation environments are warmed up for price expectations.
- **Post-shock recovery metrics**: Day 0 of each scenario corresponds to the original policy directly facing the post-shock network. After 300 days of relearning, the average of the last three validation assessments is used. Re-adaptation gain is computed by difference-in-differences:  
  (scenario endpoint − scenario day 0) − (control endpoint − control day 0),  
  which removes changes caused by relearning itself. Then,  
  `recovered_share` = re-adaptation gain / shock-induced decline.  
  Capacity removed by the shock is not restored, so only the "pricing re-adaptation" component can be recovered. In small-scale trials this component is small (intermediary exit recovers only about 5–10% of the decline), indicating that the losses from shocks are primarily structural and that the main-text policy is already relatively robust to changes in the network itself.
- **Network formation during training**: Trading-network metrics are recorded during deterministic validation both in the middle of training and at the end (`val_net_*`: primary/secondary edge counts, active seller share, seller/buyer degree, and retail-flow HHI) and plotted as `Fig_network_formation` (`FigS_robust_network_formation` for multiple seeds), describing how the trading network "grows."
- **Exploitability test**: Each intermediary deviates individually. For prosumers and pure buyers, 5% of each group are randomly sampled to form a group that shares one deviation policy (the learning signal for a single retail participant would be too weak among more than one thousand participants). The deviating agents are retrained for 1000 days, and the deviator payoff is checked on the validation set every 100 days. The best version, including the original untrained policy, is retained, so the payoff increment is ≥ 0, as required by the definition of a best response. The validation curve is stored in `exploit/<agent type>/br_val_curve.csv`.
- **Result reuse**: Jobs with exactly the same parameters and seed automatically reuse existing results. For example, the MARL experiment in the main text is the configuration selected during tuning, and seeds 0, 1, and 2 of the multi-seed experiments are the jobs already trained during tuning verification; neither is retrained. The `reused_from` column in the run-check table identifies the source.

## 3. How to Run

1. Run `pip install -r requirements.txt`. If an NVIDIA GPU is available, install the CUDA version of PyTorch using the command on the official PyTorch website.
2. From inside this folder, open `run_all.ipynb` in Jupyter. The working directory must be this folder.
3. First keep `SMOKE = True` and run from top to bottom once (a few minutes) to confirm that every cell produces data and figures. Smoke-test outputs are written to `results_smoke/` and `figures_smoke/`.
4. Set `SMOKE = False` and run from top to bottom. The two tuning cells in Stage 1 (search followed by verification) must finish before Stages 2 and 3 can begin. If they are run out of order, the code reports an error.
5. To resume after an interruption, rerun the same cell: completed jobs are skipped automatically. If the parameters of an existing job differ from the current parameters, the code reports an error instead of mixing old and new results.
6. The final cell only reads results and regenerates all figures and tables, so it can be run independently at any time.

**Parallelism**: Ensure that `WORKERS × THREADS ≤ the number of physical cores`; oversubscription can slow execution by several times. For a 24-core machine, the recommended settings are `WORKERS = 3–6` and `THREADS = 4`.

**Computational cost** (based on measurements from the previous round, with N = 3000, 2800 days, approximately 28 minutes per job, and three jobs running in parallel):

- There are approximately 106 training jobs: 31 for tuning (25 search runs + 6 verification runs; the seed 0–2 jobs for the main text and multi-seed experiments reuse their results), 2 additional multi-seed jobs (seeds 3 and 4), 15 ablation jobs, 9 attribution jobs, 40 MARL sensitivity jobs, and 9 structural jobs (N = 5000 is slower). With three jobs running in parallel, this takes approximately 30–35 hours.
- Evaluation only, without training: 24 network-shock scenarios (including 4 scale-migration scenarios) × 5 methods, approximately 2–3 hours.
- Relearning: post-shock recovery uses 6 × 300 days, and exploitability uses 5 × 1000 days, for approximately 2–3 hours in total.
- Total: approximately 35–40 hours. Approximately 20 GB of disk space is also required: new datasets must be generated whenever ξ, b, the share of pure buyers, or N changes.

## 4. Outputs and Their Locations in the Paper

| Output | Purpose |
|---|---|
| `tables/table_methods_main.csv`, `figures/main/Fig_methods` | Main-text method comparison (Table 3; daily paired Wilcoxon tests are provided as supplementary information) |
| `figures/main/Fig_network_formation` | Formation of the trading network during training (edge count, active sellers, and retail concentration) |
| `tables/table_network_main.csv`, `table_intermediaries_main_marl.csv` | Network metrics and intermediary table |
| `figures/main/Fig_training`, `tables/convergence_main.csv` | Convergence curves (training and deterministic validation; gray shading denotes the stabilization phase) and convergence determination for the stabilization phase |
| `figures/main/Fig_flow_network` | Sankey flow diagram |
| `figures/main/Fig_hourly_spread` | Hourly price spreads (TOU versus flat tariff) and transaction volume |
| `figures/main/Fig_price_ratio` | Transaction price / ECC retail price and transaction price / buyer bid cap |
| `figures/main/Fig_substitution` | Inter-tier substitution network (row percentages) |
| `figures/main/Fig_bidirectional`, `Fig_surplus` | Bidirectional nodes and surplus distribution |
| `tables/table_efficiency_main.csv`, `figures/efficiency/` | Efficiency ratios |
| `tables/table_ablation.csv`, `table_attribution.csv`, `figures/ablation/` | Ablations and 2 × 2 attribution |
| `tables/table_tuning.csv`, `figures/tuning/FigS_tuning` | Joint tuning: scatter plots of each parameter against validation welfare and multi-seed verification of the top three configurations (supplementary information) |
| `tables/table_methods_robust.csv`, `figures/robust/` | Multi-seed robustness (main conclusions use seed-level paired comparisons: mean ± standard deviation of MARL minus each benchmark, and the number of seeds in which MARL performs better) |
| `tables/table_sens_<parameter>.csv`, `figures/sens_<parameter>/` | Sensitivity analyses |
| `tables/table_sens_n_consumers.csv`, `table_sens_n_intermediaries.csv`, `figures/struct_*` | Structural experiments |
| `tables/table_network_main.csv`, `figures/main/Fig_network_structure` | Trading-network structure: edge count, degree, relationship persistence, concentration, and daily transaction-volume volatility |
| `tables/table_netshock.csv`, `figures/network/` | Network shocks: intermediary exit/entry, prosumer/buyer exit, card-type layer criticality, and scale migration |
| `tables/table_recovery.csv`, `recovery_curves.csv`, `figures/network/Fig_recovery` | Post-shock recovery trajectories and re-adaptation gains (difference-in-differences) |
| `tables/table_exploitability.csv`, `overhead.csv` | Exploitability and computational overhead |
| `tables/calibration_*.csv`, `figures/00_data/` | Calibration self-checks, electricity prices and demand, and VRAM distribution |
