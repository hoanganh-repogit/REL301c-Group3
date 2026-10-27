# Phase 6 — LaTeX Report Plan

This document fixes the research direction and the plan for writing the report.
All LaTeX sources, tables, figures, and references of the report live in
`latex/`.

## 1. Research direction

The project **no longer compares DQN with PPO**. The only object of study is
**Dueling Double DQN**:

- The agent is trained **only on Level 4**.
- After training, the network is frozen and run greedily on the other levels.
- No fine-tuning, no gradient updates, and no data from the test levels.
- The goal is to evaluate **zero-shot cross-level generalization**: whether a
  policy learned in one environment transfers to boards with different sizes
  and obstacle layouts.

PPO and the old Phase 4 experiments are outside the main research question.
They may be mentioned very briefly as project history or in an appendix, but not
in the title, the main Methodology, the main results tables, or the Conclusion.

## 2. Title

### Official title

> **Evaluating Cross-Level Generalization of a Dueling Double DQN Trained in a
> Single Snake Environment**

Vietnamese title (for course submission):

> **Đánh giá khả năng khái quát hóa đa cấp độ của Dueling Double DQN huấn luyện
> trên một môi trường trong trò chơi Snake**

The title reflects the design: one algorithm, one training environment, and
several evaluation environments. It avoids phrases such as "achieves mastery" or
"works on every level", because those are conclusions that must be verified by
evaluation.

### Short title

> **Cross-Level Generalization of DQN in Snake**

## 3. Problem statement

A Dueling Double DQN is trained only on Level 4 (30×30 board, 20 obstacle cells,
no score limit). The 27-dimensional state and three relative actions make the
network's input and output independent of board size. The same checkpoint can
therefore be loaded and run on Levels 1, 2, 3, 5, and 6 without changing the
architecture.

However, **running technically does not mean generalizing well**. The report may
draw conclusions only after repeated evaluation over many episodes, with fixed
seeds and quantitative metrics.

### Research questions

**RQ1.** How well does a Dueling Double DQN trained only on Level 4 perform on
its own training environment under greedy control?

**RQ2.** Without fine-tuning, how well does that policy transfer to Levels 1, 2,
3, 5, and 6?

**RQ3.** How do score, success rate, stability, and episode termination causes
change across levels with different board sizes and obstacle layouts?

### Objectives

1. Build a Snake environment parameterized by level.
2. Train a Dueling Double DQN stably on Level 4 only.
3. Evaluate the same policy on six levels without retraining.
4. Analyze the strengths, failure modes, and limits of single-environment
   training for cross-level transfer.

## 4. Level scope

| Level | Role in the study | Board | Target | Notes |
|---|---|---:|---:|---|
| Level 1 | Unseen evaluation | 15×15 | 25 | No obstacles |
| Level 2 | Unseen evaluation | 20×20 | 35 | 8 obstacle cells |
| Level 3 | Unseen evaluation | 24×24 | 50 | 16 obstacle cells |
| Level 4 | **Training + in-domain evaluation** | 30×30 | No limit | 20 obstacle cells |
| Level 5 | Held-out evaluation | 32×32 | 100 | 34 staggered obstacle cells |
| Level 6 | Challenge evaluation | 36×36 | 150 | 68 gate-shaped obstacle cells |

In the current continuous run the agent only sees Level 4. Levels 1–3 are
therefore also unseen for this checkpoint, even though their configs keep
`is_holdout: false` from the old design. The report must distinguish:

- **In-domain:** Level 4.
- **Unseen transfer:** Levels 1, 2, 3, 5, and 6.
- **Explicit held-out:** Level 5, according to the repository metadata.

## 5. Training status

Continuous DQN seed 0 stopped during episode 19,095:

| Property | Value |
|---|---:|
| Training level | Level 4 |
| Completed episodes logged | 19,094 |
| Episode running when stopped | 19,095 |
| Environment steps | 14,293,725 |
| Best checkpoint episode | 17,000 |
| Best rolling mean (100 training episodes) | 52.09 |
| Best single training episode | 157 |
| State dimension | 27 |
| Action space | 3 relative actions |

The values 52.09 and 157 are **training statistics**, not evaluation results.
The run has only one training seed. An independent greedy evaluation has been
completed on six levels with 100 seeds per policy–level combination. Variability
in the results tables therefore describes episodes of one learned policy, not
variability across independent training runs.

Two checkpoints are evaluated:

- `results/level4_continuous_dqn/seed_0/checkpoint_best.pt`
- `results/level4_continuous_dqn/seed_0/final_model.pt`

The main checkpoint for the report must be chosen by a predefined rule.
`checkpoint_best.pt` is the main model because it is selected by the 100-episode
rolling mean; `final_model.pt` serves as a sensitivity check.

## 6. Evaluation protocol

### 6.1 Required settings

- Fully greedy (`epsilon = 0`), no model updates.
- The same list of evaluation seeds for every level and checkpoint.
- At least 100 episodes per level; 200 if time allows.
- Store raw per-episode results, not only means.
- Record checkpoint hash and path, level config, run date, and code version.
- Evaluation runs without rendering; rendering is only for illustrations.

### 6.2 Evaluation matrix

| Checkpoint | L1 | L2 | L3 | L4 | L5 | L6 |
|---|---:|---:|---:|---:|---:|---:|
| Best DQN | 100–200 | 100–200 | 100–200 | 100–200 | 100–200 | 100–200 |
| Final DQN | 100–200 | 100–200 | 100–200 | 100–200 | 100–200 | 100–200 |
| Random baseline | 100–200 | 100–200 | 100–200 | 100–200 | 100–200 | 100–200 |

The random policy is the minimum baseline needed to show that the learned policy
beats random behavior. An untrained-network baseline may be added if time
allows, but it is not required.

### 6.3 Metrics to report

For every level:

- Mean, standard deviation, median, min/max score.
- P25, P75, and P90 score.
- Mean episode steps and steps per food.
- Death rates by wall, obstacle, self-collision, and timeout.

For Levels 1, 2, 3, 5, and 6, which have a target:

- Target success rate.
- Normalized score: `min(score / target_score, 1)`.

For Level 4, which has no target:

- Do not invent a success threshold after seeing the results.
- Use the score distribution and survival/episode steps as in-domain metrics.

Do not use a single "generalization gap" between L4 and the other levels, because
L4 has no target, board sizes differ, and episode termination is not uniform.
Per-level comparison with success rate and normalized score is more honest.

### 6.4 Conditions for each claim

- "The model runs on other levels": the checkpoint loads and completes episodes.
- "The model transfers": it clearly beats the random baseline on quantitative
  metrics.
- "The model generalizes well": stable performance on many unseen levels, not
  just one good episode or Level 1 alone.
- "The model generalizes to harder levels": evidence on Levels 5 and 6.

Without enough evaluation, the report may only say "the design supports running
on multiple levels", not "the model generalizes well".

## 7. Report structure

### 1. Introduction

Written after the Results:

- Problem: RL agents often depend on the distribution of their training
  environment.
- Gap: does a Snake policy trained in a single environment transfer to other
  board sizes and obstacle layouts?
- State RQ1–RQ3.
- Contributions:
  1. A Snake environment with six parameterized levels.
  2. A Dueling Double DQN with a board-size-independent state and action space.
  3. A zero-shot evaluation protocol from one training level to five unseen
     levels.
  4. An analysis of score, success, and failure modes by environment difficulty.

Do not include PPO or the old Phase 4 results in the contributions.

### 2. Related Work

Three groups close to the topic:

1. DQN, Double DQN, and the Dueling Network Architecture.
2. Generalization and zero-shot transfer in reinforcement learning.
3. State representation and domain invariance in games or navigation.

Every citation must have its authors, year, venue, and DOI/URL checked before it
is added to `references.bib`.

### 3. Methodology

#### 3.1 Snake environments

- Game rules, rewards, termination, and food spawning.
- Table of the six levels and their training/evaluation roles.
- Emphasize that only Level 4 appears in the replay buffer during training.

#### 3.2 State and action representation

- Explain all 27 features by group, not only as a list.
- Three relative actions: straight, turn right, turn left.
- Why the design allows one network to be shared across board sizes.

#### 3.3 Dueling Double DQN

- Dueling value/advantage decomposition.
- Double-DQN target.
- Replay buffer, epsilon-greedy, target network, Huber loss, and gradient
  clipping.
- Hyperparameter table taken from the config of the continuous run.

#### 3.4 Single-environment training

- Continuous Level 4 training, no target score.
- Episode termination conditions and the manual stopping rule of the run.
- How the best checkpoint is saved using the 100-episode rolling mean.
- State clearly that there is only one training seed, as a limitation.

#### 3.5 Cross-level evaluation

- Frozen checkpoint, greedy policy, no fine-tuning.
- Seed list, number of episodes, baseline, and metrics.
- The rule for choosing the best checkpoint is fixed before looking at test
  results.

### 4. Results

Describe only what is observed, without explaining causes yet:

1. **Level 4 training curve:** raw score and rolling mean over environment steps.
2. **Main results table:** Best DQN on L1–L6.
3. **Best versus final checkpoint:** check whether the final model degrades.
4. **DQN versus random baseline:** per level.
5. **Success/normalized score:** for the five levels with a target.
6. **Failure modes:** wall, obstacle, self-collision, timeout.
7. **Score distribution:** boxplot/violin plot instead of only a mean bar chart.

Each table/figure gets 2–3 sentences describing specific numbers. Do not use a
few nicely rendered episodes as the main evidence.

### 5. Discussion

- Answer RQ1–RQ3 in turn.
- Analyze transfer from L4 to smaller levels and to larger/harder levels.
- Relate the results to the 27-feature state and relative actions, without
  claiming causality, since there is no ablation.
- Analyze how failure modes change with board and obstacle layout.
- Threats to validity:
  - Only one training seed.
  - Only one training environment.
  - The levels differ in size, obstacles, and target at the same time, so the
    variables are not isolated.
  - The best checkpoint is selected from the training rolling mean.
  - No ablation of state features and reward shaping.
  - Different targets/termination make raw scores not directly comparable.

### 6. Conclusion

Three short paragraphs:

1. Which agent was trained, where it was trained, and how it was evaluated.
2. Answer the RQs with real numbers from the evaluation.
3. Next steps: more training seeds, more held-out layouts, ablations, domain
   randomization, and curriculum learning as future work.

### Abstract

Written last, following five points: context → problem → method → results with
numbers → significance. Do not mention PPO. Do not use the best training episode
(157) as a main result; the abstract must use greedy evaluation on unseen levels.

## 8. Order of work

### P6.1 — Freeze the design

- [x] Remove PPO from the scope.
- [x] Fix Level 4 as the only training environment.
- [x] Fix L1, L2, L3, L5, L6 as unseen evaluation environments.
- [x] Fix the new title.
- [x] Record commit hash, checkpoint hash, and config snapshot in the metadata.

### P6.2 — Complete the evaluation

- [x] Write a headless evaluation script for L1–L6.
- [x] Evaluate `checkpoint_best.pt` with 100 seeds per level.
- [x] Evaluate `final_model.pt` with the same protocol.
- [x] Run the random-policy baseline.
- [x] Export raw CSV, summary CSV, and metadata JSON.
- [x] Check the smoke test and the aggregate values.

### P6.3 — Generate tables and figures automatically

- [x] Training curve over environment steps.
- [x] L1–L6 results table.
- [x] Results chart per level/checkpoint.
- [x] Score distribution boxplot.
- [x] Failure-mode chart.
- [x] Best-versus-final checkpoint comparison.

Analysis scripts live in `latex/scripts/` and their output in
`latex/generated/`; numbers are never copied by hand into several `.tex` files.

### P6.4 — Write the core sections

- [x] Methodology.
- [x] Experimental Setup.
- [x] Results.
- [x] Discussion and Threats to Validity.

### P6.5 — Finish the paper

- [x] Verify the references and write Related Work.
- [x] Write the Introduction based on the real results.
- [x] Write the Conclusion, then the Abstract.
- [x] Check every number in the prose against the CSV files.
- [x] Add Wilson 95% CIs for the success rates.
- [x] Generate the rows of the main results table directly from the raw CSV.
- [x] Switch to the IEEEtran (journal) class and the `IEEEtran` citation style.
- [x] Clean LaTeX build: no errors, no citation/cross-reference warnings, no
  overfull boxes (checked with Tectonic on 2026-10-27).
- [x] Fill in author names, affiliation, and e-mail.
- [x] Timeout analysis by score, McNemar test for best/final, epsilon = 0.01
  experiment on Levels 4, 5, 6, and the six-level layout figure.

## 9. LaTeX folder structure

```text
latex/
├── README.md
├── main.tex                  # the whole report in one file
├── references.bib
├── generated/                # PDF/PNG figures and table rows, generated from results/
├── scripts/
│   ├── generate_report_assets.py
│   ├── generate_presentation.py
│   └── preview_presentation.py
└── slides/
    ├── template.pptx         # original template, do not edit
    ├── Snake_DQN_Research_Presentation.pptx
    ├── Snake_DQN_Research_Presentation.pdf
    ├── speaker_notes_vi.md
    └── preview/
```

## 10. Completion criteria

Phase 6 is complete when:

- The report builds from scratch.
- The same frozen DQN checkpoint is evaluated on L1–L6.
- Results are based on many episodes with reproducible seeds and a random
  baseline.
- Every table and figure traces back to the raw CSV files.
- The conclusions clearly separate "runs", "transfers", and "generalizes well".
- PPO and the old Phase 4 results no longer distort the main research story.

## 11. References and build

All citations are kept in `references.bib`. `main.tex` uses the `IEEEtran`
class (journal) and the `IEEEtran` BibTeX style; the reference list is never
written by hand in the paper.

Current source groups:

- Q-learning and the foundational reinforcement learning textbook.
- DQN, Double DQN, and the Dueling Network Architecture.
- Generalization benchmarks: CoinRun and Procgen; a survey of zero-shot
  generalization.
- Reproducibility and statistical evaluation in deep RL.
- Time limits in RL (the basis for treating timeouts as truncation).
- Statistics: Wilson interval, McNemar test, Holm correction.
- Deep reinforcement learning applied directly to Snake.

Before building, regenerate the figures and tables if the results in `results/`
change:

```powershell
py -3.14 latex/scripts/generate_report_assets.py
```

Full build with citations, in this order:

```powershell
cd latex
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Without a local LaTeX installation, use Overleaf (upload `main.tex`,
`references.bib`, and the `generated/` folder) or Tectonic:
`tectonic -X compile main.tex`.

LaTeX Workshop usually runs this build sequence automatically. Before
submission, re-check every DOI/URL, make sure there are no citation or reference
warnings, and keep only sources that are actually cited in the text.
