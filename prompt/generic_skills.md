# Generic skills: apply to every RTM-Optimizer prompt

Every milestone prompt in `prompt/` (for example `milestone1_skills.md`) builds on this file. Follow these rules in every session, whatever the task. If a milestone prompt conflicts with this file, the milestone prompt wins for that milestone only.

## 1. Prompt and session logging

- Log file: `RTM-Optimizer/logs/session.log`. Append to it; never rewrite or delete earlier entries.
- Add one entry for **every** user prompt, including prompts sent mid-task. Use this format:

  ```
  --------------------------------------------------------------------------------
  [YYYY-MM-DD] SESSION <session id> — Entry <n> (<short title>)
  --------------------------------------------------------------------------------
  USER PROMPT (verbatim):
  <exact text of the prompt, typos included>

  DECISIONS:   what was assumed or chosen, and why (placeholders flagged)
  ACTIONS / FILES:   files created, changed, moved or deleted
  RESULTS:     key numbers, with units (fill time, run time, mesh size, ...)
  STATUS:      done / awaiting team input (list what is needed)
  ```
- Number entries in sequence, continuing from the last entry in the file.
- Save attachments the user provides (datasheets, screenshots) to `logs/attachments/YYYY-MM-DD_<name>.<ext>`, and refer to them from the entry.

## 2. Run-time recording (performance log)

- Every solver run must append one row to `outputs/performance_log.csv`. This file is for performance analysis only and is kept separate from the results.
- Measure wall-clock time with `time.perf_counter()`. Every column header carries its unit in brackets:

  | Column | Unit | Meaning |
  |---|---|---|
  | `timestamp` | ISO 8601 | when the run finished |
  | `experiment` | — | experiment folder name (Section 3) |
  | `n_nodes [-]`, `fan_steps [-]` | count | problem size |
  | `mesh_time [s]` | s | gmsh meshing and conversion to dolfinx |
  | `assembly_time [s]` | s | FE operator and control-volume area assembly |
  | `fan_solve_time [s]` | s | FAN filling loop, including every pressure solve |
  | `solver_total_time [s]` | s | **total solver time** = assembly + FAN loop |
  | `plotting_time [s]` | s | figure generation |
  | `total_run_time [s]` | s | whole run, start to finish |
  | `time_per_fan_step [ms]` | ms | FAN loop time divided by the number of steps |

- The same timers are written to the experiment's `data/summary.json` under `run_time`. Print them to the console, and copy the solver and total run times into the session-log entry.
- Results and parameters (fill times, K, φ, μ, Δp, mesh settings) go to `outputs/run_history.csv`, also one row per run with units in the headers. Do not mix run times into this file.
- When a new column is added, the CSV helper rewrites the file with the combined header, so earlier rows are kept.

## 3. Naming convention for experiments and outputs

- Name each experiment `experiment_<#>_<short_description>`:
  - `<#>` is a whole number, assigned in order. Numbers are only reused after the old experiment has been abandoned and its outputs deleted.
  - `<short_description>` is snake_case, 1–4 words, and says what differs from the baseline. Symbols such as `K` may keep their case. Examples: `experiment_1_uniform_K`, `experiment_2_two_region_K_20pct_diff`.
- Register each experiment in the `EXPERIMENTS` dict in `main.py`, with a comment saying what it tests and whether its values are measured, placeholder or HYPOTHETICAL. The name is checked against `EXPERIMENT_RE` in `rtm/config.py`.
- Folder layout:

  ```
  outputs/
    run_history.csv                  results/params, one row per run (all experiments)
    performance_log.csv              run times, one row per run (all experiments)
    geometry_check.png               utility output (utils/plot_geometry.py), not an experiment
    experiment_<#>_<desc>/
      figures/  figN_<what>.png      PNG at 150 dpi or more; units on the axes; K, φ, μ, Δp in the caption
      data/     *.csv, summary.json  SI units, t = 0 when resin enters at the gate
      runs/<sub_run>/{figures,data}/ only for sweeps: one folder per sweep member
  ```
- All physical values and experiment settings live in **`prompt/experiment_configuration.md`**: a shared-parameter table, plus one row per experiment (name, K layout, K values with units, purpose). Milestone prompts point to it rather than repeating it. Update it, together with `main.py` / `rtm/config.py`, whenever an experiment is added or changed, and log the change.
- Abandoned experiments: delete their output folders, do not archive them. Their parameters and results stay in `logs/session.log`, `run_history.csv` and `performance_log.csv`.
- Put helper scripts that are not experiments (geometry checks, converters) in `utils/`. They write to `outputs/`, using paths resolved from the project root.
- Put prompts in `prompt/`, named `milestone<N>_skills.md`. This file is `prompt/generic_skills.md`.

## 4. Running

- dolfinx is not installed on the host. Run everything in the Docker image `dolfinx/dolfinx:stable` through `./run.sh`:
  - `./run.sh` runs experiment 1.
  - `./run.sh --experiment 2` or `--experiment experiment_2_two_region_K` runs one experiment.
  - `./run.sh --all` runs every registered experiment.
- Regenerate outputs with the code; never edit figures or CSVs by hand.

## 5. Standing technical rules

- Use SI units throughout. Comment every unit conversion (for example, "mm only for display").
- Never invent [TBD] physical values. Make each one a named config parameter, label it PLACEHOLDER or HYPOTHETICAL on every figure, and list it as a needed input.
- Do not load, overlay or fit experiment data, and do not run validation (ANSYS, video, MMS) unless a milestone prompt asks for it. The team owns that work.
- Before overwriting or deleting outputs, check what is there. Keep superseded code under `legacy/`.
