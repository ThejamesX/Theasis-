# Copilot Instructions

## Project Overview

This repository is a Python-based hybrid truck simulation and calibration workspace built around VECTO-style inputs and outputs. The code models ECMS, A-ECMS, P-ECMS, and DP strategies, and it reads domain files from the `Driving Cycle/`, `Engine/`, and `Emotor/` folders.

## Working Rules

- Prefer minimal, physics-aware changes. Do not alter torque split logic, battery dynamics, map interpolation behavior, or unit conversions unless the task explicitly asks for it.
- Preserve existing CSV and plot outputs unless a change to those artifacts is requested.
- Use paths relative to the script location with `os.path.dirname(os.path.abspath(__file__))` or `os.path.join(...)`. The project is intended to run from the repository root on Windows.
- Keep edits ASCII-only unless a file already uses non-ASCII characters.
- Match the existing style: plain Python scripts, direct function entry points, and pragmatic inline comments only when they clarify non-obvious physics or data handling.

## Main Entry Points

- `main.py`: runs a single ECMS-style simulation for one cycle and strategy.
- `run_batch.py`: runs batch comparisons across cycles, capacities, and strategies, and writes summary output.
- `run_dp.py`: runs the dynamic programming baseline and generates plots.
- `get_baselines.py`, `run_ice_verification.py`, `check_maps.py`, `debug_pecms_end.py`: analysis and verification utilities.
- `Calibration/`: calibration and tuning scripts.
- `A_ECMS_Implementation/`: adaptive ECMS implementation and calibration helpers.
- `P_ECMS/`: predictive ECMS supervisor and horizon prediction logic.

## Data Layout

- `Driving Cycle/`: `.vmod` cycle files used as simulation inputs.
- `Engine/`: engine maps and drivetrain files such as `.vmap`, `.vgbx`, `.veng`, `.vveh`.
- `Emotor/`: motor and battery files such as `.vemo`, `.vem`, `.vreess`, `.vbatv`, `.vbatr`.
- `output/`: generated plots and CSV results.

## Dependency Expectations

The code imports `numpy`, `pandas`, `scipy`, and `matplotlib`. There is no `pyproject.toml`, `requirements.txt`, or `setup.py` in the repo, so assume dependencies are managed manually in the active Python environment.

## Implementation Conventions

- Treat `VectoLoader` and `P2HybridTruck` as the core data/model boundary.
- Preserve the existing column names and renaming logic for `.vmod` files, especially `time`, `dt`, `velocity_kmh`, `rpm_ice`, `power_wheel_kw`, and `altitude_m`.
- Keep command-line style scripts runnable as standalone modules.
- When adding new results, follow the existing naming scheme in `output/` and avoid overwriting unrelated artifacts.

## Validation

- Prefer running the relevant script directly to verify behavior after a change.
- Use `test_physics.py` and `test_gravity_opt.py` as lightweight regression checks when touching vehicle physics or supervisor logic.
- If a change affects interpolation, battery dynamics, or load reconstruction, validate the affected simulation path end-to-end.

## Agent Interaction & Conversation Preferences

- **Plan First:** For multi-step tasks, create a concise TODO plan and track it (the assistant must use `manage_todo_list`).
- **Pre-Tool Preambles:** Before executing repository-modifying or external tools, provide a 1-2 sentence preamble that explains what will run and why.
- **Progress Cadence:** After 3–5 tool calls or when creating/editing several files, provide a short progress update and next step.
- **File References:** When mentioning files or symbols in messages, reference the file path and wrap the path as a workspace link (for example: [.github/copilot-instructions.md](.github/copilot-instructions.md)).
- **Edit Method:** Use `apply_patch` (or equivalent) for file edits so changes are auditable and reproducible.
- **Concise Tone:** Keep responses concise, practical, and teammate-like. Prefer short bullets and explicit next steps.
- **Do Not Volunteer Model:** Do not disclose the assistant/model name unless explicitly asked. If asked about the model, reply: "GPT-5 mini".

If any of these preferences should apply only to certain file types or directories, describe the scope here so the assistant can follow it precisely.