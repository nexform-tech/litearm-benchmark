# Architecture

This document describes the architecture of `litearm-benchmark`.

## Context

The tool measures how closely the torque estimated by the arm's current loop
matches the torque predicted by a rigid-body inverse-dynamics model. It is a
PC-side analysis tool; the firmware remains the source of the motion and torque
estimates. The inverse-dynamics core is a line-by-line port of the firmware's
`rnea_core_m`.

## Components

| Module | Responsibility | Depends on |
|---|---|---|
| `collect` | Acquire joint states and current-loop torque at up to 100 Hz, run scenarios, persist sessions | `litearm-python` (lazy, hardware path only) |
| `model` | Firmware model snapshot (rigid-body parameters, `jm`, payload, gravity) | Standard library |
| `rnea` | World-frame Newton–Euler inverse dynamics, mass matrix, gravity torque | Standard library |
| `filters` | Uniform resampling, zero-phase Butterworth, central difference, metrics helpers | Standard library |
| `analyze` | Residuals, per-joint metrics, strata, delay scan, correlations, diagnosis | `rnea`, `filters` |
| `plots` | Pure-SVG figures (no Matplotlib) | Standard library |
| `report` | Chinese Markdown report | `analyze` |
| `cli` | `run` / `collect` / `analyze` subcommands and the safety gate | All of the above |

## Data flow

1. Acquire joint states and current-loop torque into a timestamped session on a
   uniform grid (`status.csv`), with the model snapshot (`model.json`) and run
   metadata (`meta.json`).
2. Resample non-uniform arrivals linearly onto a uniform grid.
3. Filter `q`/`dq` and measured `tau` with zero-phase Butterworth filters; derive
   `ddq` by central difference then low-pass.
4. Compute inverse-dynamics torque per sample and the residual
   `tau_meas − tau_model`.
5. Compute per-joint metrics, speed strata, delay scan, and residual
   correlations; write `metrics.json` and `timeseries.csv`.
6. Generate SVG figures and the Markdown report; a multi-scenario run also gets
   `summary.md`.

## Boundaries

- This repository owns collection, inverse-dynamics evaluation, plotting, and
  reporting.
- The SDK, firmware, and arm description remain in their own repositories.
- The SDK is an optional extra (`.[sdk]`) and is imported lazily so the offline
  path has no third-party dependency.

## Decisions

- Sampling stays at or below the 100 Hz status stream.
- Analysis is standard-library only, including SVG plotting; figures must not
  need Matplotlib or a GUI stack.
- Raw data and generated reports are kept out of the repository by default
  (`runs/`). The frozen synthetic reference under `examples/reference_sine/` is a
  deliberate, deterministic exception used as a smoke test and documentation
  example.
- The firmware model is a regression regressor and may be non-positive-definite;
  it is used for forward evaluation only and must never be inverted.
