# Architecture

This document describes the architecture of `litearm-benchmark`.

## Context

The toolkit measures how closely the torque estimated by the arm's current loop matches
the torque predicted by an inverse-dynamics model. It is a PC-side analysis tool; the
firmware remains the source of the motion and torque estimates.

## Components

| Component | Responsibility | Depends on |
|---|---|---|
| Acquisition | Read joint position, velocity, and torque at up to 100 Hz | `litearm-python` |
| Dynamics | Compute inverse-dynamics torque from the arm model | Arm model parameters / URDF |
| Analysis | Compare measured and modelled torque, compute error metrics | NumPy |
| Reporting | Plot time series and error distributions | Matplotlib |

## Data flow

1. Acquire joint states and current-loop torque into a timestamped dataset.
2. Derive acceleration from velocity (filtered) or from the commanded trajectory.
3. Compute inverse-dynamics torque for each sample.
4. Compare without and plot per-joint error metrics.

## Boundaries

- This repository owns data collection, dynamics evaluation, and reporting.
- The SDK, firmware, and arm description remain in their own repositories.

## Decisions

- Sampling stays at or below the 100 Hz status stream so no sample is interpolated.
- Raw data and generated reports are kept out of the main repository by default.
