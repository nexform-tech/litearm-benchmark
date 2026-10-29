# litearm-benchmark

Current-loop torque estimation accuracy evaluation for the LiteArm 7-DOF arm.
Collect joint states and current-loop torque from the arm, replay rigid-body
inverse dynamics, and report per-joint error, plots, and error-source
attribution — from one command.

[中文文档](README.zh-CN.md)

## Scope

- Joint-level acquisition at up to 100 Hz: position, velocity, current-loop torque,
  MOSFET/coil temperatures, and per-joint fault flags.
- Inverse-dynamics torque from the firmware's online rigid-body model (RNEA).
- Per-joint error metrics (RMSE / MAE / bias / peak / R² / correlation), plots,
  delay scan, and evidence-ranked error-source diagnosis.
- Three excitation scenarios: static hold, sine sweep, and slow multi-pose moves.

The tool output (CLI text, report, figures) is in Chinese. This README and the
`docs/` pages are in English.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate

# Offline analysis and tests: no hardware, no third-party runtime dependencies.
pip install -e ".[test]"

# Hardware collection only: additionally installs the optional litearm SDK.
pip install -e ".[sdk]"
```

The SDK is optional and imported lazily, so offline analysis and the test suite
work without it. Without the SDK, a collection command exits with a message
pointing to `pip install -e ".[sdk]"` and the offline `analyze` path.

## Quick start

```bash
# Print the collection plan and the current pose. No motion.
litearm-tau

# Really collect: hold + sine + poses, then analyze and plot each scenario.
litearm-tau --go

# Collect one scenario only.
litearm-tau collect --scenario sine --go

# Offline analysis of a session directory or a multi-scenario run directory.
litearm-tau analyze runs/20261029_120000_tau_eval --outdir /tmp/tau
```

`python3 -m litearm_benchmark.torque_eval ...` is equivalent to `litearm-tau`
when the package is not installed as a script.

Useful flags:

| Flag | Meaning |
|---|---|
| `--go` | Enable, move, and collect. Without it the command is read-only. |
| `--scenario all\|hold\|sine\|poses` | Scenarios to run (default `all`). |
| `--fs` | Recorded sample rate, 1..100 Hz (default 100). |
| `--amp`, `--freq` | Sine amplitude (rad) and frequency (Hz); amplitude is narrowed to keep the soft-limit margin. |
| `--sine-joints 1,2,3` | Limit the sine excitation to selected joints (1-based). |
| `--gravity x,y,z` | Analysis gravity vector (default `0,0,-9.81`). Use `0,9.81,0` when the bench is mounted with gravity along +Y. |
| `--cutoff-dq`, `--cutoff-ddq`, `--cutoff-tau` | Zero-phase low-pass cutoffs (Hz). |
| `--ambient-c` | Manually recorded ambient temperature; the firmware only reports winding/MOS temperature. |
| `--no-plot`, `--no-report` | Skip SVG figures or the Markdown report. |

## Scenarios

- **hold** — static hold (~3 s) at the current pose. Isolates gravity model and
  current-loop zero offset; `ddq = dq = 0`, so the residual is a static bias.
- **sine** — continuous servo sine (~10 s, 0.2 Hz) with a commanded rate of
  50 Hz. Excites inertia, Coriolis, and friction terms; amplitude is
  automatically reduced so every selected joint keeps a 0.15 rad soft-limit
  margin.
- **poses** — slow multi-pose sequence (four target poses, 1.2 s dwell) with
  automatic amplitude narrowing. Exposes direction-dependent friction and
  gravity error at different configurations.

Every planned target passes a soft-limit margin check before anything moves.

## Collected data

A session directory contains:

- `status.csv` — one row per uniform sample: `t, seq, mode, mode_name, flags,
  joint_fault`, then `q{i}, dq{i}, tau{i}, tmos{i}, tcoil{i}, err{i}` for each
  joint.
- `meta.json` — scenario description, sample rate, sample count, time span,
  ambient temperature (if given), and other run metadata.
- `model.json` — the firmware model snapshot read back from the arm: rigid-body
  parameters, motor inertia (`jm`), payload, gravity, firmware version, and a
  geometry fingerprint.

Non-uniform arrival intervals are linearly resampled onto a uniform grid before
analysis; the report flags that this happened.

## Processing chain

1. Resample `status.csv` onto a uniform grid at the recorded rate.
2. Zero-phase Butterworth filtering (`filtfilt`): `q`/`dq` at 15 Hz, measured
   `tau` at 10 Hz. `ddq` is obtained by central difference on filtered `dq`,
   then low-passed at 5 Hz.
3. Trim both ends (`max(2/cutoff_ddq, 0.1) s`, capped at span/5) to avoid
   zero-phase filter transients.
4. Replay world-frame Newton–Euler RNEA (a line-by-line port of the firmware's
   `rnea_core_m`): `tau = M·ddq + C·dq + G + jm·ddq + friction`.
5. Residual `tau_meas − tau_model` per joint, in N·m; a `scaled` variant applies
   the firmware `gravity_scale`/`inertia_scale` conventions.
6. Metrics, speed strata, delay scan, and residual correlations; then plots and
   the Markdown report.

## Outputs

Analysis writes into the session (or the given `--outdir`):

- `timeseries.csv` — filtered `q, dq, ddq`, measured/model/scaled torque, and
  the G/C/I/friction terms per joint.
- `metrics.json` — aggregate and per-joint metrics, speed strata, delay scan,
  residual correlations, component RMS, and the diagnosis list.
- `figures/*.svg` — 33 stdlib-generated SVG figures: per-joint torque and
  residual traces, RMSE/bias bars, parity, delay scan, and residual histogram.
- `report.md` — Chinese Markdown report with the metric tables, method notes,
  diagnosis, figure index, and limitations.
- A multi-scenario run directory also gets `summary.md` comparing all scenarios.

## Diagnosis

The report lists error sources ranked by evidence, not as a verdict. Typical
signals:

| Signal | Evidence | First things to check |
|---|---|---|
| Model matches measurement | Mean relative RMSE < 2 % | Collection and RNEA chain; on synthetic data this is the expected result |
| Timing / filter delay | Best time shift lowers RMSE notably | `cutoff_ddq` too low, or current-loop reporting delay |
| Friction / low-speed reversal | Residual correlates with the friction term | Stribeck parameters; look at hold/slow segments and direction changes |
| Gravity model / current-loop zero bias | Static residual with `ddq = dq = 0` | Real gravity vector (`--gravity`), `get_ff_scalar(6)`, current-loop calibration |
| Inertia / acceleration term | Residual correlates with the inertia term | `jm`/inertia scale; compare sine `ddq` amplitude against residual sign |
| No single dominant term | Mixed errors | Per-joint `metrics.json` and residual distributions |

Each item comes with a concrete suggestion in `report.md`.

## Reference example

`examples/make_synthetic.py` synthesizes a deterministic session from the
factory model itself, so the residual is pure numerical noise. It exercises the
whole collect → analyze → plot → report pipeline without hardware:

```bash
python3 examples/make_synthetic.py --outdir runs/synthetic_sine
```

The committed reference output is in
[`examples/reference_sine/`](examples/reference_sine/) (report:
[report.md](examples/reference_sine/report.md)). Ordinary run output goes to the
git-ignored `runs/` directory. The committed reference is a deliberate exception
kept small and deterministic; regenerate it instead of analyzing it in place,
because analysis rewrites files in the target directory.

## Safety

- Commands are read-only by default: they print the plan and current pose. Any
  enable, motion, or collection requires `--go`.
- Every planned target is checked against soft limits with a 0.15 rad margin
  before execution; the amplitude is reduced automatically, and planning fails
  if no safe amplitude remains.
- Collection deliberately does **not** disable the arm afterward (disabling at
  the pose can make it sag under its own weight). Keep the emergency stop at
  hand.
- If the arm reports a fault bit, collection refuses to run.

## Limitations

- `tau` is the **current-loop estimate**, not a joint torque sensor. Current-loop
  calibration error goes straight into the conclusions.
- The firmware model is a regression regressor: link masses are often 1,
  inertia entries can be negative, and `M` is not positive definite. It is used
  for forward evaluation only — never for `M⁻¹` (simulation or control).
- The static gravity check is only meaningful while the arm is still; moving
  segments are not treated as static.
- Ambient temperature is not available from the firmware; record it manually
  with `--ambient-c` if it matters.
- Conclusions apply to the tested configuration and speed range. Extending to
  limit-adjacent or high-speed regimes requires new data.

## Tests

```bash
pip install -e ".[test]"
pytest
```

Tests are offline: no hardware, and no NumPy/Matplotlib (plots are pure SVG).
When running from a source checkout without installing, use
`PYTHONPATH=src pytest` with pytest 7 or newer.

## Documentation

- [Documentation index](docs/index.md)
- [Setup](docs/setup.md)
- [Architecture](docs/architecture.md)

## License

MIT
