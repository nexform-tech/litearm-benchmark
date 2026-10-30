# Setup

This guide installs and verifies `litearm-benchmark` on a development machine.

## Prerequisites

- Python 3.9 or newer
- Git
- A LiteArm arm and USB connection only when collecting data from hardware

## 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate

# Offline analysis and tests.
pip install -e ".[test]"

# Hardware collection only: adds the optional litearm SDK.
pip install -e ".[sdk]"
```

The analysis, plotting, and reporting code uses the Python standard library only.
The SDK is imported lazily on the hardware path, so it is not needed offline.

## 2. Build

The project is a pure-Python package; `pip install -e` is the build step.

## 3. Verify

```bash
pytest
```

Expected result: 23 passed. The suite is offline and needs no hardware.

## 4. Verify the command-line entries

```bash
litearm-tau --help
python3 -m litearm_benchmark.torque_eval --help
```

Without the SDK, running a collection command exits with a clear message and a
non-zero status instead of a traceback. Offline analysis still works:

```bash
litearm-tau analyze examples/reference_sine --outdir /tmp/tau_ref
ls /tmp/tau_ref/reference_sine/figures | wc -l     # 33
```

## Troubleshooting

**`pytest` is not found**

Activate the virtual environment first, or install the test extra with
`pip install -e ".[test]"`.

**`pytest` reports `Unknown config option: pythonpath`**

The `pyproject.toml` `pythonpath` option needs pytest 7 or newer. Upgrade pytest,
or run from a source checkout with `PYTHONPATH=src pytest`.

**`litearm-tau` is not found**

Reinstall with `pip install -e ".[test]"` (or `.[sdk]` for hardware), or use
`python3 -m litearm_benchmark.torque_eval`.

**The package imports from a stale location**

Reinstall in editable mode after changing `pyproject.toml`.
