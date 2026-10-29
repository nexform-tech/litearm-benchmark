# Setup

This guide installs and verifies `litearm-benchmark` on a development machine.

## Prerequisites

- Python 3.9 or newer
- Git
- A `litearm-python` SDK environment when collecting data from hardware

## 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
```

## 2. Build

The project is a pure-Python package; `pip install -e` is the build step.

## 3. Verify

```bash
pytest
```

Expected result: all tests pass.

## Troubleshooting

**`pytest` is not found**

Activate the virtual environment first, or install the test extra with
`pip install -e ".[test]"`.

**The package imports from a stale location**

Reinstall in editable mode after changing `pyproject.toml`.
