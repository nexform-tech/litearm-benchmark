# litearm-benchmark

Benchmark and torque-estimation accuracy evaluation tools for the LiteArm robotic manipulator series.

## Scope

- Joint-level data acquisition at up to 100 Hz: position, velocity, and current-loop torque.
- Inverse-dynamics torque computation from the arm model.
- Per-joint error analysis and plots (RMSE / MAE / peak error).

The repository is being initialized. The first milestone is the current-loop torque
estimation accuracy tool tracked on the team delivery board.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
```

## Usage

```bash
pytest
```

The command-line entry point will be added with the first implementation.

## Documentation

- [Documentation index](docs/index.md)
- [Setup](docs/setup.md)
- [Architecture](docs/architecture.md)

## License

MIT
