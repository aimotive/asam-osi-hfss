# Antenna interface draft

Jones matrix based antenna model, and a draft proposal to carry it in the
ASAM OSI radar interface (see [docs/osi_antenna_extension.md](docs/osi_antenna_extension.md)).

## Setup

```sh
git submodule update --init          # OSI fork: v3.8.0 + antenna extension
uv sync
uv run python scripts/build_proto.py # generates src/osi3
```

The submodule `third_party/open-simulation-interface-aim` tracks the branch
`feat/antenna-model` of the fork `aimotive/open-simulation-interface`.
`scripts/build_proto.py --upstream-only` builds the plain upstream `v3.8.0`
bindings from the same submodule.

## Notebooks

- `notebooks/antenna/` — the antenna element model and a ULA application.
- `notebooks/osi/01_osi_radar_sensor_simulation.ipynb` — radar sensor
  simulation with the current OSI interface.
- `notebooks/osi/02_osi_hfss_antenna_extension.ipynb` — the proposed OSI
  antenna extension end-to-end.
