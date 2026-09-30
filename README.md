# High-fidelity antenna model for ASAM OSI (draft proposal)

**Status:** draft proposal for discussion in the ASAM OSI project, not an ASAM-endorsed extension.

ASAM OSI describes a radar antenna only as a scalar power diagram in dB (`tx_antenna_diagram` / `rx_antenna_diagram`). This repository proposes a complex, polarimetric, Jones matrix based antenna model with the full array geometry, carried in the OSI radar interface, so that sensor models can simulate MIMO, beamforming, direction-of-arrival estimation and polarimetry.

It contains

- the proposed OSI messages, as a branch of an OSI fork (submodule, see below),
- a Python antenna model mirroring those messages (`src/antenna`),
- example notebooks for the current OSI radar chain and for the proposal.

## Reading order

1. [docs/osi_antenna_extension.md](docs/osi_antenna_extension.md): motivation, proposed fields, conventions, compatibility, open questions, and the class diagram [docs/osi_antenna_extension.puml](docs/osi_antenna_extension.puml).
2. `notebooks/osi/01_osi_radar_sensor_simulation.ipynb`: radar sensor simulation with the current OSI interface and its limitations.
3. `notebooks/osi/02_osi_hfss_antenna_extension.ipynb`: the proposal end to end (DOA from the antenna model, polarimetric target, 3x4 TDM-MIMO).
4. `notebooks/antenna/`: the antenna element model and a ULA application.

The notebooks are committed with their outputs and can be read without running them.

## Setup

Prerequisites: [uv](https://docs.astral.sh/uv/) and Python 3.12.

```sh
git clone --recursive https://github.com/aimotive/asam-osi-hfss.git
cd asam-osi-hfss
uv sync
uv run python scripts/build_proto.py # generates the osi3 bindings in src/osi3
```

For an existing clone, run `git submodule update --init` instead of `--recursive`.

The submodule `third_party/open-simulation-interface-aim` tracks the branch `feat/antenna-model` of the fork `aimotive/open-simulation-interface` (OSI v3.8.0 plus the antenna extension). `scripts/build_proto.py --upstream-only` builds the plain upstream `v3.8.0` bindings from the same submodule.
