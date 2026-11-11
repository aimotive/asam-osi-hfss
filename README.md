# High-fidelity sensor model for ASAM OSI (draft proposal)

**Status:** draft proposal for discussion in the ASAM OSI project, not an ASAM-endorsed extension.

Today, ASAM OSI describes a radar antenna only as a simple power diagram in dB. That is not enough to simulate
modern radars, which use many antennas (MIMO), estimate the direction of a target from phase differences
between antennas, and react to polarization. For this reason a new interface is introduced to describe signal level data.

This repository proposes a richer antenna model for OSI: complex (amplitude and phase), polarimetric, and with
the full geometry of the antenna array. It also proposes two related interfaces of a radar simulation: the
propagation paths from the environment simulation, and the raw radar data handed from the signal generator to
the signal processing (DSP).

## What is in here

- **Example notebooks** in the `notebooks` folder. They show what the interfaces carry, both for today's OSI and
  for the proposal. They are committed with their outputs, so you can read them on GitHub without running
  anything.
- **The simulation modules** in the `bin` folder: an environment simulation, an antenna model, two radar signal
  generators, a radar DSP and a few sensor models. They are provided as ready-built programs only; their
  source is not part of this repository.
- **Example input data** in the `data` folder: scenes, radar configurations and antenna descriptions.
- **A small Python package** (`osi_hfss` in the `src` folder) that loads the data, runs the modules and draws
  the figures of the notebooks.

## Where to start

Open the notebook `notebooks/osi/03_osi_hf_radar_simulation.ipynb`. It runs a complete radar simulation for two
very different example radars and compares the detected objects with where they really are.

The simulation is a chain of modules. Each module only receives OSI messages and only produces OSI messages:

1. The environment simulation turns a scene into propagation paths, without any antenna applied.
2. The antenna module describes the radar's antenna array.
3. A signal generator combines both into the raw radar data, as a radar chip would deliver it.
4. The DSP turns the raw data into detections: distance, speed and direction of each object.

The other notebooks look at one step each:

- `notebooks/radar`: the propagation paths, the two signal generators (FMCW and PMCW) and the DSP.
- `notebooks/antenna`: how an antenna model is built, and the antenna arrays of the example radars.
- `notebooks/osi`: the radar chain with today's OSI and its limits, and the antenna proposal step by step.

## The modules

Every module is a program in the `bin` folder, named after what it does, for example `environment-simulation`,
`fmcw-signal-generator` or `radar-dsp`. A module reads its inputs from files (OSI messages, plus small JSON files
for settings that OSI does not cover) and writes its output to a file. `--help` shows the options of a module:

```sh
bin/radar-dsp --help
```

From Python, the `osi_hfss.modules` package runs the modules directly on OSI messages, which is what the
notebooks do. Because the modules only see their inputs, changing an input changes the result. The main
notebook shows this by removing a car from the scene and running the chain again.

Each module can be replaced by another implementation, for example one from a different supplier, as long as
it has the same name, the same command line options, and reads and writes the same messages.

The programs are built for Linux on x86-64. On other systems, use the development container described below.

## The example data

The `data` folder contains only the inputs of the simulation.

The file `data/pipeline.json` lists, for every produced file, which module creates it and from which inputs.
To experiment, change an input (for example the radar cross-section of an object in a scene file, or the number
of chirps in a signal generator's settings) and run the notebook again: everything that depends on it is
rebuilt automatically.

The `tests` folder holds the final detections of every example as a reference. The tests check that the
modules still produce them.

## Setup

### With the development container (any platform)

The easiest way is VS Code with the Dev Containers extension and Docker. Open this repository in VS Code and
choose **Reopen in Container**. The container comes with everything needed; on first start it installs the
Python environment, prepares the example data and runs the tests. Then open any notebook.

### Without a container (Linux x86-64)

You need [uv](https://docs.astral.sh/uv/) and Python 3.12, on a Linux as recent as Ubuntu 24.04.

```sh
git clone --recursive https://github.com/aimotive/asam-osi-hfss.git
cd asam-osi-hfss
uv sync
uv run python scripts/build_proto.py
uv run pytest
```

The second-to-last command generates the Python code for the OSI messages. If you cloned without
`--recursive`, run `git submodule update --init` first.

## About the OSI version

The submodule contains OSI v3.8.0 with the proposed extensions added.
