# High-fidelity antenna model as an ASAM OSI extension (draft)

## Motivation

OSI v3.8.0 describes the radar antenna only through
`RadarSensorViewConfiguration.tx_antenna_diagram` / `rx_antenna_diagram`:
a scalar power response in dB per (horizontal, vertical) angle. The
environment simulation folds these diagrams into
`RadarSensorView.Reflection.signal_strength`.

This is not sufficient for high-fidelity radar simulation, which needs

- complex (amplitude and phase) patterns per channel for MIMO, beamforming and
  direction-of-arrival estimation,
- polarisation (co- and cross-polar Jones matrix) of antenna and target,
- the positions of all Tx and Rx elements,
- an antenna-free propagation channel, so the sensor model can apply its own
  antenna,
- separate departure (Tx) and arrival (Rx) directions.

## Proposal

Class diagram: [`osi_antenna_extension.puml`](osi_antenna_extension.puml).

| message | new field | number | purpose |
|---|---|---|---|
| `RadarSensorViewConfiguration` | `antenna_model : AntennaModel` | 12 | detailed Tx/Rx antenna |
| `RadarSensorViewConfiguration` | `antenna_application : AntennaApplication` | 13 | who applies the antenna |
| `RadarSensorView.Reflection` | `arrival_horizontal_angle`, `arrival_vertical_angle` | 6, 7 | direction at the Rx antenna |
| `RadarSensorView.Reflection` | `polarimetric_response : JonesMatrix` | 8 | antenna-free 2×2 channel |

Upstream documents `source_horizontal_angle` / `source_vertical_angle` as the
direction *at the Tx antenna*; the proposal keeps that meaning (departure) and
adds the arrival direction.

`AntennaApplication`:

- `UNKNOWN` / `ENVIRONMENT` — upstream behaviour, `signal_strength` includes
  the antenna diagrams.
- `SENSOR_MODEL` — `signal_strength` is the antenna-free path gain, the
  reflection carries `polarimetric_response` and both directions, and the
  sensor model applies `antenna_model`.

## Mapping to `src/antenna`

`src/antenna/element.py` mirrors `osi_antenna.proto` one to one: same class
and field names, same enum values, same units (rad, dB, Hz, m) and the same
angle convention (horizontal / vertical angle 0 = boresight).

| `antenna.element` | `osi3` |
|---|---|
| `AntennaModel` | `AntennaModel` |
| `PolarizationBasis` | `AntennaModel.PolarizationBasis` |
| `AntennaElementType`, `Mode` | `AntennaElementType`, `AntennaElementType.Mode` |
| `FrequencyInstance` | `FrequencyInstance` |
| `JonesPattern` | `JonesPattern` |
| `ComplexGrid` (numpy arrays, shape `(n_h, n_v)`) | `ComplexGrid` (packed, row-major `[h][v]`) |
| `AntennaElement` | `AntennaElement` |
| `Vector3d`, `Orientation3d` | `Vector3d`, `Orientation3d` |
| `ComplexValue` | `ComplexValue` |

The Python classes add evaluation on top: `AntennaModel.jones`,
`AntennaModel.response`, `AntennaModel.receive`, `AntennaModel.elements(mode)` and
`AntennaModel.resampled(step)`.

[`src/osi_hfss/antenna_adapter.py`](../src/osi_hfss/antenna_adapter.py) is a
plain field copy (`to_proto`, `from_proto`); the only other function,
`to_legacy_diagram`, derives the upstream dB diagram.

## Compatibility

- Only new optional fields and new messages; old consumers ignore them
  (proto2 unknown fields).
- Producers should keep filling `tx_antenna_diagram` / `rx_antenna_diagram`,
  derived from `antenna_model` as the co-polar magnitude in dB
  (`to_legacy_diagram`).
- An environment simulation that does not support `SENSOR_MODEL` leaves
  `antenna_application` unset in the echoed `view_configuration`; the sensor
  model detects the fallback.

## Open questions

- Pattern size vs. grid resolution (2° grid over ±90° ≈ 0.5 MB per element type);
  external pattern references or harmonic expansions.
- Interpolation between grid points and between frequency instances.
- Additional per-ray phase beyond `time_of_flight` (currently carried by the
  phase of `polarimetric_response`).
- Rotated elements: `AntennaElement.orientation` is defined, but rotating the
  direction and the polarization basis is not implemented yet (the Python
  model rejects non-zero orientations).
- Transceiver elements (one element used for Tx and Rx) cannot be expressed
  with `AntennaElementType.Mode`.
- Near-field / bistatic Tx and Rx positions (arrival ≠ departure direction)
  for multi-bounce paths.

## Examples

- [`notebooks/osi/01_osi_radar_sensor_simulation.ipynb`](../notebooks/osi/01_osi_radar_sensor_simulation.ipynb) — upstream OSI radar chain.
- [`notebooks/osi/02_osi_hfss_antenna_extension.ipynb`](../notebooks/osi/02_osi_hfss_antenna_extension.ipynb) — proposed extension end-to-end (DOA from the antenna model, polarimetric target, 3x4 TDM-MIMO virtual array).
