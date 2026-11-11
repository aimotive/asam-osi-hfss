import io
import json
import os
import platform
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from osi3.osi_antenna_pb2 import AntennaModel
from osi3.osi_groundtruth_pb2 import GroundTruth
from osi3.osi_radarcpi_pb2 import RadarCpiData
from osi3.osi_sensordata_pb2 import SensorData
from osi3.osi_sensorview_pb2 import SensorView
from osi3.osi_sensorviewconfiguration_pb2 import SensorViewConfiguration

ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT / "bin"
MODULES = ("environment-simulation", "antenna-model", "sensor-view-configuration", "fmcw-signal-generator", "pmcw-signal-generator",
           "radar-dsp", "legacy-sensor-model", "doa-sensor-model", "mimo-doa-sensor-model", "beam-scan")


def bin_dir() -> Path:
    return Path(os.environ.get("OSI_HFSS_BIN", BIN))


def executable(module: str) -> Path:
    if module not in MODULES:
        raise ValueError(f"unknown module {module!r}, one of {MODULES}")
    return bin_dir() / module


def available() -> bool:
    if "OSI_HFSS_BIN" not in os.environ and (platform.system(), platform.machine()) != ("Linux", "x86_64"):
        return False
    return all(os.access(executable(m), os.X_OK) for m in MODULES)


def run(module: str, *args: str, inputs: dict | None = None, outputs: tuple[str, ...] = ()) -> tuple[bytes, dict]:
    path = executable(module)
    if not available():
        raise FileNotFoundError(f"{path} cannot run here: the shipped modules need Linux x86-64, use the devcontainer")
    with tempfile.TemporaryDirectory() as tmp:
        argv = [str(path), *args]
        for name, value in (inputs or {}).items():
            if value is None:
                continue
            file = Path(tmp) / f"{name}.input"
            if isinstance(value, dict):
                file.write_text(json.dumps(value))
            else:
                file.write_bytes(value.SerializeToString())
            argv += [f"--{name}", str(file)]
        for name in outputs:
            argv += [f"--{name}", str(Path(tmp) / f"{name}.out")]
        result = subprocess.run(argv, capture_output=True, cwd=tmp, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"{module} failed:\n{result.stderr.decode()}")
        return result.stdout, {name: (Path(tmp) / f"{name}.out").read_bytes() for name in outputs}


def _npz(content: bytes) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(content)) as arrays:
        return dict(arrays)


def environment_simulation(ground_truth: GroundTruth, configuration: SensorViewConfiguration, scene: dict) -> SensorView:
    inputs = {"ground-truth": ground_truth, "configuration": configuration, "scene": scene}
    return SensorView.FromString(run("environment-simulation", inputs=inputs)[0])


def antenna_model(config: dict) -> AntennaModel:
    return AntennaModel.FromString(run("antenna-model", inputs={"config": config})[0])


def sensor_view_configuration(base: SensorViewConfiguration, antenna: AntennaModel) -> SensorViewConfiguration:
    return SensorViewConfiguration.FromString(
        run("sensor-view-configuration", inputs={"base": base, "antenna": antenna})[0])


def _generator(module: str, sensor_view, antenna, config, raw: bool, noise_free: bool, encoding: str,
               outputs: tuple[str, ...] = ()) -> tuple[RadarCpiData, dict]:
    flags = (["--raw"] if raw else []) + (["--noise-free"] if noise_free else []) + ["--encoding", encoding]
    inputs = {"sensor-view": sensor_view, "antenna": antenna, "config": config}
    stdout, files = run(module, *flags, inputs=inputs, outputs=outputs)
    return RadarCpiData.FromString(stdout), {name: _npz(content) for name, content in files.items()}


def fmcw_signal_generator(sensor_view: SensorView, antenna: AntennaModel, config: dict | None = None,
                          raw: bool = False, noise_free: bool = False, encoding: str = "float32") -> RadarCpiData:
    return _generator("fmcw-signal-generator", sensor_view, antenna, config, raw, noise_free, encoding)[0]


def pmcw_signal_generator(sensor_view: SensorView, antenna: AntennaModel, config: dict | None = None,
                          raw: bool = False, noise_free: bool = False, encoding: str = "float32") -> RadarCpiData:
    return _generator("pmcw-signal-generator", sensor_view, antenna, config, raw, noise_free, encoding)[0]


def pmcw_signal_generator_with_analysis(sensor_view: SensorView, antenna: AntennaModel, config: dict | None = None,
                                        raw: bool = False) -> tuple[RadarCpiData, dict]:
    cpi, files = _generator("pmcw-signal-generator", sensor_view, antenna, config, raw, False, "float32", ("analysis",))
    return cpi, files["analysis"]


def radar_dsp(cpi: RadarCpiData, antenna: AntennaModel | None = None) -> SensorData:
    return radar_dsp_with_analysis(cpi, antenna)[0]


def radar_dsp_with_analysis(cpi: RadarCpiData, antenna: AntennaModel | None = None) -> tuple[SensorData, dict]:
    inputs = {"cpi": cpi} | ({"antenna": antenna} if antenna is not None else {})
    stdout, files = run("radar-dsp", inputs=inputs, outputs=("analysis",))
    return SensorData.FromString(stdout), _npz(files["analysis"])


def legacy_sensor_model(sensor_view: SensorView) -> SensorData:
    return SensorData.FromString(run("legacy-sensor-model", inputs={"sensor-view": sensor_view})[0])


def doa_sensor_model(sensor_view: SensorView, snr_threshold_db: float | None = None) -> tuple[SensorData, dict]:
    args = [] if snr_threshold_db is None else ["--snr-threshold", repr(float(snr_threshold_db))]
    stdout, files = run("doa-sensor-model", *args, inputs={"sensor-view": sensor_view}, outputs=("analysis",))
    return SensorData.FromString(stdout), _npz(files["analysis"])


def mimo_doa_sensor_model(sensor_view: SensorView, transmitters: int | None = None) -> tuple[SensorData, dict]:
    args = [] if transmitters is None else ["--transmitters", str(transmitters)]
    stdout, files = run("mimo-doa-sensor-model", *args, inputs={"sensor-view": sensor_view}, outputs=("analysis",))
    return SensorData.FromString(stdout), _npz(files["analysis"])


def beam_scan(antenna: AntennaModel, steer_deg: float, taper: str = "uniform", frequency: float | None = None,
              mode: str = "receive") -> dict[str, np.ndarray]:
    args = ["--steer", str(steer_deg), "--taper", taper, "--mode", mode]
    if frequency is not None:
        args += ["--frequency", repr(float(frequency))]
    return _npz(run("beam-scan", *args, inputs={"antenna": antenna})[0])
