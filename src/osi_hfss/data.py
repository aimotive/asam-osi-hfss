import json
from pathlib import Path

import numpy as np

from osi3.osi_antenna_pb2 import AntennaModel
from osi3.osi_groundtruth_pb2 import GroundTruth
from osi3.osi_radarcpi_pb2 import RadarCpiData
from osi3.osi_sensordata_pb2 import SensorData
from osi3.osi_sensorview_pb2 import SensorView
from osi3.osi_sensorviewconfiguration_pb2 import SensorViewConfiguration
from osi_hfss import pipeline

DATA_DIR = pipeline.DATA_DIR

MESSAGE_TYPES = {
    "antennamodel": AntennaModel,
    "groundtruth": GroundTruth,
    "radarcpidata": RadarCpiData,
    "sensordata": SensorData,
    "sensorview": SensorView,
    "sensorviewconfiguration": SensorViewConfiguration,
}


def read(path: Path):
    match path.suffix:
        case ".bin":
            return MESSAGE_TYPES[path.name.split(".")[-2]].FromString(path.read_bytes())
        case ".json":
            return json.loads(path.read_text())
        case ".npz":
            with np.load(path) as arrays:
                return dict(arrays)
    raise ValueError(f"unknown data file type {path.name}")


def load(name: str):
    return read(pipeline.path(name))
