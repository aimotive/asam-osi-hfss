import re

import numpy as np

from osi3 import osi_version_pb2
from osi3.osi_antenna_pb2 import AntennaElementType
from osi3.osi_radarcpi_pb2 import RadarCpiFrame

SPEED_OF_LIGHT = 299_792_458.0

RECEIVE = AntennaElementType.MODE_RECEIVE
TRANSMIT = AntennaElementType.MODE_TRANSMIT

_DTYPES = {
    RadarCpiFrame.DATA_ENCODING_FLOAT32: "<f4",
    RadarCpiFrame.DATA_ENCODING_FLOAT64: "<f8",
    RadarCpiFrame.DATA_ENCODING_INT16: "<i2",
}
_GRIDS = ("pattern_00", "pattern_01", "pattern_10", "pattern_11")


def interface_version():
    return osi_version_pb2.DESCRIPTOR.GetOptions().Extensions[osi_version_pb2.current_interface_version]


def complex_value(value) -> complex:
    return value.amplitude * np.exp(1j * value.phase)


def jones(matrix) -> np.ndarray:
    return np.array([complex_value(getattr(matrix, n)) for n in ("m00", "m01", "m10", "m11")]).reshape(2, 2)


def axis_values(axis) -> np.ndarray:
    return axis.offset + axis.spacing * np.arange(axis.size)


def cpi_cube(frame) -> np.ndarray:
    scale = frame.scale_factor if frame.HasField("scale_factor") else 1.0
    values = np.frombuffer(frame.data, dtype=_DTYPES[frame.data_encoding]).astype(float) * scale
    shape = (len(frame.channel), frame.slow_axis.size, frame.fast_axis.size)
    if frame.sample_format == RadarCpiFrame.SAMPLE_FORMAT_COMPLEX_IQ:
        values = values.reshape(*shape, 2)
        return values[..., 0] + 1j * values[..., 1]
    return values.reshape(shape).astype(complex)


def enum_name(message, field: str) -> str:
    enum = message.DESCRIPTOR.fields_by_name[field].enum_type
    prefix = re.sub(r"(?<!^)(?=[A-Z])", "_", enum.name).upper() + "_"
    return enum.values_by_number[getattr(message, field)].name.removeprefix(prefix)


_REFLECTION_FIELDS = ("signal_strength", "time_of_flight", "doppler_shift", "source_horizontal_angle",
                      "source_vertical_angle", "arrival_horizontal_angle", "arrival_vertical_angle")


def reflection_arrays(reflections) -> dict[str, np.ndarray]:
    reflections = list(reflections)
    arrays = {name: np.array([getattr(r, name) for r in reflections], dtype=float) for name in _REFLECTION_FIELDS}
    arrays["polarimetric_response"] = np.array([jones(r.polarimetric_response) for r in reflections]).reshape(-1, 2, 2)
    arrays["number_of_interactions"] = np.array(
        [r.number_of_interactions if r.HasField("number_of_interactions") else 1 for r in reflections], dtype=int)
    arrays["object_id"] = np.array([r.object_id.value if r.HasField("object_id") else -1 for r in reflections], dtype=int)
    return arrays


def unit_vectors(horizontal, vertical) -> np.ndarray:
    horizontal, vertical = np.asarray(horizontal, dtype=float), np.asarray(vertical, dtype=float)
    cos_v = np.cos(vertical)
    return np.stack([cos_v * np.cos(horizontal), cos_v * np.sin(horizontal), np.sin(vertical)], axis=-1)


def element_type(antenna, element):
    return next(t for t in antenna.element_type if t.id.value == element.element_type_id.value)


def elements(antenna, mode) -> list:
    return [e for e in antenna.element if element_type(antenna, e).mode == mode]


def position(element) -> np.ndarray:
    return np.array([element.position.x, element.position.y, element.position.z])


def closest_frequency_instance(type_, frequency: float):
    return min(type_.frequency_instance, key=lambda f: abs(f.frequency - frequency))


def pattern_grid(pattern, name: str) -> np.ndarray:
    grid = getattr(pattern, name)
    shape = (len(pattern.horizontal_angle), len(pattern.vertical_angle))
    return (np.array(grid.amplitude) * np.exp(1j * np.array(grid.phase))).reshape(shape)


def element_jones(antenna, element, frequency: float, horizontal, vertical=0.0) -> np.ndarray:
    instance = closest_frequency_instance(element_type(antenna, element), frequency)
    pattern = instance.radiation
    h, v = np.array(pattern.horizontal_angle), np.array(pattern.vertical_angle)
    horizontal, vertical = np.broadcast_arrays(np.asarray(horizontal, dtype=float), np.asarray(vertical, dtype=float))
    h_idx = np.argmin(np.abs(h - horizontal[..., None]), axis=-1)
    v_idx = np.argmin(np.abs(v - vertical[..., None]), axis=-1)
    factor = 10 ** (instance.gain / 20) * np.exp(1j * instance.phase)
    values = [pattern_grid(pattern, name)[h_idx, v_idx] for name in _GRIDS]
    return factor * np.stack(values, axis=-1).reshape(*horizontal.shape, 2, 2)
