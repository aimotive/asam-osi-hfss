"""Field-by-field copy between `antenna.element` and `osi3.AntennaModel`.

Both use the same names, units and conventions, so no conversion happens here.
"""

import numpy as np

from antenna.element import (
    AntennaElement,
    AntennaElementType,
    AntennaModel,
    ComplexGrid,
    Direction,
    FrequencyInstance,
    JonesPattern,
    Mode,
    Orientation3d,
    PolarizationBasis,
    Vector3d,
)
from osi3 import osi_antenna_pb2, osi_sensorviewconfiguration_pb2

GRIDS = ("pattern_00", "pattern_11", "pattern_01", "pattern_10")


def to_proto(model: AntennaModel) -> osi_antenna_pb2.AntennaModel:
    msg = osi_antenna_pb2.AntennaModel(
        polarization_basis=model.polarization_basis.value
    )
    msg.id.value = model.id
    for element_type in model.element_type:
        msg_type = msg.element_type.add(mode=element_type.mode.value)
        msg_type.id.value = element_type.id
        for f in element_type.frequency_instance:
            msg_f = msg_type.frequency_instance.add(
                frequency=f.frequency, gain=f.gain, phase=f.phase
            )
            msg_f.radiation.horizontal_angle.extend(f.radiation.horizontal_angle)
            msg_f.radiation.vertical_angle.extend(f.radiation.vertical_angle)
            for name in GRIDS:
                grid = getattr(f.radiation, name)
                getattr(msg_f.radiation, name).amplitude.extend(grid.amplitude.ravel())
                getattr(msg_f.radiation, name).phase.extend(grid.phase.ravel())
    for e in model.element:
        msg_e = msg.element.add(index=e.index)
        msg_e.element_type_id.value = e.element_type_id
        msg_e.position.x, msg_e.position.y, msg_e.position.z = (
            e.position.x,
            e.position.y,
            e.position.z,
        )
        o = msg_e.orientation
        o.roll, o.pitch, o.yaw = (
            e.orientation.roll,
            e.orientation.pitch,
            e.orientation.yaw,
        )
    return msg


def _jones_pattern(msg: osi_antenna_pb2.JonesPattern) -> JonesPattern:
    shape = (len(msg.horizontal_angle), len(msg.vertical_angle))
    grids = {
        name: ComplexGrid(
            amplitude=np.reshape(getattr(msg, name).amplitude, shape),
            phase=np.reshape(getattr(msg, name).phase, shape),
        )
        for name in GRIDS
    }
    return JonesPattern(msg.horizontal_angle, msg.vertical_angle, **grids)


def from_proto(msg: osi_antenna_pb2.AntennaModel) -> AntennaModel:
    return AntennaModel(
        id=msg.id.value,
        polarization_basis=PolarizationBasis(msg.polarization_basis),
        element_type=[
            AntennaElementType(
                id=t.id.value,
                mode=Mode(t.mode),
                frequency_instance=[
                    FrequencyInstance(
                        frequency=f.frequency,
                        gain=f.gain,
                        phase=f.phase,
                        radiation=_jones_pattern(f.radiation),
                    )
                    for f in t.frequency_instance
                ],
            )
            for t in msg.element_type
        ],
        element=[
            AntennaElement(
                index=e.index,
                element_type_id=e.element_type_id.value,
                position=Vector3d(e.position.x, e.position.y, e.position.z),
                orientation=Orientation3d(
                    e.orientation.roll, e.orientation.pitch, e.orientation.yaw
                ),
            )
            for e in msg.element
        ],
    )


def to_legacy_diagram(
    model: AntennaModel,
    element: AntennaElement,
    frequency: float,
    horizontal_angles: np.ndarray,
    vertical_angles: np.ndarray,
    floor_db: float = -60.0,
) -> list[
    osi_sensorviewconfiguration_pb2.RadarSensorViewConfiguration.AntennaDiagramEntry
]:
    """Derive the scalar OSI antenna diagram (co-polar response in dB)."""
    Entry = (
        osi_sensorviewconfiguration_pb2.RadarSensorViewConfiguration.AntennaDiagramEntry
    )
    floor = 10.0 ** (floor_db / 20.0)
    entries = []
    for h in horizontal_angles:
        for v in vertical_angles:
            co_polar = model.jones(element, frequency, Direction(h, v))[0, 0]
            response = 20.0 * np.log10(max(abs(co_polar), floor))
            entries.append(
                Entry(
                    horizontal_angle=float(h),
                    vertical_angle=float(v),
                    response=response,
                )
            )
    return entries
