"""Minimal deterministic "environment simulation" for the OSI notebooks.

It is not a ray tracer: every object is treated as a single point scatterer at
its bounding box centre, which yields one `RadarSensorView.Reflection` per
object. The host vehicle reference point is `host.base.position`, the sensor
is placed relative to it by `mounting_position` (yaw only).
"""

from dataclasses import dataclass

import numpy as np

from antenna.element import SPEED_OF_LIGHT
from osi3 import (
    osi_groundtruth_pb2,
    osi_object_pb2,
    osi_sensorview_pb2,
    osi_sensorviewconfiguration_pb2,
    osi_version_pb2,
)

RadarSensorViewConfiguration = (
    osi_sensorviewconfiguration_pb2.RadarSensorViewConfiguration
)
Reflection = osi_sensorview_pb2.RadarSensorView.Reflection

HOST_ID = 0


@dataclass(frozen=True)
class Target:
    """A scene object together with radar properties not contained in OSI."""

    id: int
    position: tuple[float, float, float]
    velocity: tuple[float, float, float]
    dimension: tuple[float, float, float]
    rcs_dbsm: float
    # Normalised 2x2 scattering matrix S[received, transmitted] in HV basis.
    scattering: np.ndarray


PLATE = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)


def dihedral(rotation_rad: float) -> np.ndarray:
    """Scattering matrix of a dihedral rotated around the line of sight."""
    c, s = np.cos(2 * rotation_rad), np.sin(2 * rotation_rad)
    return np.array([[c, s], [s, -c]], dtype=complex)


DEFAULT_TARGETS = [
    Target(1, (33.0, -15.0, 0.5), (15.0, 0.0, 0.0), (4.5, 1.8, 1.5), 10.0, PLATE),
    Target(2, (60.0, 5.5, 0.5), (25.0, 0.0, 0.0), (4.8, 1.9, 1.6), 12.0, PLATE),
    Target(3, (22.0, 12.0, 0.5), (0.0, 0.0, 0.0), (4.2, 1.8, 1.5), 8.0, PLATE),
    # Pedestrian outside the field of view.
    Target(4, (4.0, 8.0, 0.5), (0.0, -1.0, 0.0), (0.5, 0.5, 1.8), -5.0, PLATE),
]


def interface_version():
    return osi_version_pb2.DESCRIPTOR.GetOptions().Extensions[
        osi_version_pb2.current_interface_version
    ]


def make_ground_truth(
    targets: list[Target] = DEFAULT_TARGETS,
    host_speed: float = 20.0,
    timestamp: float = 0.0,
) -> osi_groundtruth_pb2.GroundTruth:
    ground_truth = osi_groundtruth_pb2.GroundTruth()
    ground_truth.version.CopyFrom(interface_version())
    ground_truth.timestamp.seconds = int(timestamp)
    ground_truth.timestamp.nanos = int((timestamp % 1) * 1e9)
    ground_truth.host_vehicle_id.value = HOST_ID

    host = ground_truth.moving_object.add()
    host.id.value = HOST_ID
    host.type = osi_object_pb2.MovingObject.TYPE_VEHICLE
    host.base.position.x, host.base.position.y, host.base.position.z = 0.0, 0.0, 0.0
    host.base.velocity.x = host_speed
    (
        host.base.dimension.length,
        host.base.dimension.width,
        host.base.dimension.height,
    ) = (
        4.6,
        1.9,
        1.5,
    )

    for target in targets:
        obj = ground_truth.moving_object.add()
        obj.id.value = target.id
        obj.type = osi_object_pb2.MovingObject.TYPE_VEHICLE
        obj.base.position.x, obj.base.position.y, obj.base.position.z = target.position
        obj.base.velocity.x, obj.base.velocity.y, obj.base.velocity.z = target.velocity
        (
            obj.base.dimension.length,
            obj.base.dimension.width,
            obj.base.dimension.height,
        ) = target.dimension
    return ground_truth


@dataclass(frozen=True)
class TargetGeometry:
    id: int
    range: float  # m
    azimuth: float  # rad, OSI horizontal angle
    elevation: float  # rad, OSI vertical angle
    range_rate: float  # m/s, positive = moving away


def sensor_frame_geometry(
    config: RadarSensorViewConfiguration,
    ground_truth: osi_groundtruth_pb2.GroundTruth,
) -> list[TargetGeometry]:
    """Range, angles and range rate of every non-host object w.r.t. the sensor."""
    objects = {o.id.value: o for o in ground_truth.moving_object}
    host = objects[ground_truth.host_vehicle_id.value]

    host_yaw = host.base.orientation.yaw
    sensor_yaw = host_yaw + config.mounting_position.orientation.yaw

    def rot(yaw: float) -> np.ndarray:
        c, s = np.cos(yaw), np.sin(yaw)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def vec(v) -> np.ndarray:
        return np.array([v.x, v.y, v.z])

    mount = vec(config.mounting_position.position)
    sensor_position = vec(host.base.position) + rot(host_yaw) @ mount
    sensor_velocity = vec(host.base.velocity)
    to_sensor = rot(sensor_yaw).T

    geometries = []
    for obj_id, obj in objects.items():
        if obj_id == host.id.value:
            continue
        relative_position = to_sensor @ (vec(obj.base.position) - sensor_position)
        relative_velocity = to_sensor @ (vec(obj.base.velocity) - sensor_velocity)
        distance = float(np.linalg.norm(relative_position))
        line_of_sight = relative_position / distance
        geometries.append(
            TargetGeometry(
                id=obj_id,
                range=distance,
                azimuth=float(np.arctan2(relative_position[1], relative_position[0])),
                elevation=float(np.arcsin(line_of_sight[2])),
                range_rate=float(relative_velocity @ line_of_sight),
            )
        )
    return geometries


def in_field_of_view(config: RadarSensorViewConfiguration, g: TargetGeometry) -> bool:
    return (
        abs(g.azimuth) <= config.field_of_view_horizontal / 2
        and abs(g.elevation) <= config.field_of_view_vertical / 2
    )


def path_gain_db(frequency: float, rcs_dbsm: float, distance: float) -> float:
    """Antenna-free monostatic radar equation: lambda^2 sigma / ((4 pi)^3 R^4)."""
    wavelength = SPEED_OF_LIGHT / frequency
    rcs = 10.0 ** (rcs_dbsm / 10.0)
    return float(
        10.0 * np.log10(wavelength**2 * rcs / ((4 * np.pi) ** 3 * distance**4))
    )


def diagram_lookup(
    diagram: list[RadarSensorViewConfiguration.AntennaDiagramEntry],
    horizontal: float,
    vertical: float,
) -> float:
    """Nearest-neighbour lookup of an OSI antenna diagram, returns dB."""
    angles = np.array([[e.horizontal_angle, e.vertical_angle] for e in diagram])
    idx = int(np.argmin(np.sum((angles - [horizontal, vertical]) ** 2, axis=1)))
    return diagram[idx].response


def _kinematics(reflection: Reflection, g: TargetGeometry, frequency: float) -> None:
    reflection.time_of_flight = 2.0 * g.range / SPEED_OF_LIGHT
    reflection.doppler_shift = -2.0 * g.range_rate * frequency / SPEED_OF_LIGHT
    reflection.source_horizontal_angle = g.azimuth
    reflection.source_vertical_angle = g.elevation


def legacy_reflections(
    config: RadarSensorViewConfiguration,
    ground_truth: osi_groundtruth_pb2.GroundTruth,
    targets: list[Target] = DEFAULT_TARGETS,
) -> list[Reflection]:
    """Upstream OSI semantics: antenna diagrams folded into signal_strength."""
    rcs = {t.id: t.rcs_dbsm for t in targets}
    reflections = []
    for g in sensor_frame_geometry(config, ground_truth):
        if not in_field_of_view(config, g):
            continue
        reflection = Reflection()
        _kinematics(reflection, g, config.emitter_frequency)
        reflection.signal_strength = (
            diagram_lookup(config.tx_antenna_diagram, g.azimuth, g.elevation)
            + diagram_lookup(config.rx_antenna_diagram, g.azimuth, g.elevation)
            + path_gain_db(config.emitter_frequency, rcs[g.id], g.range)
        )
        reflections.append(reflection)
    return reflections


def _fill_jones(matrix, values: np.ndarray) -> None:
    for name, value in zip(("m00", "m01", "m10", "m11"), values.ravel()):
        getattr(matrix, name).amplitude = float(np.abs(value))
        getattr(matrix, name).phase = float(np.angle(value))


def hfss_reflections(
    config: RadarSensorViewConfiguration,
    ground_truth: osi_groundtruth_pb2.GroundTruth,
    targets: list[Target] = DEFAULT_TARGETS,
) -> list[Reflection]:
    """Proposed semantics: antenna-free polarimetric channel per reflection.

    Monostatic and single bounce, so arrival and departure angles coincide;
    they are still filled separately as required by the proposal.
    """
    by_id = {t.id: t for t in targets}
    reflections = []
    for g in sensor_frame_geometry(config, ground_truth):
        if not in_field_of_view(config, g):
            continue
        target = by_id[g.id]
        reflection = Reflection()
        _kinematics(reflection, g, config.emitter_frequency)
        reflection.arrival_horizontal_angle = g.azimuth
        reflection.arrival_vertical_angle = g.elevation
        gain_db = path_gain_db(config.emitter_frequency, target.rcs_dbsm, g.range)
        reflection.signal_strength = gain_db
        _fill_jones(
            reflection.polarimetric_response,
            10.0 ** (gain_db / 20.0) * target.scattering,
        )
        reflections.append(reflection)
    return reflections


def jones_from_osi(matrix) -> np.ndarray:
    """`osi3.JonesMatrix` -> complex 2x2 numpy array."""
    return np.array(
        [
            getattr(matrix, n).amplitude * np.exp(1j * getattr(matrix, n).phase)
            for n in ("m00", "m01", "m10", "m11")
        ]
    ).reshape(2, 2)
