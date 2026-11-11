from dataclasses import dataclass

import numpy as np

from osi3.osi_antenna_pb2 import AntennaElementType, AntennaModel
from osi_hfss.decode import SPEED_OF_LIGHT


@dataclass(frozen=True)
class ExpectedPosition:
    object_id: int
    distance: float
    azimuth: float 
    radial_velocity: float


def expected_positions(rays) -> list[ExpectedPosition]:
    if hasattr(rays, "radar_sensor_view"):
        rays = rays.radar_sensor_view[0]
    frequency = rays.view_configuration.emitter_frequency
    return [
        ExpectedPosition(
            object_id=r.object_id.value,
            distance=SPEED_OF_LIGHT * r.time_of_flight / 2,
            azimuth=r.arrival_horizontal_angle,
            radial_velocity=r.doppler_shift * SPEED_OF_LIGHT / (2 * frequency),
        )
        for r in rays.reflection
        if r.number_of_interactions == 1
    ]


def closest(detection, expected: list[ExpectedPosition], tolerance=(1.0, 1.0, np.radians(5.0))) -> ExpectedPosition:
    def cost(e: ExpectedPosition) -> float:
        delta = (
            e.distance - detection.position.distance,
            e.radial_velocity - detection.radial_velocity,
            e.azimuth - detection.position.azimuth,
        )
        return sum((x / t) ** 2 for x, t in zip(delta, tolerance))

    return min(expected, key=cost)


def detection_table(sensor_data, expected: list[ExpectedPosition], names: dict[int, str] | None = None) -> str:
    lines = [f"{'object':>12} | {'detected: range [m]':>19} {'az [deg]':>8} {'v_r [m/s]':>9} {'SNR [dB]':>8} | "
             f"{'expected: range [m]':>19} {'az [deg]':>8} {'v_r [m/s]':>9}"]
    for d in sensor_data.feature_data.radar_sensor[0].detection:
        e = closest(d, expected)
        name = (names or {}).get(e.object_id, str(e.object_id))
        lines.append(
            f"{name:>12} | {d.position.distance:19.2f} {np.degrees(d.position.azimuth):8.2f} {d.radial_velocity:9.2f} "
            f"{d.snr:8.1f} | {e.distance:19.2f} {np.degrees(e.azimuth):8.2f} {e.radial_velocity:9.2f}"
        )
    return "\n".join(lines)


def sensor_to_world(mounting_position, distance, azimuth) -> tuple:
    a = mounting_position.orientation.yaw + np.asarray(azimuth)
    p = mounting_position.position
    return p.x + np.asarray(distance) * np.cos(a), p.y + np.asarray(distance) * np.sin(a)


def array_positions(antenna: AntennaModel) -> dict[str, np.ndarray]:
    modes = {t.id.value: t.mode for t in antenna.element_type}

    def positions(mode):
        return np.array([e.position.y for e in antenna.element if modes[e.element_type_id.value] == mode])

    tx, rx = positions(AntennaElementType.MODE_TRANSMIT), positions(AntennaElementType.MODE_RECEIVE)
    return {"tx": tx, "rx": rx, "virtual": np.add.outer(tx, rx).ravel()}


def diagram_grid(diagram) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h = np.unique([e.horizontal_angle for e in diagram])
    v = np.unique([e.vertical_angle for e in diagram])
    return h, v, np.array([e.response for e in diagram]).reshape(len(h), len(v))
