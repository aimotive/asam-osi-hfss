from dataclasses import dataclass
from typing import Literal

import numpy as np

from .element import AntennaElementInstance, Direction

Axis = Literal["x", "y", "z"]
Taper = Literal["uniform", "hann"]


@dataclass(frozen=True)
class BeamScanResult:
    scan_angles_deg: np.ndarray
    power_linear: np.ndarray
    power_db: np.ndarray
    peak_angle_deg: float


class ULA:
    def __init__(
        self,
        elements: list[AntennaElementInstance],
        frequency: float,
        axis: Axis = "x",
    ):
        self.elements = elements
        self.frequency = float(frequency)
        self.wavelength = 3e8 / self.frequency
        self.axis = axis
        self.k = 2.0 * np.pi / self.wavelength
        self.positions = np.array(
            [self._axis_position(el) for el in elements], dtype=float
        )

    def _axis_position(self, element: AntennaElementInstance) -> float:
        p = element.position
        if self.axis == "x":
            return p.x
        if self.axis == "y":
            return p.y
        return p.z

    def _spatial_phase(self, angle_deg: float) -> np.ndarray:
        theta = np.deg2rad(angle_deg)
        return np.exp(1j * self.k * self.positions * np.sin(theta))

    def element_response_vector(
        self, azimuth_deg: float, elevation_deg: float = 0.0
    ) -> np.ndarray:
        direction = Direction(
            azimuth=float(azimuth_deg), elevation=float(elevation_deg)
        )
        elem = np.zeros(len(self.elements), dtype=np.complex128)

        for i, el in enumerate(self.elements):
            s = el.sample(frequency=self.frequency, direction=direction)
            elem[i] = s.amplitude * np.exp(1j * s.phase)

        return elem * self._spatial_phase(azimuth_deg)

    def _taper(self, kind: Taper) -> np.ndarray:
        n = len(self.elements)
        if kind == "uniform":
            return np.ones(n, dtype=float)
        if kind == "hann":
            return np.hanning(n).astype(float)
        raise ValueError("Unsupported taper")

    def steering_weights(
        self,
        steer_angle_deg: float,
        taper: Taper = "uniform",
        elevation_deg: float = 0.0,
    ) -> np.ndarray:
        a0 = self.element_response_vector(steer_angle_deg, elevation_deg)
        w = a0 * self._taper(taper)
        norm = np.linalg.norm(w)
        if norm < 1e-15:
            raise ValueError("Degenerate steering vector")
        return w / norm

    def beam_scan(
        self,
        steer_angle_deg: float,
        scan_angles_deg: np.ndarray,
        taper: Taper = "uniform",
        elevation_deg: float = 0.0,
    ) -> BeamScanResult:
        w = self.steering_weights(
            steer_angle_deg=steer_angle_deg,
            taper=taper,
            elevation_deg=elevation_deg,
        )

        power = np.zeros_like(scan_angles_deg, dtype=float)
        for i, ang in enumerate(scan_angles_deg):
            a = self.element_response_vector(float(ang), elevation_deg)
            y = np.vdot(w, a)
            power[i] = np.abs(y) ** 2

        power /= np.max(power) + 1e-15
        power_db = 10.0 * np.log10(np.maximum(power, 1e-12))
        peak_idx = int(np.argmax(power))

        return BeamScanResult(
            scan_angles_deg=scan_angles_deg,
            power_linear=power,
            power_db=power_db,
            peak_angle_deg=float(scan_angles_deg[peak_idx]),
        )
