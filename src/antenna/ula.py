from dataclasses import dataclass
from typing import Literal

import numpy as np

from .element import AntennaModel, Direction, Mode, Radian

Taper = Literal["uniform", "hann"]


@dataclass(frozen=True)
class BeamScanResult:
    scan_angles: np.ndarray  # rad
    power_linear: np.ndarray
    power_db: np.ndarray
    peak_angle: Radian


class ULA:
    """Linear array formed by the elements of one mode of an `AntennaModel`."""

    def __init__(
        self, model: AntennaModel, frequency: float, mode: Mode = Mode.RECEIVE
    ):
        self.model = model
        self.elements = model.elements(mode)
        self.frequency = float(frequency)
        self.wavelength = 3e8 / self.frequency
        self.k = 2.0 * np.pi / self.wavelength
        self.positions = np.array([e.position.to_array() for e in self.elements])

    def _spatial_phase(self, direction: Direction) -> np.ndarray:
        return np.exp(1j * self.k * self.positions @ direction.unit_vector())

    def element_response_vector(
        self, horizontal: Radian, vertical: Radian = 0.0
    ) -> np.ndarray:
        direction = Direction(horizontal=float(horizontal), vertical=float(vertical))
        elem = np.array(
            [
                self.model.response(e, self.frequency, direction)[0]
                for e in self.elements
            ]
        )
        return elem * self._spatial_phase(direction)

    def _taper(self, kind: Taper) -> np.ndarray:
        n = len(self.elements)
        if kind == "uniform":
            return np.ones(n, dtype=float)
        if kind == "hann":
            return np.hanning(n).astype(float)
        raise ValueError("Unsupported taper")

    def steering_weights(
        self,
        steer_angle: Radian,
        taper: Taper = "uniform",
        vertical: Radian = 0.0,
    ) -> np.ndarray:
        a0 = self.element_response_vector(steer_angle, vertical)
        w = a0 * self._taper(taper)
        norm = np.linalg.norm(w)
        if norm < 1e-15:
            raise ValueError("Degenerate steering vector")
        return w / norm

    def beam_scan(
        self,
        steer_angle: Radian,
        scan_angles: np.ndarray,
        taper: Taper = "uniform",
        vertical: Radian = 0.0,
    ) -> BeamScanResult:
        w = self.steering_weights(
            steer_angle=steer_angle,
            taper=taper,
            vertical=vertical,
        )

        power = np.zeros_like(scan_angles, dtype=float)
        for i, ang in enumerate(scan_angles):
            a = self.element_response_vector(float(ang), vertical)
            y = np.vdot(w, a)
            power[i] = np.abs(y) ** 2

        power /= np.max(power) + 1e-15
        power_db = 10.0 * np.log10(np.maximum(power, 1e-12))
        peak_idx = int(np.argmax(power))

        return BeamScanResult(
            scan_angles=scan_angles,
            power_linear=power,
            power_db=power_db,
            peak_angle=float(scan_angles[peak_idx]),
        )
