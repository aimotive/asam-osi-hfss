from dataclasses import dataclass
from enum import Enum
import numpy as np

type Degree = float


@dataclass(frozen=True)
class Direction:
    azimuth: Degree
    elevation: Degree


class AnglePatternBase:
    def __init__(self, az_angles: list[Degree], el_angles: list[Degree]):
        self.az_angles = np.asarray(az_angles, dtype=float)
        self.el_angles = np.asarray(el_angles, dtype=float)
        self.pattern: np.ndarray = self.create_pattern()

    def find_closest_indices(self, direction: Direction) -> tuple[int, int]:
        az_idx = int(np.argmin(np.abs(self.az_angles - direction.azimuth)))
        el_idx = int(np.argmin(np.abs(self.el_angles - direction.elevation)))
        return az_idx, el_idx

    def find_closest_direction(self, direction: Direction) -> Direction:
        az_idx, el_idx = self.find_closest_indices(direction)
        return Direction(
            azimuth=self.az_angles[az_idx], elevation=self.el_angles[el_idx]
        )

    def get_by_angle(self, direction: Direction) -> float:
        az_idx, el_idx = self.find_closest_indices(direction)
        return self.pattern[az_idx, el_idx]

    def create_pattern(self) -> np.ndarray:
        raise NotImplementedError


class SimpleAmplitudePattern(AnglePatternBase):
    def create_pattern(self) -> np.ndarray:
        return (
            np.abs(np.square(np.cos(np.radians(self.az_angles))))[:, np.newaxis]
            * np.abs(np.square(np.cos(np.radians(self.el_angles - 90))))[np.newaxis, :]
        )


class SimplePhasePattern(AnglePatternBase):
    def create_pattern(self) -> np.ndarray:
        return np.exp(
            np.square(
                (np.arctan(np.radians(self.az_angles))[:, np.newaxis])
                * (np.arctan(np.radians(self.el_angles - 90))[np.newaxis, :])
            )
        )


class ZeroPhasePattern(AnglePatternBase):
    def create_pattern(self) -> np.ndarray:
        return np.zeros((len(self.az_angles), len(self.el_angles)))


class ZeroAmplitudePattern(AnglePatternBase):
    def create_pattern(self) -> np.ndarray:
        return np.zeros((len(self.az_angles), len(self.el_angles)))


@dataclass(frozen=True)
class Position:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class RadiationPattern:
    amplitude: AnglePatternBase
    phase: AnglePatternBase


@dataclass(frozen=True)
class RadiationProperties:
    hh_pattern: RadiationPattern
    vv_pattern: RadiationPattern
    hv_pattern: RadiationPattern


@dataclass(frozen=True)
class FrequencyInstance:
    frequency: float
    radiation: RadiationProperties
    gain: float
    phase: float
    axial_ratio: float


class RxTMode(Enum):
    RECEIVE = 0
    TRANSMIT = 1


@dataclass(frozen=True)
class AntennaElement:
    frequency_instances: list[FrequencyInstance]
    mode: RxTMode


@dataclass(frozen=True)
class ComplexSample:
    amplitude: float
    phase: Degree


class AntennaElementInstance:
    def __init__(self, idx: int, position: Position, antenna_element: AntennaElement):
        self.idx = idx
        self.position = position
        self.antenna_element = antenna_element

    def sample(self, frequency: float, direction: Direction) -> ComplexSample:
        assert len(self.antenna_element.frequency_instances) == 1, (
            "Only single frequency instances are supported now."
        )
        assert self.antenna_element.frequency_instances[0].axial_ratio == 1.0, (
            "Only linear polarisation is supported now."
        )
        assert np.allclose(
            self.antenna_element.frequency_instances[
                0
            ].radiation.vv_pattern.amplitude.pattern,
            0.0,
        ), "Only horizontal polarisation is supported now."
        assert np.allclose(
            self.antenna_element.frequency_instances[
                0
            ].radiation.hv_pattern.amplitude.pattern,
            0.0,
        ), "Only co-polarised radiation is supported now."

        amplitude_pattern_instance = self.antenna_element.frequency_instances[
            0
        ].radiation.hh_pattern.amplitude.get_by_angle(direction)

        amplitude = (
            amplitude_pattern_instance
            * self.antenna_element.frequency_instances[0].gain
        )

        phase_pattern_instance = self.antenna_element.frequency_instances[
            0
        ].radiation.hh_pattern.phase.get_by_angle(direction)

        phase = (
            phase_pattern_instance + self.antenna_element.frequency_instances[0].phase
        )
        return ComplexSample(amplitude=amplitude, phase=phase)
