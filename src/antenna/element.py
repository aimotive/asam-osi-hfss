from dataclasses import dataclass
from enum import Enum
import numpy as np

type Degree = float


class AnglePatternBase:
    def __init__(self, angles: list[Degree]):
        self.angles = angles
        self.pattern = self.create_pattern()

    def get_by_angle(self, angle: Degree) -> float:
        if angle < self.angles[0] or angle > self.angles[-1]:
            raise ValueError("Angle is out of bounds.")
        return np.interp(angle, self.angles, self.pattern)

    def create_pattern(self) -> list[float]:
        raise NotImplementedError


class SimpleAmplitudePattern(AnglePatternBase):
    def create_pattern(self) -> list[float]:
        return np.abs(np.square(np.cos(np.radians(self.angles))))


class SimplePhasePattern(AnglePatternBase):
    def create_pattern(self) -> list[Degree]:
        return np.arctan(self.angles)


class ZeroPhasePattern(AnglePatternBase):
    def create_pattern(self) -> list[Degree]:
        return np.zeros_like(self.angles)


class ZeroAmplitudePattern(AnglePatternBase):
    def create_pattern(self) -> list[float]:
        return np.zeros_like(self.angles)


@dataclass(frozen=True)
class Position:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class Direction:
    azimuth: Degree
    elevation: Degree


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
        ].radiation.hh_pattern.amplitude.get_by_angle(direction.azimuth)

        amplitude = (
            amplitude_pattern_instance
            * self.antenna_element.frequency_instances[0].gain
        )

        phase_pattern_instance = self.antenna_element.frequency_instances[
            0
        ].radiation.hh_pattern.phase.get_by_angle(direction.azimuth)

        phase = (
            phase_pattern_instance + self.antenna_element.frequency_instances[0].phase
        )
        return ComplexSample(amplitude=amplitude, phase=phase)
