from dataclasses import dataclass
from enum import Enum
import numpy as np

type Degree = float


@dataclass(frozen=True)
class Direction:
    azimuth: Degree
    elevation: Degree


class IAnglePattern:
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


class SimpleAmplitudePattern(IAnglePattern):
    def create_pattern(self) -> np.ndarray:
        return (
            np.abs(np.square(np.cos(np.radians(self.az_angles))))[:, np.newaxis]
            * np.abs(np.square(np.cos(np.radians(self.el_angles - 90))))[np.newaxis, :]
        )


class SimplePhasePattern(IAnglePattern):
    def create_pattern(self) -> np.ndarray:
        return np.exp(
            np.square(
                (np.arctan(np.radians(self.az_angles))[:, np.newaxis])
                * (np.arctan(np.radians(self.el_angles - 90))[np.newaxis, :])
            )
        )


class ZeroPhasePattern(IAnglePattern):
    def create_pattern(self) -> np.ndarray:
        return np.zeros((len(self.az_angles), len(self.el_angles)))


class ZeroAmplitudePattern(IAnglePattern):
    def create_pattern(self) -> np.ndarray:
        return np.zeros((len(self.az_angles), len(self.el_angles)))


@dataclass(frozen=True)
class Position:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class RadiationPattern:
    amplitude: IAnglePattern
    phase: IAnglePattern


@dataclass(frozen=True)
class ComplexSample:
    amplitude: float
    phase: Degree


class IRadiationProperties:
    def __init__(
        self,
        pattern_00: RadiationPattern,
        pattern_11: RadiationPattern,
        pattern_01: RadiationPattern,
        pattern_10: RadiationPattern,
    ):
        self.pattern_00 = pattern_00
        self.pattern_11 = pattern_11
        self.pattern_01 = pattern_01
        self.pattern_10 = pattern_10
        assert (
            self.pattern_00.amplitude.az_angles.shape[0]
            == self.pattern_00.amplitude.pattern.shape[0]
        )
        assert (
            self.pattern_00.amplitude.el_angles.shape[0]
            == self.pattern_00.amplitude.pattern.shape[1]
        )
        self.az_angles = self.pattern_00.amplitude.az_angles
        self.el_angles = self.pattern_00.amplitude.el_angles

    def _find_closest_indices(self, direction: Direction) -> tuple[int, int]:
        az_idx = (np.abs(self.az_angles - direction.azimuth)).argmin()
        el_idx = (np.abs(self.el_angles - direction.elevation)).argmin()
        return az_idx, el_idx

    def _get_value(
        self,
        excitation_0: ComplexSample,
        excitation_1: ComplexSample,
        direction: Direction,
    ) -> tuple[ComplexSample, ComplexSample]:
        az_idx, el_idx = self._find_closest_indices(direction)
        amplitude_00 = self.pattern_00.amplitude.pattern[az_idx, el_idx]
        phase_00 = self.pattern_00.phase.pattern[az_idx, el_idx]
        amplitude_11 = self.pattern_11.amplitude.pattern[az_idx, el_idx]
        phase_11 = self.pattern_11.phase.pattern[az_idx, el_idx]
        amplitude_01 = self.pattern_01.amplitude.pattern[az_idx, el_idx]
        phase_01 = self.pattern_01.phase.pattern[az_idx, el_idx]
        amplitude_10 = self.pattern_10.amplitude.pattern[az_idx, el_idx]
        phase_10 = self.pattern_10.phase.pattern[az_idx, el_idx]

        result_amplitude_0 = excitation_0.amplitude * amplitude_00 * np.exp(
            1j * np.radians(excitation_0.phase + phase_00)
        ) + excitation_1.amplitude * amplitude_10 * np.exp(
            1j * np.radians(excitation_1.phase + phase_10)
        )
        result_phase_0 = np.angle(result_amplitude_0, deg=True)

        result_amplitude_1 = excitation_0.amplitude * amplitude_01 * np.exp(
            1j * np.radians(excitation_0.phase + phase_01)
        ) + excitation_1.amplitude * amplitude_11 * np.exp(
            1j * np.radians(excitation_1.phase + phase_11)
        )
        result_phase_1 = np.angle(result_amplitude_1, deg=True)

        return (
            ComplexSample(amplitude=np.abs(result_amplitude_0), phase=result_phase_0),
            ComplexSample(amplitude=np.abs(result_amplitude_1), phase=result_phase_1),
        )


class LinearRadiationProperties(IRadiationProperties):
    def __init__(
        self,
        hh_pattern: RadiationPattern,
        vv_pattern: RadiationPattern,
        hv_pattern: RadiationPattern,
        vh_pattern: RadiationPattern,
    ):
        # HH -> 00
        # VV -> 11
        # HV -> 01
        # VH -> 10
        super().__init__(
            pattern_00=hh_pattern,
            pattern_11=vv_pattern,
            pattern_01=hv_pattern,
            pattern_10=vh_pattern,
        )

    def get_value(
        self,
        excitation_h: ComplexSample,
        excitation_v: ComplexSample,
        direction: Direction,
    ) -> tuple[ComplexSample, ComplexSample]:
        return super()._get_value(excitation_h, excitation_v, direction)


@dataclass(frozen=True)
class FrequencyInstance:
    frequency: float
    radiation: IRadiationProperties
    gain: float
    phase: float


class RxTMode(Enum):
    RECEIVE = 0
    TRANSMIT = 1


@dataclass(frozen=True)
class AntennaElement:
    frequency_instances: list[FrequencyInstance]
    mode: RxTMode


class AntennaElementInstance:
    def __init__(self, idx: int, position: Position, antenna_element: AntennaElement):
        self.idx = idx
        self.position = position
        self.antenna_element = antenna_element

    def sample(
        self, frequency: float, direction: Direction
    ) -> tuple[ComplexSample, ComplexSample]:
        assert self.antenna_element.frequency_instances[0].frequency == frequency, (
            "Only single frequency instances are supported now."
        )
        frequency_instance = self.antenna_element.frequency_instances[0]
        radiation = frequency_instance.radiation
        assert isinstance(radiation, LinearRadiationProperties), (
            "Only linear polarisation is supported now."
        )

        response_0, response_1 = radiation.get_value(
            excitation_h=ComplexSample(amplitude=1.0, phase=0.0),
            excitation_v=ComplexSample(amplitude=0.0, phase=0.0),
            direction=direction,
        )
        return (
            ComplexSample(
                amplitude=response_0.amplitude * frequency_instance.gain,
                phase=response_0.phase + frequency_instance.phase,
            ),
            ComplexSample(
                amplitude=response_1.amplitude * frequency_instance.gain,
                phase=response_1.phase + frequency_instance.phase,
            ),
        )
