"""Jones matrix based antenna model.

The structure mirrors `osi_antenna.proto` of the osi-hfss OSI fork one to one:
same message names, field names, enum values and units (rad, dB, Hz, m).
Angles follow the OSI `Spherical3d` convention: horizontal (azimuth) and
vertical (elevation) angle 0 is boresight.
"""

from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property

import numpy as np

type Radian = float


class PolarizationBasis(Enum):
    UNKNOWN = 0
    OTHER = 1
    # index 0 = horizontal (H), index 1 = vertical (V)
    LINEAR_HV = 2
    # index 0 = left hand (L), index 1 = right hand (R)
    CIRCULAR_LR = 3


class Mode(Enum):
    UNKNOWN = 0
    OTHER = 1
    RECEIVE = 2
    TRANSMIT = 3


@dataclass(frozen=True)
class Direction:
    horizontal: Radian
    vertical: Radian

    def unit_vector(self) -> np.ndarray:
        cos_v = np.cos(self.vertical)
        return np.array(
            [
                cos_v * np.cos(self.horizontal),
                cos_v * np.sin(self.horizontal),
                np.sin(self.vertical),
            ]
        )


@dataclass(frozen=True)
class Vector3d:
    x: float
    y: float
    z: float

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


@dataclass(frozen=True)
class Orientation3d:
    roll: Radian = 0.0
    pitch: Radian = 0.0
    yaw: Radian = 0.0


@dataclass(frozen=True)
class ComplexValue:
    amplitude: float
    phase: Radian


@dataclass(frozen=True, eq=False)
class ComplexGrid:
    """Complex values in polar form, shape (len(horizontal), len(vertical))."""

    amplitude: np.ndarray
    phase: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "amplitude", np.asarray(self.amplitude, dtype=float))
        object.__setattr__(self, "phase", np.asarray(self.phase, dtype=float))
        assert self.amplitude.shape == self.phase.shape

    @property
    def values(self) -> np.ndarray:
        return self.amplitude * np.exp(1j * self.phase)

    def value_at(self, index: tuple[int, int]) -> complex:
        return self.amplitude[index] * np.exp(1j * self.phase[index])


@dataclass(frozen=True, eq=False)
class JonesPattern:
    """Complex polarimetric far-field pattern on a regular angular grid.

    For an excitation e = (e_0, e_1) the radiated field is E = J @ e with
    J = [[pattern_00, pattern_10], [pattern_01, pattern_11]].
    """

    horizontal_angle: np.ndarray
    vertical_angle: np.ndarray
    pattern_00: ComplexGrid
    pattern_11: ComplexGrid
    pattern_01: ComplexGrid
    pattern_10: ComplexGrid

    def __post_init__(self):
        for name in ("horizontal_angle", "vertical_angle"):
            angles = np.asarray(getattr(self, name), dtype=float)
            assert np.all(np.diff(angles) > 0), f"{name} must be strictly increasing"
            object.__setattr__(self, name, angles)
        shape = (len(self.horizontal_angle), len(self.vertical_angle))
        for grid in self._patterns():
            assert grid.amplitude.shape == shape

    def _patterns(self) -> tuple[ComplexGrid, ...]:
        return (self.pattern_00, self.pattern_11, self.pattern_01, self.pattern_10)

    def closest_indices(self, direction: Direction) -> tuple[int, int]:
        h_idx = int(np.argmin(np.abs(self.horizontal_angle - direction.horizontal)))
        v_idx = int(np.argmin(np.abs(self.vertical_angle - direction.vertical)))
        return h_idx, v_idx

    def jones(self, direction: Direction) -> np.ndarray:
        """2x2 Jones matrix J[out, in] (nearest neighbour lookup)."""
        idx = self.closest_indices(direction)
        return np.array(
            [
                [self.pattern_00.value_at(idx), self.pattern_10.value_at(idx)],
                [self.pattern_01.value_at(idx), self.pattern_11.value_at(idx)],
            ]
        )

    def resampled(self, step: Radian) -> "JonesPattern":
        def indices(angles: np.ndarray) -> np.ndarray:
            targets = np.arange(angles[0], angles[-1] + 1e-12, step)
            return np.unique([int(np.argmin(np.abs(angles - t))) for t in targets])

        h_idx = indices(self.horizontal_angle)
        v_idx = indices(self.vertical_angle)

        def sub(grid: ComplexGrid) -> ComplexGrid:
            return ComplexGrid(
                amplitude=grid.amplitude[np.ix_(h_idx, v_idx)],
                phase=grid.phase[np.ix_(h_idx, v_idx)],
            )

        return JonesPattern(
            horizontal_angle=self.horizontal_angle[h_idx],
            vertical_angle=self.vertical_angle[v_idx],
            pattern_00=sub(self.pattern_00),
            pattern_11=sub(self.pattern_11),
            pattern_01=sub(self.pattern_01),
            pattern_10=sub(self.pattern_10),
        )


@dataclass(frozen=True)
class FrequencyInstance:
    frequency: float  # Hz
    gain: float  # dB, amplitude (20 * log10 of the linear factor)
    phase: Radian
    radiation: JonesPattern
    axial_ratio: float | None = None  # dB

    def jones(self, direction: Direction) -> np.ndarray:
        factor = 10.0 ** (self.gain / 20.0) * np.exp(1j * self.phase)
        return self.radiation.jones(direction) * factor

    def resampled(self, step: Radian) -> "FrequencyInstance":
        return FrequencyInstance(
            frequency=self.frequency,
            gain=self.gain,
            phase=self.phase,
            radiation=self.radiation.resampled(step),
            axial_ratio=self.axial_ratio,
        )


@dataclass(frozen=True)
class AntennaElementType:
    id: int
    mode: Mode
    frequency_instance: list[FrequencyInstance]

    def closest_frequency_instance(self, frequency: float) -> FrequencyInstance:
        return min(self.frequency_instance, key=lambda f: abs(f.frequency - frequency))


@dataclass(frozen=True)
class AntennaElement:
    index: int
    element_type_id: int
    position: Vector3d  # m, sensor coordinates
    orientation: Orientation3d = field(default_factory=Orientation3d)


@dataclass(frozen=True)
class AntennaModel:
    id: int
    polarization_basis: PolarizationBasis
    element_type: list[AntennaElementType]
    element: list[AntennaElement]

    @cached_property
    def _types(self) -> dict[int, AntennaElementType]:
        return {t.id: t for t in self.element_type}

    def type_of(self, element: AntennaElement) -> AntennaElementType:
        return self._types[element.element_type_id]

    def elements(self, mode: Mode) -> list[AntennaElement]:
        return [e for e in self.element if self.type_of(e).mode == mode]

    def jones(
        self, element: AntennaElement, frequency: float, direction: Direction
    ) -> np.ndarray:
        """2x2 Jones matrix of an element incl. gain and phase."""
        instance = self.type_of(element).closest_frequency_instance(frequency)
        return instance.jones(direction)

    def response(
        self,
        element: AntennaElement,
        frequency: float,
        direction: Direction,
        excitation: np.ndarray | None = None,
    ) -> np.ndarray:
        """Radiated field (component 0, component 1), default excitation (1, 0)."""
        if excitation is None:
            excitation = np.array([1.0, 0.0])
        return self.jones(element, frequency, direction) @ excitation

    def resampled(self, step: Radian) -> "AntennaModel":
        return AntennaModel(
            id=self.id,
            polarization_basis=self.polarization_basis,
            element_type=[
                AntennaElementType(
                    id=t.id,
                    mode=t.mode,
                    frequency_instance=[
                        f.resampled(step) for f in t.frequency_instance
                    ],
                )
                for t in self.element_type
            ],
            element=list(self.element),
        )


def simple_grid(horizontal: np.ndarray, vertical: np.ndarray) -> ComplexGrid:
    """cos^2 amplitude pattern with a slowly varying phase."""
    h = np.asarray(horizontal, dtype=float)[:, np.newaxis]
    v = np.asarray(vertical, dtype=float)[np.newaxis, :]
    return ComplexGrid(
        amplitude=np.square(np.cos(h)) * np.square(np.cos(v)),
        phase=np.radians(np.exp(np.square(np.arctan(h) * np.arctan(v)))),
    )


def zero_grid(horizontal: np.ndarray, vertical: np.ndarray) -> ComplexGrid:
    shape = (len(horizontal), len(vertical))
    return ComplexGrid(amplitude=np.zeros(shape), phase=np.zeros(shape))
