import numpy as np
import pytest

from osi_hfss.analysis import closest, expected_positions
from osi_hfss import pipeline
from osi_hfss.data import MESSAGE_TYPES, load

RADARS = ("front", "corner")
AZIMUTH = np.radians(np.arange(-70.0, 70.0 + 1e-9, 0.25))


def test_every_file_loads():
    files = pipeline.names()
    assert len(files) >= 50
    for name in files:
        content = load(name)
        if name.endswith(".bin"):
            assert isinstance(content, MESSAGE_TYPES[name.split(".")[-2]])


@pytest.mark.parametrize("name", RADARS)
def test_detections_are_at_the_expected_positions(name):
    detections = load(f"{name}.sensordata.bin").feature_data.radar_sensor[0].detection
    expected = expected_positions(load(f"{name}.sensorview.bin"))
    assert {closest(d, expected).object_id for d in detections} == {e.object_id for e in expected}
    for d in detections:
        e = closest(d, expected)
        assert abs(d.position.distance - e.distance) < 0.2
        assert abs(d.position.azimuth - e.azimuth) < np.radians(0.3)
        assert abs(d.radial_velocity - e.radial_velocity) < 0.4


@pytest.mark.parametrize("name", RADARS)
def test_dsp_analysis_is_consistent_with_the_detections(name):
    detections = load(f"{name}.sensordata.bin").feature_data.radar_sensor[0].detection
    analysis = load(f"{name}.dsp.npz")
    azimuth = analysis["azimuth"]
    np.testing.assert_allclose(azimuth, AZIMUTH)
    assert analysis["snr"].shape == (len(analysis["range_rate"]), len(analysis["range"]))
    assert len(analysis["spectrum_antenna"]) == len(detections)
    for d, spectrum in zip(detections, analysis["spectrum_antenna"], strict=True):
        assert azimuth[np.argmax(spectrum)] == pytest.approx(d.position.azimuth)
    expected = expected_positions(load(f"{name}.sensorview.bin"))
    ideal = load(f"{name}.phase_centre.sensordata.bin").feature_data.radar_sensor[0].detection
    bias = [abs(d.position.azimuth - closest(d, expected, (1.0, 1.0, 0.3)).azimuth) for d in ideal]
    assert max(bias) > np.radians(2.0)
