from pathlib import Path

import numpy as np
import pytest

from osi3.osi_antenna_pb2 import AntennaElementType, AntennaModel
from osi_hfss import decode, modules
from osi_hfss.data import load, read

REFERENCE_DIR = Path(__file__).parent / "reference"

pytestmark = pytest.mark.skipif(not modules.available(), reason="the modules in bin/ need Linux x86-64")

CHAINS = {"front": modules.fmcw_signal_generator, "corner": modules.pmcw_signal_generator}


def assert_messages_close(a, b, path=""):
    fa, fb = {f.name: v for f, v in a.ListFields()}, {f.name: v for f, v in b.ListFields()}
    assert set(fa) == set(fb), path
    for name, va in fa.items():
        vb, where = fb[name], f"{path}.{name}"
        if isinstance(va, bytes):
            continue
        repeated = hasattr(va, "__len__") and not isinstance(va, str)
        xs, ys = (list(va), list(vb)) if repeated else ([va], [vb])
        assert len(xs) == len(ys), where
        for x, y in zip(xs, ys):
            if hasattr(x, "ListFields"):
                assert_messages_close(x, y, where)
            elif isinstance(x, float):
                assert x == pytest.approx(y, rel=1e-9, abs=1e-12), where
            else:
                assert x == y, where


@pytest.fixture(scope="module", params=list(CHAINS))
def chain(request):
    name = request.param
    sensor_view = modules.environment_simulation(load(f"{name}.groundtruth.bin"),
                                                 load(f"{name}.sensorviewconfiguration.bin"), load(f"{name}.scene.json"))
    antenna = modules.antenna_model(load(f"{name}.antenna.json"))
    cpi = CHAINS[name](sensor_view, antenna, load(f"{name}.signal_generator.json"))
    return name, sensor_view, antenna, cpi


def test_environment_and_antenna_equal_the_pipeline(chain):
    name, sensor_view, antenna, _ = chain
    assert_messages_close(sensor_view, load(f"{name}.sensorview.bin"))
    assert_messages_close(antenna, load(f"{name}.antennamodel.bin"))


def test_signal_generator_equals_the_pipeline(chain):
    name, _, _, cpi = chain
    recorded = load(f"{name}.radarcpidata.bin")
    assert_messages_close(cpi, recorded)
    cube, expected = decode.cpi_cube(cpi.cpi_frame[0]), decode.cpi_cube(recorded.cpi_frame[0])
    np.testing.assert_allclose(cube, expected, rtol=0, atol=1e-6 * np.abs(expected).max())


def test_dsp_equals_the_pipeline(chain):
    name, _, antenna, cpi = chain
    sensor_data, analysis = modules.radar_dsp_with_analysis(cpi, antenna)
    assert_messages_close(sensor_data, load(f"{name}.sensordata.bin"))
    recorded = load(f"{name}.dsp.npz")
    for key, value in analysis.items():
        np.testing.assert_allclose(value, recorded[key], rtol=1e-6, err_msg=key)
    assert_messages_close(modules.radar_dsp(cpi), load(f"{name}.phase_centre.sensordata.bin"))


def test_changed_configuration_changes_the_output():
    sensor_view, antenna = load("front.sensorview.bin"), load("front.antennamodel.bin")
    config = load("front.signal_generator.json")
    config["loops"] = 32
    frame = modules.fmcw_signal_generator(sensor_view, antenna, config).cpi_frame[0]
    assert frame.slow_axis.size == 32


def test_pmcw_analysis_equals_the_pipeline():
    sensor_view, antenna = load("corner.sensorview.bin"), load("corner.antennamodel.bin")
    _, analysis = modules.pmcw_signal_generator_with_analysis(sensor_view, antenna, load("corner.signal_generator.json"))
    for key, value in load("corner.pmcw.npz").items():
        np.testing.assert_allclose(analysis[key], value, rtol=1e-6, atol=1e-9, err_msg=key)


def test_osi_sensor_models_equal_the_pipeline():
    legacy = modules.environment_simulation(load("legacy.groundtruth.bin"), load("legacy.sensorviewconfiguration.bin"),
                                            load("legacy.scene.json"))
    assert_messages_close(legacy, load("legacy.sensorview.bin"))
    assert_messages_close(modules.legacy_sensor_model(legacy), load("legacy.sensordata.bin"))

    extension = modules.environment_simulation(load("extension.groundtruth.bin"),
                                               load("extension.sensorviewconfiguration.bin"), load("extension.scene.json"))
    assert_messages_close(extension, load("extension.sensorview.bin"))
    sensor_data, doa = modules.doa_sensor_model(extension)
    assert_messages_close(sensor_data, load("extension.sensordata.bin"))
    np.testing.assert_allclose(doa["spectra"], load("extension.doa.npz")["spectra"], rtol=1e-6)

    mimo = modules.environment_simulation(load("extension_mimo.groundtruth.bin"),
                                          load("extension_mimo.sensorviewconfiguration.bin"), load("extension.scene.json"))
    assert_messages_close(mimo, load("extension_mimo.sensorview.bin"))
    for transmitters, name in ((1, "extension_mimo.simo"), (None, "extension_mimo")):
        sensor_data, spectrum = modules.mimo_doa_sensor_model(mimo, transmitters)
        assert_messages_close(sensor_data, load(f"{name}.sensordata.bin"))
        np.testing.assert_allclose(spectrum["spectrum_db"], load(f"{name}.npz")["spectrum_db"], atol=1e-6)


def test_errors_of_the_modules_are_reported():
    sensor_view = load("front.sensorview.bin")
    with pytest.raises(RuntimeError, match="not a serialized osi3.AntennaModel"):
        modules.fmcw_signal_generator(sensor_view, sensor_view)
    scene = load("front.scene.json")
    scene["objects"] = scene["objects"][1:]
    with pytest.raises(RuntimeError, match="no radar properties for object"):
        modules.environment_simulation(load("front.groundtruth.bin"), load("front.sensorviewconfiguration.bin"), scene)


@pytest.mark.parametrize("name", sorted(p.name for p in REFERENCE_DIR.glob("*.bin")))
def test_outputs_match_the_reference(name):
    assert_messages_close(load(name), read(REFERENCE_DIR / name))


def test_beam_scan_of_a_model_built_in_python():
    angles = np.radians(np.arange(-90.0, 90.0 + 1e-9, 1.0))
    co = (np.cos(angles)[:, None] ** 2 * np.cos(angles)[None, :] ** 2).ravel()
    model = AntennaModel(polarization_basis=AntennaModel.POLARIZATION_BASIS_LINEAR_HV)
    model.id.value = 0
    element_type = model.element_type.add(mode=AntennaElementType.MODE_RECEIVE)
    element_type.id.value = 0
    instance = element_type.frequency_instance.add(frequency=77e9, gain=20.0, phase=0.0)
    instance.radiation.horizontal_angle.extend(angles)
    instance.radiation.vertical_angle.extend(angles)
    for name in ("pattern_00", "pattern_11", "pattern_01", "pattern_10"):
        getattr(instance.radiation, name).amplitude.extend(co if name == "pattern_00" else np.zeros_like(co))
        getattr(instance.radiation, name).phase.extend(np.zeros_like(co))
    for i in range(8):
        element = model.element.add(index=i)
        element.element_type_id.value = 0
        element.position.y = i * 0.5 * decode.SPEED_OF_LIGHT / 77e9
    scan = modules.beam_scan(model, steer_deg=20.0, taper="hann")
    recorded = load("ula.beam_scan.npz")
    np.testing.assert_allclose(scan["power_db"], recorded["power_db"], atol=1e-9)
    assert float(scan["peak_angle"]) == pytest.approx(float(recorded["peak_angle"]))
