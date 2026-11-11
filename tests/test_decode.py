import numpy as np
import pytest

from osi3.osi_antenna_pb2 import AntennaElementType, AntennaModel
from osi3.osi_radarcpi_pb2 import RadarCpiChannel, RadarCpiData
from osi3.osi_sensorview_pb2 import RadarSensorView, SensorView
from osi_hfss import decode
from osi_hfss.data import load


def test_reflections_of_the_front_radar():
    view = SensorView.FromString(load("front.sensorview.bin").SerializeToString()).radar_sensor_view[0]
    rays = decode.reflection_arrays(view.reflection)
    n = len(view.reflection)
    assert rays["polarimetric_response"].shape == (n, 2, 2) and rays["time_of_flight"].shape == (n,)
    assert set(rays["number_of_interactions"]) == {1, 2}
    assert set(rays["object_id"]) == {1, 2, 3}
    power = np.sum(np.abs(rays["polarimetric_response"]) ** 2, axis=(1, 2)) / 2
    np.testing.assert_allclose(rays["signal_strength"], 10 * np.log10(power), atol=1e-9)


def test_unset_analysis_fields_of_a_reflection():
    view = RadarSensorView()
    view.reflection.add(time_of_flight=1e-7)
    rays = decode.reflection_arrays(view.reflection)
    assert rays["object_id"].tolist() == [-1] and rays["number_of_interactions"].tolist() == [1]


def test_cpi_layouts():
    raw = load("front.raw.radarcpidata.bin").cpi_frame[0]
    virtual = load("front.radarcpidata.bin").cpi_frame[0]
    assert (decode.enum_name(raw.fast_axis, "domain"), decode.enum_name(raw.slow_axis, "domain")) == ("FAST_TIME", "SLOW_TIME")
    assert decode.enum_name(virtual.fast_axis, "domain") == "RANGE"
    assert decode.enum_name(virtual.channel[0], "kind") == "VIRTUAL"
    assert decode.enum_name(raw, "sample_format") == "COMPLEX_IQ"
    assert len(virtual.channel) == len(raw.channel) * len(raw.transmitter)
    assert not any(c.HasField("tx_index") for c in raw.channel)
    assert all(c.HasField("tx_index") and c.kind == RadarCpiChannel.KIND_VIRTUAL for c in virtual.channel)
    assert decode.cpi_cube(virtual).shape == (12, 64, 256)
    np.testing.assert_allclose(decode.axis_values(virtual.fast_axis)[:2], [0.0, virtual.fast_axis.spacing])


def test_data_layout_is_channel_slow_fast_interleaved_iq():
    frame = load("front.raw.radarcpidata.bin").cpi_frame[0]
    values = np.frombuffer(frame.data, dtype="<f4")
    c, s, f = 2, 5, 17
    i = ((c * frame.slow_axis.size + s) * frame.fast_axis.size + f) * 2
    assert decode.cpi_cube(frame)[c, s, f] == pytest.approx(complex(values[i], values[i + 1]) * frame.scale_factor)


def test_int16_quantization_stays_below_the_noise_floor():
    float32 = load("front.radarcpidata.bin").cpi_frame[0]
    int16 = RadarCpiData.FromString(load("front.int16.radarcpidata.bin").SerializeToString()).cpi_frame[0]
    assert len(int16.data) == len(float32.data) // 2
    error = decode.cpi_cube(int16) - decode.cpi_cube(float32)
    assert np.abs(error).max() <= int16.scale_factor
    assert np.mean(np.abs(error) ** 2) < 1e-3 * float32.noise_power


@pytest.mark.parametrize("name, configuration_id, port, feed", [("front", 1, 0, (1.0, 0.0)), ("corner", 2, 1, (0.0, 1.0))])
def test_polarization_and_configuration_reach_the_message(name, configuration_id, port, feed):
    frame = load(f"{name}.radarcpidata.bin").cpi_frame[0]
    assert frame.configuration_id.value == configuration_id
    assert {c.rx_port for c in frame.channel} == {port}  # H port (FMCW) or V port (PMCW)
    for t in frame.transmitter:
        assert (abs(decode.complex_value(t.excitation_0)), abs(decode.complex_value(t.excitation_1))) == pytest.approx(feed)


def asymmetric_model() -> AntennaModel:
    angles = np.radians(np.arange(-90.0, 90.0 + 1e-9, 5.0))
    co = (np.cos(angles)[:, None] ** 2 * np.cos(angles)[None, :] ** 2).ravel()
    model = AntennaModel(polarization_basis=AntennaModel.POLARIZATION_BASIS_LINEAR_HV)
    element_type = model.element_type.add(mode=AntennaElementType.MODE_TRANSMIT)
    element_type.id.value = 0
    instance = element_type.frequency_instance.add(frequency=77e9, gain=20.0, phase=0.0)
    instance.radiation.horizontal_angle.extend(angles)
    instance.radiation.vertical_angle.extend(angles)
    for name, amplitude in (("pattern_00", co), ("pattern_01", 0.3 * co), ("pattern_10", 0 * co), ("pattern_11", 0 * co)):
        getattr(instance.radiation, name).amplitude.extend(amplitude)
        getattr(instance.radiation, name).phase.extend(np.zeros_like(co))
    element = model.element.add(index=0)
    element.element_type_id.value = 0
    return model


def test_pattern_rt_maps_excitation_t_to_field_r():
    model = AntennaModel.FromString(asymmetric_model().SerializeToString())
    (element,) = decode.elements(model, decode.TRANSMIT)
    j = decode.element_jones(model, element, 77e9, 0.0, 0.0)
    np.testing.assert_allclose(j @ [0, 1], [3.0, 0.0], atol=1e-12)  # V excitation -> H field 0.3, gain 20 dB
    np.testing.assert_allclose(j @ [1, 0], [10.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(j.T @ [1, 0], [10.0, 3.0], atol=1e-12)  # reciprocity: H field received on the V port
    np.testing.assert_allclose(np.abs(decode.element_jones(model, element, 77e9, np.radians([0.0, 61.0]))[:, 0, 0]),
                               [10.0, 10 * np.cos(np.radians(60.0)) ** 2])


def test_antenna_models():
    front = load("front.antennamodel.bin")
    assert len(decode.elements(front, decode.TRANSMIT)) == 3 and len(decode.elements(front, decode.RECEIVE)) == 4
    rx0 = decode.elements(front, decode.RECEIVE)[0]
    j = decode.element_jones(front, rx0, 76.75e9, 0.0)
    assert abs(j[0, 0]) == pytest.approx(10.0) and abs(j[1, 0]) == pytest.approx(1.0)  # H co-polar, -20 dB cross
    corner = load("corner.antennamodel.bin")
    j = decode.element_jones(corner, decode.elements(corner, decode.RECEIVE)[0], 79e9, 0.0)
    assert abs(j[1, 1]) == pytest.approx(10.0) and abs(j[0, 0]) == 0.0  # V co-polar
