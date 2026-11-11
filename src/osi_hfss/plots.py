import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Wedge

from osi_hfss.analysis import (
    ExpectedPosition,
    array_positions,
    closest,
    diagram_grid,
    sensor_to_world,
)
from osi_hfss.decode import SPEED_OF_LIGHT, axis_values, cpi_cube, reflection_arrays, unit_vectors

HOST_REAR_OVERHANG = (
    1.0  # m, the host's box starts this far behind its reference point (rear axle)
)


def _axes(ax, figsize):
    return ax if ax is not None else plt.subplots(figsize=figsize)[1]


def _detections(sensor_data):
    return sensor_data.feature_data.radar_sensor[0].detection


def plot_scene(
    ground_truth,
    mounting_position,
    field_of_view: float,
    ax=None,
    reach: float = 80.0,
    names: dict[int, str] | None = None,
    velocities: bool = False,
    xlim=None,
    ylim=None,
    title: str = "",
):
    ax = _axes(ax, (10, 6))
    mount = mounting_position.position
    yaw = np.degrees(mounting_position.orientation.yaw)
    half_fov = np.degrees(field_of_view / 2)
    ax.add_patch(
        Wedge(
            (mount.x, mount.y),
            reach,
            yaw - half_fov,
            yaw + half_fov,
            alpha=0.08,
            color="tab:blue",
        )
    )
    for i, obj in enumerate(ground_truth.moving_object):
        b = obj.base
        host = obj.id.value == ground_truth.host_vehicle_id.value
        x0 = b.position.x - (HOST_REAR_OVERHANG if host else b.dimension.length / 2)
        color = "tab:gray" if host else f"C{i - 1}"
        ax.add_patch(
            Rectangle(
                (x0, b.position.y - b.dimension.width / 2),
                b.dimension.length,
                b.dimension.width,
                color=color,
                alpha=0.6,
            )
        )
        label = (
            "host" if host else (names or {}).get(obj.id.value, f"id {obj.id.value}")
        )
        ax.annotate(
            label,
            (x0 + b.dimension.length / 2, b.position.y + b.dimension.width / 2 + 0.6),
            ha="center",
        )
        if velocities and (b.velocity.x, b.velocity.y) != (0, 0):
            ax.annotate(
                "",
                (b.position.x + b.velocity.x, b.position.y + b.velocity.y),
                (b.position.x, b.position.y),
                arrowprops=dict(arrowstyle="->", color=color),
            )
    ax.plot(mount.x, mount.y, "k^", markersize=10, label="radar")
    ax.set_xlim(*(xlim or (-5, 70)))
    ax.set_ylim(*(ylim or (-25, 25)))
    ax.set_aspect("equal")
    ax.set_xlabel("x world (m)")
    ax.set_ylabel("y world (m)")
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    return ax


def plot_detections_world(
    sensor_data, mounting_position, ax, style: str = "rx", label: str = "RadarDetection"
):
    detections = _detections(sensor_data)
    x, y = sensor_to_world(
        mounting_position,
        [d.position.distance for d in detections],
        [d.position.azimuth for d in detections],
    )
    ax.plot(x, y, style, markersize=10, mew=2, label=label)
    return ax


def plot_detections(
    sensor_data,
    expected: list[ExpectedPosition],
    field_of_view: float,
    title: str = "",
    ax=None,
):
    ax = _axes(ax, (11, 5))
    ax.plot(0, 0, "k^", markersize=10, label="sensor")
    ax.plot(
        [e.distance * np.cos(e.azimuth) for e in expected],
        [e.distance * np.sin(e.azimuth) for e in expected],
        "rs",
        mfc="none",
        markersize=13,
        label="expected (direct reflections)",
    )
    detections = _detections(sensor_data)
    xy = np.array(
        [
            [
                d.position.distance * np.cos(d.position.azimuth),
                d.position.distance * np.sin(d.position.azimuth),
            ]
            for d in detections
        ]
    )
    ax.plot(xy[:, 0], xy[:, 1], "bx", markersize=10, label="detections (SensorData)")
    for d, (x, y) in zip(detections, xy):
        ax.annotate(
            f"object {closest(d, expected).object_id}, v_r {d.radial_velocity:+.1f} m/s",
            (x, y),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=8,
        )
    reach = 1.25 * max(e.distance for e in expected)
    for s in (-1, 1):
        ax.plot(
            [0, reach * np.cos(s * field_of_view / 2)],
            [0, reach * np.sin(s * field_of_view / 2)],
            "k:",
            lw=0.8,
        )
    ax.set_xlim(-0.05 * reach, reach)
    ax.set_ylim(-0.5 * reach, 0.5 * reach)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlabel("x sensor (m)")
    ax.set_ylabel("y sensor (m)")
    ax.set_title(title)
    return ax


def _direct_positions(rays: dict[str, np.ndarray]) -> dict[int, np.ndarray]:
    direct = rays["number_of_interactions"] == 1
    positions = (
        SPEED_OF_LIGHT
        * rays["time_of_flight"][direct, None]
        / 2
        * unit_vectors(rays["arrival_horizontal_angle"][direct], rays["arrival_vertical_angle"][direct])
    )
    return {int(i): p for i, p in zip(rays["object_id"][direct], positions)}


def plot_ray_geometry(reflections, config, axes=None):
    if axes is None:
        axes = plt.subplots(
            1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.2, 1]}
        )[1]
    top, side = axes
    truth = _direct_positions(reflection_arrays(reflections))
    top.plot(0, 0, "k^", markersize=10, label="sensor")
    for i, (oid, p) in enumerate(truth.items()):
        top.plot([0, p[0]], [0, p[1]], "-", color=f"C{i}")
        top.plot(*p[:2], "s", color=f"C{i}", markersize=9, label=f"object {oid}")
    fov = config.field_of_view_horizontal / 2
    for s in (-1, 1):
        top.plot([0, 70 * np.cos(s * fov)], [0, 70 * np.sin(s * fov)], "k:", lw=0.8)
    top.set_xlabel("x sensor (m)")
    top.set_ylabel("y sensor (m)")
    top.set_aspect("equal")
    top.set_title("Top view")
    top.legend(fontsize=8)
    top.grid(True, alpha=0.3)

    h = config.mounting_position.position.z
    oid, p = next(iter(truth.items()))
    d, z = np.hypot(*p[:2]), p[2]
    side.axhline(-h, color="saddlebrown", lw=2, label="ground")
    side.plot([0, d], [0, z], "-", color="C0", label="direct")
    x_bounce = d * h / (2 * h + z)  # specular point of the image path
    side.plot([d, x_bounce, 0], [z, -h, 0], "--", color="C0", label="via ground")
    side.plot(0, 0, "k^", markersize=10)
    side.plot(d, z, "s", color="C0", markersize=9)
    side.set_xlabel("horizontal distance (m)")
    side.set_ylabel("z sensor (m)")
    side.set_title(f"Side view, object {oid}")
    side.legend(fontsize=8)
    side.grid(True, alpha=0.3)
    return axes


def plot_object_channels(reflections, frequency: np.ndarray, ax=None):
    ax = _axes(ax, (10, 4))
    rays = reflection_arrays(reflections)
    for i, oid in enumerate(_direct_positions(rays)):
        for n, style in ((1, ":"), (2, "-")):
            paths = (rays["object_id"] == oid) & (rays["number_of_interactions"] <= n)
            phase = np.exp(-2j * np.pi * frequency[:, None] * rays["time_of_flight"][paths])
            h_hh = phase @ rays["polarimetric_response"][paths, 0, 0]
            ax.plot(
                frequency / 1e9,
                20 * np.log10(np.abs(h_hh)),
                style,
                color=f"C{i}",
                label=f"object {oid}, {'direct only' if n == 1 else 'direct + ground'}",
            )
    ax.set_xlabel("frequency (GHz)")
    ax.set_ylabel("|H_HH| (dB)")
    ax.set_title("Antenna-free HH channel per object")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=3)
    return ax


def plot_array(antenna, title: str = "", ax=None):
    ax = _axes(ax, (10, 2.6))
    p = array_positions(antenna)
    ax.plot(p["tx"] * 1e3, [2] * len(p["tx"]), "rv", markersize=10, label="Tx")
    ax.plot(p["rx"] * 1e3, [1] * len(p["rx"]), "b^", markersize=10, label="Rx")
    ax.plot(
        p["virtual"] * 1e3,
        [0] * len(p["virtual"]),
        "ks",
        markersize=7,
        label="virtual (p_tx + p_rx)",
    )
    ax.set_yticks([0, 1, 2], ["virtual", "Rx", "Tx"])
    ax.set_ylim(-0.7, 2.7)
    ax.set_xlabel("y in sensor frame (mm)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    ax.figure.tight_layout()
    return ax


def plot_antenna_diagram(diagram, axes=None):
    if axes is None:
        axes = plt.subplots(1, 2, figsize=(13, 4.5))[1]
    cut, grid = axes
    h, v, response = diagram_grid(diagram)
    cut.plot(np.degrees(h), response[:, len(v) // 2])
    cut.set_title("rx_antenna_diagram, vertical_angle = 0")
    cut.set_xlabel("horizontal_angle (deg)")
    cut.set_ylabel("response (dB)")
    cut.set_ylim(-40, 25)
    cut.grid(True, alpha=0.3)
    im = grid.imshow(
        response.T,
        origin="lower",
        aspect="auto",
        vmin=-20,
        vmax=20,
        extent=[
            np.degrees(h[0]),
            np.degrees(h[-1]),
            np.degrees(v[0]),
            np.degrees(v[-1]),
        ],
    )
    grid.set_title("rx_antenna_diagram")
    grid.set_xlabel("horizontal_angle (deg)")
    grid.set_ylabel("vertical_angle (deg)")
    grid.figure.colorbar(im, ax=grid, label="response (dB)")
    grid.figure.tight_layout()
    return axes


def plot_chirp_sequence(config: dict, chirps: int = 7, ax=None):
    ax = _axes(ax, (11, 3.5))
    period = config["chirp_repetition_interval"]
    for m in range(chirps):
        t0, t = m * period, np.array([0, config["ramp_duration"]])
        tx_id = config["tx_sequence"][m % len(config["tx_sequence"])]
        ax.plot(
            (t0 + t) * 1e6,
            (config["start_frequency"] + config["slope"] * t) / 1e9,
            color=f"C{tx_id}",
            lw=2,
            label=f"Tx{tx_id}" if m < len(config["tx_sequence"]) else None,
        )
        adc = (
            t0
            + config["adc_start_time"]
            + np.array([0, config["samples_per_chirp"] / config["sample_rate"]])
        )
        ax.axvspan(
            *(adc * 1e6),
            color="grey",
            alpha=0.15,
            label="ADC window" if m == 0 else None,
        )
    ax.set_xlabel("time (us)")
    ax.set_ylabel("frequency (GHz)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", ncol=4)
    ax.set_title(f"FMCW TDM-MIMO chirp sequence (first {chirps} chirps)")
    return ax


def plot_beat_signal(frame, noise_free, ax=None):
    ax = _axes(ax, (7, 4))
    t = axis_values(frame.fast_axis)
    ax.plot(t * 1e6, cpi_cube(frame)[0, 0].real * 1e6, lw=0.8, label="with noise")
    ax.plot(t * 1e6, cpi_cube(noise_free)[0, 0].real * 1e6, lw=1.2, label="noise free")
    ax.set_xlabel("fast time (us)")
    ax.set_ylabel("Re s (µ√W)")
    ax.set_title("Beat signal, Rx0, chirp 0 (Tx0)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_range_profile(frame, expected: list[ExpectedPosition], ax=None):
    ax = _axes(ax, (7, 4))
    ax.plot(
        axis_values(frame.fast_axis),
        10 * np.log10(np.mean(np.abs(cpi_cube(frame)[0]) ** 2, axis=0) / frame.noise_power),
    )
    for e in expected:
        ax.axvline(e.distance, color="k", ls=":")
    c = frame.channel[0]
    ax.set_xlabel("range (m)")
    ax.set_ylabel("SNR (dB)")
    ax.grid(True, alpha=0.3)
    ax.set_title(
        f"Range profile, Tx{c.tx_index}-Rx{c.rx_index} (expected ranges dotted)"
    )
    return ax


def plot_virtual_phase(frame, target: ExpectedPosition, ax=None):
    ax = _axes(ax, (8, 3.5))
    range_bin = int(round((target.distance - frame.fast_axis.offset) / frame.fast_axis.spacing))
    phase = np.unwrap(np.angle(cpi_cube(frame)[:, 0, range_bin]))
    ax.plot([c.position.y * 1e3 for c in frame.channel], np.degrees(phase), "o-")
    ax.set_xlabel("virtual phase centre y (mm)")
    ax.set_ylabel("phase (deg)")
    ax.grid(True, alpha=0.3)
    ax.set_title(
        f"Phase across virtual channels, range bin of object {target.object_id} "
        f"(azimuth {np.degrees(target.azimuth):.1f} deg)"
    )
    return ax


def plot_pmcw_codes(waveform: dict, code_correlation_db: np.ndarray, axes=None):
    if axes is None:
        axes = plt.subplots(1, 2, figsize=(14, 3.8))[1]
    chips, correlation = axes
    codes, bins = np.array(waveform["codes"]), waveform["range_bins"]
    for i in range(len(codes)):
        chips.step(
            np.arange(60), codes[i, :60] * 0.4 + 1.2 * i, where="post", color=f"C{i}"
        )
    chips.set_yticks(
        1.2 * np.arange(len(codes)), [f"Tx{t}" for t in waveform["tx_indices"]]
    )
    chips.set_xlabel("chip")
    chips.set_title("First 60 chips of the Tx codes")
    for i, db in enumerate(code_correlation_db):
        correlation.plot(db, color=f"C{i}", label=f"Tx{waveform['tx_indices'][i]}")
        correlation.axvspan(i * bins, (i + 1) * bins, color=f"C{i}", alpha=0.08)
    correlation.set_ylim(-70, 3)
    correlation.set_xlabel("lag (chips)")
    correlation.set_ylabel("dB")
    correlation.legend(ncol=4, fontsize=8)
    correlation.set_title("Correlation of every Tx code with the Tx0 code")
    correlation.figure.tight_layout()
    return axes


def plot_pmcw_raw(raw, ax=None):
    ax = _axes(ax, (7, 4))
    ax.plot(axis_values(raw.fast_axis)[:200] * 1e9, cpi_cube(raw)[0, 0, :200].real * 1e6, lw=0.8)
    ax.set_xlabel("fast time (ns)")
    ax.set_ylabel("Re s (µ√W)")
    ax.set_title("Raw accumulated chips, Rx0, slow sample 0")
    ax.grid(True, alpha=0.3)
    return ax


def plot_pmcw_correlation(lag_range: np.ndarray, snr_db: np.ndarray, waveform: dict, ax=None):
    ax = _axes(ax, (7, 4))
    bins, resolution, power = waveform["range_bins"], waveform["range_resolution"], snr_db
    ax.plot(lag_range, power, lw=0.8)
    for i in range(len(waveform["tx_indices"])):
        ax.axvspan(
            i * bins * resolution,
            (i + 1) * bins * resolution,
            color=f"C{i}",
            alpha=0.08,
        )
        ax.text(
            (i + 0.5) * bins * resolution,
            power.max() + 2,
            f"Tx{waveform['tx_indices'][i]}",
            ha="center",
        )
    ax.set_xlabel("lag x range resolution (m)")
    ax.set_ylabel("SNR (dB)")
    ax.set_ylim(top=power.max() + 6)
    ax.set_title("Rx0 correlated with the Tx0 code: four Tx windows")
    ax.grid(True, alpha=0.3)
    return ax


def plot_range_doppler(
    ranges,
    range_rates,
    snr,
    sensor_data=None,
    expected=(),
    ax=None,
    title: str = "",
    xlim=None,
    ylim=None,
):
    ax = _axes(ax, (11, 5))
    snr_db = 10 * np.log10(snr)
    im = ax.pcolormesh(
        ranges,
        range_rates,
        snr_db,
        vmin=0,
        vmax=snr_db.max(),
        shading="nearest",
        cmap="viridis",
    )
    if sensor_data is not None:
        detections = _detections(sensor_data)
        ax.plot([d.position.distance for d in detections], [-d.radial_velocity for d in detections], "wo",
                mfc="none", markersize=12, label="detections")
    if expected:
        ax.plot([e.distance for e in expected], [-e.radial_velocity for e in expected], "r+", markersize=12,
                label="expected")
    ax.set_xlabel("range (m)")
    ax.set_ylabel("range rate (m/s)")
    ax.set_title(title)
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper right")
    ax.figure.colorbar(im, ax=ax, label="SNR (dB)")
    return ax


def plot_spectrum(ax, azimuth, power, *args, **kwargs):
    return ax.plot(
        np.degrees(azimuth), 10 * np.log10(power / power.max()), *args, **kwargs
    )[0]


def format_spectrum_axes(ax, title: str = "", ylim=(-30, 1), ncol: int = 3):
    ax.set_ylim(*ylim)
    ax.set_xlabel("azimuth (deg)")
    ax.set_ylabel("normalised Bartlett (dB)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=ncol, loc='center')
    return ax
