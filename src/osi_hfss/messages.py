

def describe(message) -> str:
    """One line on what an interface message carries."""
    match type(message).__name__:
        case "SensorView":
            view = message.radar_sensor_view[0]
            return (f"{len(view.reflection)} reflections at {view.view_configuration.emitter_frequency / 1e9:.2f} GHz, "
                    f"{len(message.global_ground_truth.moving_object)} objects in the ground truth")
        case "RadarSensorView":
            return (f"{len(message.reflection)} antenna-free reflections at "
                    f"{message.view_configuration.emitter_frequency / 1e9:.2f} GHz")
        case "AntennaModel":
            return f"{len(message.element)} elements, {len(message.element_type)} element types (embedded patterns)"
        case "RadarCpiData":
            frame = message.cpi_frame[0]
            kind = type(frame.channel[0]).Kind.Name(frame.channel[0].kind)
            domains = [type(a).Domain.Name(a.domain) for a in (frame.fast_axis, frame.slow_axis)]
            return (f"{len(message.cpi_frame)} CPI frame, {len(frame.channel)} channels {kind}, "
                    f"{frame.slow_axis.size} x {frame.fast_axis.size} cells ({domains[1]} x {domains[0]})")
        case "SensorData":
            return f"{sum(len(r.detection) for r in message.feature_data.radar_sensor)} radar detections"
    return type(message).__name__


def over_osi(message, verbose: bool = True):
    """Serialize and parse again: only the OSI message crosses the module boundary."""
    received = type(message).FromString(message.SerializeToString())
    if verbose:
        print(f"  {type(message).__name__:16s} {message.ByteSize() / 1e3:10.1f} kB")
    return received


# Fields that are large or not needed to understand a message; `preview` shows their size only.
BULKY_FIELDS = {"antenna_model", "global_ground_truth", "tx_antenna_diagram", "rx_antenna_diagram", "data"}


def preview(message, max_repeated: int = 2, indent: str = "") -> str:
    """Readable text form of an OSI message for the examples.

    Like protobuf's text format, but floats are shortened, repeated fields
    show their first `max_repeated` entries and `BULKY_FIELDS` only their size.
    """
    lines = []
    for field, value in message.ListFields():
        name = f"{indent}{field.name}"
        repeated = hasattr(value, "__len__") and not isinstance(value, (str, bytes))
        if field.name in BULKY_FIELDS:
            size = len(value) if isinstance(value, bytes) else (
                sum(v.ByteSize() for v in value) if repeated else value.ByteSize())
            lines.append(f"{name}: <{size / 1e6:.2f} MB>")
            continue
        items = list(value) if repeated else [value]
        for item in items[:max_repeated]:
            if hasattr(item, "ListFields"):
                lines.append(f"{name} {{")
                lines.append(preview(item, max_repeated, indent + "  "))
                lines.append(f"{indent}}}")
            elif field.enum_type is not None:
                lines.append(f"{name}: {field.enum_type.values_by_number[item].name}")
            elif isinstance(item, float):
                lines.append(f"{name}: {item:.6g}")
            else:
                lines.append(f"{name}: {item}")
        if len(items) > max_repeated:
            lines.append(f"{name}: ... {len(items) - max_repeated} more")
    return "\n".join(line for line in lines if line)
