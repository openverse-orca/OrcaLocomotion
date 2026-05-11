from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import math
import tempfile
import xml.etree.ElementTree as ET


_REF_ATTRS = {
    "body",
    "body1",
    "body2",
    "joint",
    "joint1",
    "joint2",
    "site",
    "site1",
    "site2",
    "objname",
    "geom",
    "geom1",
    "geom2",
    "tendon",
}


def build_local_mjcf_batch(
    *,
    source_xml_path: str | Path,
    agent_names: list[str],
    spacing: float = 2.0,
    output_dir: str | Path | None = None,
) -> str:
    """Create a local batched MuJoCo XML by cloning one robot body per agent."""

    if not agent_names:
        raise ValueError("At least one agent name is required to build a local MJCF batch.")

    source_path = Path(source_xml_path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Local MJCF source does not exist: {source_path}")

    tree = ET.parse(source_path)
    root = tree.getroot()
    root.set("model", f"{root.get('model', 'orca_rl')}_batch_{len(agent_names)}")
    _absolutize_asset_files(root, source_path)

    target_worldbody, robot_templates = _remove_robot_templates(root)
    actuator_section, actuator_templates = _collect_and_clear_section(root, "actuator")
    sensor_section, sensor_templates = _collect_and_clear_section(root, "sensor")
    contact_section, contact_templates = _collect_and_clear_section(root, "contact")
    tendon_section, tendon_templates = _collect_and_clear_section(root, "tendon")
    equality_section, equality_templates = _collect_and_clear_section(root, "equality")
    _remove_sections(root, "keyframe")

    grid_width = int(math.ceil(math.sqrt(len(agent_names))))
    x0 = -0.5 * spacing * (grid_width - 1)
    y0 = -0.5 * spacing * (grid_width - 1)
    for index, agent_name in enumerate(agent_names):
        name_map: dict[str, str] = {}
        bodies = [deepcopy(body) for body in robot_templates]
        for body in bodies:
            _prefix_named_elements(body, agent_name, name_map)
            _rewrite_refs(body, name_map)
            _offset_root_body(body, x0 + spacing * (index % grid_width), y0 + spacing * (index // grid_width))
            target_worldbody.append(body)

        _append_prefixed_templates(tendon_section, tendon_templates, agent_name, name_map)
        _append_prefixed_templates(equality_section, equality_templates, agent_name, name_map)
        _append_prefixed_templates(contact_section, contact_templates, agent_name, name_map, keep_if_ref_touched=True)
        _append_prefixed_templates(actuator_section, actuator_templates, agent_name, name_map)
        _append_prefixed_templates(sensor_section, sensor_templates, agent_name, name_map)

    output_root = Path(output_dir).expanduser() if output_dir is not None else Path(tempfile.gettempdir()) / "orca_rl_mjcf"
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"{source_path.stem}_batch_{len(agent_names)}.xml"
    tree.write(output_path, encoding="utf-8", xml_declaration=False)
    return str(output_path.resolve())


def resolve_existing_xml_path(path_or_auto: str | Path, candidates: list[str | Path]) -> str:
    if str(path_or_auto).strip().lower() not in {"", "auto"}:
        path = Path(path_or_auto).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Configured local XML path does not exist: {path}")
        return str(path)

    for candidate in candidates:
        if not str(candidate).strip():
            continue
        path = Path(candidate).expanduser()
        if path.exists():
            return str(path.resolve())
    raise FileNotFoundError(
        "Cannot find a local G1 MJCF source. Set `scene_binding.local_xml_path` or `ORCA_RL_G1_XML`."
    )


def _absolutize_asset_files(root: ET.Element, source_path: Path) -> None:
    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir") if compiler is not None else None
    mesh_root = source_path.parent / meshdir if meshdir else source_path.parent
    if compiler is not None and meshdir and not Path(meshdir).is_absolute():
        compiler.set("meshdir", str(mesh_root.resolve()))

    for element in root.findall(".//*[@file]"):
        file_value = element.get("file")
        if not file_value:
            continue
        file_path = Path(file_value)
        if file_path.is_absolute():
            continue
        base = mesh_root if element.tag == "mesh" else source_path.parent
        element.set("file", str((base / file_path).resolve()))


def _remove_robot_templates(root: ET.Element) -> tuple[ET.Element, list[ET.Element]]:
    worldbodies = root.findall("worldbody")
    if not worldbodies:
        raise ValueError("Local MJCF source has no <worldbody> section.")

    target_worldbody: ET.Element | None = None
    robot_templates: list[ET.Element] = []
    for worldbody in worldbodies:
        for child in list(worldbody):
            if child.tag == "body" and _looks_like_robot_root(child):
                if target_worldbody is None:
                    target_worldbody = worldbody
                robot_templates.append(deepcopy(child))
                worldbody.remove(child)

    if target_worldbody is None or not robot_templates:
        raise ValueError("Local MJCF source has no cloneable robot root body with a free joint.")
    return target_worldbody, robot_templates


def _looks_like_robot_root(body: ET.Element) -> bool:
    for element in body.iter():
        if element.tag == "freejoint":
            return True
        if element.tag == "joint" and element.get("type") == "free":
            return True
    return False


def _collect_and_clear_section(root: ET.Element, section_name: str) -> tuple[ET.Element, list[ET.Element]]:
    sections = root.findall(section_name)
    if sections:
        first = sections[0]
    else:
        first = ET.SubElement(root, section_name)

    templates: list[ET.Element] = []
    for section in sections:
        templates.extend(deepcopy(list(section)))
        for child in list(section):
            section.remove(child)
    for section in sections[1:]:
        root.remove(section)
    return first, templates


def _remove_sections(root: ET.Element, section_name: str) -> None:
    for section in root.findall(section_name):
        root.remove(section)


def _prefix_named_elements(element: ET.Element, prefix: str, name_map: dict[str, str]) -> None:
    for child in element.iter():
        name = child.get("name")
        if name:
            prefixed = f"{prefix}_{name}"
            name_map[name] = prefixed
            child.set("name", prefixed)


def _rewrite_refs(element: ET.Element, name_map: dict[str, str]) -> bool:
    touched = False
    for child in element.iter():
        for attr in _REF_ATTRS:
            value = child.get(attr)
            if value in name_map:
                child.set(attr, name_map[value])
                touched = True
    return touched


def _offset_root_body(body: ET.Element, dx: float, dy: float) -> None:
    pos = _parse_float_list(body.get("pos"), default=[0.0, 0.0, 0.0])
    pos[0] += dx
    pos[1] += dy
    body.set("pos", _format_float_list(pos))


def _append_prefixed_templates(
    section: ET.Element,
    templates: list[ET.Element],
    prefix: str,
    name_map: dict[str, str],
    *,
    keep_if_ref_touched: bool = False,
) -> None:
    for template in templates:
        clone = deepcopy(template)
        _prefix_named_elements(clone, prefix, name_map)
        ref_touched = _rewrite_refs(clone, name_map)
        if keep_if_ref_touched and not ref_touched:
            continue
        section.append(clone)


def _parse_float_list(text: str | None, *, default: list[float]) -> list[float]:
    if not text:
        return list(default)
    values = [float(value) for value in text.split()]
    while len(values) < len(default):
        values.append(default[len(values)])
    return values


def _format_float_list(values: list[float]) -> str:
    return " ".join(f"{value:.8g}" for value in values)
