from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv


@dataclass(frozen=True)
class AssetUiHint:
    display_name: str
    asset_search_name: str


ASSET_UI_HINTS: dict[str, AssetUiHint] = {
    "g1": AssetUiHint(display_name="G1 humanoid", asset_search_name="g1"),
    "G1": AssetUiHint(display_name="G1 humanoid", asset_search_name="g1"),
}


@dataclass(frozen=True)
class SceneModelNames:
    bodies: set[str]
    joints: set[str]
    actuators: set[str]
    sites: set[str]
    sensors: set[str]


@dataclass(frozen=True)
class SuffixTemplate:
    model_name: str
    joints: list[str]
    actuators: list[str]
    sites: list[str]
    bodies: list[str]
    sensors: list[str]


@dataclass(frozen=True)
class InstanceMatch:
    prefix: str
    matched_names: dict[str, dict[str, str]]
    missing_suffixes: dict[str, list[str]]

    @property
    def is_complete(self) -> bool:
        return all(len(missing) == 0 for missing in self.missing_suffixes.values())

    @property
    def agent_name(self) -> str:
        return self.prefix


@dataclass(frozen=True)
class SceneScanReport:
    model_name: str
    complete_matches: list[InstanceMatch]
    partial_matches: list[InstanceMatch]
    scene_names: SceneModelNames

    @property
    def detected_count(self) -> int:
        return len(self.complete_matches)


class SceneProbeEnv(OrcaGymLocalEnv):
    """Minimal OrcaGym env used only for reading model dictionaries."""

    _headless = True
    _render_mode = "none"
    _is_subenv = True


def scan_scene_for_template(
    orcagym_addr: str,
    time_step: float,
    template: SuffixTemplate,
) -> SceneScanReport:
    scene_names = probe_scene_model(orcagym_addr=orcagym_addr, time_step=time_step)
    return match_robot_instances(template, scene_names)


def probe_scene_model(orcagym_addr: str, time_step: float) -> SceneModelNames:
    probe_env = SceneProbeEnv(
        frame_skip=1,
        orcagym_addr=orcagym_addr,
        agent_names=["SceneProbe"],
        time_step=time_step,
    )
    try:
        site_dict = probe_env.model.get_site_dict() if hasattr(probe_env.model, "get_site_dict") else {}
        sensor_dict = getattr(probe_env.model, "_sensor_dict", {})
        return SceneModelNames(
            bodies=set(probe_env.model.get_body_names()),
            joints=set(probe_env.model.get_joint_dict().keys()),
            actuators=set(probe_env.model.get_actuator_dict().keys()),
            sites=set(site_dict.keys()),
            sensors=set(sensor_dict.keys()),
        )
    finally:
        probe_env.close()


def build_suffix_template(
    model_name: str,
    joints: Iterable[str] = (),
    actuators: Iterable[str] = (),
    sites: Iterable[str] = (),
    bodies: Iterable[str] = (),
    sensors: Iterable[str] = (),
) -> SuffixTemplate:
    return SuffixTemplate(
        model_name=model_name,
        joints=list(joints),
        actuators=list(actuators),
        sites=list(sites),
        bodies=list(bodies),
        sensors=list(sensors),
    )


def match_robot_instances(template: SuffixTemplate, scene_names: SceneModelNames) -> SceneScanReport:
    category_available_names = {
        "joints": scene_names.joints,
        "actuators": scene_names.actuators,
        "sites": scene_names.sites,
        "bodies": scene_names.bodies,
        "sensors": scene_names.sensors,
    }
    category_required_suffixes = {
        "joints": template.joints,
        "actuators": template.actuators,
        "sites": template.sites,
        "bodies": template.bodies,
        "sensors": template.sensors,
    }
    category_matches = {
        category: _collect_matches_by_prefix(required_suffixes, category_available_names[category])
        for category, required_suffixes in category_required_suffixes.items()
        if required_suffixes
    }

    candidate_prefixes = set()
    for matches_by_prefix in category_matches.values():
        candidate_prefixes.update(matches_by_prefix.keys())

    instance_matches = [
        _build_instance_match(prefix, category_matches, template)
        for prefix in sorted(candidate_prefixes)
    ]
    return SceneScanReport(
        model_name=template.model_name,
        complete_matches=[match for match in instance_matches if match.is_complete],
        partial_matches=[match for match in instance_matches if not match.is_complete],
        scene_names=scene_names,
    )


def require_complete_matches(
    report: SceneScanReport,
    *,
    min_count: int = 1,
    max_count: int | None = None,
    allow_empty_prefix: bool = False,
    orcagym_addr: str | None = None,
) -> list[InstanceMatch]:
    _log_scene_scan_report(report)

    if not report.complete_matches:
        if report.partial_matches:
            first_partial = report.partial_matches[0]
            detail = []
            for category, missing_suffixes in first_partial.missing_suffixes.items():
                if missing_suffixes:
                    detail.append(f"{category} missing {missing_suffixes[:8]}")
            _emit_terminal_hint(
                _build_ui_hint_message(
                    report,
                    problem_message=f"{report.model_name} joints or actuators do not fully match.",
                    min_count=min_count,
                    max_count=max_count,
                )
            )
            raise ValueError(f"{report.model_name} scene match is incomplete: {'; '.join(detail)}")

        _emit_terminal_hint(
            _build_ui_hint_message(
                report,
                problem_message=f"No complete {report.model_name} instance was found in the current layout.",
                min_count=min_count,
                max_count=max_count,
            )
        )
        raise ValueError(f"Cannot find robot model in scene: {report.model_name}")

    if len(report.complete_matches) < min_count:
        raise ValueError(
            f"{report.model_name} complete match count is too small: "
            f"need {min_count}, found {len(report.complete_matches)}"
        )

    if max_count is not None and len(report.complete_matches) > max_count:
        raise ValueError(
            f"{report.model_name} complete match count is too large: "
            f"max {max_count}, found {len(report.complete_matches)} "
            f"{[match.agent_name for match in report.complete_matches]}"
        )

    if not allow_empty_prefix:
        unnamed_matches = [match for match in report.complete_matches if not match.prefix]
        if unnamed_matches:
            raise ValueError(
                f"{report.model_name} contains an un-prefixed complete instance. "
                "The runtime cannot map it to OrcaGym agent_names."
            )

    return report.complete_matches


def ordered_match_names(match: InstanceMatch, category: str, suffixes: Iterable[str]) -> list[str]:
    category_matches = match.matched_names.get(category, {})
    return [category_matches[suffix] for suffix in suffixes if suffix in category_matches]


def _split_tokens(name: str) -> list[str]:
    return [token for token in name.split("_") if token]


def _match_prefix(full_name: str, suffix: str) -> str | None:
    full_tokens = _split_tokens(full_name)
    suffix_tokens = _split_tokens(suffix)
    if not suffix_tokens or len(full_tokens) < len(suffix_tokens):
        return None
    if full_tokens[-len(suffix_tokens) :] != suffix_tokens:
        return None
    return "_".join(full_tokens[: -len(suffix_tokens)])


def _collect_matches_by_prefix(required_suffixes: list[str], available_names: set[str]) -> dict[str, dict[str, str]]:
    matches_by_prefix: dict[str, dict[str, str]] = defaultdict(dict)
    for suffix in required_suffixes:
        for full_name in available_names:
            prefix = _match_prefix(full_name, suffix)
            if prefix is not None:
                matches_by_prefix[prefix][suffix] = full_name
    return dict(matches_by_prefix)


def _build_instance_match(
    prefix: str,
    category_matches: dict[str, dict[str, dict[str, str]]],
    template: SuffixTemplate,
) -> InstanceMatch:
    required_by_category = {
        "joints": template.joints,
        "actuators": template.actuators,
        "sites": template.sites,
        "bodies": template.bodies,
        "sensors": template.sensors,
    }
    missing_suffixes = {}
    matched_names = {}
    for category, required_suffixes in required_by_category.items():
        matched = category_matches.get(category, {}).get(prefix, {})
        matched_names[category] = matched
        missing_suffixes[category] = [suffix for suffix in required_suffixes if suffix not in matched]
    return InstanceMatch(prefix=prefix, matched_names=matched_names, missing_suffixes=missing_suffixes)


def _log_scene_scan_report(report: SceneScanReport) -> None:
    if report.complete_matches:
        print(
            f"[orca_rl.scene] {report.model_name}: complete matches "
            f"{[match.agent_name for match in report.complete_matches]}"
        )
    else:
        print(f"[orca_rl.scene] {report.model_name}: no complete match")

    for match in report.partial_matches[:8]:
        missing_parts = []
        for category, missing_suffixes in match.missing_suffixes.items():
            if missing_suffixes:
                missing_parts.append(f"{category}={missing_suffixes[:6]}")
        if missing_parts:
            print(f"[orca_rl.scene] partial {match.prefix or '<root>'}: {'; '.join(missing_parts)}")


def _build_ui_hint_message(
    report: SceneScanReport,
    *,
    problem_message: str,
    min_count: int,
    max_count: int | None,
) -> str:
    hint = ASSET_UI_HINTS.get(report.model_name)
    message_parts = [problem_message]
    if hint is not None:
        message_parts.append(f"Search and drag asset `{hint.asset_search_name}` into the OrcaLab layout.")
        message_parts.append("If it is not listed, check whether the corresponding asset package is subscribed.")
    if max_count == 1:
        message_parts.append("Keep exactly one matching instance in the layout.")
    elif max_count is not None:
        message_parts.append(f"Keep at most {max_count} matching instances in the layout.")
    elif min_count > 1:
        message_parts.append(f"At least {min_count} complete matching instances are required.")
    else:
        message_parts.append("At least one complete matching instance is required.")
    return " ".join(message_parts)


def _emit_terminal_hint(message: str) -> None:
    print(f"[orca_rl.scene hint] {message}")
